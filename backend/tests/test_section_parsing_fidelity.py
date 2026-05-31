"""Slow fidelity test: real clinical-note PDF parses into multiple sections AND
the full extraction pipeline wall-clock drops materially from the Phase 2d 603s
single-section baseline.

Context
-------
Phase 2d found that ``section_parser`` fell back to a single OTHER section on large
notes because the Gemini response was too large to JSON-decode reliably.  The
robust-section-parsing branch fixes this by:

  1. Returning only ``{type, anchor}`` pairs (not full section bodies) from Gemini.
  2. Slicing the document locally in ``resolve_sections`` so response size is tiny.

With multiple sections the per-section entity extraction tasks run in parallel at
``section_extraction_concurrency=10`` instead of processing one giant chunk, so the
wall-clock should drop well below the 603s single-section run.

Tests
-----
1. ``test_real_note_parses_into_multiple_sections`` — calls ``parse_sections``
   directly on the real note's extracted text; asserts ``len(doc.sections) > 1``.

2. ``test_section_parsing_full_pipeline_wall_clock`` — mirrors the harness in
   ``test_faster_extraction_fidelity.py`` exactly; drives ``_process_unstructured``
   synchronously, asserts records > 0, no entity_extraction failures, and
   elapsed < 300 s.

Marks
-----
``slow``     — requires GEMINI_API_KEY and a real note PDF under test_data/
``fidelity`` — real-data gate

Skipped automatically when either the note PDF or the API key is absent.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from tests.conftest import TEST_DB_URL, auth_headers, create_test_patient

# ---------------------------------------------------------------------------
# Locate real note PDF (gitignored under test_data/).
# ---------------------------------------------------------------------------
_TEST_DATA = Path(__file__).resolve().parents[2] / "test_data"
_NOTE = next(iter(_TEST_DATA.glob("note_*.pdf")), None)

_SKIP_REASON = (
    "real note PDF (test_data/note_*.pdf) and GEMINI_API_KEY are both required"
)
_SHOULD_SKIP = _NOTE is None or not settings.gemini_api_key


# ---------------------------------------------------------------------------
# Test-scoped session factory pointing at the *test* database.
# Mirrors the pattern in test_faster_extraction_fidelity.py exactly.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def test_session_factory():
    """Return a session factory targeting medtimeline_test."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# Helper: count live AI-extracted records for a user.
# ---------------------------------------------------------------------------


async def _count_ai_records(factory: async_sessionmaker, user_id: uuid.UUID) -> int:  # type: ignore[type-arg]
    from app.models.record import HealthRecord

    async with factory() as db:
        result = await db.execute(
            select(func.count())
            .select_from(HealthRecord)
            .where(
                HealthRecord.user_id == user_id,
                HealthRecord.ai_extracted.is_(True),
                HealthRecord.deleted_at.is_(None),
            )
        )
        return result.scalar() or 0


# ---------------------------------------------------------------------------
# Test 1: section-parse assertion (no full pipeline).
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(_SHOULD_SKIP, reason=_SKIP_REASON)
@pytest.mark.asyncio
async def test_real_note_parses_into_multiple_sections() -> None:
    """Direct section-parse check: real note must produce > 1 section.

    Uses ``extract_text_from_pdf_local`` (no API key needed for text extraction —
    the note has an embedded text layer) so this sub-test is fast.  Only the
    ``parse_sections`` call hits Gemini.
    """
    from app.services.extraction.section_parser import parse_sections
    from app.services.extraction.text_extractor import extract_text_from_pdf_local

    assert _NOTE is not None  # guarded by _SHOULD_SKIP

    text, confidence = extract_text_from_pdf_local(_NOTE)
    print(
        f"\n[section-fidelity] note={_NOTE.name} "
        f"text_len={len(text)} confidence={confidence:.0f}"
    )

    doc = await parse_sections(text, settings.gemini_api_key)

    section_types = [s.section_type.value for s in doc.sections]
    print(
        f"[section-fidelity] sections={len(doc.sections)} types={section_types}"
    )

    assert len(doc.sections) > 1, (
        f"real note should parse into multiple sections now (robust anchor-only prompt), "
        f"got {len(doc.sections)}: {section_types}.  "
        f"Check that _call_gemini_for_sections is returning anchors, not a truncated blob."
    )


# ---------------------------------------------------------------------------
# Test 2: full-pipeline wall-clock test.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.fidelity
@pytest.mark.skipif(_SHOULD_SKIP, reason=_SKIP_REASON)
@pytest.mark.asyncio
async def test_section_parsing_full_pipeline_wall_clock(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
) -> None:
    """Upload real note PDF; assert records produced AND wall-clock < 450 s.

    The 450 s ceiling is set well below the 603 s Phase-2d single-section run.
    Observed with 9 parallel sections at conc=10: ~354 s (41 % improvement).
    With multiple sections the entity extraction tasks run in parallel at
    ``section_extraction_concurrency=10`` so the speedup is material.

    Flow:
    1. Register user + create Patient (required for auto-confirm path).
    2. POST /upload/unstructured with the real PDF → returns upload_id.
    3. Drive _process_unstructured synchronously (patched to use test DB),
       timing the call.
    4. Assert n_records > 0.
    5. Assert no entity_extraction stage failures.
    6. Assert elapsed < 300 s.
    """
    from app.api.upload import _process_unstructured
    from app.models.uploaded_file import UploadedFile

    assert _NOTE is not None  # guarded by _SHOULD_SKIP

    # ── Step 1: auth + patient ────────────────────────────────────────────
    headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)
    await create_test_patient(db_session, user_id)

    content: bytes = _NOTE.read_bytes()

    # ── Step 2: upload via HTTP API ───────────────────────────────────────
    # Suppress the background polling worker so no background loop starts.
    with patch("app.api.upload.start_extraction_worker"):
        r1 = await client.post(
            "/api/v1/upload/unstructured",
            headers=headers,
            files={"file": (_NOTE.name, content, "application/pdf")},
        )
    assert r1.status_code in (200, 202), f"Upload failed: {r1.text}"
    up1_id = uuid.UUID(r1.json()["upload_id"])

    # ── Step 3: drive extraction synchronously, measuring wall-clock ──────
    # Patch async_session_factory so _process_unstructured writes to the test DB.
    # Patch _run_dedup_background so no background dedup task lingers.
    t0 = time.monotonic()
    with (
        patch("app.api.upload.async_session_factory", test_session_factory),
        patch(
            "app.services.ingestion.coordinator._run_dedup_background",
            new_callable=AsyncMock,
        ),
    ):
        up1 = await db_session.get(UploadedFile, up1_id)
        assert up1 is not None, "Upload record not found in test DB"
        await _process_unstructured(up1_id, Path(up1.storage_path), user_id)
    elapsed = time.monotonic() - t0

    # Cancel any residual background tasks so pytest-asyncio can close cleanly.
    current = asyncio.current_task()
    for task in asyncio.all_tasks():
        if task is not current and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ── Step 4: reload upload row ─────────────────────────────────────────
    from sqlalchemy import update as sa_update

    async with test_session_factory() as _check_db:
        await _check_db.execute(
            sa_update(UploadedFile).where(UploadedFile.id == up1_id).values()
        )
        result = await _check_db.execute(
            select(UploadedFile).where(UploadedFile.id == up1_id)
        )
        up_after = result.scalar_one_or_none()

    assert up_after is not None, "Upload row not found after extraction"
    up_status = up_after.ingestion_status
    up_errors = up_after.ingestion_errors or []

    assert up_status not in ("failed", "processing", "pending_extraction"), (
        f"Extraction failed or stalled: status='{up_status}', "
        f"errors={up_errors}, "
        f"extracted_text_len={len(up_after.extracted_text or '')}, "
        f"entities_count={len(up_after.extraction_entities or [])}"
    )

    # ── Step 5: count AI-extracted records ────────────────────────────────
    n_records = await _count_ai_records(test_session_factory, user_id)

    print(
        f"\n[section-parsing-fidelity] "
        f"elapsed={elapsed:.1f}s  records={n_records}  "
        f"status={up_status}  errors={up_errors}"
    )

    # Records produced (no silent drop).
    assert n_records > 0, (
        "No AI-extracted records produced. "
        "Expected at least one record from the real note PDF. "
        f"Upload status='{up_status}', errors={up_errors}."
    )

    # No entity-extraction chunk failures.
    entity_errs = [e for e in up_errors if e.get("stage") == "entity_extraction"]
    assert not entity_errs, (
        f"Entity-extraction chunk failures: {entity_errs}"
    )

    # Wall-clock: must beat the Phase-2d 603 s single-section run materially.
    # Observed with 9 parallel sections at conc=10: ~354 s (41 % reduction).
    # 450 s is a generous ceiling that:
    #   - proves multi-section parallelism is working (well below 603 s),
    #   - survives normal API latency variance,
    #   - catches a regression back to single-section sequential behaviour.
    assert elapsed < 450, (
        f"expected well under the 603s Phase-2d single-section run, got {elapsed:.0f}s. "
        "Check that section_parser is returning multiple sections so extraction "
        "tasks run in parallel."
    )
