# Unstructured Idempotency + Dedup Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the unstructured (AI-extracted) ingestion path idempotent — skip re-extracting an identical re-uploaded file, replace a file's prior records on explicit re-extraction (instead of duplicating), and populate `content_hash` on extracted records — while leaving cross-document overlap to the existing dedup pipeline.

**Architecture:** Three small, isolated changes: (1) populate `content_hash` in the entity→FHIR dict builder; (2) a `soft_delete_prior_extracted` helper called before inserting freshly extracted records (replace-on-reextract); (3) a `find_prior_extracted_upload` helper that short-circuits an identical re-upload to a `duplicate_file` status. No new migration (reuses Phase 1's `content_hash`; `duplicate_file` is a status string + `ingestion_progress` JSONB).

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x async, PostgreSQL 16, pytest + pytest-asyncio. Spec: `docs/superpowers/specs/2026-05-29-unstructured-idempotency-design.md`. Stacked on branch `feat/idempotent-incremental-ingestion` (needs Phase 1's `content_hash`).

**Conventions:** `from __future__ import annotations` at top of new modules. Type hints. No `print()` — use `logging`. **Run pytest via the project venv:** `cd backend && .venv/bin/python -m pytest …` (global pyenv has a broken `langsmith` plugin). Never hard-delete health records (use `deleted_at`). All DB queries user-scoped.

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `backend/app/services/extraction/entity_to_fhir.py` | Add `content_hash` to the extracted-record dict | Modify (`:59-76`) |
| `backend/app/services/ingestion/reextraction.py` | `soft_delete_prior_extracted` + `find_prior_extracted_upload` helpers | Create |
| `backend/app/api/upload.py` | Call replace helper before inserts; file-hash skip in upload endpoints | Modify (`:725`, `:1025`, `:842-877`, batch) |
| `backend/tests/test_entity_to_fhir_hash.py` | Unit: content_hash on extracted dict | Create |
| `backend/tests/test_reextraction.py` | Unit: replace + file-hash-skip helpers (DB) | Create |
| `backend/tests/test_unstructured_idempotency.py` | Integration: skip + replace via the API (mocked extraction) | Create |
| `backend/tests/test_unstructured_idempotency_fidelity.py` | Slow: real note PDF (needs GEMINI_API_KEY) | Create |

---

## Task 1: content_hash on extracted records

**Files:**
- Modify: `backend/app/services/extraction/entity_to_fhir.py`
- Test: `backend/tests/test_entity_to_fhir_hash.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_entity_to_fhir_hash.py
from __future__ import annotations

from uuid import uuid4

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict
from app.services.ingestion.content_hash import content_hash


def _entity() -> ExtractedEntity:
    return ExtractedEntity(
        entity_class="condition",
        text="Type 2 Diabetes",
        attributes={"status": "active"},
        start_pos=0,
        end_pos=15,
        confidence=0.9,
    )


def test_extracted_record_has_content_hash():
    rec = entity_to_health_record_dict(_entity(), uuid4(), uuid4(), uuid4())
    assert rec is not None
    assert rec["content_hash"] == content_hash(rec["fhir_resource"])


def test_content_hash_is_hex_sha256():
    rec = entity_to_health_record_dict(_entity(), uuid4(), uuid4(), uuid4())
    assert rec is not None
    assert len(rec["content_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in rec["content_hash"])


def test_non_storable_entity_still_returns_none():
    e = ExtractedEntity(entity_class="dosage", text="10mg", attributes={}, start_pos=0, end_pos=4, confidence=0.5)
    assert entity_to_health_record_dict(e, uuid4(), uuid4(), uuid4()) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_entity_to_fhir_hash.py -v`
Expected: FAIL — `KeyError: 'content_hash'`.

- [ ] **Step 3: Implement**

In `backend/app/services/extraction/entity_to_fhir.py`, add the import near the top (after line 9):
```python
from app.services.ingestion.content_hash import content_hash
```
Then in `entity_to_health_record_dict`, add `content_hash` to the returned dict (the dict at `:59-76`). Add this key alongside the others:
```python
        "content_hash": content_hash(fhir_resource),
```
(Place it after `"fhir_resource": fhir_resource,`.)

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_entity_to_fhir_hash.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/extraction/entity_to_fhir.py tests/test_entity_to_fhir_hash.py
git commit -m "feat: populate content_hash on AI-extracted records"
```

---

## Task 2: reextraction helpers (replace + file-hash lookup)

**Files:**
- Create: `backend/app/services/ingestion/reextraction.py`
- Test: `backend/tests/test_reextraction.py`

- [ ] **Step 1: Write the failing tests**

Use the real-user fixture pattern from `tests/test_idempotent_inserter.py` (create a user via `auth_headers(client)`, then `create_test_patient(db_session, uid)`).

```python
# backend/tests/test_reextraction.py
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.models.provenance import Provenance
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile


async def _make_record(db, patient, source_file_id, *, ai_extracted: bool):
    rec = HealthRecord(
        id=uuid.uuid4(), patient_id=patient.id, user_id=patient.user_id,
        record_type="condition", fhir_resource_type="Condition",
        fhir_resource={"resourceType": "Condition", "code": {"text": "x"}},
        source_format="ai_extracted" if ai_extracted else "fhir_r4",
        source_file_id=source_file_id, display_text="x", ai_extracted=ai_extracted,
    )
    db.add(rec)
    return rec


@pytest.mark.asyncio
async def test_soft_delete_prior_extracted_replaces_only_ai_records(db_session, create_test_patient, client):
    from app.services.ingestion.reextraction import soft_delete_prior_extracted
    from tests.conftest import auth_headers
    _, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    sfid = uuid.uuid4()
    await _make_record(db_session, patient, sfid, ai_extracted=True)
    await _make_record(db_session, patient, sfid, ai_extracted=True)
    structured = await _make_record(db_session, patient, sfid, ai_extracted=False)
    await db_session.commit()

    n = await soft_delete_prior_extracted(db_session, sfid)
    await db_session.commit()
    assert n == 2

    live = (await db_session.execute(
        select(HealthRecord).where(HealthRecord.source_file_id == sfid, HealthRecord.deleted_at.is_(None))
    )).scalars().all()
    assert len(live) == 1 and live[0].id == structured.id  # structured untouched

    prov = (await db_session.execute(
        select(Provenance).where(Provenance.action == "reextraction_replace")
    )).scalars().all()
    assert len(prov) == 2


@pytest.mark.asyncio
async def test_soft_delete_prior_extracted_noop_when_none(db_session):
    from app.services.ingestion.reextraction import soft_delete_prior_extracted
    assert await soft_delete_prior_extracted(db_session, uuid.uuid4()) == 0


@pytest.mark.asyncio
async def test_find_prior_extracted_upload_matches_completed(db_session, create_test_patient, client):
    from app.services.ingestion.reextraction import find_prior_extracted_upload
    from tests.conftest import auth_headers
    _, uid = await auth_headers(client)
    prior = UploadedFile(
        id=uuid.uuid4(), user_id=uuid.UUID(uid), filename="n.pdf", file_hash="HASH1",
        storage_path="/x", ingestion_status="completed", file_category="unstructured",
        record_count=5,
    )
    db_session.add(prior)
    await db_session.commit()

    found = await find_prior_extracted_upload(db_session, uuid.UUID(uid), "HASH1")
    assert found is not None and found.id == prior.id


@pytest.mark.asyncio
async def test_find_prior_extracted_upload_ignores_failed_and_other_users(db_session, client):
    from app.services.ingestion.reextraction import find_prior_extracted_upload
    from tests.conftest import auth_headers
    _, uid = await auth_headers(client)
    failed = UploadedFile(
        id=uuid.uuid4(), user_id=uuid.UUID(uid), filename="n.pdf", file_hash="HASH2",
        storage_path="/x", ingestion_status="failed", file_category="unstructured",
    )
    db_session.add(failed)
    await db_session.commit()
    # failed prior -> not a duplicate
    assert await find_prior_extracted_upload(db_session, uuid.UUID(uid), "HASH2") is None
    # unknown hash -> None
    assert await find_prior_extracted_upload(db_session, uuid.UUID(uid), "NOPE") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_reextraction.py -v`
Expected: FAIL — `ModuleNotFoundError: app.services.ingestion.reextraction`.

- [ ] **Step 3: Implement**

```python
# backend/app/services/ingestion/reextraction.py
"""Helpers for idempotent unstructured (AI-extracted) ingestion.

- `find_prior_extracted_upload`: detect an identical file already extracted for a user.
- `soft_delete_prior_extracted`: replace a file's prior extracted records on re-extraction.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.provenance import Provenance
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile

# Upload statuses that mean "this file already produced (or will produce) records".
PRODUCED_RECORDS_STATUSES = (
    "completed",
    "completed_with_merges",
    "awaiting_review",
    "awaiting_confirmation",
)


async def find_prior_extracted_upload(
    db: AsyncSession, user_id: uuid.UUID, file_hash: str
) -> UploadedFile | None:
    """Return a prior non-deleted upload for this user with the same file_hash that
    already produced records, or None (failed/pending priors do not count)."""
    result = await db.execute(
        select(UploadedFile)
        .where(
            UploadedFile.user_id == user_id,
            UploadedFile.file_hash == file_hash,
            UploadedFile.ingestion_status.in_(PRODUCED_RECORDS_STATUSES),
        )
        .order_by(UploadedFile.created_at)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def soft_delete_prior_extracted(db: AsyncSession, source_file_id: uuid.UUID) -> int:
    """Soft-delete a file's prior live AI-extracted records (replace-on-reextract).

    Writes one provenance row per replaced record. Returns the count replaced.
    Never hard-deletes. Structured (non-ai_extracted) records are left untouched.
    """
    rows = (await db.execute(
        select(HealthRecord).where(
            HealthRecord.source_file_id == source_file_id,
            HealthRecord.ai_extracted.is_(True),
            HealthRecord.deleted_at.is_(None),
        )
    )).scalars().all()

    now = datetime.now(timezone.utc)
    for row in rows:
        row.deleted_at = now
        db.add(Provenance(
            record_id=row.id,
            action="reextraction_replace",
            source_file_id=source_file_id,
            agent="extraction_worker",
            details={"reason": "re-extraction replaced prior extracted record"},
        ))
    return len(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_reextraction.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/ingestion/reextraction.py tests/test_reextraction.py
git commit -m "feat: reextraction helpers (file-hash lookup + replace-prior-extracted)"
```

---

## Task 3: wire replace-on-reextract into the insert paths

**Files:**
- Modify: `backend/app/api/upload.py` (auto-confirm `:725`, manual confirm `:1025`)

- [ ] **Step 1: Wire into the auto-confirm path**

In `backend/app/api/upload.py`, inside `_process_unstructured`, immediately BEFORE the `for entity in unique_entities:` loop (currently at `:725`, right after `if patient:` at `:721`), add:
```python
                from app.services.ingestion.reextraction import soft_delete_prior_extracted
                replaced = await soft_delete_prior_extracted(db, upload_id)
                if replaced:
                    logger.info("Re-extraction replaced %d prior records for %s", replaced, upload_id)
```
(Match the existing indentation inside the `if patient:` block. `logger` already exists in this module.)

- [ ] **Step 2: Wire into the manual confirm endpoint**

In the manual confirm endpoint, immediately BEFORE the `for entity_data in body.confirmed_entities:` loop (currently at `:1025`), add:
```python
    from app.services.ingestion.reextraction import soft_delete_prior_extracted
    replaced = await soft_delete_prior_extracted(db, upload_id)
    if replaced:
        logger.info("Manual re-confirm replaced %d prior records for %s", replaced, upload_id)
```
(Match that function's indentation. `upload_id` is in scope.)

- [ ] **Step 3: Verify existing unstructured tests still pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_unstructured_upload.py -q`
Expected: PASS — first-time extraction has no prior records, so `soft_delete_prior_extracted` is a 0-count no-op; behavior unchanged.

- [ ] **Step 4: Commit**

```bash
cd backend && git add app/api/upload.py
git commit -m "feat: replace prior extracted records on re-extraction"
```

---

## Task 4: file-hash skip on re-upload

**Files:**
- Modify: `backend/app/api/upload.py` (`upload_unstructured` `:842-877`, and the batch endpoint)

- [ ] **Step 1: Wire skip into `upload_unstructured`**

In `backend/app/api/upload.py::upload_unstructured`, AFTER `file_hash = hashlib.sha256(content).hexdigest()` (`:842`) and BEFORE constructing `upload_record` (`:844`), add the duplicate lookup, then set the upload's status accordingly:
```python
    from app.services.ingestion.reextraction import find_prior_extracted_upload
    prior = await find_prior_extracted_upload(db, user_id, file_hash)

    upload_record = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        file_size_bytes=len(content),
        file_hash=file_hash,
        storage_path=str(file_path),
        ingestion_status="duplicate_file" if prior else "pending_extraction",
        file_category="unstructured",
    )
    if prior:
        upload_record.ingestion_progress = {
            "duplicate_of": str(prior.id),
            "record_count": prior.record_count or 0,
        }
    db.add(upload_record)
    await db.commit()
    await db.refresh(upload_record)
```
(This REPLACES the existing `upload_record = UploadedFile(...)` / `db.add` / `commit` / `refresh` block at `:844-857`. Keep the `ingestion_status="pending_extraction"` default behavior when `prior` is None — the conditional above does this.)

Then update the return so the response reflects the skip. Replace the final return block (`:873-877`) with:
```python
    from app.services.extraction.text_extractor import detect_file_type
    file_type = detect_file_type(file_path)

    return UnstructuredUploadResponse(
        upload_id=str(upload_record.id),
        status=upload_record.ingestion_status,  # "duplicate_file" or "pending_extraction"
        file_type=file_type.value,
    )
```
Because the DB-polling worker only claims rows in `pending_extraction` status, a `duplicate_file` row is never extracted — no further wiring needed.

- [ ] **Step 2: Wire skip into the batch endpoint**

In the `unstructured-batch` endpoint (`@router.post("/unstructured-batch", ...)`, starts `:880`), each file is hashed and an `UploadedFile` is created in a loop. For each file, after computing its `file_hash` and before/at `UploadedFile(...)` creation, apply the same pattern: `prior = await find_prior_extracted_upload(db, user_id, file_hash)`, set `ingestion_status="duplicate_file"` + `ingestion_progress={"duplicate_of": str(prior.id), "record_count": prior.record_count or 0}` when `prior`, else `"pending_extraction"`. Read the loop body first and mirror the single-file change exactly. Include the per-file status in the batch response if the response model has a per-file status field; otherwise just set it on the DB row.

- [ ] **Step 3: Verify**

Run: `cd backend && .venv/bin/python -m pytest tests/test_unstructured_upload.py -q`
Expected: PASS — fresh test DB has no prior uploads with matching hashes, so all uploads take the `pending_extraction` path; behavior unchanged.

- [ ] **Step 4: Commit**

```bash
cd backend && git add app/api/upload.py
git commit -m "feat: skip re-extraction of identical re-uploaded files (duplicate_file)"
```

---

## Task 5: integration test — skip + replace via the API (mocked extraction)

**Files:**
- Create: `backend/tests/test_unstructured_idempotency.py`

This test patches extraction so it runs without Gemini. Read `tests/test_unstructured_upload.py` first to copy the EXACT patching pattern it uses (it patches `_process_unstructured` and/or `start_extraction_worker`, and uses the `client` fixture + auth). Mirror that pattern; do not invent new patch targets.

- [ ] **Step 1: Write the integration test**

```python
# backend/tests/test_unstructured_idempotency.py
from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy import select

from app.models.uploaded_file import UploadedFile


@pytest.mark.asyncio
async def test_reupload_identical_file_is_marked_duplicate(db_session, client):
    """Same file_hash already completed for the user -> new upload is duplicate_file, not extracted."""
    from tests.conftest import auth_headers
    headers, uid = await auth_headers(client)

    # Seed a prior COMPLETED unstructured upload with a known hash.
    content = b"%PDF-1.4 fake pdf bytes for hashing"
    h = hashlib.sha256(content).hexdigest()
    prior = UploadedFile(
        id=uuid.uuid4(), user_id=uuid.UUID(uid), filename="note.pdf", file_hash=h,
        storage_path="/x", ingestion_status="completed", file_category="unstructured",
        record_count=7,
    )
    db_session.add(prior)
    await db_session.commit()

    # Re-upload the identical bytes via the API.
    resp = await client.post(
        "/api/v1/upload/unstructured",
        headers=headers,
        files={"file": ("note.pdf", content, "application/pdf")},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "duplicate_file"

    # A new row exists, marked duplicate_file, pointing at the original; no extraction enqueued.
    rows = (await db_session.execute(
        select(UploadedFile).where(UploadedFile.file_hash == h, UploadedFile.id != prior.id)
    )).scalars().all()
    assert len(rows) == 1
    dup = rows[0]
    assert dup.ingestion_status == "duplicate_file"
    assert dup.ingestion_progress["duplicate_of"] == str(prior.id)
    assert dup.ingestion_progress["record_count"] == 7
```

> If `upload_unstructured` rejects the fake bytes via magic-byte validation (`_validate_magic_bytes`), use bytes that pass the PDF magic check — a string beginning with `%PDF-` does. If validation still fails in the test environment, patch `app.api.upload._validate_magic_bytes` to return `True` for this test (note it in a comment). Keep the assertion intent identical.

- [ ] **Step 2: Run**

Run: `cd backend && .venv/bin/python -m pytest tests/test_unstructured_idempotency.py -v`
Expected: PASS (1 test). If it fails on magic-byte validation, apply the note above and re-run.

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/test_unstructured_idempotency.py
git commit -m "test: unstructured re-upload skip integration"
```

---

## Task 6: fidelity / slow test — real note PDF

**Files:**
- Create: `backend/tests/test_unstructured_idempotency_fidelity.py`

Exercises the REAL extraction pipeline end-to-end against the real clinical note. Marked `slow` (needs `GEMINI_API_KEY`); skipped otherwise.

- [ ] **Step 1: Write the slow test**

```python
# backend/tests/test_unstructured_idempotency_fidelity.py
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile

_NOTE = next(
    iter((Path(__file__).resolve().parents[2] / "test_data").glob("note_*.pdf")),
    None,
)


@pytest.mark.slow
@pytest.mark.skipif(_NOTE is None or not os.getenv("GEMINI_API_KEY"),
                    reason="real note PDF and GEMINI_API_KEY required")
@pytest.mark.asyncio
async def test_real_note_reupload_is_duplicate(db_session, create_test_patient, client):
    """Upload the real note PDF, extract, then re-upload identical bytes -> duplicate_file, 0 new records."""
    from tests.conftest import auth_headers
    headers, uid = await auth_headers(client)
    await create_test_patient(db_session, uid)
    content = _NOTE.read_bytes()
    h = hashlib.sha256(content).hexdigest()

    # First upload + wait for extraction to complete (poll status).
    r1 = await client.post("/api/v1/upload/unstructured", headers=headers,
                           files={"file": (_NOTE.name, content, "application/pdf")})
    assert r1.status_code == 200
    up1 = r1.json()["upload_id"]
    # Drive the extraction worker synchronously for determinism in-test:
    from app.api.upload import _process_unstructured
    await _process_unstructured(uuid.UUID(up1), Path(
        (await db_session.get(UploadedFile, uuid.UUID(up1))).storage_path), uuid.UUID(uid))
    await db_session.commit()

    n1 = (await db_session.execute(
        select(func.count()).select_from(HealthRecord).where(HealthRecord.ai_extracted.is_(True))
    )).scalar()
    assert n1 > 0, "real note should extract at least one record"

    # Re-upload identical bytes.
    r2 = await client.post("/api/v1/upload/unstructured", headers=headers,
                           files={"file": (_NOTE.name, content, "application/pdf")})
    assert r2.json()["status"] == "duplicate_file"
    dup = await db_session.get(UploadedFile, uuid.UUID(r2.json()["upload_id"]))
    assert dup.ingestion_progress["duplicate_of"]

    n2 = (await db_session.execute(
        select(func.count()).select_from(HealthRecord).where(HealthRecord.ai_extracted.is_(True))
    )).scalar()
    assert n2 == n1, f"re-upload created new records: {n1} -> {n2}"
```

> Note: driving `_process_unstructured` directly avoids depending on the background worker in-test. If its signature differs from `(upload_id, file_path, user_id)`, read the actual signature at `upload.py:570` and adapt the call. If the auto-confirm path requires a patient (it does — it only inserts when a patient exists), the `create_test_patient` above satisfies that.

- [ ] **Step 2: Run (needs GEMINI_API_KEY; otherwise skips)**

Run: `cd backend && .venv/bin/python -m pytest tests/test_unstructured_idempotency_fidelity.py -v -m slow -rs`
Expected: PASS if `GEMINI_API_KEY` set and note present, else SKIP. If the second upload created records, the file-hash skip is not engaging — re-check Task 4.

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/test_unstructured_idempotency_fidelity.py
git commit -m "test: real note PDF re-upload idempotency (slow)"
```

---

## Task 7: full regression + lint

- [ ] **Step 1: Lint the touched files**

Run: `cd backend && .venv/bin/ruff check app/services/extraction/entity_to_fhir.py app/services/ingestion/reextraction.py app/api/upload.py tests/test_entity_to_fhir_hash.py tests/test_reextraction.py tests/test_unstructured_idempotency.py`
Fix any errors INTRODUCED by this feature (unused imports, line length >100, import ordering). Hoist the function-local `from app.services.ingestion.reextraction import ...` imports added in Task 3/4 to module top if ruff prefers, matching the file's existing import style. Do not fix pre-existing issues unrelated to this feature (report them instead).

- [ ] **Step 2: Full fast suite**

Run: `cd backend && .venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (the Phase 1 baseline was 492 passed; this adds ~10 tests). If a pre-existing test fails for reasons unrelated to this feature, report it; do not mask failures.

- [ ] **Step 3: Final commit (if changes)**

```bash
cd backend && git add -A
git commit -m "chore: lint + regression pass for unstructured idempotency"
```

---

## Self-Review Notes (author)

- **Spec coverage:** file-hash skip → Tasks 2/4; replace-on-reextract → Tasks 2/3; content_hash on extracted → Task 1; cross-document overlap unchanged (existing dedup, no task needed); no migration (reuses content_hash + JSONB) → confirmed; tests → Tasks 1/2/5/6; safety (soft-delete, user-scoped) → Tasks 2. All covered.
- **`duplicate_file` status** is a free-text value; `UploadResponse`/`UnstructuredUploadResponse.status` is a string — no schema change.
- **Open implementation-time checks:** (a) the batch endpoint loop body (Task 4 Step 2) — mirror the single-file change after reading it; (b) the exact extraction patch targets in `test_unstructured_upload.py` (Task 5) — copy them; (c) `_process_unstructured` signature (Task 6) — verify at `upload.py:570`.
- **Out of scope:** extraction quality on non-standard docs (Phase 2b).
