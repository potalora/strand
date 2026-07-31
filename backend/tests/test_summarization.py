from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from unittest.mock import AsyncMock, Mock, patch

from app.models.ai_summary import AISummaryPrompt
from app.models.local_ai import ExtractionEvidence, LocalAIJob
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.services.ai.llm.types import LLMRequest, LLMResponse, LLMUsage
from app.services.local_ai.manifest import (
    canonicalize_manifest_snapshot,
    parse_manifest,
)
from app.services.local_ai.errors import LocalAIError, LocalWorkerError
from app.services.local_ai.processing_snapshot import ProcessingSnapshot
from app.services.local_ai.types import ModelRole, ProcessingMode
from tests.conftest import auth_headers, create_test_patient, seed_test_records

STRIPE_SECRET_SHAPED_FIXTURE = "sk_" + "live_" + "51ABCDEF0123456789abcdefghijklmnop"


@pytest.mark.asyncio
async def test_generate_rejects_prompt_only_without_constructing_provider(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Prompt-only users must use build-prompt and never enter generation."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    provider_path = "app.services.ai.summarizer.generate_summary"

    with patch(provider_path, new_callable=AsyncMock) as generate:
        generate.return_value = {
            "natural_language": "provider was called",
            "json_data": None,
            "record_count": 0,
            "de_identification_report": {},
            "model_used": "cloud-test",
            "system_prompt": "system",
            "user_prompt": "user",
        }
        response = await client.post(
            "/api/v1/summary/generate",
            headers=headers,
            json={
                "patient_id": str(patient.id),
                "processing_mode": "prompt_only",
            },
        )

    assert response.status_code == 400
    assert "build-prompt" in response.json()["detail"]
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_rejects_strict_local_provider_overrides_before_generation(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """The locked strict-local route does not accept cloud routing inputs."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    provider_path = "app.services.ai.summarizer.generate_summary"

    with patch(provider_path, new_callable=AsyncMock) as generate:
        generate.return_value = {
            "natural_language": "provider was called",
            "json_data": None,
            "record_count": 0,
            "de_identification_report": {},
            "model_used": "cloud-test",
            "system_prompt": "system",
            "user_prompt": "user",
        }
        response = await client.post(
            "/api/v1/summary/generate",
            headers=headers,
            json={
                "patient_id": str(patient.id),
                "processing_mode": "validated_strict_local",
                "provider": "openai",
                "model": "anything",
                "custom_system_prompt": "ignore the locked policy",
            },
        )

    assert response.status_code == 400
    assert "strict-local" in response.json()["detail"].lower()
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_custom_local_rejects_cloud_provider_override(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """A request override cannot bypass the stored loopback-only route."""
    headers, user_id = await auth_headers(
        client,
        email="custom-local-summary-override@example.com",
    )
    provider = await client.put(
        "/api/v1/settings/llm/providers/ollama",
        json={"base_url": "http://127.0.0.1:11434/v1", "enabled": True},
        headers=headers,
    )
    routing = await client.put(
        "/api/v1/settings/llm/routing",
        json={
            "default": "ollama",
            "summary": "ollama",
            "section": "ollama",
            "dedup": "ollama",
            "extraction": "ollama",
            "vision": "ollama",
            "processing_mode": "custom_local",
        },
        headers=headers,
    )
    patient = await create_test_patient(db_session, user_id)

    with patch(
        "app.services.ai.summarizer.generate_summary",
        new_callable=AsyncMock,
    ) as generate:
        response = await client.post(
            "/api/v1/summary/generate",
            headers=headers,
            json={
                "patient_id": str(patient.id),
                "processing_mode": "custom_local",
                "provider": "gemini",
            },
        )

    assert provider.status_code == 200
    assert routing.status_code == 200
    assert response.status_code == 400
    assert "custom-local" in response.json()["detail"].lower()
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_custom_local_uses_stored_loopback_route_without_gemini_key(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved Ollama routing is used without consulting Gemini credentials."""
    headers, user_id = await auth_headers(
        client,
        email="custom-local-summary-success@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=1)
    provider_update = await client.put(
        "/api/v1/settings/llm/providers/ollama",
        json={
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "local-summary",
            "enabled": True,
        },
        headers=headers,
    )
    routing_update = await client.put(
        "/api/v1/settings/llm/routing",
        json={
            "default": "ollama",
            "summary": "ollama",
            "section": "ollama",
            "dedup": "ollama",
            "extraction": "ollama",
            "vision": "ollama",
            "processing_mode": "custom_local",
        },
        headers=headers,
    )
    local_provider = _mock_provider("Locally organized record.", model="local-summary")
    local_provider.name = "ollama"
    constructed = []

    def get_local_provider(operation, config):
        constructed.append((operation, config))
        assert config.processing_mode is ProcessingMode.CUSTOM_LOCAL
        assert config.routing["summary"] == "ollama"
        assert config.providers["ollama"].base_url == "http://127.0.0.1:11434/v1"
        return local_provider

    monkeypatch.setattr("app.services.ai.summarizer.get_provider", get_local_provider)
    monkeypatch.setattr("app.services.ai.summarizer.settings.gemini_api_key", "")

    response = await client.post(
        "/api/v1/summary/generate",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "processing_mode": "custom_local",
            "output_format": "natural_language",
        },
    )

    assert provider_update.status_code == 200
    assert routing_update.status_code == 200
    assert response.status_code == 200, response.text
    assert constructed and constructed[0][0] == "summary"
    local_provider.complete.assert_awaited_once()
    data = response.json()
    assert data["model_used"] == "local-summary"
    assert data["processing_mode"] == "custom_local"
    assert data["model_provenance"] == {
        "processing_mode": "custom_local",
        "provider": "ollama",
        "model": "local-summary",
    }
    assert set(data["typed_response"]) == {"sections", "uncertainties"}
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(AISummaryPrompt.id == UUID(data["id"]))
        )
    ).scalar_one()
    assert prompt.processing_mode == "custom_local"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preflight_case",
    [
        "success",
        "over_limit",
        "unsupported_qualifier",
        "count_failure",
        "cancel_before_count",
        "cancel_during_count",
    ],
)
async def test_strict_local_summary_commits_then_queues_background_work(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preflight_case: str,
) -> None:
    """The strict route commits its recoverable job, then uses only its lock."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    manifest_payload = _strict_manifest()
    manifest_snapshot, digest = canonicalize_manifest_snapshot(manifest_payload)
    manifest = parse_manifest(manifest_snapshot)
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="strict-local.pdf",
        mime_type="application/pdf",
        file_hash="f" * 64,
        storage_path="/private/strict-local.pdf",
        processing_mode="validated_strict_local",
    )
    db_session.add(upload)
    await db_session.flush()
    survivor = HealthRecord(
        user_id=UUID(user_id),
        patient_id=patient.id,
        record_type="condition",
        fhir_resource_type="Condition",
        fhir_resource={
            "resourceType": "Condition",
            "code": {"text": "Hypertension"},
        },
        source_format="fhir",
        source_file_id=None,
        display_text="Hypertension",
        status="active" if preflight_case == "unsupported_qualifier" else None,
        ai_extracted=False,
    )
    db_session.add(survivor)
    await db_session.flush()
    record = HealthRecord(
        user_id=UUID(user_id),
        patient_id=patient.id,
        record_type="condition",
        fhir_resource_type="Condition",
        fhir_resource={"resourceType": "Condition"},
        source_format="local_ai",
        source_file_id=upload.id,
        display_text="Hypertension",
        ai_extracted=True,
        is_duplicate=True,
        merged_into_id=survivor.id,
    )
    db_session.add(record)
    await db_session.flush()
    db_session.add(
        ExtractionEvidence(
            user_id=UUID(user_id),
            upload_id=upload.id,
            health_record_id=record.id,
            excerpt="Hypertension from archived strict evidence.",
            field_paths=["conditions[0].name"],
            source_metadata={"evidence_id": "ev1_test"},
        )
    )
    await db_session.commit()

    async def resolve_snapshot(*_args, **_kwargs) -> ProcessingSnapshot:
        return ProcessingSnapshot(
            mode=ProcessingMode.VALIDATED_STRICT_LOCAL,
            manifest_snapshot=manifest_snapshot,
            manifest_sha256=digest,
            schema_version="clinical-document-extraction.v1",
        )

    async def revalidate_snapshot(*_args, **_kwargs) -> None:
        return None

    class Store:
        packs_dir = tmp_path / "packs"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def active_manifest(self):
            return manifest

    commits = 0
    inference_events: list[str] = []
    worker_session: AsyncSession | None = None
    original_commit = db_session.commit

    async def counted_commit() -> None:
        nonlocal commits
        commits += 1
        await original_commit()

    async def run(role: ModelRole, payload: dict) -> dict:
        inference_events.append("generate")
        assert commits >= 1
        assert role is ModelRole.SUMMARY
        assert set(payload) == {
            "job_id",
            "requested_scope",
            "facts",
            "evidence",
            "uncertainty_labels",
            "safety_rules",
            "manifest_path",
            "model_dir",
            "manifest_identity",
            "max_output_tokens",
        }
        assert Path(payload["manifest_path"]).is_absolute()
        assert Path(payload["model_dir"]).is_absolute()
        assert payload["manifest_identity"]["revision"] == "0" * 40
        assert payload["max_output_tokens"] == 301
        fact = payload["facts"][0]
        assert fact["record_id"] == str(survivor_id)
        assert [item["excerpt"] for item in payload["evidence"]] == [
            "Hypertension from archived strict evidence."
        ]
        return {
            "sections": [
                {
                    "heading": "Conditions",
                    "claims": [
                        {
                            "fact_id": fact["fact_id"],
                            "field_paths": ["/diagnosis"],
                            "evidence_ids": fact["evidence_ids"],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        }

    async def count_reference_tokens(payload: dict) -> int:
        inference_events.append("count")
        assert set(payload["reference_document"]) == {"sections", "uncertainties"}
        assert worker_session is not None
        stage_row = (
            await worker_session.execute(
                select(LocalAIJob.stage, LocalAIJob.progress).where(
                    LocalAIJob.id == job_id
                )
            )
        ).one()
        assert stage_row.stage == "preflight_output_fit"
        assert stage_row.progress == {
            "stage": "preflight_output_fit",
            "model_role": "summary",
        }
        if preflight_case == "count_failure":
            raise LocalWorkerError("Local worker failed.")
        if preflight_case == "cancel_during_count":
            await worker_session.execute(
                update(LocalAIJob)
                .where(LocalAIJob.id == job_id)
                .values(cancel_requested=True)
            )
            await worker_session.commit()
            raise LocalWorkerError("Local worker cancelled.")
        return 897 if preflight_case == "over_limit" else 173

    enqueue = Mock()
    monkeypatch.setattr("app.api.summary.resolve_new_job_snapshot", resolve_snapshot)
    monkeypatch.setattr(
        "app.api.summary.revalidate_strict_snapshot_admission",
        revalidate_snapshot,
    )
    model_run = AsyncMock(side_effect=run)
    token_count = AsyncMock(side_effect=count_reference_tokens)
    if preflight_case == "cancel_before_count":
        from app.services.ai import summarizer as summarizer_module

        original_cancel_check = summarizer_module._is_strict_summary_cancelled
        cancellation_checks = 0

        async def cancel_before_tokenizer(
            session: AsyncSession,
            candidate_job_id: UUID,
        ) -> bool:
            nonlocal cancellation_checks
            cancellation_checks += 1
            if cancellation_checks == 2:
                stage_row = (
                    await session.execute(
                        select(LocalAIJob.stage, LocalAIJob.progress).where(
                            LocalAIJob.id == candidate_job_id
                        )
                    )
                ).one()
                assert stage_row.stage == "preflight_output_fit"
                assert stage_row.progress == {
                    "stage": "preflight_output_fit",
                    "model_role": "summary",
                }
                await session.execute(
                    update(LocalAIJob)
                    .where(LocalAIJob.id == candidate_job_id)
                    .values(cancel_requested=True)
                )
                await session.commit()
                return True
            return await original_cancel_check(session, candidate_job_id)

        monkeypatch.setattr(
            summarizer_module,
            "_is_strict_summary_cancelled",
            cancel_before_tokenizer,
        )
    monkeypatch.setattr("app.services.local_ai.artifact_store.ArtifactStore", Store)
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.run", model_run
    )
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.count_summary_tokens",
        token_count,
    )
    monkeypatch.setattr(
        "app.services.ai.summarizer.settings.local_ai_scratch_dir",
        str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "app.api.summary.local_summary_runner.enqueue",
        enqueue,
        raising=False,
    )
    monkeypatch.setattr(db_session, "commit", counted_commit)

    response = await client.post(
        "/api/v1/summary/generate",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "full",
            "processing_mode": "validated_strict_local",
            "output_format": "both",
        },
    )

    assert response.status_code == 202, response.text
    data = response.json()
    assert data["processing_mode"] == "validated_strict_local"
    assert data["kind"] == "summary"
    assert data["status"] == "queued"
    assert data["stage"] == "queued"
    prompt = (await db_session.execute(select(AISummaryPrompt))).scalar_one()
    job = (await db_session.execute(select(LocalAIJob))).scalar_one()
    assert data["id"] == str(prompt.id)
    assert data["job_id"] == str(job.id)
    assert job.status == "queued"
    assert job.kind == "summary"
    enqueue.assert_called_once_with(job.id)
    model_run.assert_not_awaited()
    job_id = job.id
    prompt_id = prompt.id
    patient_id = patient.id
    survivor_id = survivor.id
    await db_session.rollback()

    from app.services.ai.summarizer import generate_grounded_local_summary

    session_factory = async_sessionmaker(
        db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        worker_session = session
        assert (await session.get(LocalAIJob, job_id)).status == "queued"
        if preflight_case == "success":
            await generate_grounded_local_summary(
                session,
                user_id=UUID(user_id),
                patient_id=patient_id,
                job_id=job_id,
                summary_type="full",
            )
        else:
            with pytest.raises(LocalAIError):
                await generate_grounded_local_summary(
                    session,
                    user_id=UUID(user_id),
                    patient_id=patient_id,
                    job_id=job_id,
                    summary_type="full",
                )

        durable_job = await session.get(LocalAIJob, job_id)
        durable_prompt = await session.get(AISummaryPrompt, prompt_id)

    assert durable_job is not None
    assert durable_prompt is not None
    if preflight_case == "success":
        model_run.assert_awaited_once()
        token_count.assert_awaited_once()
        assert inference_events == ["count", "generate"]
        token_payload = token_count.await_args.args[0]
        assert set(token_payload) == {
            "job_id",
            "reference_document",
            "manifest_path",
            "model_dir",
            "manifest_identity",
        }
        assert durable_job.status == "completed"
        assert durable_prompt.response_text is not None
    elif preflight_case in {"over_limit", "count_failure"}:
        token_count.assert_awaited_once()
        model_run.assert_not_awaited()
        assert inference_events == ["count"]
        assert durable_job.status == "failed"
        assert durable_job.failure["stage"] == "preflight_output_fit"
        assert durable_job.failure["cloud_fallback_attempted"] is False
        assert durable_prompt.response_text is None
        assert durable_prompt.typed_response is None
    elif preflight_case == "unsupported_qualifier":
        token_count.assert_not_awaited()
        model_run.assert_not_awaited()
        assert inference_events == []
        assert durable_job.status == "failed"
        assert durable_job.failure["stage"] == "preflight"
        assert durable_job.failure["cloud_fallback_attempted"] is False
        assert durable_prompt.response_text is None
        assert durable_prompt.typed_response is None
    elif preflight_case == "cancel_before_count":
        token_count.assert_not_awaited()
        model_run.assert_not_awaited()
        assert inference_events == []
        assert durable_job.status == "cancelled"
        assert durable_job.failure is None
        assert durable_prompt.response_text is None
        assert durable_prompt.typed_response is None
    else:
        token_count.assert_awaited_once()
        model_run.assert_not_awaited()
        assert inference_events == ["count"]
        assert durable_job.status == "cancelled"
        assert durable_job.failure is None
        assert durable_prompt.response_text is None
        assert durable_prompt.typed_response is None


@pytest.mark.asyncio
async def test_two_sessions_atomically_claim_one_strict_summary_attempt(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Two stale sessions still admit exactly one durable model attempt."""
    from app.services.ai.summarizer import _claim_strict_summary_job

    _headers, user_id = await auth_headers(
        client,
        email="strict-summary-atomic-claim@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    snapshot, digest = canonicalize_manifest_snapshot(_strict_manifest())
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=UUID(user_id),
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={},
        system_prompt="Locked policy",
        user_prompt="Grounded facts",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=0,
        generated_at=datetime.now(timezone.utc),
    )
    job = LocalAIJob(
        user_id=UUID(user_id),
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        manifest_sha256=digest,
        status="queued",
        stage="queued",
        progress={"stage": "queued"},
    )
    db_session.add_all([prompt, job])
    await db_session.commit()
    job_id = job.id
    await db_session.rollback()

    model_call = AsyncMock()
    session_factory = async_sessionmaker(
        db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session_a, session_factory() as session_b:
        assert (await session_a.get(LocalAIJob, job_id)).status == "queued"
        assert (await session_b.get(LocalAIJob, job_id)).status == "queued"

        async def claim_and_run(session: AsyncSession):
            claim = await _claim_strict_summary_job(
                session,
                job_id=job_id,
                user_id=UUID(user_id),
            )
            if claim is not None:
                await model_call()
            return claim

        tasks = [
            asyncio.create_task(claim_and_run(session_a)),
            asyncio.create_task(claim_and_run(session_b)),
        ]
        try:
            claims = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    assert sum(claim is not None for claim in claims) == 1
    model_call.assert_awaited_once()
    await db_session.refresh(job)
    assert (job.status, job.stage) == ("processing", "preflight")
    assert job.started_at is not None


def test_generate_openapi_reserves_strict_local_for_202() -> None:
    """Generated clients see strict-local acceptance only under HTTP 202."""
    from app.main import app

    responses = app.openapi()["paths"]["/api/v1/summary/generate"]["post"]["responses"]
    completed_schema = json.dumps(
        responses["200"]["content"]["application/json"]["schema"]
    )
    accepted_schema = json.dumps(
        responses["202"]["content"]["application/json"]["schema"]
    )

    assert "StrictLocalGenerateSummaryResponse" not in completed_schema
    assert "StrictLocalSummaryAccepted" in accepted_schema


@pytest.mark.asyncio
async def test_startup_requeues_and_resumes_strict_local_summary_job(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process restart makes an interrupted summary runnable again."""
    from app.main import _recover_strict_local_summary_jobs_on_startup
    from app.services.ai.summarizer import resume_grounded_local_summary_jobs

    _headers, user_id = await auth_headers(
        client,
        email="strict-summary-recovery@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    manifest_snapshot, digest = canonicalize_manifest_snapshot(_strict_manifest())
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=UUID(user_id),
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={
            "category": None,
            "date_from": None,
            "date_to": None,
            "record_ids": [],
            "output_format": "both",
        },
        system_prompt="Locked policy",
        user_prompt="Grounded facts",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=0,
        generated_at=datetime.now(timezone.utc),
    )
    job = LocalAIJob(
        user_id=UUID(user_id),
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=manifest_snapshot,
        manifest_sha256=digest,
        status="processing",
        stage="summary",
        progress={"stage": "summary"},
    )
    db_session.add_all([prompt, job])
    await db_session.commit()

    resumable = await _recover_strict_local_summary_jobs_on_startup(db_session)
    await db_session.commit()
    await db_session.refresh(job)

    assert resumable == [job.id]
    assert (job.status, job.stage) == ("queued", "recovery")

    generate = AsyncMock(return_value={})
    monkeypatch.setattr(
        "app.services.ai.summarizer.generate_grounded_local_summary",
        generate,
    )
    await resume_grounded_local_summary_jobs(
        resumable,
        session_factory=async_sessionmaker(
            db_session.bind,
            class_=AsyncSession,
            expire_on_commit=False,
        ),
    )

    generate.assert_awaited_once()
    assert generate.await_args.kwargs["job_id"] == job.id
    assert generate.await_args.kwargs["summary_type"] == "full"


@pytest.mark.asyncio
async def test_old_summary_attempt_cannot_terminalize_a_new_claim(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Finalization is conditional on the attempt's durable claim timestamp."""
    from app.services.ai.summarizer import _finish_strict_summary_job

    _headers, user_id = await auth_headers(
        client,
        email="strict-summary-stale-finalizer@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    snapshot, digest = canonicalize_manifest_snapshot(_strict_manifest())
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=UUID(user_id),
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={},
        system_prompt="Locked policy",
        user_prompt="Grounded facts",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=0,
        generated_at=datetime.now(timezone.utc),
    )
    old_started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new_started_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    job = LocalAIJob(
        user_id=UUID(user_id),
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        manifest_sha256=digest,
        status="processing",
        stage="summary",
        progress={"stage": "summary"},
        started_at=old_started_at,
    )
    db_session.add_all([prompt, job])
    await db_session.commit()

    async with AsyncSession(db_session.bind, expire_on_commit=False) as stale_session:
        stale_job = await stale_session.get(LocalAIJob, job.id)
        assert stale_job is not None
        job.status = "queued"
        job.stage = "recovery"
        await db_session.commit()
        job.status = "processing"
        job.stage = "preflight"
        job.started_at = new_started_at
        await db_session.commit()

        await _finish_strict_summary_job(
            stale_session,
            job_id=job.id,
            error=RuntimeError("old attempt failed"),
            claim_started_at=old_started_at,
        )

    await db_session.refresh(job)
    assert (job.status, job.stage, job.started_at) == (
        "processing",
        "preflight",
        new_started_at,
    )


@pytest.mark.asyncio
async def test_strict_local_summary_cancellation_finalizes_job_before_reraising(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Task cancellation rolls back, persists cancellation, then propagates."""
    headers, user_id = await auth_headers(
        client,
        email="strict-summary-cancel@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    manifest_snapshot, digest = canonicalize_manifest_snapshot(_strict_manifest())
    manifest = parse_manifest(manifest_snapshot)
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=UUID(user_id),
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={},
        system_prompt="Locked policy",
        user_prompt="Grounded facts",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=0,
        generated_at=datetime.now(timezone.utc),
    )
    job = LocalAIJob(
        user_id=UUID(user_id),
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=manifest_snapshot,
        manifest_sha256=digest,
        status="queued",
        stage="queued",
        progress={"stage": "queued"},
    )
    record = HealthRecord(
        user_id=UUID(user_id),
        patient_id=patient.id,
        record_type="condition",
        fhir_resource_type="Condition",
        fhir_resource={"resourceType": "Condition"},
        source_format="fhir",
        display_text="Hypertension",
    )
    db_session.add_all([prompt, job, record])
    await db_session.commit()

    class Store:
        packs_dir = tmp_path / "packs"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def active_manifest(self):
            return manifest

    async def cancel(*_args, **_kwargs):
        job.cancel_requested = True
        await db_session.commit()
        raise asyncio.CancelledError

    monkeypatch.setattr("app.services.local_ai.artifact_store.ArtifactStore", Store)
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.run", cancel
    )
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.count_summary_tokens",
        AsyncMock(return_value=100),
    )
    monkeypatch.setattr(
        "app.services.ai.summarizer.settings.local_ai_scratch_dir",
        str(tmp_path / "scratch"),
    )
    from app.services.ai.summarizer import generate_grounded_local_summary

    with pytest.raises(asyncio.CancelledError):
        await generate_grounded_local_summary(
            db_session,
            user_id=UUID(user_id),
            patient_id=patient.id,
            job_id=job.id,
            summary_type="full",
        )

    await db_session.refresh(job)
    assert job.status == "cancelled"
    assert job.stage == "cancelled"
    assert job.completed_at is not None
    assert job.failure is None


@pytest.mark.asyncio
async def test_final_summary_completion_honors_cancellation_under_its_terminal_lock(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cancellation after model output must win over the final completion write."""
    from types import SimpleNamespace

    from app.services.ai.summarizer import generate_grounded_local_summary

    _headers, user_id = await auth_headers(
        client,
        email="strict-summary-final-window-cancel@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    manifest_snapshot, digest = canonicalize_manifest_snapshot(_strict_manifest())
    manifest = parse_manifest(manifest_snapshot)
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=UUID(user_id),
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={},
        system_prompt="Locked policy",
        user_prompt="Grounded facts",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=0,
        generated_at=datetime.now(timezone.utc),
    )
    job = LocalAIJob(
        user_id=UUID(user_id),
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=manifest_snapshot,
        manifest_sha256=digest,
        status="queued",
        stage="queued",
        progress={"stage": "queued"},
    )
    db_session.add_all(
        [
            prompt,
            job,
            HealthRecord(
                user_id=UUID(user_id),
                patient_id=patient.id,
                record_type="condition",
                fhir_resource_type="Condition",
                fhir_resource={"resourceType": "Condition"},
                source_format="fhir",
                display_text="Hypertension",
            ),
        ]
    )
    await db_session.commit()

    class Store:
        packs_dir = tmp_path / "packs"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def active_manifest(self):
            return manifest

    async def complete_model(*_args, **_kwargs) -> dict:
        return {}

    async def persist_cancellation() -> None:
        async with async_sessionmaker(
            db_session.bind,
            class_=AsyncSession,
            expire_on_commit=False,
        )() as cancellation_session:
            await cancellation_session.execute(
                update(LocalAIJob)
                .where(LocalAIJob.id == job.id)
                .values(cancel_requested=True)
            )
            await cancellation_session.commit()

    cancellation_task: asyncio.Task[None] | None = None

    def cancel_after_model_output(*_args, **_kwargs):
        nonlocal cancellation_task
        cancellation_task = asyncio.create_task(persist_cancellation())
        return SimpleNamespace(
            document=SimpleNamespace(
                model_dump=lambda **_kwargs: {"sections": [], "uncertainties": []}
            ),
            markdown="Cancelled summary must not be persisted.",
        )

    original_execute = db_session.execute

    async def execute_after_cancellation(*args, **kwargs):
        nonlocal cancellation_task
        if cancellation_task is not None:
            task, cancellation_task = cancellation_task, None
            await task
        return await original_execute(*args, **kwargs)

    monkeypatch.setattr("app.services.local_ai.artifact_store.ArtifactStore", Store)
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.run",
        complete_model,
    )
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.count_summary_tokens",
        AsyncMock(return_value=100),
    )
    monkeypatch.setattr(
        "app.services.local_ai.grounded_summary.validate_and_render_summary",
        cancel_after_model_output,
    )
    monkeypatch.setattr(
        "app.services.ai.summarizer.settings.local_ai_scratch_dir",
        str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(db_session, "execute", execute_after_cancellation)

    with pytest.raises(LocalAIError, match="cancelled"):
        await generate_grounded_local_summary(
            db_session,
            user_id=UUID(user_id),
            patient_id=patient.id,
            job_id=job.id,
            summary_type="full",
        )

    await db_session.refresh(job)
    assert (job.status, job.stage, job.cancel_requested) == (
        "cancelled",
        "cancelled",
        True,
    )
    assert job.failure is None


def _mock_provider(
    text: str,
    model: str = "gemini-3.5-flash",
    *,
    grounded: bool = True,
) -> AsyncMock:
    """Build a provider returning references by default or explicit invalid text."""
    prov = AsyncMock()
    prov.name = "gemini"
    if not grounded:
        prov.complete.return_value = LLMResponse(
            text=text,
            finish_reason="stop",
            model=model,
            usage=LLMUsage(10, 5, 15),
            raw=None,
        )
        return prov

    async def complete(request: LLMRequest) -> LLMResponse:
        content = request.messages[0].content
        assert isinstance(content, str)
        transport = json.loads(content.rsplit("INPUT_JSON=", 1)[1])
        fact = transport["facts"][0]
        evidence_by_id = {item["evidence_id"]: item for item in transport["evidence"]}
        evidence_id = fact["evidence_ids"][0]
        output = {
            "sections": [
                {
                    "heading": "Overview",
                    "claims": [
                        {
                            "fact_id": fact["fact_id"],
                            "field_paths": evidence_by_id[evidence_id]["field_paths"],
                            "evidence_ids": [evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        }
        return LLMResponse(
            text=json.dumps(output),
            finish_reason="stop",
            model=model,
            usage=LLMUsage(10, 5, 15),
            raw=None,
        )

    prov.complete.side_effect = complete
    return prov


def _strict_manifest() -> dict:
    """Return the smallest complete locked pack accepted by manifest validation."""

    def artifact(role: str, character: str) -> dict:
        return {
            "role": role,
            "repository": f"owner/{role}",
            "revision": "0" * 40,
            "quantization": "4bit",
            "license": "apache-2.0",
            "attribution": f"https://huggingface.co/owner/{role}",
            "decode_limits": {"max_input_tokens": 4096, "max_output_tokens": 1024},
            "files": [
                {
                    "path": f"{role}/model.safetensors",
                    "sha256": character * 64,
                    "size": 10,
                }
            ],
        }

    return {
        "schema_version": 1,
        "pack_revision": "apple-m4-16gb-v1",
        "platform": "apple_silicon",
        "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        "validation_suite_version": "local-ai-fixtures-v1",
        "artifacts": [
            artifact("ocr", "a"),
            artifact("extraction", "b"),
            artifact("summary", "c"),
        ],
    }


# ---------- Unit tests (no API calls) ----------


@pytest.mark.parametrize(
    ("reference_tokens", "manifest_limit", "expected"),
    [(1, 1024, 256), (173, 1024, 301), (3968, 8192, 4096)],
)
def test_strict_local_summary_output_cap_is_exact_and_bounded(
    reference_tokens: int,
    manifest_limit: int,
    expected: int,
) -> None:
    from app.services.ai.summarizer import _bounded_summary_output_tokens

    assert (
        _bounded_summary_output_tokens(
            reference_tokens=reference_tokens,
            manifest_max_output_tokens=manifest_limit,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("category", "expected"),
    [("family_history", "condition"), ("imaging", "imaging_study")],
)
def test_strict_local_category_scope_normalizes_record_aliases(
    category: str,
    expected: str,
) -> None:
    from app.services.ai.summarizer import _grounded_requested_scope

    scope = _grounded_requested_scope(
        summary_type="full",
        category=category,
        date_from=None,
        date_to=None,
        record_ids=None,
    )

    assert scope == {"summary_type": "category", "category": expected}


@pytest.mark.asyncio
async def test_deduped_records_only(client: AsyncClient, db_session: AsyncSession):
    """Verify that duplicate records are excluded from summary."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    records = await seed_test_records(db_session, user_id, patient.id, count=5)

    # Mark one as duplicate
    records[0].is_duplicate = True
    await db_session.commit()

    from app.services.ai.summarizer import _fetch_deduped_records

    deduped = await _fetch_deduped_records(db_session, UUID(user_id), patient.id)
    assert len(deduped) == 4  # 5 total - 1 duplicate


@pytest.mark.asyncio
async def test_duplicate_warning_counts(client: AsyncClient, db_session: AsyncSession):
    """Verify duplicate warning math: total - deduped = excluded."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    records = await seed_test_records(db_session, user_id, patient.id, count=5)

    # Mark 2 as duplicates
    records[0].is_duplicate = True
    records[1].is_duplicate = True
    await db_session.commit()

    from app.services.ai.summarizer import _count_records

    total = await _count_records(
        db_session, UUID(user_id), patient.id, deduped_only=False
    )
    deduped = await _count_records(
        db_session, UUID(user_id), patient.id, deduped_only=True
    )

    assert total == 5
    assert deduped == 3
    assert total - deduped == 2


@pytest.mark.asyncio
async def test_phi_scrubbing_before_summary(
    client: AsyncClient, db_session: AsyncSession
):
    """Verify that PHI is scrubbed from record text before summarization."""
    from app.services.ai.phi_scrubber import scrub_phi

    text = "Patient John Doe, SSN 123-45-6789, email john@example.com was seen for follow-up."
    scrubbed, report = scrub_phi(text)

    assert "123-45-6789" not in scrubbed
    assert "john@example.com" not in scrubbed
    assert "[SSN]" in scrubbed
    assert "[EMAIL]" in scrubbed


@pytest.mark.asyncio
async def test_generate_endpoint_no_api_key(
    client: AsyncClient, db_session: AsyncSession
):
    """Verify generate returns error when API key is not configured."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=3)

    # Temporarily clear the API key
    from app.config import settings

    original_key = settings.gemini_api_key
    settings.gemini_api_key = ""
    try:
        resp = await client.post(
            "/api/v1/summary/generate",
            json={
                "patient_id": str(patient.id),
                "summary_type": "full",
                "output_format": "natural_language",
                "processing_mode": "cloud_assisted",
            },
            headers=headers,
        )
        assert resp.status_code == 400
        assert "GEMINI_API_KEY" in resp.json()["detail"]
    finally:
        settings.gemini_api_key = original_key


@pytest.mark.asyncio
async def test_generate_endpoint_no_records(
    client: AsyncClient, db_session: AsyncSession
):
    """Verify generate returns error when no records exist."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)

    from app.config import settings

    original_key = settings.gemini_api_key
    settings.gemini_api_key = "test-key"
    try:
        resp = await client.post(
            "/api/v1/summary/generate",
            json={
                "patient_id": str(patient.id),
                "summary_type": "full",
                "output_format": "natural_language",
                "processing_mode": "cloud_assisted",
            },
            headers=headers,
        )
        assert resp.status_code == 400
        assert "No records" in resp.json()["detail"]
    finally:
        settings.gemini_api_key = original_key


@pytest.mark.asyncio
async def test_generate_patient_not_found(
    client: AsyncClient, db_session: AsyncSession
):
    """Verify generate returns 404 for non-existent patient."""
    headers, user_id = await auth_headers(client)

    resp = await client.post(
        "/api/v1/summary/generate",
        json={
            "patient_id": "00000000-0000-0000-0000-000000000000",
            "summary_type": "full",
            "output_format": "natural_language",
            "processing_mode": "cloud_assisted",
        },
        headers=headers,
    )
    assert resp.status_code == 404


# ---------- Integration tests (facade mocked; no live API) ----------
#
# These drive the full /summary/generate endpoint but patch the LLM facade
# (``get_provider``) so the canned provider response stands in for a live call.
# generate_summary still resolves the gemini path (provider=None), so a dummy
# key satisfies the in-function guard.


@pytest.mark.asyncio
async def test_generate_natural_language(client: AsyncClient, db_session: AsyncSession):
    """Generate NL summary via the facade and verify response shape."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=5)

    nl_text = (
        "Health record overview organized by category. The patient has several "
        "documented conditions, medications, and lab results across visits."
    )

    from app.config import settings

    original_key = settings.gemini_api_key
    settings.gemini_api_key = "test-key"
    try:
        with patch(
            "app.services.ai.summarizer.get_provider",
            return_value=_mock_provider(nl_text),
        ):
            resp = await client.post(
                "/api/v1/summary/generate",
                json={
                    "patient_id": str(patient.id),
                    "summary_type": "full",
                    "output_format": "natural_language",
                    "processing_mode": "cloud_assisted",
                },
                headers=headers,
            )
    finally:
        settings.gemini_api_key = original_key

    assert resp.status_code == 200
    data = resp.json()
    assert data["natural_language"] is not None
    assert len(data["natural_language"]) > 50
    assert data["record_count"] == 5
    assert data["model_used"] == "gemini-3.5-flash"
    assert data["processing_mode"] == "cloud_assisted"
    assert data["model_provenance"] == {
        "processing_mode": "cloud_assisted",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
    }
    assert set(data["typed_response"]) == {"sections", "uncertainties"}


@pytest.mark.asyncio
async def test_generate_route_never_returns_or_persists_remote_model_identity(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Remote model metadata cannot cross the API or persistence boundary."""
    headers, user_id = await auth_headers(
        client,
        email="remote-model-metadata@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=1)
    remote_model = "https://attacker.example/model?patient=secret"

    from app.config import settings

    expected_model = settings.gemini_model
    original_key = settings.gemini_api_key
    settings.gemini_api_key = "test-key"
    try:
        with patch(
            "app.services.ai.summarizer.get_provider",
            return_value=_mock_provider("Organized record facts.", model=remote_model),
        ):
            response = await client.post(
                "/api/v1/summary/generate",
                headers=headers,
                json={
                    "patient_id": str(patient.id),
                    "summary_type": "full",
                    "processing_mode": "cloud_assisted",
                    "output_format": "natural_language",
                },
            )
    finally:
        settings.gemini_api_key = original_key

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model_used"] == expected_model
    assert remote_model not in str(data)
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(AISummaryPrompt.id == UUID(data["id"]))
        )
    ).scalar_one()
    assert prompt.target_model == expected_model
    assert prompt.api_model_used == expected_model
    assert remote_model not in str(prompt.model_provenance)


@pytest.mark.asyncio
async def test_generate_rejects_unsafe_routed_output_without_persistence(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Unsafe provider prose returns only a safe error and creates no summary."""
    headers, user_id = await auth_headers(
        client,
        email="unsafe-routed-output@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=1)
    unsafe_outputs = [
        "You should stop metformin and increase the dose.",
        "Recommended: start insulin.",
        "Plan: start insulin.",
        "Start insulin now.",
        "Initiate insulin now.",
        "The patient is diagnosed with pneumonia.",
        "You may want to start insulin.",
        "You might consider beginning insulin.",
        "I suspect pneumonia.",
        "- Start insulin.",
        "Stаrt insulin.",
        "```text\nSt\\x61rt insulin.\n```",
        '```json\n{"D\\u006fuble the metformin dose.": true}\n```',
        'prefix {"D\\u006fuble the metformin dose.": true}',
        "**Start** insulin.",
        "<b>Start</b> insulin.",
        "Sтart insulin.",
        "You may want to **start** insulin.",
        "I **suspect** pneumonia.",
        "Starting insulin may help.",
        "Insulin may help.",
    ]

    from app.config import settings

    original_key = settings.gemini_api_key
    settings.gemini_api_key = "test-key"
    try:
        for unsafe_output in unsafe_outputs:
            with patch(
                "app.services.ai.summarizer.get_provider",
                return_value=_mock_provider(unsafe_output, grounded=False),
            ):
                response = await client.post(
                    "/api/v1/summary/generate",
                    headers=headers,
                    json={
                        "patient_id": str(patient.id),
                        "summary_type": "full",
                        "processing_mode": "cloud_assisted",
                        "output_format": "natural_language",
                    },
                )

            assert response.status_code == 400
            assert response.json()["detail"].startswith("Summary output")
            assert unsafe_output not in response.text
    finally:
        settings.gemini_api_key = original_key

    persisted = list(
        (
            await db_session.execute(
                select(AISummaryPrompt).where(
                    AISummaryPrompt.user_id == UUID(user_id),
                    AISummaryPrompt.patient_id == patient.id,
                )
            )
        )
        .scalars()
        .all()
    )
    assert persisted == []


@pytest.mark.parametrize(
    ("case_name", "unsafe_model"),
    [
        ("path", "models/qwen/model.gguf"),
        ("aws", "AKIAIOSFODNN7EXAMPLE"),
        ("stripe", STRIPE_SECRET_SHAPED_FIXTURE),
        ("opaque", "AbCdEfGh1234567890IjKlMnOpQrStUv"),
    ],
)
@pytest.mark.asyncio
async def test_generate_unsafe_configured_model_identity_persists_unreported_only(
    client: AsyncClient,
    db_session: AsyncSession,
    case_name: str,
    unsafe_model: str,
) -> None:
    """Path- and credential-shaped models never enter API or stored provenance."""
    headers, user_id = await auth_headers(
        client,
        email=f"unsafe-configured-model-{case_name}@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=1)

    from app.config import settings

    original_key = settings.gemini_api_key
    original_model = settings.gemini_model
    settings.gemini_api_key = "test-key"
    settings.gemini_model = unsafe_model
    try:
        with patch(
            "app.services.ai.summarizer.get_provider",
            return_value=_mock_provider("Records organized by category."),
        ):
            response = await client.post(
                "/api/v1/summary/generate",
                headers=headers,
                json={
                    "patient_id": str(patient.id),
                    "summary_type": "full",
                    "processing_mode": "cloud_assisted",
                    "output_format": "natural_language",
                },
            )
    finally:
        settings.gemini_api_key = original_key
        settings.gemini_model = original_model

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model_used"] == "unreported"
    assert data["model_provenance"] is None
    assert unsafe_model not in str(data)
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(AISummaryPrompt.id == UUID(data["id"]))
        )
    ).scalar_one()
    assert prompt.target_model == "unreported"
    assert prompt.api_model_used == "unreported"
    assert prompt.model_provenance is None


@pytest.mark.asyncio
async def test_generate_json_format(client: AsyncClient, db_session: AsyncSession):
    """Generate JSON summary via the facade and verify valid JSON returned."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=3)

    json_text = (
        '{"summary": "Brief overview", "categories": {"conditions": [], '
        '"medications": []}, "timeline_highlights": []}'
    )

    from app.config import settings

    original_key = settings.gemini_api_key
    settings.gemini_api_key = "test-key"
    try:
        with patch(
            "app.services.ai.summarizer.get_provider",
            return_value=_mock_provider(json_text),
        ):
            resp = await client.post(
                "/api/v1/summary/generate",
                json={
                    "patient_id": str(patient.id),
                    "summary_type": "full",
                    "output_format": "json",
                    "processing_mode": "cloud_assisted",
                },
                headers=headers,
            )
    finally:
        settings.gemini_api_key = original_key

    assert resp.status_code == 200
    data = resp.json()
    assert data["json_data"] is not None
    assert isinstance(data["json_data"], dict)


@pytest.mark.asyncio
async def test_generate_both_formats(client: AsyncClient, db_session: AsyncSession):
    """Generate both NL + JSON summary via the facade."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=3)

    both_text = (
        '{"natural_language": "A markdown summary of the records.", '
        '"structured_data": {"summary": "overview", "categories": {}}}'
    )

    from app.config import settings

    original_key = settings.gemini_api_key
    settings.gemini_api_key = "test-key"
    try:
        with patch(
            "app.services.ai.summarizer.get_provider",
            return_value=_mock_provider(both_text),
        ):
            resp = await client.post(
                "/api/v1/summary/generate",
                json={
                    "patient_id": str(patient.id),
                    "summary_type": "full",
                    "output_format": "both",
                    "processing_mode": "cloud_assisted",
                },
                headers=headers,
            )
    finally:
        settings.gemini_api_key = original_key

    assert resp.status_code == 200
    data = resp.json()
    # At least one of the two should be populated
    assert data["natural_language"] is not None or data["json_data"] is not None
