"""Database-free legacy runtime-admission and state-transition regressions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.local_ai import _retry_local_ai_job, cancel_local_ai_job
from app.api.upload import (
    _mark_cancelled,
    _recover_stuck_files,
    _refresh_strict_local_job_lease,
    _run_strict_local_ingestion_for_upload,
    cancel_extraction,
)
from app.main import (
    _recover_strict_local_summary_jobs_on_startup,
    _recover_unstructured_jobs_on_startup,
)
from app.models.local_ai import LocalAIJob, _reject_persisted_job_identity_changes
from app.services.local_ai.errors import (
    LocalAIError,
    LocalPolicyError,
    RuntimeIdentityRequiredError,
)
from app.services.local_ai.manifest import canonicalize_manifest_snapshot, parse_manifest
from app.services.local_ai.processing_snapshot import (
    fail_active_legacy_ingestion_jobs,
    fail_legacy_runtime_identity_required,
)
from app.schemas.upload import CancelExtractionRequest
from app.services.ai.summarizer import (
    _claim_strict_summary_job,
    _finish_strict_summary_job,
    generate_grounded_local_summary,
    requeue_interrupted_summary_jobs,
    resume_grounded_local_summary_jobs,
)


def _legacy_snapshot() -> tuple[dict[str, object], str]:
    raw: dict[str, object] = {
        "schema_version": 1,
        "pack_revision": "apple-m4-16gb-v1",
        "platform": "apple_silicon",
        "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        "validation_suite_version": "local-ai-fixtures-v1",
        "artifacts": [
            {
                "role": role,
                "repository": f"owner/{role}",
                "revision": str(index) * 40,
                "quantization": "4bit",
                "license": "apache-2.0",
                "attribution": f"https://huggingface.co/owner/{role}",
                "decode_limits": {
                    "max_input_tokens": 32768,
                    "max_output_tokens": 4096,
                },
                "files": [
                    {
                        "path": "model.safetensors",
                        "sha256": hashlib.sha256(role.encode()).hexdigest(),
                        "size": len(role),
                    }
                ],
            }
            for index, role in enumerate(("ocr", "extraction", "summary"), start=1)
        ],
    }
    diagnostic = parse_manifest(raw, allow_legacy_diagnostic=True)
    return diagnostic.canonical_snapshot, diagnostic.canonical_sha256


def _persisted_legacy_job(*, kind: str = "ingestion") -> LocalAIJob:
    snapshot, digest = _legacy_snapshot()
    current = deepcopy(snapshot)
    current["schema_version"] = 2
    current["runtime"].update(
        worker_identity_scheme="local-ai-worker-bundle.v1",
        worker_bundle_sha256="a" * 64,
    )
    current_snapshot, current_digest = canonicalize_manifest_snapshot(current)
    job = LocalAIJob(
        user_id=uuid4(),
        upload_id=uuid4() if kind == "ingestion" else None,
        summary_prompt_id=uuid4() if kind == "summary" else None,
        kind=kind,
        processing_mode="validated_strict_local",
        manifest_snapshot=current_snapshot,
        manifest_sha256=current_digest,
        status="queued",
        stage="queued",
        progress={},
        failure=None,
        audit_metadata={},
        cancel_requested=False,
        started_at=None,
        completed_at=None,
        created_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    job.id = uuid4()
    # Emulate a row loaded from the pre-upgrade database without invoking the
    # constructor's v2-only admission validation.
    job.__dict__["manifest_snapshot"] = deepcopy(snapshot)
    job.__dict__["manifest_sha256"] = digest
    return job


def test_runtime_identity_required_error_has_fixed_policy_code() -> None:
    error = RuntimeIdentityRequiredError(
        "Strict-local worker runtime identity is required."
    )

    assert error.code == "runtime_identity_required"
    assert error.retryable is False


def test_legacy_job_terminalization_preserves_snapshot_and_digest() -> None:
    job = _persisted_legacy_job()
    original_snapshot_bytes = json.dumps(job.manifest_snapshot, sort_keys=True).encode()
    original_digest = job.manifest_sha256
    completed_at = datetime(2026, 8, 14, tzinfo=timezone.utc)

    assert fail_legacy_runtime_identity_required(job, completed_at=completed_at)

    assert json.dumps(job.manifest_snapshot, sort_keys=True).encode() == original_snapshot_bytes
    assert job.manifest_sha256 == original_digest
    assert job.status == "failed"
    assert job.stage == "failed"
    assert job.progress == {"stage": "failed"}
    assert job.failure == {
        "stage": "failed",
        "code": "runtime_identity_required",
        "retryable": False,
        "checkpoint_preserved": False,
        "cloud_fallback_attempted": False,
    }
    assert job.completed_at == completed_at


def test_legacy_terminalization_helper_is_one_way_and_not_reauthorizing() -> None:
    job = _persisted_legacy_job()

    assert fail_legacy_runtime_identity_required(job)
    assert fail_legacy_runtime_identity_required(job) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("progress", {"stage": "failed", "current": 1}),
        (
            "failure",
            {
                "stage": "failed",
                "code": "runtime_identity_required",
                "retryable": False,
                "checkpoint_preserved": False,
                "cloud_fallback_attempted": False,
                "extra": "forbidden",
            },
        ),
        ("completed_at", datetime(2026, 8, 15, tzinfo=timezone.utc)),
        ("cancel_requested", True),
        ("audit_metadata", {"changed": True}),
    ],
)
def test_terminalized_legacy_job_rejects_every_later_mutation(
    field: str,
    value: object,
) -> None:
    job = _persisted_legacy_job()
    state = job._sa_instance_state
    state._commit_all(job.__dict__)
    assert fail_legacy_runtime_identity_required(job)
    _reject_persisted_job_identity_changes(None, None, job)
    state._commit_all(job.__dict__)

    setattr(job, field, value)

    with pytest.raises(Exception, match="transition"):
        _reject_persisted_job_identity_changes(None, None, job)


@pytest.mark.parametrize("kind", ["ingestion", "summary"])
def test_active_legacy_job_cannot_race_into_completed(kind: str) -> None:
    job = _persisted_legacy_job(kind=kind)
    state = job._sa_instance_state
    state._commit_all(job.__dict__)

    job.status = "completed"
    job.stage = "completed"
    job.progress = {"stage": "completed"}
    job.completed_at = datetime(2026, 8, 14, tzinfo=timezone.utc)

    with pytest.raises(Exception, match="transition"):
        _reject_persisted_job_identity_changes(None, None, job)


@pytest.mark.asyncio
async def test_failed_legacy_summary_job_cannot_be_retried_or_mutated() -> None:
    job = _persisted_legacy_job(kind="summary")
    fail_legacy_runtime_identity_required(job)
    original = deepcopy(
        {key: value for key, value in job.__dict__.items() if key != "_sa_instance_state"}
    )

    with pytest.raises(HTTPException) as captured:
        await _retry_local_ai_job(object(), job=job)

    assert captured.value.status_code == 409
    assert {
        key: value for key, value in job.__dict__.items() if key != "_sa_instance_state"
    } == original


@pytest.mark.asyncio
async def test_retry_rejects_preexisting_retryable_legacy_summary_without_mutation() -> None:
    job = _persisted_legacy_job(kind="summary")
    job.status = "failed"
    job.stage = "failed"
    job.failure = {"code": "local_worker_error", "retryable": True}
    original = deepcopy(
        {key: value for key, value in job.__dict__.items() if key != "_sa_instance_state"}
    )

    with pytest.raises(HTTPException) as captured:
        await _retry_local_ai_job(object(), job=job)

    assert captured.value.status_code == 409
    assert {
        key: value for key, value in job.__dict__.items() if key != "_sa_instance_state"
    } == original


class _RowsResult:
    def __init__(self, rows: list[object], *, rowcount: int = 0) -> None:
        self.rows = rows
        self.rowcount = rowcount

    def scalars(self) -> _RowsResult:
        return self

    def all(self) -> list[object]:
        return self.rows

    def first(self) -> object | None:
        return self.rows[0] if self.rows else None

    def scalar_one_or_none(self) -> object | None:
        return self.rows[0] if self.rows else None


class _FakeSession:
    def __init__(self, batches: list[list[object] | Exception]) -> None:
        self.batches = list(batches)
        self.executed: list[str] = []
        self.commits = 0
        self.flushes = 0
        self.rollbacks = 0
        self.get_values: list[object] = []
        self.get_calls: list[dict[str, object]] = []

    async def execute(
        self,
        statement: object,
        _parameters: object | None = None,
    ) -> _RowsResult:
        self.executed.append(str(statement))
        batch = self.batches.pop(0) if self.batches else []
        if isinstance(batch, Exception):
            raise batch
        return _RowsResult(batch)

    async def commit(self) -> None:
        self.commits += 1

    async def flush(self) -> None:
        self.flushes += 1

    async def refresh(self, _value: object) -> None:
        return None

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def get(
        self,
        _model: object,
        _identity: object,
        **kwargs: object,
    ) -> object | None:
        self.get_calls.append(kwargs)
        return self.get_values.pop(0) if self.get_values else None


class _SessionContext:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_interrupted_summary_recovery_fails_legacy_instead_of_requeueing() -> None:
    job = _persisted_legacy_job(kind="summary")
    session = _FakeSession([[job], []])

    resumed = await requeue_interrupted_summary_jobs(
        session_factory=lambda: _SessionContext(session)
    )

    assert resumed == []
    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert "FOR UPDATE" in session.executed[0]


@pytest.mark.asyncio
async def test_startup_summary_recovery_fails_legacy_instead_of_resuming() -> None:
    job = _persisted_legacy_job(kind="summary")
    session = _FakeSession([[job], []])

    resumed = await _recover_strict_local_summary_jobs_on_startup(session)

    assert resumed == []
    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert "FOR UPDATE" in session.executed[0]


@pytest.mark.asyncio
async def test_processing_legacy_cancel_fails_without_worker_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _persisted_legacy_job(kind="summary")
    job.status = "processing"
    job.stage = "summary"
    session = _FakeSession([[job], [job]])
    cancellation_calls: list[str] = []

    async def _audit(*_args: object, **_kwargs: object) -> None:
        return None

    async def _cancel_registered(job_id: str) -> bool:
        cancellation_calls.append(job_id)
        return True

    monkeypatch.setattr("app.api.local_ai.log_audit_event", _audit)
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
        _cancel_registered,
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "scheme": "http",
            "server": ("test", 80),
            "query_string": b"",
        }
    )

    await cancel_local_ai_job(job.id, request, job.user_id, session)

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert cancellation_calls == []


@pytest.mark.parametrize("status", ["queued", "processing"])
@pytest.mark.asyncio
async def test_direct_ingestion_cancel_pairs_legacy_upload_failure_atomically(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    job = _persisted_legacy_job()
    job.status = status
    job.stage = status
    upload = _legacy_upload(job)
    session = _FakeSession([[job], [upload], [job]])
    cancellation_calls: list[str] = []

    async def _audit(*_args: object, **_kwargs: object) -> None:
        return None

    async def _cancel_registered(job_id: str) -> bool:
        cancellation_calls.append(job_id)
        return True

    monkeypatch.setattr("app.api.local_ai.log_audit_event", _audit)
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
        _cancel_registered,
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "scheme": "http",
            "server": ("test", 80),
            "query_string": b"",
        }
    )

    await cancel_local_ai_job(job.id, request, job.user_id, session)

    assert len(session.executed) == 3
    assert "uploaded_files" in session.executed[1]
    assert "local_ai_jobs" in session.executed[2]
    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert job.cancel_requested is False
    assert upload.ingestion_status == "failed"
    assert upload.progress_stage is None
    assert upload.progress_detail is None
    assert upload.ingestion_errors == [
        {"error_type": "runtime_identity_required"}
    ]
    assert upload.processing_completed_at == job.completed_at
    assert cancellation_calls == []


@pytest.mark.asyncio
async def test_summary_failure_terminalizer_prefers_runtime_identity_failure() -> None:
    job = _persisted_legacy_job(kind="summary")
    job.status = "processing"
    job.stage = "summary"
    session = _FakeSession([[job]])

    await _finish_strict_summary_job(
        session,
        job_id=job.id,
        error=LocalAIError("synthetic"),
    )

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"


@pytest.mark.asyncio
async def test_summary_resume_fails_legacy_before_loading_prompt() -> None:
    job = _persisted_legacy_job(kind="summary")
    session = _FakeSession([])
    session.get_values = [job]

    await resume_grounded_local_summary_jobs(
        [job.id],
        session_factory=lambda: _SessionContext(session),
    )

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert session.get_values == []
    assert session.get_calls[0].get("with_for_update") is True


@pytest.mark.asyncio
async def test_summary_admission_fails_legacy_before_evidence_or_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _persisted_legacy_job(kind="summary")
    session = _FakeSession([[job]])

    async def _not_claimed(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "app.services.ai.summarizer._claim_strict_summary_job",
        _not_claimed,
    )

    with pytest.raises(RuntimeIdentityRequiredError):
        await generate_grounded_local_summary(
            session,
            user_id=job.user_id,
            patient_id=uuid4(),
            job_id=job.id,
            summary_type="full",
        )

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert "FOR UPDATE" in session.executed[0]


def _legacy_upload(job: LocalAIJob) -> SimpleNamespace:
    return SimpleNamespace(
        id=job.upload_id,
        user_id=job.user_id,
        processing_mode="validated_strict_local",
        processing_manifest=deepcopy(job.manifest_snapshot),
        ingestion_status="processing",
        cancel_requested=False,
        progress_stage="local_ocr",
        progress_detail={"stage": "ocr"},
        ingestion_errors=[],
        processing_completed_at=None,
    )


@pytest.mark.asyncio
async def test_mark_cancelled_fails_legacy_instead_of_cancelling() -> None:
    job = _persisted_legacy_job()
    upload = _legacy_upload(job)
    session = _FakeSession([[upload.id], [job]])

    await _mark_cancelled(session, upload)

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert upload.ingestion_status == "failed"


@pytest.mark.asyncio
async def test_bulk_cancel_fails_legacy_instead_of_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _persisted_legacy_job()
    upload = _legacy_upload(job)
    session = _FakeSession([[upload], [job]])
    cancellation_calls: list[str] = []

    async def _cancel_registered(job_id: str) -> bool:
        cancellation_calls.append(job_id)
        return True

    async def _audit(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
        _cancel_registered,
    )
    monkeypatch.setattr("app.api.upload.log_audit_event", _audit)

    await cancel_extraction(
        CancelExtractionRequest(upload_ids=[str(upload.id)]),
        job.user_id,
        session,
    )

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert upload.ingestion_status == "failed"
    assert cancellation_calls == []


@pytest.mark.asyncio
async def test_ingestion_admission_fails_legacy_before_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _persisted_legacy_job()
    upload = _legacy_upload(job)
    session = _FakeSession([[upload.id], [job]])
    session.get_values = [job, upload]

    async def _terminal_lock(*_args: object, **_kwargs: object) -> bool:
        return False

    class _NoCheckpoints:
        def __init__(self, _db: object) -> None:
            pass

        async def count_pages(self, _job_id: object) -> int:
            return 0

    monkeypatch.setattr("app.api.upload._lock_strict_terminal_state", _terminal_lock)
    monkeypatch.setattr("app.api.upload.CheckpointStore", _NoCheckpoints, raising=False)

    with pytest.raises(RuntimeIdentityRequiredError):
        await _run_strict_local_ingestion_for_upload(
            session,
            upload,
            Path("/synthetic/not-opened.pdf"),
            job.user_id,
        )

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"


@pytest.mark.asyncio
async def test_mark_cancelled_rollback_fallback_fails_legacy() -> None:
    job = _persisted_legacy_job()
    upload = _legacy_upload(job)
    session = _FakeSession([RuntimeError("synthetic poisoned session"), [], [job]])

    await _mark_cancelled(session, upload)

    assert session.rollbacks == 1
    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"


@pytest.mark.asyncio
async def test_stuck_file_recovery_terminalizes_legacy_before_raw_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _persisted_legacy_job()
    upload = _legacy_upload(job)
    session = _FakeSession([[job], [upload], [job]])
    monkeypatch.setattr(
        "app.api.upload.async_session_factory",
        lambda: _SessionContext(session),
    )

    await _recover_stuck_files()

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert upload.ingestion_status == "failed"
    assert "FOR UPDATE" not in session.executed[0]
    assert "uploaded_files" in session.executed[1]
    assert "FOR UPDATE" in session.executed[1]
    assert "local_ai_jobs" in session.executed[2]
    assert "FOR UPDATE" in session.executed[2]
    job_updates = [
        statement
        for statement in session.executed
        if "UPDATE local_ai_jobs" in statement
    ]
    assert job_updates
    assert all(
        "manifest_snapshot" in statement and "schema_version" in statement
        for statement in job_updates
    )


@pytest.mark.asyncio
async def test_startup_ingestion_recovery_terminalizes_legacy_before_raw_updates() -> None:
    job = _persisted_legacy_job()
    upload = _legacy_upload(job)
    session = _FakeSession([[job], [upload], [job]])

    await _recover_unstructured_jobs_on_startup(session)

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert upload.ingestion_status == "failed"
    assert "FOR UPDATE" not in session.executed[0]
    assert "uploaded_files" in session.executed[1]
    assert "FOR UPDATE" in session.executed[1]
    assert "local_ai_jobs" in session.executed[2]
    assert "FOR UPDATE" in session.executed[2]
    job_updates = [
        statement
        for statement in session.executed
        if "UPDATE local_ai_jobs" in statement
    ]
    assert job_updates
    assert all(
        "manifest_snapshot" in statement and "schema_version" in statement
        for statement in job_updates
    )


@pytest.mark.parametrize("recovery_path", ["periodic", "startup"])
@pytest.mark.asyncio
async def test_cancel_requested_legacy_recovery_preserves_paired_failure(
    monkeypatch: pytest.MonkeyPatch,
    recovery_path: str,
) -> None:
    job = _persisted_legacy_job()
    upload = _legacy_upload(job)
    upload.cancel_requested = True
    original_snapshot = deepcopy(job.manifest_snapshot)
    original_digest = job.manifest_sha256

    class _CancellationSqlSession(_FakeSession):
        async def execute(
            self,
            statement: object,
            parameters: object | None = None,
        ) -> _RowsResult:
            result = await super().execute(statement, parameters)
            rendered = str(statement)
            if (
                rendered.lstrip().startswith("UPDATE uploaded_files")
                and "ingestion_status = 'cancelled'" in rendered
                and "IS DISTINCT FROM '2'" not in rendered
            ):
                upload.ingestion_status = "cancelled"
                upload.ingestion_errors = []
            return result

    session = _CancellationSqlSession([[job], [upload], [job]])
    if recovery_path == "periodic":
        monkeypatch.setattr(
            "app.api.upload.async_session_factory",
            lambda: _SessionContext(session),
        )
        await _recover_stuck_files()
    else:
        await _recover_unstructured_jobs_on_startup(session)

    assert job.status == "failed"
    assert job.failure["code"] == "runtime_identity_required"
    assert upload.ingestion_status == "failed"
    assert upload.ingestion_errors == [
        {"error_type": "runtime_identity_required"}
    ]
    assert job.manifest_snapshot == original_snapshot
    assert job.manifest_sha256 == original_digest
    assert "FOR UPDATE" not in session.executed[0]
    assert "uploaded_files" in session.executed[1]
    assert "FOR UPDATE" in session.executed[1]
    assert "local_ai_jobs" in session.executed[2]
    assert "FOR UPDATE" in session.executed[2]
    cancellation_updates = [
        statement
        for statement in session.executed
        if statement.lstrip().startswith("UPDATE uploaded_files")
        and "ingestion_status = 'cancelled'" in statement
    ]
    assert len(cancellation_updates) == 1
    assert "NOT EXISTS" in cancellation_updates[0]
    assert "IS DISTINCT FROM '2'" in cancellation_updates[0]


@pytest.mark.asyncio
async def test_legacy_ingestion_recovery_locks_pairs_in_stable_upload_order() -> None:
    first = _persisted_legacy_job()
    second = _persisted_legacy_job()
    ordered_jobs = sorted([first, second], key=lambda item: (item.upload_id, item.id))
    uploads = {job.upload_id: _legacy_upload(job) for job in ordered_jobs}
    jobs = {job.id: job for job in ordered_jobs}

    class _OrderingSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__([])
            self.locked_upload_ids: list[object] = []
            self.discovery_done = False

        async def execute(
            self,
            statement: object,
            _parameters: object | None = None,
        ) -> _RowsResult:
            rendered = str(statement)
            self.executed.append(rendered)
            parameters = statement.compile().params
            if not self.discovery_done:
                self.discovery_done = True
                discovered = ordered_jobs if "ORDER BY" in rendered else ordered_jobs[::-1]
                return _RowsResult(discovered)
            upload_id = next(
                (
                    value
                    for value in parameters.values()
                    if isinstance(value, type(first.upload_id)) and value in uploads
                ),
                None,
            )
            if "uploaded_files" in rendered:
                self.locked_upload_ids.append(upload_id)
                return _RowsResult([uploads[upload_id]])
            job_id = next(
                (
                    value
                    for value in parameters.values()
                    if isinstance(value, type(first.id)) and value in jobs
                ),
                None,
            )
            return _RowsResult([jobs[job_id]])

    session = _OrderingSession()

    await fail_active_legacy_ingestion_jobs(
        session,
        completed_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert session.locked_upload_ids == [job.upload_id for job in ordered_jobs]


class _LegacyRaceSession(_FakeSession):
    def __init__(self, job: LocalAIJob) -> None:
        super().__init__([])
        self.job = job

    async def execute(
        self,
        statement: object,
        _parameters: object | None = None,
    ) -> _RowsResult:
        rendered = str(statement)
        self.executed.append(rendered)
        if "manifest_snapshot" in rendered:
            return _RowsResult([])
        if rendered.lstrip().startswith("UPDATE"):
            return _RowsResult([self.job.id])
        return _RowsResult([self.job])


@pytest.mark.asyncio
async def test_ingestion_lease_cannot_refresh_legacy_race() -> None:
    job = _persisted_legacy_job()
    job.status = "processing"
    job.started_at = datetime(2026, 8, 14, tzinfo=timezone.utc)
    session = _LegacyRaceSession(job)

    with pytest.raises(LocalPolicyError):
        await _refresh_strict_local_job_lease(
            session,
            upload_id=job.upload_id,
            job_id=job.id,
            user_id=job.user_id,
            claim_started_at=job.started_at,
        )


@pytest.mark.asyncio
async def test_summary_claim_cannot_promote_legacy_race() -> None:
    job = _persisted_legacy_job(kind="summary")
    session = _LegacyRaceSession(job)

    claimed = await _claim_strict_summary_job(
        session,
        job_id=job.id,
        user_id=job.user_id,
    )

    assert claimed is None
