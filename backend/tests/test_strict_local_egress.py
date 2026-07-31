from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.local_ai.errors import LocalWorkerError
from app.services.local_ai.manifest import parse_manifest
from app.services.local_ai.pipeline import MemoryCheckpointStore, StrictLocalPipeline
from app.services.local_ai.types import ModelRole
from app.utils.file_utils import encrypt_stream
from tests.test_strict_local_pipeline import (
    _Manager,
    _job_and_upload,
    _rasterizer,
)


def _encrypted_source(tmp_path: Path) -> tuple[Path, str]:
    plaintext = b"%PDF-1.4\nstrict local egress test\n%%EOF\n"
    source = tmp_path / "encrypted.pdf"
    source.write_bytes(b"".join(encrypt_stream([plaintext])))
    return source, hashlib.sha256(plaintext).hexdigest()


def test_fresh_strict_pipeline_import_does_not_load_langextract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import app.services.local_ai.pipeline; "
                "assert 'langextract' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_fresh_upload_api_import_does_not_load_cloud_sdk_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import app.api.upload; "
                "forbidden = ('openai', 'anthropic', 'google.genai'); "
                "loaded = sorted(name for name in sys.modules "
                "if any(name == root or name.startswith(root + '.') "
                "for root in forbidden)); "
                "assert not loaded, loaded"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_strict_pipeline_constructs_no_provider_and_attempts_no_egress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, digest = _encrypted_source(tmp_path)
    job, upload = _job_and_upload(file_hash=digest)
    manager = _Manager()
    external_attempts: list[object] = []

    def fail_provider(*_args, **_kwargs):
        raise AssertionError("cloud provider was constructed")

    def fail_connect(*args, **_kwargs):
        external_attempts.append((args, _kwargs))
        raise AssertionError("network connection was attempted")

    monkeypatch.setenv("GEMINI_API_KEY", "configured-but-forbidden")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-forbidden")
    monkeypatch.setattr("app.services.ai.llm.registry.get_provider", fail_provider)
    monkeypatch.setattr("socket.socket.connect", fail_connect)
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
    )

    await pipeline.run_ingestion(job, upload, source)

    assert external_attempts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_role", [ModelRole.OCR, ModelRole.EXTRACTION])
async def test_local_worker_failure_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_role: ModelRole,
) -> None:
    source, digest = _encrypted_source(tmp_path)
    job, upload = _job_and_upload(file_hash=digest)
    provider_calls = 0

    def fail_provider(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("cloud provider was constructed")

    monkeypatch.setattr("app.services.ai.llm.registry.get_provider", fail_provider)
    pipeline = StrictLocalPipeline(
        manager=_Manager(failure_role=failure_role),
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
    )

    with pytest.raises(LocalWorkerError):
        await pipeline.run_ingestion(job, upload, source)

    assert provider_calls == 0
    assert upload.processing_mode == "validated_strict_local"
