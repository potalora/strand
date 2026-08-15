from __future__ import annotations

from pathlib import Path

import pytest

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import parse_manifest
from app.services.local_ai.pipeline import MemoryCheckpointStore, StrictLocalPipeline
from tests.test_strict_local_pipeline import (
    _Manager,
    _job_and_upload,
    _rasterizer,
)


@pytest.mark.asyncio
async def test_pipeline_purges_plaintext_scratch_after_validation_failure(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()

    class MalformedExtractionManager(_Manager):
        async def run_attested(self, manifest, role, payload, on_progress=None):
            self._require_attested_manifest(manifest)
            if role.value == "extraction":
                return {"patient": {"name": "Private Name"}}
            return await super().run_attested(manifest, role, payload, on_progress)

    pipeline = StrictLocalPipeline(
        manager=MalformedExtractionManager(),
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
    )

    with pytest.raises(LocalValidationError):
        await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert not (tmp_path / "scratch" / str(job.id)).exists()
