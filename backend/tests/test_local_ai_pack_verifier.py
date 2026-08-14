"""Synthetic runtime/architecture fixture gate for model-pack activation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.pack_verifier import verify_pack_candidate
from app.services.local_ai.runtime_identity import WorkerRuntimeIdentity
from app.services.local_ai.types import ModelRole


def _manifest() -> LocalAIManifest:
    return LocalAIManifest(
        schema_version=2,
        pack_revision="apple-m4-16gb-v2",
        platform="apple_silicon",
        runtime={
            "name": "mlx-vlm",
            "version": "0.5.0",
            "worker_identity_scheme": "local-ai-worker-bundle.v1",
            "worker_bundle_sha256": "a" * 64,
        },
        validation_suite_version="local-ai-fixtures-v1",
        artifacts=tuple(
            ManifestArtifact(
                role=role,
                repository=f"owner/{role.value}",
                revision=str(index) * 40,
                quantization="4bit",
                license="apache-2.0",
                attribution=f"https://huggingface.co/owner/{role.value}",
                decode_limits={
                    "max_input_tokens": 32768,
                    "max_output_tokens": 4096,
                },
                files=(
                    ManifestFile(
                        path="model.safetensors",
                        sha256=hashlib.sha256(role.value.encode()).hexdigest(),
                        size=len(role.value),
                    ),
                ),
            )
            for index, role in enumerate(ModelRole, start=1)
        ),
    )


def _write_pack(path: Path, manifest: LocalAIManifest) -> None:
    for artifact in manifest.artifacts:
        for model_file in artifact.files:
            target = path / artifact.role.value / model_file.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(artifact.role.value.encode())


class FakeFixtureManager:
    def __init__(self, *, bad_ocr: bool = False) -> None:
        self.bad_ocr = bad_ocr
        self.started = False
        self.stopped = False
        self.roles: list[ModelRole] = []
        self.payloads: list[dict] = []
        self.manifests: list[LocalAIManifest] = []
        self.runtime_identity = WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256="a" * 64,
        )

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def run_attested(
        self,
        manifest: LocalAIManifest,
        role: ModelRole,
        payload: dict,
        _on_progress=None,
    ):
        self.manifests.append(manifest)
        self.roles.append(role)
        self.payloads.append(payload)
        if role is ModelRole.OCR:
            return {
                "markdown": "Hemoglobin A1c six point eight"
                if self.bad_ocr
                else "Hemoglobin A1c 6.8 %",
                "page_number": 1,
            }
        if role is ModelRole.EXTRACTION:
            return {
                "labs": [
                    {
                        "fact_id": "lab-1",
                        "name": "Hemoglobin A1c",
                        "value": "6.8",
                        "unit": "%",
                        "assertion": "present",
                        "verbatim": "Hemoglobin A1c 6.8 %",
                        "page_number": 1,
                        "evidence_excerpt": "Hemoglobin A1c 6.8 %",
                    }
                ]
            }
        fact = payload["facts"][0]
        evidence = payload["evidence"][0]
        return {
            "sections": [
                {
                    "heading": "Observations",
                    "claims": [
                        {
                            "fact_id": fact["fact_id"],
                            "field_paths": ["/name", "/value", "/unit"],
                            "evidence_ids": [evidence["evidence_id"]],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        }


@pytest.mark.asyncio
async def test_verifier_rejects_runtime_identity_mismatch_before_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    pack = tmp_path / "pack"
    _write_pack(pack, manifest)
    manager = FakeFixtureManager()
    manager.runtime_identity = WorkerRuntimeIdentity(
        scheme="local-ai-worker-bundle.v1",
        bundle_sha256="b" * 64,
    )
    monkeypatch.setattr(
        "app.services.local_ai.pack_verifier.platform_profile",
        lambda: ("apple_silicon", True),
    )

    with pytest.raises(LocalValidationError, match="runtime identity"):
        await verify_pack_candidate(
            manifest,
            pack,
            manager=manager,
            scratch_root=tmp_path / "scratch",
        )

    assert manager.started is False
    assert manager.roles == []


@pytest.mark.asyncio
async def test_default_verifier_shares_one_normalized_bare_command_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.local_ai import pack_verifier as verifier_module

    manifest = _manifest()
    pack = tmp_path / "pack"
    _write_pack(pack, manifest)
    project = tmp_path / "worker-project"
    project.mkdir()
    manager = FakeFixtureManager()
    resolved: list[tuple[tuple[str, ...], Path]] = []
    constructed: list[tuple[tuple[str, ...], Path]] = []

    def resolve(command, project_dir):
        normalized = (tuple(command), Path(project_dir))
        resolved.append(normalized)
        return manager.runtime_identity

    def construct(command, *, worker_project_dir):
        normalized = (tuple(command), Path(worker_project_dir))
        constructed.append(normalized)
        return manager

    monkeypatch.setattr(verifier_module.settings, "local_ai_worker_command", "python3")
    monkeypatch.setattr(
        verifier_module.settings,
        "local_ai_worker_project_dir",
        str(project),
    )
    monkeypatch.setattr(verifier_module, "resolve_worker_runtime_identity", resolve)
    monkeypatch.setattr(verifier_module, "LocalModelManager", construct)
    monkeypatch.setattr(
        verifier_module,
        "platform_profile",
        lambda: ("apple_silicon", True),
    )

    await verify_pack_candidate(
        manifest,
        pack,
        scratch_root=tmp_path / "scratch",
    )

    assert resolved == constructed
    assert len(resolved) == 1
    command, normalized_project = resolved[0]
    assert Path(command[0]).is_absolute()
    assert Path(command[0]).name == "python3"
    assert normalized_project == project.absolute()
    assert manager.manifests == [manifest, manifest, manifest]


@pytest.mark.asyncio
async def test_verifier_loads_all_roles_serially_and_runs_grounded_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    pack = tmp_path / "pack"
    _write_pack(pack, manifest)
    manager = FakeFixtureManager()
    monkeypatch.setattr(
        "app.services.local_ai.pack_verifier.platform_profile",
        lambda: ("apple_silicon", True),
    )

    await verify_pack_candidate(
        manifest,
        pack,
        manager=manager,
        scratch_root=tmp_path / "scratch",
    )

    assert manager.started is True
    assert manager.stopped is True
    assert manager.roles == [
        ModelRole.OCR,
        ModelRole.EXTRACTION,
        ModelRole.SUMMARY,
    ]
    assert manager.manifests == [manifest, manifest, manifest]
    assert all(
        Path(payload["model_dir"]) == pack.resolve() for payload in manager.payloads
    )
    assert all(
        payload["manifest_identity"]["revision"]
        == next(
            artifact.revision
            for artifact in manifest.artifacts
            if artifact.role is role
        )
        for role, payload in zip(manager.roles, manager.payloads, strict=True)
    )
    assert "scratch_dir" in manager.payloads[0]
    assert "scratch_dir" in manager.payloads[1]
    assert "scratch_dir" not in manager.payloads[2]


@pytest.mark.asyncio
async def test_verifier_canonicalizes_relative_scratch_paths_for_worker_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    pack = tmp_path / "pack"
    _write_pack(pack, manifest)
    manager = FakeFixtureManager()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "app.services.local_ai.pack_verifier.platform_profile",
        lambda: ("apple_silicon", True),
    )

    await verify_pack_candidate(
        manifest,
        pack,
        manager=manager,
        scratch_root=Path("relative-scratch"),
    )

    assert all(
        Path(payload["scratch_dir"]).is_absolute() for payload in manager.payloads[:2]
    )
    assert Path(manager.payloads[0]["image_path"]).is_absolute()
    assert Path(manager.payloads[0]["manifest_path"]).is_absolute()


@pytest.mark.asyncio
async def test_verifier_rejects_critical_numeric_ocr_drift_and_stops_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    pack = tmp_path / "pack"
    _write_pack(pack, manifest)
    manager = FakeFixtureManager(bad_ocr=True)
    monkeypatch.setattr(
        "app.services.local_ai.pack_verifier.platform_profile",
        lambda: ("apple_silicon", True),
    )

    with pytest.raises(LocalValidationError, match="fixture"):
        await verify_pack_candidate(
            manifest,
            pack,
            manager=manager,
            scratch_root=tmp_path / "scratch",
        )

    assert manager.stopped is True
    assert manager.roles == [ModelRole.OCR]


@pytest.mark.asyncio
async def test_verifier_fails_closed_on_incompatible_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    pack = tmp_path / "pack"
    _write_pack(pack, manifest)
    manager = FakeFixtureManager()
    monkeypatch.setattr(
        "app.services.local_ai.pack_verifier.platform_profile",
        lambda: ("unsupported", False),
    )

    with pytest.raises(LocalValidationError, match="platform"):
        await verify_pack_candidate(
            manifest,
            pack,
            manager=manager,
            scratch_root=tmp_path / "scratch",
        )

    assert manager.started is False
