"""Security and atomicity tests for the local-AI artifact store."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.types import ModelRole


def _manifest(
    pack_revision: str,
    *,
    contents: dict[ModelRole, bytes] | None = None,
) -> tuple[LocalAIManifest, dict[ModelRole, bytes]]:
    file_contents = contents or {
        ModelRole.OCR: b"ocr-weights",
        ModelRole.EXTRACTION: b"extraction-weights",
        ModelRole.SUMMARY: b"summary-weights",
    }
    artifacts = tuple(
        ManifestArtifact(
            role=role,
            repository=f"owner/{role.value}",
            revision=str(index) * 40,
            quantization="4bit",
            license="apache-2.0",
            attribution=f"https://huggingface.co/owner/{role.value}",
            decode_limits={"max_input_tokens": 4096, "max_output_tokens": 1024},
            files=(
                ManifestFile(
                    path="weights/model.safetensors",
                    sha256=hashlib.sha256(file_contents[role]).hexdigest(),
                    size=len(file_contents[role]),
                ),
            ),
        )
        for index, role in enumerate(ModelRole, start=1)
    )
    return (
        LocalAIManifest(
            schema_version=1,
            pack_revision=pack_revision,
            platform="apple_silicon",
            runtime={"name": "mlx-vlm", "version": "0.5.0"},
            validation_suite_version="fixtures-v1",
            artifacts=artifacts,
        ),
        file_contents,
    )


def _write_manifest_files(
    root: Path,
    manifest: LocalAIManifest,
    contents: dict[ModelRole, bytes],
) -> None:
    for artifact in manifest.artifacts:
        for file in artifact.files:
            path = root / artifact.role.value / file.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[artifact.role])


def _install(
    store: ArtifactStore,
    revision: str,
) -> tuple[LocalAIManifest, dict[ModelRole, bytes]]:
    manifest, contents = _manifest(revision)
    stage = store.stage(revision)
    _write_manifest_files(stage, manifest, contents)
    store.activate(stage, manifest)
    return manifest, contents


def test_activation_is_atomic_and_preserves_previous_pack(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    manifest, contents = _manifest("second")
    stage = store.stage("second")
    _write_manifest_files(stage, manifest, contents)
    target = stage / ModelRole.OCR.value / "weights/model.safetensors"
    target.write_bytes(b"x" * len(contents[ModelRole.OCR]))

    with pytest.raises(LocalValidationError, match="SHA-256"):
        store.activate(stage, manifest)

    assert store.active_revision() == "first"
    assert not (tmp_path / "packs" / "second").exists()


def test_activation_rejects_a_stale_active_pointer_without_replacing_it(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    stale = {"pack_revision": "missing", "manifest_sha256": "a" * 64}
    (tmp_path / "active.json").write_text(json.dumps(stale), encoding="utf-8")
    manifest, contents = _manifest("new")
    stage = store.stage("new")
    _write_manifest_files(stage, manifest, contents)

    with pytest.raises(LocalValidationError, match="verified"):
        store.activate(stage, manifest)

    assert json.loads((tmp_path / "active.json").read_text()) == stale
    assert not (tmp_path / "previous.json").exists()


def test_successive_activation_preserves_previous_and_rollback_verifies_it(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    _install(store, "second")

    store.rollback()

    assert store.active_revision() == "first"
    assert (
        json.loads((tmp_path / "previous.json").read_text())["pack_revision"]
        == "second"
    )


def test_rollback_rejects_a_pack_changed_after_activation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    _install(store, "second")
    installed = tmp_path / "packs" / "first" / "ocr" / "weights/model.safetensors"
    installed.write_bytes(b"changed")

    with pytest.raises(LocalValidationError, match="verified"):
        store.rollback()

    assert store.active_revision() == "second"


def test_store_rejects_symlinked_artifact_and_symlinked_parent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    manifest, contents = _manifest("bad")
    stage = store.stage("bad")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.safetensors").write_bytes(contents[ModelRole.OCR])
    role_dir = stage / ModelRole.OCR.value
    role_dir.symlink_to(outside, target_is_directory=True)
    _write_manifest_files(
        stage,
        replace(
            manifest,
            artifacts=tuple(
                artifact
                for artifact in manifest.artifacts
                if artifact.role is not ModelRole.OCR
            ),
        ),
        contents,
    )

    with pytest.raises(LocalValidationError, match="symlink"):
        store.verify(stage, manifest)


def test_store_rejects_hardlinks_extras_missing_files_and_non_regular_files(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    manifest, contents = _manifest("checks")

    missing = store.stage("missing")
    _write_manifest_files(missing, manifest, contents)
    (missing / "summary" / "weights/model.safetensors").unlink()
    with pytest.raises(LocalValidationError, match="file set"):
        store.verify(missing, replace(manifest, pack_revision="missing"))

    extras = store.stage("extras")
    _write_manifest_files(extras, manifest, contents)
    (extras / "ocr" / "extra.json").write_text("{}")
    with pytest.raises(LocalValidationError, match="file set"):
        store.verify(extras, replace(manifest, pack_revision="extras"))

    hardlink = store.stage("hardlink")
    _write_manifest_files(hardlink, manifest, contents)
    alias = hardlink / "alias"
    os.link(hardlink / "ocr" / "weights/model.safetensors", alias)
    with pytest.raises(LocalValidationError, match="hard link"):
        store.verify(hardlink, replace(manifest, pack_revision="hardlink"))

    fifo = store.stage("fifo")
    _write_manifest_files(fifo, manifest, contents)
    os.mkfifo(fifo / "extra")
    with pytest.raises(LocalValidationError, match="regular"):
        store.verify(fifo, replace(manifest, pack_revision="fifo"))


def test_store_revalidates_manifest_paths_sizes_hashes_and_aggregate_limit(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    manifest, contents = _manifest("bounds")
    stage = store.stage("bounds")
    _write_manifest_files(stage, manifest, contents)

    duplicate = replace(
        manifest.artifacts[0],
        files=manifest.artifacts[0].files * 2,
    )
    with pytest.raises(LocalValidationError, match="duplicate"):
        store.verify(
            stage, replace(manifest, artifacts=(duplicate, *manifest.artifacts[1:]))
        )

    escaped = replace(
        manifest.artifacts[0],
        files=(replace(manifest.artifacts[0].files[0], path="../escape.json"),),
    )
    with pytest.raises(LocalValidationError, match="path"):
        store.verify(
            stage, replace(manifest, artifacts=(escaped, *manifest.artifacts[1:]))
        )

    wrong_size = replace(
        manifest.artifacts[0],
        files=(replace(manifest.artifacts[0].files[0], size=1),),
    )
    with pytest.raises(LocalValidationError, match="size"):
        store.verify(
            stage, replace(manifest, artifacts=(wrong_size, *manifest.artifacts[1:]))
        )

    too_large = replace(
        manifest.artifacts[0],
        files=(replace(manifest.artifacts[0].files[0], size=8 * 1024**3 + 1),),
    )
    with pytest.raises(LocalValidationError, match="limit"):
        store.verify(
            stage, replace(manifest, artifacts=(too_large, *manifest.artifacts[1:]))
        )


def test_store_rejects_every_malformed_manifest_trust_field_before_activation(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    manifest, contents = _manifest("metadata")
    stage = store.stage("metadata")
    _write_manifest_files(stage, manifest, contents)
    first = manifest.artifacts[0]
    artifact_mutations = (
        replace(first, quantization=""),
        replace(first, license="proprietary"),
        replace(first, license="Apache-2.0"),
        replace(first, attribution=""),
        replace(
            first,
            decode_limits={"max_input_tokens": 0, "max_output_tokens": 1024},
        ),
    )

    for malformed in artifact_mutations:
        with pytest.raises(LocalValidationError, match="metadata"):
            store.verify(
                stage,
                replace(manifest, artifacts=(malformed, *manifest.artifacts[1:])),
            )

    with pytest.raises(LocalValidationError, match="manifest"):
        store.verify(stage, replace(manifest, validation_suite_version="bad\nvalue"))


def test_store_requires_exact_staging_revision_association(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    manifest, contents = _manifest("pack")
    stage = store.stage("pack-two")
    _write_manifest_files(stage, manifest, contents)

    with pytest.raises(LocalValidationError, match="revision"):
        store.verify(stage, manifest)


def test_store_rejects_symlinked_root_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(LocalValidationError, match="symlink"):
        ArtifactStore(alias / "models")

    assert not (real_parent / "models").exists()


@pytest.mark.parametrize("revision", ["../escape", "/absolute", "", "UPPER", "a/b"])
def test_store_validates_pack_revisions_before_path_use(
    tmp_path: Path,
    revision: str,
) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(LocalValidationError, match="revision"):
        store.stage(revision)

    assert not (tmp_path.parent / "escape").exists()


def test_stage_is_owner_only_and_rejects_an_installed_revision(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "installed")

    with pytest.raises(LocalValidationError, match="already exists"):
        store.stage("installed")

    stage = store.stage("new")
    assert stage.stat().st_mode & 0o777 == 0o700


def test_role_removal_clears_affected_pointer_and_preserves_other_pack(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    _install(store, "second")

    store.remove(ModelRole.SUMMARY)

    assert store.active_revision() is None
    assert not (tmp_path / "packs" / "second" / "summary").exists()
    assert (tmp_path / "packs" / "first" / "summary").is_dir()
    assert (
        json.loads((tmp_path / "previous.json").read_text())["pack_revision"] == "first"
    )


def test_complete_removal_clears_only_model_store_content(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "installed")
    abandoned = store.stage("abandoned")
    (tmp_path / "operations" / "keep.json").write_text("{}", encoding="utf-8")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")

    store.remove()

    assert not (tmp_path / "active.json").exists()
    assert not (tmp_path / "previous.json").exists()
    assert list((tmp_path / "packs").iterdir()) == []
    assert not abandoned.exists()
    assert (tmp_path / "operations" / "keep.json").exists()
    assert unrelated.read_text() == "keep"
