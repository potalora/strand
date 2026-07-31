"""Security and atomicity tests for the local-AI artifact store."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import app.services.local_ai.artifact_store as artifact_store_module
from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.types import ModelRole
from app.services.local_ai.validation_receipt import (
    _issue_runtime_validation_receipt,
)


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
    store.activate_validated(
        stage,
        manifest,
        _issue_runtime_validation_receipt(manifest),
    )
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
        store.activate_validated(
            stage,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )

    assert store.active_revision() == "first"
    assert not (tmp_path / "packs" / "second").exists()


def test_active_manifest_returns_only_the_fully_verified_active_pack(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    manifest, _contents = _install(store, "first")

    assert store.active_manifest() == manifest


def test_active_manifest_returns_none_without_an_active_pack(tmp_path: Path) -> None:
    assert ArtifactStore(tmp_path).active_manifest() is None


def test_store_has_no_public_hash_only_activation_method(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    assert not hasattr(store, "activate")
    assert hasattr(store, "activate_validated")


def test_activation_uses_one_authoritative_atomic_state_file(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    _install(store, "second")

    state = json.loads((tmp_path / "activation-state.json").read_text())

    assert state["active"]["pack_revision"] == "second"
    assert state["previous"]["pack_revision"] == "first"
    assert not (tmp_path / "active.json").exists()
    assert not (tmp_path / "previous.json").exists()


def test_state_failure_restores_prior_pointer_and_removes_inactive_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    manifest, contents = _manifest("second")
    stage = store.stage("second")
    _write_manifest_files(stage, manifest, contents)

    def fail_state(**_kwargs: object) -> None:
        raise LocalValidationError("injected state failure")

    monkeypatch.setattr(store, "_write_state", fail_state)

    with pytest.raises(LocalValidationError, match="injected"):
        store.activate_validated(
            stage,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )

    assert store.active_revision() == "first"
    assert (tmp_path / "packs" / "first").is_dir()
    assert not (tmp_path / "packs" / "second").exists()
    assert len(list(store.validations_dir.glob("*.json"))) == 1


def test_receipt_write_failure_preserves_prior_active_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    manifest, contents = _manifest("second")
    stage = store.stage("second")
    _write_manifest_files(stage, manifest, contents)
    original_write = store._write_json_atomic

    def fail_receipt_write(path: Path, payload: dict[str, object]) -> None:
        if path.parent == store.validations_dir:
            raise LocalValidationError("injected receipt write failure")
        original_write(path, payload)

    monkeypatch.setattr(store, "_write_json_atomic", fail_receipt_write)

    with pytest.raises(LocalValidationError, match="receipt write"):
        store.activate_validated(
            stage,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )

    assert store.active_revision() == "first"
    assert store.active_manifest() is not None
    assert not (store.packs_dir / "second").exists()


def test_state_fsync_failure_restores_prior_validated_active_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    manifest, contents = _manifest("second")
    stage = store.stage("second")
    _write_manifest_files(stage, manifest, contents)
    original_fsync = artifact_store_module._fsync_directory
    failed_once = False

    def fail_after_state_replace(path: Path) -> None:
        nonlocal failed_once
        if path == tmp_path and not failed_once:
            failed_once = True
            raise LocalValidationError("injected state fsync failure")
        original_fsync(path)

    monkeypatch.setattr(
        artifact_store_module,
        "_fsync_directory",
        fail_after_state_replace,
    )

    with pytest.raises(LocalValidationError, match="injected"):
        store.activate_validated(
            stage,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )

    assert store.active_revision() == "first"
    assert (tmp_path / "packs" / "first").is_dir()
    assert not (tmp_path / "packs" / "second").exists()


def test_post_rename_verification_failure_does_not_update_activation_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    original_state = (tmp_path / "activation-state.json").read_bytes()
    manifest, contents = _manifest("second")
    stage = store.stage("second")
    _write_manifest_files(stage, manifest, contents)
    destination = tmp_path / "packs" / "second"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(contents[ModelRole.OCR])
    original_replace = os.replace
    replaced = False

    def replace_then_mutate(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        nonlocal replaced
        original_replace(source, target)
        if Path(source) == stage and Path(target) == destination:
            model = destination / "ocr" / "weights" / "model.safetensors"
            model.unlink()
            model.symlink_to(outside)
            replaced = True

    monkeypatch.setattr(os, "replace", replace_then_mutate)

    with pytest.raises(LocalValidationError):
        store.activate_validated(
            stage,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )

    assert replaced
    assert store.active_revision() == "first"
    assert (tmp_path / "activation-state.json").read_bytes() == original_state
    assert not destination.exists()


def test_stage_removes_unreferenced_orphan_destination_for_honest_retry(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    orphan = store.packs_dir / "retry-revision"
    orphan.mkdir()
    (orphan / "partial.bin").write_bytes(b"partial")

    staging = store.stage("retry-revision")

    assert not orphan.exists()
    assert staging.parent == store.staging_dir
    assert staging.is_dir()


def test_mutations_are_serialized_within_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    second_store = ArtifactStore(tmp_path)
    first_manifest, first_contents = _manifest("first")
    first_stage = store.stage("first")
    _write_manifest_files(first_stage, first_manifest, first_contents)
    second_manifest, second_contents = _manifest("second")
    second_stage = store.stage("second")
    _write_manifest_files(second_stage, second_manifest, second_contents)
    original_write = store._write_json_atomic
    counter_lock = threading.Lock()
    active_writers = 0
    max_active_writers = 0

    def observed_write(path: Path, payload: dict[str, object]) -> None:
        nonlocal active_writers, max_active_writers
        with counter_lock:
            active_writers += 1
            max_active_writers = max(max_active_writers, active_writers)
        time.sleep(0.02)
        try:
            original_write(path, payload)
        finally:
            with counter_lock:
                active_writers -= 1

    monkeypatch.setattr(store, "_write_json_atomic", observed_write)
    monkeypatch.setattr(second_store, "_write_json_atomic", observed_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                store.activate_validated,
                first_stage,
                first_manifest,
                _issue_runtime_validation_receipt(first_manifest),
            ),
            executor.submit(
                second_store.activate_validated,
                second_stage,
                second_manifest,
                _issue_runtime_validation_receipt(second_manifest),
            ),
        ]
        for future in futures:
            future.result()

    state = json.loads((tmp_path / "activation-state.json").read_text())
    revisions = {
        state["active"]["pack_revision"],
        state["previous"]["pack_revision"],
    }
    assert revisions == {"first", "second"}
    assert max_active_writers == 1


def test_mutations_take_owner_only_cross_process_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[int] = []

    class FakeFcntl:
        LOCK_EX = 2
        LOCK_UN = 8

        @staticmethod
        def flock(_descriptor: int, operation: int) -> None:
            events.append(operation)

    monkeypatch.setattr(artifact_store_module, "fcntl", FakeFcntl, raising=False)
    store = ArtifactStore(tmp_path)

    store.remove()

    assert events == [FakeFcntl.LOCK_EX, FakeFcntl.LOCK_UN]
    assert (tmp_path / ".activation.lock").stat().st_mode & 0o777 == 0o600


def test_activation_rejects_a_stale_active_pointer_without_replacing_it(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    stale = {
        "pack_revision": "missing",
        "manifest_sha256": "a" * 64,
        "validation_receipt_sha256": "b" * 64,
    }
    stale_state = {"active": stale, "previous": None}
    state_path = tmp_path / "activation-state.json"
    state_path.write_text(json.dumps(stale_state), encoding="utf-8")
    manifest, contents = _manifest("new")
    stage = store.stage("new")
    _write_manifest_files(stage, manifest, contents)

    with pytest.raises(LocalValidationError, match="verified"):
        store.activate_validated(
            stage,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )

    assert json.loads(state_path.read_text()) == stale_state


def test_successive_activation_preserves_previous_and_rollback_verifies_it(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    _install(store, "first")
    _install(store, "second")

    store.rollback()

    assert store.active_revision() == "first"
    state = json.loads((tmp_path / "activation-state.json").read_text())
    assert state["previous"]["pack_revision"] == "second"


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


def test_verification_rejects_file_replaced_with_symlink_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    manifest, contents = _manifest("race")
    stage = store.stage("race")
    _write_manifest_files(stage, manifest, contents)
    target = stage / "ocr" / "weights" / "model.safetensors"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(contents[ModelRole.OCR])
    original_path_open = Path.open
    original_os_open = os.open
    replaced = False

    def replace_target() -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        target.unlink()
        target.symlink_to(outside)

    def racing_path_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == target:
            replace_target()
        return original_path_open(path, *args, **kwargs)

    def racing_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "model.safetensors" and dir_fd is not None:
            replace_target()
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(Path, "open", racing_path_open)
    monkeypatch.setattr(os, "open", racing_os_open)

    with pytest.raises(LocalValidationError):
        store.activate_validated(
            stage,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )

    assert replaced
    assert store.active_revision() is None
    assert not (tmp_path / "packs" / "race").exists()


def test_verification_rejects_entry_replaced_while_descriptor_is_hashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    manifest, contents = _manifest("hash-race")
    stage = store.stage("hash-race")
    _write_manifest_files(stage, manifest, contents)
    target = stage / "ocr" / "weights" / "model.safetensors"
    target_inode = target.stat().st_ino
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(contents[ModelRole.OCR])
    original_read = os.read
    replaced = False

    def read_then_replace(descriptor: int, length: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, length)
        if not replaced and os.fstat(descriptor).st_ino == target_inode:
            target.unlink()
            target.symlink_to(outside)
            replaced = True
        return chunk

    monkeypatch.setattr(os, "read", read_then_replace)

    with pytest.raises(LocalValidationError):
        store.activate_validated(
            stage,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )

    assert replaced
    assert store.active_revision() is None
    assert not (tmp_path / "packs" / "hash-race").exists()


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
    state = json.loads((tmp_path / "activation-state.json").read_text())
    assert state["active"] is None
    assert state["previous"]["pack_revision"] == "first"


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
    state = json.loads((tmp_path / "activation-state.json").read_text())
    assert state == {"active": None, "previous": None}
    assert list((tmp_path / "packs").iterdir()) == []
    assert not abandoned.exists()
    assert (tmp_path / "operations" / "keep.json").exists()
    assert unrelated.read_text() == "keep"
