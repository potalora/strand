"""The benchmark-only candidate CLI cannot become a public release path."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.types import ModelRole
from tests.test_local_ai_artifacts import _install, _rewrite_active_as_legacy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_candidate_install_explicitly_skips_release_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.local_ai_candidate_pack as candidate_cli

    captured: dict[str, object] = {}

    async def run(action: str, **kwargs: object) -> int:
        captured["action"] = action
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(candidate_cli, "_run_lifecycle", run)
    assert await candidate_cli.execute("install") == 0
    assert captured["action"] == "install"
    assert captured["require_release"] is False


def test_candidate_cli_normalizes_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.local_ai_candidate_pack as candidate_cli

    async def interrupt(_action: str) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(candidate_cli, "execute", interrupt)

    assert candidate_cli.main(["install"]) == 130
    assert (
        capsys.readouterr().err == "ERROR: candidate model pack command interrupted.\n"
    )


def _recipe(name: str) -> str:
    content = (REPOSITORY_ROOT / "justfile").read_text(encoding="utf-8")
    marker = f"{name}:\n"
    return content.split(marker, 1)[1].split("\n\n", 1)[0]


def test_justfile_separates_candidate_and_release_required_verification() -> None:
    candidate = _recipe("local-ai-candidate-verify")
    assert "scripts/local_ai_candidate_pack.py verify" in candidate
    assert "scripts/local_ai_pack.py verify" not in candidate

    promoted = _recipe("local-ai-pack-verify")
    assert "scripts/local_ai_pack.py verify" in promoted
    assert "scripts/local_ai_candidate_pack.py" not in promoted


def test_fidelity_recipe_clears_private_fixture_inheritance() -> None:
    fidelity = _recipe("local-ai-fidelity")
    assert "env -u REAL_MEDICAL_FIXTURES_DIR" in fidelity
    assert "scripts/run_local_ai_fidelity.py" in fidelity


@pytest.mark.asyncio
async def test_candidate_verify_uses_retained_tree_without_lifecycle_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.local_ai_candidate_pack as candidate_cli

    manifest = SimpleNamespace(pack_revision="apple-m4-16gb-v2")
    binding = SimpleNamespace(path=Path("/retained/v1"))
    calls: list[tuple[str, object]] = []

    class Candidate:
        def __init__(self) -> None:
            self.manifest = manifest
            self.pack_path = binding.path

        def revalidate(self) -> None:
            calls.append(("revalidate", manifest))

    async def verify(value: object, path: object) -> None:
        calls.append(("verify", (value, path)))

    async def forbidden_lifecycle(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("candidate verify must not mutate lifecycle metadata")

    monkeypatch.setattr(
        candidate_cli,
        "resolve_retained_candidate_pack",
        lambda **_kwargs: Candidate(),
    )
    monkeypatch.setattr(
        candidate_cli,
        "platform_profile",
        lambda: ("apple_silicon", True),
        raising=False,
    )
    monkeypatch.setattr(candidate_cli, "verify_pack_candidate", verify, raising=False)
    monkeypatch.setattr(candidate_cli, "_run_lifecycle", forbidden_lifecycle)

    assert await candidate_cli.execute("verify") == 0
    assert calls == [
        ("verify", (manifest, binding.path)),
        ("revalidate", manifest),
    ]


def _retained_legacy_candidate(
    tmp_path: Path,
) -> tuple[object, Path, Path, Path, Path]:
    store = ArtifactStore(tmp_path / "models")
    legacy, _contents = _install(store, "apple-m4-16gb-v1")
    _rewrite_active_as_legacy(store, legacy)
    candidate = replace(legacy, pack_revision="apple-m4-16gb-v2")
    manifest_path = tmp_path / "apple-m4-16gb-v2.lock.json"
    manifest_path.write_text(
        json.dumps(
            asdict(candidate),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    pack_path = store.packs_dir / legacy.pack_revision
    return (
        candidate,
        manifest_path,
        store.root,
        pack_path,
        store.root / "activation-state.json",
    )


@pytest.mark.asyncio
async def test_candidate_verify_leaves_legacy_metadata_and_state_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.local_ai_candidate_pack as candidate_cli

    candidate, manifest_path, model_root, pack_path, state_path = (
        _retained_legacy_candidate(tmp_path)
    )
    metadata_path = pack_path / ".manifest.json"
    metadata_before = metadata_path.read_bytes()
    state_before = state_path.read_bytes()

    async def verify(manifest: object, observed_path: Path) -> None:
        assert manifest == candidate
        assert observed_path == pack_path

    monkeypatch.setattr(candidate_cli.settings, "local_ai_manifest_path", manifest_path)
    monkeypatch.setattr(candidate_cli.settings, "local_ai_model_dir", model_root)
    monkeypatch.setattr(
        candidate_cli,
        "platform_profile",
        lambda: ("apple_silicon", True),
    )
    monkeypatch.setattr(candidate_cli, "verify_pack_candidate", verify)

    assert await candidate_cli.execute("verify") == 0
    assert metadata_path.read_bytes() == metadata_before
    assert state_path.read_bytes() == state_before


@pytest.mark.parametrize("mutation", ["state", "metadata", "artifact"])
@pytest.mark.asyncio
async def test_candidate_verify_rejects_retained_tree_change_after_model_gate(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.local_ai_candidate_pack as candidate_cli

    _candidate, manifest_path, model_root, pack_path, state_path = (
        _retained_legacy_candidate(tmp_path)
    )

    async def mutate_after_bind(_manifest: object, _path: Path) -> None:
        if mutation == "state":
            state_path.write_text(
                json.dumps(json.loads(state_path.read_text(encoding="utf-8")), indent=2),
                encoding="utf-8",
            )
        elif mutation == "metadata":
            metadata_path = pack_path / ".manifest.json"
            metadata_path.write_text(
                json.dumps(
                    json.loads(metadata_path.read_text(encoding="utf-8")),
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            target = pack_path / ModelRole.OCR.value / "weights/model.safetensors"
            target.write_bytes(b"x" * target.stat().st_size)

    monkeypatch.setattr(candidate_cli.settings, "local_ai_manifest_path", manifest_path)
    monkeypatch.setattr(candidate_cli.settings, "local_ai_model_dir", model_root)
    monkeypatch.setattr(
        candidate_cli,
        "platform_profile",
        lambda: ("apple_silicon", True),
    )
    monkeypatch.setattr(candidate_cli, "verify_pack_candidate", mutate_after_bind)

    assert await candidate_cli.execute("verify") == 1
