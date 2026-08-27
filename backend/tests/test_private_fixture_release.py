from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

import pytest

import tests.private_fixture_release as fixture_release
from tests.private_fixture_release import (
    PrivateFixtureReleaseError,
    private_fixture_root,
)


_DATASET = "medtimeline"
_TARGET = "synthetic-test-target"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write_private(path: Path, body: bytes) -> None:
    path.write_bytes(body)
    path.chmod(0o600)


def _synthetic_release(
    approved_root: Path,
    *,
    dataset: str = _DATASET,
    top_level: str = "raw",
    payloads: dict[str, bytes] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    approved_root.mkdir(parents=True, mode=0o700)
    approved_root.chmod(0o700)
    dataset_root = approved_root / dataset
    data = dataset_root / "releases" / ("0" * 64) / "data"
    (data / top_level).mkdir(parents=True, mode=0o700)
    for directory in (
        dataset_root,
        dataset_root / "releases",
        data.parent,
        data,
        data / top_level,
    ):
        directory.chmod(0o700)
    payloads = payloads or {"example.txt": b"synthetic\n"}
    payload_paths: list[Path] = []
    for relative_path, body in sorted(payloads.items()):
        payload = data / top_level / relative_path
        payload.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        current = payload.parent
        while current != data:
            current.chmod(0o700)
            current = current.parent
        _write_private(payload, body)
        payload_paths.append(payload)

    manifest = {
        "created_at": "2026-08-27T00:00:00Z",
        "dataset": dataset,
        "file_count": len(payload_paths),
        "files": [
            {
                "mode": "0600",
                "path": payload.relative_to(data).as_posix(),
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "size": payload.stat().st_size,
            }
            for payload in payload_paths
        ],
        "hash_algorithm": "sha256",
        "schema_version": 1,
        "total_bytes": sum(payload.stat().st_size for payload in payload_paths),
    }
    manifest_bytes = _canonical(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    release = data.parents[0]
    final_release = release.parent / manifest_sha
    release.rename(final_release)
    data = final_release / "data"
    _write_private(final_release / "manifest.json", manifest_bytes)
    _write_private(final_release / "manifest.minisig", b"synthetic-signature\n")

    receipt: dict[str, object] = {
        "dataset": dataset,
        "file_count": manifest["file_count"],
        "manifest_sha256": manifest_sha,
        "schema_version": 1,
        "status": "verified",
        "target_id": _TARGET,
        "total_bytes": manifest["total_bytes"],
        "verified_at": "2026-08-27T00:00:01Z",
    }
    _write_private(final_release / "receipt.json", _canonical(receipt))
    (dataset_root / "current").symlink_to(Path("releases") / manifest_sha)
    return dataset_root, data, receipt


def _enable(
    monkeypatch: pytest.MonkeyPatch, approved_root: Path, dataset_root: Path
) -> None:
    monkeypatch.setenv("REAL_MEDICAL_FIXTURES_APPROVED_ROOT", str(approved_root))
    monkeypatch.setenv("REAL_MEDICAL_FIXTURES_DIR", str(dataset_root / "current"))
    monkeypatch.setenv("REAL_MEDICAL_FIXTURES_TARGET_ID", _TARGET)


def test_private_fixture_release_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REAL_MEDICAL_FIXTURES_DIR", raising=False)
    monkeypatch.delenv("REAL_MEDICAL_FIXTURES_APPROVED_ROOT", raising=False)

    assert private_fixture_root() is None


def test_requested_release_requires_an_approved_runtime_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_root, _, _ = _synthetic_release(tmp_path / "unapproved")
    monkeypatch.setenv("REAL_MEDICAL_FIXTURES_DIR", str(dataset_root / "current"))
    monkeypatch.delenv("REAL_MEDICAL_FIXTURES_APPROVED_ROOT", raising=False)

    with pytest.raises(PrivateFixtureReleaseError, match="approved runtime root"):
        private_fixture_root()


def test_requested_release_requires_an_approved_target_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, _, _ = _synthetic_release(approved)
    _enable(monkeypatch, approved, dataset_root)
    monkeypatch.delenv("REAL_MEDICAL_FIXTURES_TARGET_ID")

    with pytest.raises(PrivateFixtureReleaseError, match="target identity"):
        private_fixture_root()


def test_path_outside_approved_root_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    dataset_root, _, _ = _synthetic_release(tmp_path / "outside")
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="approved runtime root"):
        private_fixture_root()


def test_missing_current_pointer_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, _, _ = _synthetic_release(approved)
    (dataset_root / "current").unlink()
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="current release pointer"):
        private_fixture_root()


@pytest.mark.parametrize("replacement", [b"{}\n", b"not-json\n"])
def test_bad_receipt_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replacement: bytes,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(approved)
    _write_private(data.parent / "receipt.json", replacement)
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="receipt"):
        private_fixture_root()


def test_noncanonical_receipt_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, receipt = _synthetic_release(approved)
    _write_private(
        data.parent / "receipt.json",
        (json.dumps(receipt, indent=2) + "\n").encode("utf-8"),
    )
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="not canonical"):
        private_fixture_root()


def test_missing_receipt_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(approved)
    (data.parent / "receipt.json").unlink()
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="receipt"):
        private_fixture_root()


def test_bad_manifest_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(approved)
    manifest = data.parent / "manifest.json"
    _write_private(manifest, manifest.read_bytes() + b" ")
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="manifest"):
        private_fixture_root()


def test_dataset_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, receipt = _synthetic_release(approved)
    receipt["dataset"] = "different-dataset"
    _write_private(data.parent / "receipt.json", _canonical(receipt))
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="dataset"):
        private_fixture_root()


def test_target_identity_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, _, _ = _synthetic_release(approved)
    _enable(monkeypatch, approved, dataset_root)
    monkeypatch.setenv("REAL_MEDICAL_FIXTURES_TARGET_ID", "different-target")

    with pytest.raises(PrivateFixtureReleaseError, match="target identity"):
        private_fixture_root()


def test_receipt_aggregate_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, receipt = _synthetic_release(approved)
    receipt["total_bytes"] = int(receipt["total_bytes"]) + 1
    _write_private(data.parent / "receipt.json", _canonical(receipt))
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="aggregates"):
        private_fixture_root()


def test_valid_synthetic_current_release_is_exposed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(approved)
    _enable(monkeypatch, approved, dataset_root)

    assert private_fixture_root() == data.resolve(strict=True)


def test_data_directory_replacement_after_initial_check_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(approved)
    _enable(monkeypatch, approved, dataset_root)
    original_check = fixture_release._require_linked_directory
    replaced = False

    def replace_after_data_check(
        descriptor: int,
        expected: object,
        parent_fd: int,
        name: str,
        label: str,
    ) -> None:
        nonlocal replaced
        original_check(descriptor, expected, parent_fd, name, label)  # type: ignore[arg-type]
        if label == "private fixture data directory" and not replaced:
            replaced = True
            data.rename(data.with_name("data-before-swap"))
            data.mkdir(mode=0o700)

    monkeypatch.setattr(
        fixture_release,
        "_require_linked_directory",
        replace_after_data_check,
    )

    with pytest.raises(PrivateFixtureReleaseError, match="data directory changed"):
        private_fixture_root()


def test_release_replacement_after_initial_parent_check_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(approved)
    _enable(monkeypatch, approved, dataset_root)
    original_check = fixture_release._require_linked_directory
    release = data.parent
    replaced = False

    def replace_after_release_check(
        descriptor: int,
        expected: object,
        parent_fd: int,
        name: str,
        label: str,
    ) -> None:
        nonlocal replaced
        original_check(descriptor, expected, parent_fd, name, label)  # type: ignore[arg-type]
        if label == "private fixture release" and not replaced:
            replaced = True
            release.rename(release.with_name("release-before-swap"))
            (release / "data").mkdir(parents=True, mode=0o700)
            release.chmod(0o700)

    monkeypatch.setattr(
        fixture_release,
        "_require_linked_directory",
        replace_after_release_check,
    )

    with pytest.raises(PrivateFixtureReleaseError, match="release changed"):
        private_fixture_root()


def test_fixturectl_valid_raw_subtree_loads_private_local_ai_corpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.services.local_ai.fidelity_runner import load_private_fidelity_corpus

    corpus_bytes = _canonical(
        {
            "documents": [
                {
                    "critical_numeric_tokens": ["7.1"],
                    "expected_facts": [
                        {
                            "category": "labs",
                            "name": "Hemoglobin A1c",
                            "unit": "%",
                            "value": "7.1",
                        }
                    ],
                    "forbidden_facts": [],
                    "id": "synthetic-private-note",
                    "source_file": "synthetic-private-note.pdf",
                }
            ],
            "schema_version": 1,
            "suite_version": "local-ai-fidelity-v1",
        }
    )
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(
        approved,
        payloads={
            "local-ai-corpus-v1.json": corpus_bytes,
            "synthetic-private-note.pdf": b"%PDF-synthetic-private-fixture\n",
        },
    )
    _enable(monkeypatch, approved, dataset_root)

    guarded_data = private_fixture_root()
    assert guarded_data == data.resolve(strict=True)
    corpus = load_private_fidelity_corpus(guarded_data / "raw")
    assert corpus.documents[0].source_path == data / "raw/synthetic-private-note.pdf"


def test_distinct_fidelity_dataset_requires_explicit_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = "medtimeline-fidelity-v2"
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(
        approved,
        dataset=dataset,
        top_level="Run-v5-71a2f50",
    )
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="current link"):
        private_fixture_root()
    assert private_fixture_root(dataset) == data.resolve(strict=True)


def test_dataset_top_level_policy_is_not_inferred_from_the_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, _, _ = _synthetic_release(approved, top_level="unexpected")
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="top-level"):
        private_fixture_root()


def test_discovery_top_level_policy_matches_fixturectl_policy() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (repository_root / "tools/fixturectl/policy.json").read_text(encoding="utf-8")
    )
    fixturectl_top_levels = {
        dataset: frozenset(values["allowed_top_level"])
        for dataset, values in policy["datasets"].items()
    }

    assert fixture_release._DATASET_TOP_LEVEL == fixturectl_top_levels


@pytest.mark.parametrize(
    ("relative_path", "mode"),
    [("receipt.json", 0o644), ("manifest.json", 0o640)],
)
def test_control_files_require_owner_only_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
    mode: int,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(approved)
    (data.parent / relative_path).chmod(mode)
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="unsafe"):
        private_fixture_root()


def test_current_must_be_one_relative_immutable_release_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    dataset_root, data, _ = _synthetic_release(approved)
    current = dataset_root / "current"
    current.unlink()
    current.symlink_to(data.parent.resolve())
    _enable(monkeypatch, approved, dataset_root)

    with pytest.raises(PrivateFixtureReleaseError, match="current release pointer"):
        private_fixture_root()


def test_protected_fixture_variable_has_no_unguarded_runtime_consumers() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    excluded = {
        repository_root / "backend/tests/private_fixture_release.py",
        Path(__file__).resolve(),
    }
    offenders: list[str] = []
    for path in repository_root.rglob("*.py"):
        if path.resolve() in excluded or any(
            part in {".git", ".venv", "node_modules"} for part in path.parts
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "getenv"}
                and any(
                    isinstance(argument, ast.Constant)
                    and argument.value == "REAL_MEDICAL_FIXTURES_DIR"
                    for argument in node.args
                )
            ):
                offenders.append(path.relative_to(repository_root).as_posix())
                break
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "REAL_MEDICAL_FIXTURES_DIR"
            ):
                offenders.append(path.relative_to(repository_root).as_posix())
                break

    process_env_access = re.compile(
        r"process\.env(?:\.REAL_MEDICAL_FIXTURES_DIR|\[['\"]REAL_MEDICAL_FIXTURES_DIR['\"]\])"
    )
    for suffix in ("*.js", "*.mjs", "*.cjs", "*.ts", "*.tsx"):
        for path in repository_root.rglob(suffix):
            if any(part in {".git", ".next", "node_modules"} for part in path.parts):
                continue
            if process_env_access.search(path.read_text(encoding="utf-8")):
                offenders.append(path.relative_to(repository_root).as_posix())

    assert sorted(set(offenders)) == []


def test_legacy_developer_fixture_consumers_use_an_explicit_dev_only_variable() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    legacy_variable = "MEDTIMELINE_LEGACY_DEV_FIXTURES_DIR"

    for relative_path in (
        "scripts/e2e_full_v2.py",
        "frontend/e2e/helpers/test-data.ts",
    ):
        source = (repository_root / relative_path).read_text(encoding="utf-8")
        assert legacy_variable in source
        assert "REAL_MEDICAL_FIXTURES_DIR" not in source
