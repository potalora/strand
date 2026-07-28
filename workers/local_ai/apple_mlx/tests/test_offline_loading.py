from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file(path: Path, relative: str) -> dict[str, object]:
    target = path / relative
    return {
        "path": relative,
        "sha256": _sha256(target),
        "size": target.stat().st_size,
    }


@pytest.fixture
def synthetic_pack(tmp_path: Path) -> tuple[dict[str, object], Path]:
    artifacts: list[dict[str, object]] = []
    repositories = {
        "ocr": "sahilchachra/ovisocr2-int4-mlx",
        "extraction": "numind/NuExtract3-mlx-4bits",
        "summary": "mlx-community/Qwen3.5-9B-MLX-4bit",
    }
    for role, repository in repositories.items():
        role_dir = tmp_path / role
        role_dir.mkdir()
        (role_dir / "config.json").write_text(
            json.dumps({"model_type": f"reviewed_{role}"}),
            encoding="utf-8",
        )
        (role_dir / "model.safetensors").write_bytes(f"{role}-weights".encode())
        artifacts.append(
            {
                "role": role,
                "repository": repository,
                "revision": "a" * 40,
                "quantization": "4bit",
                "license": "apache-2.0",
                "attribution": f"https://huggingface.co/{repository}",
                "decode_limits": {
                    "max_input_tokens": 32768,
                    "max_output_tokens": 4096,
                },
                "files": [
                    _manifest_file(role_dir, "config.json"),
                    _manifest_file(role_dir, "model.safetensors"),
                ],
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "pack_revision": "test-pack",
        "platform": "apple_silicon",
        "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        "validation_suite_version": "test-v1",
        "artifacts": artifacts,
    }
    return manifest, tmp_path


def test_load_role_rehashes_artifact_and_calls_loader_with_local_path_only(
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker.common import load_role

    manifest, model_dir = synthetic_pack
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    calls: list[tuple[str, dict[str, object]]] = []

    def loader(path: str, **kwargs: object) -> tuple[object, object]:
        calls.append((path, kwargs))
        return object(), object()

    loaded = load_role(
        "ocr",
        manifest,
        model_dir,
        trust_remote_code=False,
        loader=loader,
    )

    assert calls == [
        (
            str((model_dir / "ocr").resolve()),
            {
                "lazy": False,
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    assert loaded.repository_files_used == frozenset({"config.json", "model.safetensors"})

    weights = model_dir / "ocr" / "model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)
    with pytest.raises(ValueError, match="hash"):
        load_role(
            "ocr",
            manifest,
            model_dir,
            trust_remote_code=False,
            loader=loader,
        )


def test_default_loader_bypasses_hub_resolving_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx_vlm
    from mlx_vlm import utils

    from local_ai_mlx_worker.common import _default_loader

    calls: list[tuple[str, Path, dict[str, object]]] = []
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=42))
    processor = SimpleNamespace()

    def reject_hub_resolver(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("hub-resolving entrypoint must not be called")

    def load_model(path: Path, lazy: bool, **kwargs: object) -> SimpleNamespace:
        calls.append(("model", path, {"lazy": lazy, **kwargs}))
        return model

    def load_image_processor(path: Path, **kwargs: object) -> object:
        calls.append(("image", path, kwargs))
        return object()

    def load_processor(
        path: Path,
        add_detokenizer: bool,
        eos_token_ids: int | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        calls.append(
            (
                "processor",
                path,
                {
                    "add_detokenizer": add_detokenizer,
                    "eos_token_ids": eos_token_ids,
                    **kwargs,
                },
            )
        )
        return processor

    monkeypatch.setattr(mlx_vlm, "load", reject_hub_resolver)
    monkeypatch.setattr(utils, "load_model", load_model)
    monkeypatch.setattr(utils, "load_image_processor", load_image_processor)
    monkeypatch.setattr(utils, "load_processor", load_processor)

    loaded_model, loaded_processor = _default_loader(
        str(tmp_path),
        lazy=False,
        local_files_only=True,
        trust_remote_code=False,
    )

    assert loaded_model is model
    assert loaded_processor is processor
    assert [path for _stage, path, _kwargs in calls] == [tmp_path] * 3
    assert processor.image_processor is not None


@pytest.mark.parametrize(
    "relative",
    [
        "undeclared.safetensors",
        "tokenizer.json",
        "preprocessor_config.json",
    ],
)
def test_load_role_rejects_every_undeclared_artifact_file(
    relative: str,
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker.common import load_role

    manifest, model_dir = synthetic_pack
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }.items():
        monkeypatch.setenv(name, value)
    (model_dir / "ocr" / relative).write_bytes(b"undeclared")

    with pytest.raises(ValueError, match="undeclared"):
        load_role("ocr", manifest, model_dir, loader=lambda *_args, **_kwargs: ())


def test_load_role_requires_exact_manifest_schema_and_expected_identity(
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker.common import build_manifest_identity, load_role

    manifest, model_dir = synthetic_pack
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }.items():
        monkeypatch.setenv(name, value)
    identity = build_manifest_identity(manifest, "ocr")
    mismatched = dict(identity)
    mismatched["revision"] = "b" * 40

    with pytest.raises(ValueError, match="identity"):
        load_role(
            "ocr",
            manifest,
            model_dir,
            expected_identity=mismatched,
            loader=lambda *_args, **_kwargs: (object(), object()),
        )

    extra_root = deepcopy(manifest)
    extra_root["unexpected"] = True
    with pytest.raises(ValueError, match="structure"):
        load_role(
            "ocr",
            extra_root,
            model_dir,
            loader=lambda *_args, **_kwargs: (object(), object()),
        )

    extra_artifact = deepcopy(manifest)
    extra_artifact["artifacts"][0]["unexpected"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="structure"):
        load_role(
            "ocr",
            extra_artifact,
            model_dir,
            loader=lambda *_args, **_kwargs: (object(), object()),
        )


def test_manifest_file_rejects_duplicate_json_keys(
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import load_role

    manifest, model_dir = synthetic_pack
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }.items():
        monkeypatch.setenv(name, value)
    manifest_path = tmp_path / "manifest.json"
    serialized = json.dumps(manifest)
    manifest_path.write_text(
        serialized.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest"):
        load_role(
            "ocr",
            manifest_path,
            model_dir,
            loader=lambda *_args, **_kwargs: (object(), object()),
        )


def test_role_payload_requires_and_binds_expected_manifest_identity(
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker import common

    manifest, model_dir = synthetic_pack
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }.items():
        monkeypatch.setenv(name, value)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "model_dir": str(model_dir),
    }
    with pytest.raises(ValueError, match="identity"):
        common.load_role_from_payload("ocr", payload)

    payload["manifest_identity"] = common.build_manifest_identity(manifest, "ocr")
    monkeypatch.setattr(
        common,
        "_default_loader",
        lambda *_args, **_kwargs: (object(), object()),
    )
    loaded = common.load_role_from_payload("ocr", payload)

    assert loaded.role == "ocr"


def test_load_role_rejects_auto_map_repository_python_and_remote_code(
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker.common import load_role

    manifest, model_dir = synthetic_pack
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")

    with pytest.raises(ValueError, match="remote code"):
        load_role("ocr", manifest, model_dir, trust_remote_code=True, loader=lambda _: ())

    config = model_dir / "ocr" / "config.json"
    config.write_text(json.dumps({"auto_map": {"AutoModel": "modeling.Custom"}}))
    artifact = manifest["artifacts"][0]  # type: ignore[index]
    artifact["files"][0] = _manifest_file(model_dir / "ocr", "config.json")  # type: ignore[index]
    with pytest.raises(ValueError, match="auto_map"):
        load_role("ocr", manifest, model_dir, loader=lambda _: ())

    config.write_text(json.dumps({"model_type": "reviewed_ocr"}))
    artifact["files"][0] = _manifest_file(model_dir / "ocr", "config.json")  # type: ignore[index]
    (model_dir / "ocr" / "modeling_custom.py").write_text("raise RuntimeError")
    with pytest.raises(ValueError, match="repository Python"):
        load_role("ocr", manifest, model_dir, loader=lambda _: ())


def test_load_role_accepts_declared_real_sized_tokenizer_json(
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker.common import load_role

    manifest, model_dir = synthetic_pack
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }.items():
        monkeypatch.setenv(name, value)
    tokenizer = model_dir / "ocr" / "tokenizer.json"
    tokenizer.write_text(
        json.dumps(
            {
                "model": {"vocab": {"auto_map": 0}},
                "padding": "x" * 20_000_000,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    artifact = manifest["artifacts"][0]  # type: ignore[index]
    artifact["files"].append(  # type: ignore[index]
        _manifest_file(model_dir / "ocr", "tokenizer.json")
    )

    loaded = load_role(
        "ocr",
        manifest,
        model_dir,
        loader=lambda *_args, **_kwargs: (object(), object()),
    )

    assert "tokenizer.json" in loaded.repository_files_used


def test_load_role_keeps_artifact_json_scan_bounded(
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import common

    manifest, model_dir = synthetic_pack
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }.items():
        monkeypatch.setenv(name, value)
    tokenizer = model_dir / "ocr" / "tokenizer.json"
    tokenizer.write_text('{"tokens":["bounded"]}', encoding="utf-8")
    artifact = manifest["artifacts"][0]  # type: ignore[index]
    artifact["files"].append(  # type: ignore[index]
        _manifest_file(model_dir / "ocr", "tokenizer.json")
    )
    monkeypatch.setattr(common, "MAX_ARTIFACT_JSON_BYTES", tokenizer.stat().st_size - 1)

    with pytest.raises(ValueError, match="too large"):
        common.load_role(
            "ocr",
            manifest,
            model_dir,
            loader=lambda *_args, **_kwargs: (object(), object()),
        )


def test_load_role_requires_offline_environment(
    synthetic_pack: tuple[dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker.common import load_role

    manifest, model_dir = synthetic_pack
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")

    with pytest.raises(ValueError, match="offline"):
        load_role("ocr", manifest, model_dir, loader=lambda _: ())


@contextlib.contextmanager
def deny_all_network() -> Iterator[None]:
    original_socket = socket.socket

    class DeniedSocket(original_socket):
        def connect(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("network access attempted")

        def connect_ex(self, *_args: object, **_kwargs: object) -> int:
            raise AssertionError("network access attempted")

    socket.socket = DeniedSocket
    try:
        yield
    finally:
        socket.socket = original_socket


def _locked_artifacts() -> tuple[Path, Path]:
    manifest_override = os.environ.get("LOCAL_AI_LOCKED_MANIFEST")
    model_dir_override = os.environ.get("LOCAL_AI_MODEL_DIR")
    project_root = Path(__file__).resolve().parents[4]
    manifest_path = (
        Path(manifest_override)
        if manifest_override
        else project_root / "backend" / "app" / "model_manifests" / "apple-m4-16gb-v1.lock.json"
    )
    if model_dir_override:
        model_dir = Path(model_dir_override)
    else:
        model_dir = (
            project_root / "backend" / "data" / "local-ai" / "models" / "packs" / "apple-m4-16gb-v1"
        )
    return manifest_path, model_dir


def test_default_locked_artifact_paths_match_backend_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_AI_LOCKED_MANIFEST", raising=False)
    monkeypatch.delenv("LOCAL_AI_MODEL_DIR", raising=False)
    project_root = Path(__file__).resolve().parents[4]

    manifest_path, model_dir = _locked_artifacts()

    assert manifest_path == (
        project_root / "backend" / "app" / "model_manifests" / "apple-m4-16gb-v1.lock.json"
    )
    assert model_dir == (
        project_root / "backend" / "data" / "local-ai" / "models" / "packs" / "apple-m4-16gb-v1"
    )


@pytest.mark.local_model
@pytest.mark.parametrize("role", ["ocr", "extraction", "summary"])
def test_exact_manifest_artifact_loads_without_remote_code(role: str) -> None:
    from local_ai_mlx_worker.common import load_role

    manifest_path, model_dir = _locked_artifacts()
    if not manifest_path.is_file():
        pytest.skip(f"locked manifest absent: {manifest_path}")
    if not (model_dir / role).is_dir():
        pytest.skip(f"locked {role} artifact absent: {model_dir / role}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    with deny_all_network():
        loaded = load_role(
            role,
            manifest_path,
            model_dir,
            trust_remote_code=False,
        )
    assert not any(path.endswith(".py") for path in loaded.repository_files_used)
