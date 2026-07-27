"""Strict tests for local-AI manifests and candidate catalog locking."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import load_manifest
from app.services.local_ai.types import ModelRole

lock_script = importlib.import_module("scripts.lock_local_ai_manifest")

ROLES = ("ocr", "extraction", "summary")


def _manifest_file(path: str = "model.safetensors", *, size: int = 10) -> dict[str, Any]:
    return {"path": path, "sha256": "a" * 64, "size": size}


def _artifact(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "repository": f"owner/{role}",
        "revision": "0" * 40,
        "quantization": "4bit",
        "license": "apache-2.0",
        "attribution": f"https://huggingface.co/owner/{role}",
        "decode_limits": {"max_input_tokens": 4096, "max_output_tokens": 1024},
        "files": [_manifest_file()],
    }


def _valid_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pack_revision": "apple-m4-16gb-v1",
        "platform": "apple_silicon",
        "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        "validation_suite_version": "local-ai-fixtures-v1",
        "artifacts": [_artifact(role) for role in ROLES],
    }


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_manifest_loads_exactly_one_of_each_role(tmp_path: Path) -> None:
    manifest = load_manifest(_write_json(tmp_path / "manifest.json", _valid_manifest()))

    assert manifest.schema_version == 1
    assert tuple(artifact.role for artifact in manifest.artifacts) == (
        ModelRole.OCR,
        ModelRole.EXTRACTION,
        ModelRole.SUMMARY,
    )
    assert manifest.validation_suite_version == "local-ai-fixtures-v1"


def test_manifest_requires_immutable_revision_and_sha256(tmp_path: Path) -> None:
    raw = _valid_manifest()
    raw["artifacts"][0]["revision"] = "main"
    raw["artifacts"][0]["files"][0]["sha256"] = "bad"

    with pytest.raises(LocalValidationError, match="immutable revision"):
        load_manifest(_write_json(tmp_path / "manifest.json", raw))


def test_manifest_rejects_repository_code_and_pickle(tmp_path: Path) -> None:
    raw = _valid_manifest()
    raw["artifacts"][0]["files"] = [
        _manifest_file("modeling_ovis.py"),
        _manifest_file("pytorch_model.bin"),
    ]

    with pytest.raises(LocalValidationError, match="forbidden"):
        load_manifest(_write_json(tmp_path / "manifest.json", raw))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema version"),
        (lambda value: value.update(schema_version=True), "schema version"),
        (lambda value: value.pop("artifacts"), "structure"),
        (lambda value: value.update(artifacts={}), "structure"),
        (lambda value: value["artifacts"][0].update(role="other"), "model role"),
        (
            lambda value: value["artifacts"][1].update(role="ocr"),
            "exactly one artifact",
        ),
        (
            lambda value: value["artifacts"][0].update(revision="A" * 40),
            "immutable revision",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(sha256="A" * 64),
            "SHA-256",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(size=0),
            "positive",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(size=True),
            "positive",
        ),
        (lambda value: value["artifacts"][0].update(files=[]), "nonempty"),
        (
            lambda value: value["artifacts"][0].update(
                files=[_manifest_file("a.safetensors"), _manifest_file("a.safetensors")]
            ),
            "duplicate",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(path="../model.json"),
            "path",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(path="/model.json"),
            "path",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(path="a\\model.json"),
            "path",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(path="./model.json"),
            "path",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(path="model.gguf"),
            "forbidden",
        ),
        (
            lambda value: value["artifacts"][0].update(decode_limits={"max_input_tokens": 0}),
            "decode limits",
        ),
    ],
)
def test_manifest_rejects_invalid_values_safely(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    raw = _valid_manifest()
    mutate(raw)

    with pytest.raises(LocalValidationError, match=message):
        load_manifest(_write_json(tmp_path / "manifest.json", raw))


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "null",
        "[]",
        '{"schema_version": 1}',
        '{"schema_version": "one"}',
    ],
)
def test_manifest_normalizes_malformed_json_types_and_missing_keys(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(LocalValidationError):
        load_manifest(path)


def test_manifest_enforces_file_count_and_byte_bounds(tmp_path: Path) -> None:
    too_many = _valid_manifest()
    too_many["artifacts"][0]["files"] = [
        _manifest_file(f"weights/{index}.safetensors") for index in range(63)
    ]
    with pytest.raises(LocalValidationError, match="file count"):
        load_manifest(_write_json(tmp_path / "too-many.json", too_many))

    too_large = _valid_manifest()
    too_large["artifacts"][0]["files"][0]["size"] = 8 * 1024**3 + 1
    with pytest.raises(LocalValidationError, match="file size"):
        load_manifest(_write_json(tmp_path / "too-large.json", too_large))

    oversized_pack = _valid_manifest()
    for artifact in oversized_pack["artifacts"]:
        artifact["files"] = [
            _manifest_file(f"weights/{index}.safetensors", size=8 * 1024**3)
            for index in range(1)
        ]
    with pytest.raises(LocalValidationError, match="pack size"):
        load_manifest(_write_json(tmp_path / "oversized-pack.json", oversized_pack))


def _catalog() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pack_revision": "apple-m4-16gb-v1",
        "platform": "apple_silicon",
        "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        "validation_suite_version": "local-ai-fixtures-v1",
        "candidates": [
            {
                "role": role,
                "repository": f"owner/{role}",
                "quantization": "4bit",
                "license": "apache-2.0",
                "attribution": f"https://huggingface.co/owner/{role}",
                "decode_limits": {
                    "max_input_tokens": 4096,
                    "max_output_tokens": 1024,
                },
            }
            for role in ROLES
        ],
    }


class _HFTransport:
    def __init__(
        self,
        *,
        forbidden_role: str | None = None,
        auto_map_role: str | None = None,
        bad_weight_role: str | None = None,
    ) -> None:
        self.requests: list[str] = []
        self.forbidden_role = forbidden_role
        self.auto_map_role = auto_map_role
        self.bad_weight_role = bad_weight_role
        self.config_bytes: dict[str, bytes] = {}
        self.weight_bytes: dict[str, bytes] = {}
        for role in ROLES:
            config = {"architectures": [f"{role.title()}Model"]}
            if role == auto_map_role:
                config["auto_map"] = {"AutoModel": "modeling_custom.CustomModel"}
            self.config_bytes[role] = json.dumps(config).encode()
            self.weight_bytes[role] = f"safe-{role}-weights".encode()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        parts = request.url.path.strip("/").split("/")
        if parts[:2] == ["api", "models"]:
            role = parts[3]
            config = self.config_bytes[role]
            weights = self.weight_bytes[role]
            siblings: list[dict[str, Any]] = [
                {
                    "rfilename": "config.json",
                    "size": len(config),
                    "blobId": hashlib.sha1(config, usedforsecurity=False).hexdigest(),
                },
                {
                    "rfilename": "model.safetensors",
                    "size": len(weights),
                    "lfs": {
                        "sha256": hashlib.sha256(weights).hexdigest(),
                        "size": len(weights),
                        "pointerSize": 132,
                    },
                },
                {"rfilename": ".gitattributes", "size": 10, "blobId": "f" * 40},
            ]
            if role == self.forbidden_role:
                siblings.append(
                    {"rfilename": "modeling_custom.py", "size": 20, "blobId": "e" * 40}
                )
            return httpx.Response(
                200,
                json={
                    "id": f"owner/{role}",
                    "sha": str(ROLES.index(role) + 1) * 40,
                    "cardData": {"license": "apache-2.0"},
                    "siblings": siblings,
                },
                request=request,
            )
        if parts[0] == "owner" and parts[2] == "resolve":
            role = parts[1]
            filename = "/".join(parts[4:])
            if filename == "config.json":
                return httpx.Response(200, content=self.config_bytes[role], request=request)
            if filename == "model.safetensors":
                content = self.weight_bytes[role]
                if role == self.bad_weight_role:
                    content += b"-tampered"
                return httpx.Response(200, content=content, request=request)
        return httpx.Response(404, request=request)


def _install_mock_client(monkeypatch: pytest.MonkeyPatch, transport: _HFTransport) -> None:
    real_client = httpx.Client

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(lock_script.httpx, "Client", client_factory)


def test_lock_catalog_resolves_hashes_and_writes_canonical_lock_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport()
    _install_mock_client(monkeypatch, transport)
    catalog = _write_json(tmp_path / "catalog.json", _catalog())
    output = tmp_path / "candidate.lock.json"

    count = lock_script.lock_catalog(catalog, output)

    assert count == 3
    assert output.read_bytes().endswith(b"\n")
    assert output.read_text() == (
        json.dumps(json.loads(output.read_text()), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    manifest = load_manifest(output)
    assert {artifact.role for artifact in manifest.artifacts} == set(ModelRole)
    assert all(len(artifact.revision) == 40 for artifact in manifest.artifacts)
    assert all(len(file.sha256) == 64 for artifact in manifest.artifacts for file in artifact.files)
    assert not any(".gitattributes" in file.path for artifact in manifest.artifacts for file in artifact.files)


def test_lock_catalog_preflights_all_metadata_before_any_file_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(forbidden_role="summary")
    _install_mock_client(monkeypatch, transport)
    output = tmp_path / "candidate.lock.json"

    with pytest.raises(LocalValidationError, match="forbidden"):
        lock_script.lock_catalog(_write_json(tmp_path / "catalog.json", _catalog()), output)

    assert not output.exists()
    assert len(transport.requests) == 3
    assert all("/api/models/" in request for request in transport.requests)


def test_lock_catalog_rejects_remote_code_before_weight_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(auto_map_role="extraction")
    _install_mock_client(monkeypatch, transport)
    output = tmp_path / "candidate.lock.json"

    with pytest.raises(LocalValidationError, match="remote code"):
        lock_script.lock_catalog(_write_json(tmp_path / "catalog.json", _catalog()), output)

    assert not output.exists()
    extraction_requests = [
        request for request in transport.requests if "/owner/extraction/resolve/" in request
    ]
    assert any(request.endswith("/config.json") for request in extraction_requests)
    assert not any(request.endswith("/model.safetensors") for request in transport.requests)


def test_lock_catalog_refuses_download_when_validation_cache_would_leave_low_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport()
    _install_mock_client(monkeypatch, transport)
    actual_usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: type(actual_usage)(
            40 * 1024**3,
            25 * 1024**3,
            15 * 1024**3,
        ),
    )

    with pytest.raises(LocalValidationError, match="15 GiB"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

    assert len(transport.requests) == 3
    assert all("/api/models/" in request for request in transport.requests)


def test_lock_catalog_rejects_stream_mismatch_without_output_or_temp_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(bad_weight_role="summary")
    _install_mock_client(monkeypatch, transport)
    output = tmp_path / "candidate.lock.json"

    with pytest.raises(LocalValidationError, match="metadata"):
        lock_script.lock_catalog(_write_json(tmp_path / "catalog.json", _catalog()), output)

    assert not output.exists()
    assert list(tmp_path.glob(".local-ai-lock-*")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["candidates"][0].update(role="unknown"),
        lambda value: value["candidates"][1].update(role="ocr"),
        lambda value: value["candidates"][0].update(repository="../bad"),
    ],
)
def test_lock_catalog_rejects_bad_catalog_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    catalog_value = _catalog()
    mutation(catalog_value)
    network_called = False

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        nonlocal network_called
        network_called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr(lock_script.httpx, "Client", forbidden_client)

    with pytest.raises(LocalValidationError):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", catalog_value),
            tmp_path / "candidate.lock.json",
        )

    assert network_called is False


def test_lock_catalog_rejects_invalid_pack_revision_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_value = _catalog()
    catalog_value["pack_revision"] = "../mutable"
    network_called = False

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        nonlocal network_called
        network_called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr(lock_script.httpx, "Client", forbidden_client)

    with pytest.raises(LocalValidationError, match="pack revision"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", catalog_value),
            tmp_path / "candidate.lock.json",
        )

    assert network_called is False


def test_lock_catalog_rejects_oversized_declared_metadata_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            content=b"{}",
            headers={
                "Content-Length": str(lock_script.MAX_HF_METADATA_BYTES + 1),
            },
            request=request,
        )

    _install_mock_client(monkeypatch, transport)  # type: ignore[arg-type]

    with pytest.raises(LocalValidationError, match="metadata response is too large"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

    assert len(requests) == 1


def test_lock_catalog_rejects_oversized_actual_metadata_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[str] = []
    content = b'{"padding":"' + b"x" * lock_script.MAX_HF_METADATA_BYTES + b'"}'

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            stream=httpx.ByteStream(content),
            request=request,
        )

    _install_mock_client(monkeypatch, transport)  # type: ignore[arg-type]

    with pytest.raises(LocalValidationError, match="metadata response is too large"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

    assert len(requests) == 1


def test_lock_script_is_directly_executable_from_backend() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/lock_local_ai_manifest.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--catalog" in result.stdout
    assert "--output" in result.stdout
