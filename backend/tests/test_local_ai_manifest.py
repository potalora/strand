"""Strict tests for local-AI manifests and candidate catalog locking."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    is_secret_shaped_manifest_text,
    load_manifest,
    parse_manifest,
)
from app.services.local_ai.types import ModelRole

lock_script = importlib.import_module("scripts.lock_local_ai_manifest")

ROLES = ("ocr", "extraction", "summary")
LICENSE_SOURCE_CONTENT = b"Apache License\nVersion 2.0, January 2004\n"
LICENSE_SOURCE_REVISION = "f" * 40
LICENSE_SOURCE_SHA256 = hashlib.sha256(LICENSE_SOURCE_CONTENT).hexdigest()


@pytest.fixture(autouse=True)
def _stable_validation_disk_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep non-capacity lock tests independent of the host's free disk."""

    usage_type = type(shutil.disk_usage(tmp_path))
    ample_usage = usage_type(
        80 * 1024**3,
        20 * 1024**3,
        60 * 1024**3,
    )
    monkeypatch.setattr(
        lock_script.shutil,
        "disk_usage",
        lambda _: ample_usage,
    )


def _pathological_json_nesting(depth: int = 2_000) -> bytes:
    return b"[" * depth + b"0" + b"]" * depth


def _manifest_file(
    path: str = "model.safetensors", *, size: int = 10
) -> dict[str, Any]:
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


@pytest.mark.parametrize(
    ("field", "mutate", "canary"),
    [
        (
            "pack revision",
            lambda value, canary: value.update(pack_revision=canary),
            "sk-proj-pack-manifest-canary",
        ),
        (
            "repository",
            lambda value, canary: value["artifacts"][0].update(
                repository=f"owner/{canary}"
            ),
            "sk-proj-repository-manifest-canary",
        ),
        (
            "revision",
            lambda value, canary: value["artifacts"][0].update(revision=canary),
            "sk-proj-revision-manifest-canary",
        ),
        (
            "quantization",
            lambda value, canary: value["artifacts"][0].update(quantization=canary),
            "sk_" + "live_quantization_manifest_canary",
        ),
        (
            "runtime",
            lambda value, canary: value["runtime"].update(name=canary),
            "api_key=runtime-manifest-canary",
        ),
        (
            "validation suite",
            lambda value, canary: value.update(validation_suite_version=canary),
            "secret-validation-manifest-canary",
        ),
        (
            "attribution",
            lambda value, canary: value["artifacts"][0].update(
                attribution=f"authorization=Bearer-{canary}"
            ),
            "attribution-manifest-canary",
        ),
        (
            "path",
            lambda value, canary: value["artifacts"][0]["files"][0].update(
                path=f"weights/{canary}.safetensors"
            ),
            "sk-proj-path-manifest-canary",
        ),
    ],
)
def test_manifest_rejects_secret_shaped_metadata_without_echo(
    field: str,
    mutate: Any,
    canary: str,
) -> None:
    raw = _valid_manifest()
    mutate(raw, canary)

    with pytest.raises(LocalValidationError) as exc_info:
        parse_manifest(raw)

    assert field
    assert canary not in str(exc_info.value)


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
            lambda value: value["artifacts"][0]["files"][0].update(
                path="../model.json"
            ),
            "path",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(path="/model.json"),
            "path",
        ),
        (
            lambda value: value["artifacts"][0]["files"][0].update(
                path="a\\model.json"
            ),
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
            lambda value: value["artifacts"][0].update(
                decode_limits={"max_input_tokens": 0}
            ),
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


def test_manifest_normalizes_pathological_json_nesting(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(_pathological_json_nesting())

    with pytest.raises(LocalValidationError, match="JSON"):
        load_manifest(path)


def test_manifest_and_schema_reject_uppercase_file_suffix(tmp_path: Path) -> None:
    raw = _valid_manifest()
    raw["artifacts"][0]["files"][0]["path"] = "model.JSON"

    with pytest.raises(LocalValidationError, match="lowercase"):
        load_manifest(_write_json(tmp_path / "manifest.json", raw))

    schema_path = Path(__file__).parents[1] / "app/model_manifests/schema-v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    pattern = schema["properties"]["artifacts"]["items"]["properties"]["files"][
        "items"
    ]["properties"]["path"]["pattern"]
    assert re.search(pattern, "model.JSON") is None
    assert re.search(pattern, "model.json") is not None


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
                "revision": str(ROLES.index(role) + 1) * 40,
                "quantization": "4bit",
                "license": "apache-2.0",
                "attribution": f"https://huggingface.co/owner/{role}",
                **(
                    {
                        "license_source": {
                            "repository": "owner/extraction-base",
                            "revision": LICENSE_SOURCE_REVISION,
                            "path": "LICENSE",
                            "sha256": LICENSE_SOURCE_SHA256,
                            "size": len(LICENSE_SOURCE_CONTENT),
                        }
                    }
                    if role == "extraction"
                    else {}
                ),
                "decode_limits": {
                    "max_input_tokens": 4096,
                    "max_output_tokens": 1024,
                },
            }
            for role in ROLES
        ],
    }


@pytest.mark.parametrize(
    ("field", "mutation", "canary"),
    [
        (
            "pack revision",
            lambda value, canary: value.update(pack_revision=canary),
            "sk-proj-catalog-canary",
        ),
        (
            "validation suite",
            lambda value, canary: value.update(validation_suite_version=canary),
            "secret-validation-catalog-canary",
        ),
        (
            "runtime",
            lambda value, canary: value["runtime"].update(name=canary),
            "api_key=runtime-catalog-canary",
        ),
        (
            "repository",
            lambda value, canary: value["candidates"][0].update(
                repository=f"owner/{canary}"
            ),
            "sk-proj-repository-catalog-canary",
        ),
        (
            "quantization",
            lambda value, canary: value["candidates"][0].update(quantization=canary),
            "sk_" + "live_quantization_catalog_canary",
        ),
        (
            "attribution",
            lambda value, canary: value["candidates"][0].update(
                attribution=f"authorization=Bearer-{canary}"
            ),
            "attribution-catalog-canary",
        ),
        (
            "license source repository",
            lambda value, canary: value["candidates"][1]["license_source"].update(
                repository=f"owner/{canary}"
            ),
            "sk-proj-license-source-catalog-canary",
        ),
        (
            "standalone token label",
            lambda value, canary: value.update(pack_revision=canary),
            "token-mycredential1234",
        ),
        (
            "concatenated password label",
            lambda value, canary: value.update(validation_suite_version=canary),
            "passwordhunter2",
        ),
        (
            "concatenated secret label",
            lambda value, canary: value["runtime"].update(version=canary),
            "secretcanary123",
        ),
        (
            "repository token label",
            lambda value, canary: value["candidates"][0].update(
                repository=f"owner/{canary}"
            ),
            "token-mycredential1234",
        ),
    ],
)
def test_lock_catalog_rejects_secret_shaped_strings_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutation: Any,
    canary: str,
) -> None:
    catalog = _catalog()
    mutation(catalog, canary)
    network_called = False

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        nonlocal network_called
        network_called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr(lock_script.httpx, "Client", forbidden_client)

    with pytest.raises(LocalValidationError) as exc_info:
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", catalog),
            tmp_path / "candidate.lock.json",
        )

    assert field
    assert network_called is False
    assert canary not in str(exc_info.value)


@pytest.mark.parametrize(
    "value",
    [
        "apple-m4-16gb-v1",
        "local-ai-fixtures-v1",
        "mlx-vlm",
        "0.5.0",
        "owner/ocr",
        "sahilchachra/ovisocr2-int4-mlx",
        "938d8919941c6e7efd3c7150eff7fe9d12afa631",
        "https://huggingface.co/sahilchachra/ovisocr2-int4-mlx",
        "https://huggingface.co/numind/NuExtract3-mlx-4bits",
        "tokenizer_config.json",
    ],
)
def test_manifest_secret_filter_preserves_pinned_public_identity(value: str) -> None:
    assert is_secret_shaped_manifest_text(value) is False


class _HFTransport:
    def __init__(
        self,
        *,
        forbidden_role: str | None = None,
        auto_map_role: str | None = None,
        bad_weight_role: str | None = None,
        oversized_json_role: str | None = None,
        uppercase_json_role: str | None = None,
        nested_config_role: str | None = None,
        recursive_config_role: str | None = None,
        large_tokenizer_role: str | None = None,
        large_vocab_role: str | None = None,
        missing_license_role: str | None = None,
        tampered_license_source: bool = False,
    ) -> None:
        self.requests: list[str] = []
        self.forbidden_role = forbidden_role
        self.auto_map_role = auto_map_role
        self.bad_weight_role = bad_weight_role
        self.oversized_json_role = oversized_json_role
        self.uppercase_json_role = uppercase_json_role
        self.missing_license_role = missing_license_role
        self.tampered_license_source = tampered_license_source
        self.config_bytes: dict[str, bytes] = {}
        self.tokenizer_bytes: dict[str, bytes] = {}
        self.vocab_bytes: dict[str, bytes] = {}
        self.weight_bytes: dict[str, bytes] = {}
        for role in ROLES:
            config = {"architectures": [f"{role.title()}Model"]}
            if role == auto_map_role:
                config["auto_map"] = {"AutoModel": "modeling_custom.CustomModel"}
            if role == nested_config_role:
                nested: Any = "leaf"
                for _ in range(80):
                    nested = {"child": nested}
                config["nested"] = nested
            if role == recursive_config_role:
                self.config_bytes[role] = _pathological_json_nesting()
            else:
                self.config_bytes[role] = json.dumps(config).encode()
            if role == large_tokenizer_role:
                self.tokenizer_bytes[role] = json.dumps(
                    {"vocab": list(range(lock_script.MAX_JSON_STRUCTURE_NODES + 1))}
                ).encode()
            if role == large_vocab_role:
                self.vocab_bytes[role] = json.dumps(
                    {
                        f"token-{index}": index
                        for index in range(lock_script.MAX_JSON_STRUCTURE_NODES + 1)
                    }
                ).encode()
            self.weight_bytes[role] = f"safe-{role}-weights".encode()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        parts = request.url.path.strip("/").split("/")
        if parts[:2] == ["api", "models"]:
            role = parts[3]
            config = self.config_bytes[role]
            weights = self.weight_bytes[role]
            config_path = (
                "config.JSON" if role == self.uppercase_json_role else "config.json"
            )
            config_size = (
                lock_script.MAX_JSON_METADATA_BYTES + 1
                if role == self.oversized_json_role
                else len(config)
            )
            siblings: list[dict[str, Any]] = [
                {
                    "rfilename": config_path,
                    "size": config_size,
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
            tokenizer = self.tokenizer_bytes.get(role)
            if tokenizer is not None:
                siblings.append(
                    {
                        "rfilename": "tokenizer.json",
                        "size": len(tokenizer),
                        "blobId": hashlib.sha1(
                            tokenizer, usedforsecurity=False
                        ).hexdigest(),
                    }
                )
            vocab = self.vocab_bytes.get(role)
            if vocab is not None:
                siblings.append(
                    {
                        "rfilename": "vocab.json",
                        "size": len(vocab),
                        "blobId": hashlib.sha1(
                            vocab, usedforsecurity=False
                        ).hexdigest(),
                    }
                )
            if role == self.forbidden_role:
                siblings.append(
                    {"rfilename": "modeling_custom.py", "size": 20, "blobId": "e" * 40}
                )
            return httpx.Response(
                200,
                json={
                    "id": f"owner/{role}",
                    "sha": str(ROLES.index(role) + 1) * 40,
                    "cardData": (
                        {}
                        if role == self.missing_license_role
                        else {"license": "apache-2.0"}
                    ),
                    "siblings": siblings,
                },
                request=request,
            )
        if (
            parts[:2] == ["owner", "extraction-base"]
            and parts[2:4] == ["resolve", LICENSE_SOURCE_REVISION]
            and parts[4:] == ["LICENSE"]
        ):
            content = LICENSE_SOURCE_CONTENT
            if self.tampered_license_source:
                content += b"tampered"
            return httpx.Response(200, content=content, request=request)
        if parts[0] == "owner" and parts[2] == "resolve":
            role = parts[1]
            filename = "/".join(parts[4:])
            if filename == "config.json":
                return httpx.Response(
                    200, content=self.config_bytes[role], request=request
                )
            if filename == "tokenizer.json":
                return httpx.Response(
                    200, content=self.tokenizer_bytes[role], request=request
                )
            if filename == "vocab.json":
                return httpx.Response(
                    200, content=self.vocab_bytes[role], request=request
                )
            if filename == "model.safetensors":
                content = self.weight_bytes[role]
                if role == self.bad_weight_role:
                    content += b"-tampered"
                return httpx.Response(200, content=content, request=request)
        return httpx.Response(404, request=request)


def _install_mock_client(
    monkeypatch: pytest.MonkeyPatch, transport: _HFTransport
) -> None:
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
        json.dumps(
            json.loads(output.read_text()), sort_keys=True, separators=(",", ":")
        )
        + "\n"
    )
    manifest = load_manifest(output)
    assert {artifact.role for artifact in manifest.artifacts} == set(ModelRole)
    assert all(len(artifact.revision) == 40 for artifact in manifest.artifacts)
    assert all(
        len(file.sha256) == 64
        for artifact in manifest.artifacts
        for file in artifact.files
    )
    assert not any(
        ".gitattributes" in file.path
        for artifact in manifest.artifacts
        for file in artifact.files
    )
    extraction = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.role == ModelRole.EXTRACTION
    )
    assert f"license-sha256={LICENSE_SOURCE_SHA256}" in extraction.attribution


def test_lock_catalog_accepts_missing_quant_license_only_with_verified_pinned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(missing_license_role="extraction")
    _install_mock_client(monkeypatch, transport)

    lock_script.lock_catalog(
        _write_json(tmp_path / "catalog.json", _catalog()),
        tmp_path / "candidate.lock.json",
    )

    license_request = next(
        index
        for index, request in enumerate(transport.requests)
        if "/owner/extraction-base/resolve/" in request
    )
    first_model_file_request = next(
        index
        for index, request in enumerate(transport.requests)
        if "/resolve/" in request and "/owner/extraction-base/resolve/" not in request
    )
    assert license_request < first_model_file_request


def test_lock_catalog_rejects_missing_license_metadata_without_attestation_before_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    catalog["candidates"][1].pop("license_source")
    transport = _HFTransport(missing_license_role="extraction")
    _install_mock_client(monkeypatch, transport)

    with pytest.raises(LocalValidationError, match="license metadata"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", catalog),
            tmp_path / "candidate.lock.json",
        )

    assert len(transport.requests) == 2
    assert all("/api/models/" in request for request in transport.requests)


def test_lock_catalog_rejects_tampered_license_attestation_before_model_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(
        missing_license_role="extraction",
        tampered_license_source=True,
    )
    _install_mock_client(monkeypatch, transport)

    with pytest.raises(LocalValidationError, match="[Ll]icense attestation"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

    assert not any(
        "/resolve/" in request and "/owner/extraction-base/resolve/" not in request
        for request in transport.requests
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda source: source.update(revision="main"),
        lambda source: source.update(path="../LICENSE"),
        lambda source: source.update(sha256="A" * 64),
        lambda source: source.update(size=0),
    ],
)
def test_lock_catalog_rejects_invalid_license_attestation_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    catalog = _catalog()
    mutation(catalog["candidates"][1]["license_source"])
    network_called = False

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        nonlocal network_called
        network_called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr(lock_script.httpx, "Client", forbidden_client)

    with pytest.raises(LocalValidationError, match="license source"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", catalog),
            tmp_path / "candidate.lock.json",
        )

    assert network_called is False


def test_lock_catalog_preflights_all_metadata_before_any_file_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(forbidden_role="summary")
    _install_mock_client(monkeypatch, transport)
    output = tmp_path / "candidate.lock.json"

    with pytest.raises(LocalValidationError, match="forbidden"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()), output
        )

    assert not output.exists()
    assert len(transport.requests) == 3
    assert all("/api/models/" in request for request in transport.requests)


def test_lock_catalog_rejects_oversized_repository_json_before_file_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(oversized_json_role="summary")
    _install_mock_client(monkeypatch, transport)

    with pytest.raises(LocalValidationError, match="JSON metadata"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

    assert len(transport.requests) == 3
    assert all("/api/models/" in request for request in transport.requests)


def test_lock_catalog_rejects_uppercase_suffix_before_file_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(uppercase_json_role="summary")
    _install_mock_client(monkeypatch, transport)

    with pytest.raises(LocalValidationError, match="lowercase"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

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
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()), output
        )

    assert not output.exists()
    extraction_requests = [
        request
        for request in transport.requests
        if "/owner/extraction/resolve/" in request
    ]
    assert any(request.endswith("/config.json") for request in extraction_requests)
    assert not any(
        request.endswith("/model.safetensors") for request in transport.requests
    )


def test_lock_catalog_accepts_large_static_tokenizer_json_but_still_inspects_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(large_tokenizer_role="ocr")
    _install_mock_client(monkeypatch, transport)
    output = tmp_path / "candidate.lock.json"

    lock_script.lock_catalog(
        _write_json(tmp_path / "catalog.json", _catalog()),
        output,
    )

    manifest = load_manifest(output)
    ocr = next(
        artifact for artifact in manifest.artifacts if artifact.role is ModelRole.OCR
    )
    assert any(file.path == "tokenizer.json" for file in ocr.files)
    assert any(request.endswith("/config.json") for request in transport.requests)


def test_lock_catalog_accepts_large_static_vocab_json_but_still_inspects_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(large_vocab_role="ocr")
    _install_mock_client(monkeypatch, transport)
    output = tmp_path / "candidate.lock.json"

    lock_script.lock_catalog(
        _write_json(tmp_path / "catalog.json", _catalog()),
        output,
    )

    manifest = load_manifest(output)
    ocr = next(
        artifact for artifact in manifest.artifacts if artifact.role is ModelRole.OCR
    )
    assert any(file.path == "vocab.json" for file in ocr.files)
    assert any(request.endswith("/config.json") for request in transport.requests)


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        (
            "tokenizer.json",
            {"auto_map": {"AutoTokenizer": "custom.Tokenizer"}},
        ),
        (
            "vocab.json",
            {"trust_remote_code": True},
        ),
    ],
)
def test_static_tokenizer_json_rejects_remote_code_keys(
    tmp_path: Path,
    filename: str,
    payload: dict[str, Any],
) -> None:
    path = _write_json(tmp_path / filename, payload)

    with pytest.raises(LocalValidationError, match="remote code"):
        lock_script._inspect_downloaded_json(path, path.stat().st_size)


def test_static_tokenizer_allows_bounded_bpe_merge_containers() -> None:
    tokenizer = {
        "model": {
            "type": "BPE",
            "merges": [["left", "right"]] * (lock_script.MAX_JSON_STRUCTURE_NODES + 1),
        }
    }

    assert lock_script._static_json_contains_remote_code_requirement(tokenizer) is False


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
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()), output
        )

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


def test_lock_catalog_rejects_mutable_candidate_revision_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_value = _catalog()
    catalog_value["candidates"][0]["revision"] = "main"
    network_called = False

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        nonlocal network_called
        network_called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr(lock_script.httpx, "Client", forbidden_client)

    with pytest.raises(LocalValidationError, match="candidate revision"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", catalog_value),
            tmp_path / "candidate.lock.json",
        )

    assert network_called is False


def test_lock_catalog_rejects_repository_revision_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_value = _catalog()
    catalog_value["candidates"][0]["revision"] = "9" * 40
    transport = _HFTransport()
    _install_mock_client(monkeypatch, transport)

    with pytest.raises(LocalValidationError, match="candidate revision"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", catalog_value),
            tmp_path / "candidate.lock.json",
        )

    assert len(transport.requests) == 1
    assert f"/revision/{'9' * 40}" in transport.requests[0]


def test_lock_catalog_normalizes_pathological_catalog_nesting_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(_pathological_json_nesting())
    network_called = False

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        nonlocal network_called
        network_called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr(lock_script.httpx, "Client", forbidden_client)

    with pytest.raises(LocalValidationError, match="JSON"):
        lock_script.lock_catalog(catalog, tmp_path / "candidate.lock.json")

    assert network_called is False


def test_lock_catalog_normalizes_pathological_hf_metadata_nesting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            stream=httpx.ByteStream(_pathological_json_nesting()),
            request=request,
        )

    _install_mock_client(monkeypatch, transport)  # type: ignore[arg-type]

    with pytest.raises(LocalValidationError, match="metadata"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

    assert len(requests) == 1


def test_lock_catalog_normalizes_pathological_downloaded_json_nesting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(recursive_config_role="ocr")
    _install_mock_client(monkeypatch, transport)

    with pytest.raises(LocalValidationError, match="JSON metadata"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

    model_file_requests = [
        request
        for request in transport.requests
        if "/resolve/" in request and "/owner/extraction-base/resolve/" not in request
    ]
    assert len(model_file_requests) == 1
    assert model_file_requests[-1].endswith("/config.json")


def test_lock_catalog_bounds_downloaded_json_structure_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _HFTransport(nested_config_role="ocr")
    _install_mock_client(monkeypatch, transport)

    with pytest.raises(LocalValidationError, match="JSON structure"):
        lock_script.lock_catalog(
            _write_json(tmp_path / "catalog.json", _catalog()),
            tmp_path / "candidate.lock.json",
        )

    model_file_requests = [
        request
        for request in transport.requests
        if "/resolve/" in request and "/owner/extraction-base/resolve/" not in request
    ]
    assert len(model_file_requests) == 1
    assert model_file_requests[-1].endswith("/config.json")


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


def test_local_ai_env_is_documented_only_in_canonical_example() -> None:
    backend_root = Path(__file__).parents[1]
    repository_root = backend_root.parent
    canonical_example = (repository_root / ".env.example").read_text(encoding="utf-8")

    assert not (backend_root / ".env.example").exists()
    expected_lines = {
        "LOCAL_AI_ENABLED=false",
        "LOCAL_AI_MODEL_DIR=./data/local-ai/models",
        "LOCAL_AI_SCRATCH_DIR=./data/local-ai/scratch",
        "LOCAL_AI_MANIFEST_PATH=./app/model_manifests/apple-m4-16gb-v1.lock.json",
        (
            "LOCAL_AI_WORKER_COMMAND="
            "../workers/local_ai/apple_mlx/.venv/bin/local-ai-mlx-worker"
        ),
        "LOCAL_AI_MAX_FILES=64",
        "LOCAL_AI_MAX_FILE_BYTES=8589934592",
        "LOCAL_AI_MAX_PACK_BYTES=21474836480",
        "LOCAL_AI_WORKER_TIMEOUT_SECONDS=1800",
        "LOCAL_AI_WORKER_HARD_TIMEOUT_SECONDS=7200",
        "LOCAL_AI_MAX_PAGE_PIXELS=40000000",
    }
    assert expected_lines <= set(canonical_example.splitlines())


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
