"""Document-free, bounded downloader tests for local-AI model artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.downloader import DownloadProgress, download_manifest
from app.services.local_ai.errors import LocalAIError, LocalValidationError
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.types import ModelRole


def _manifest(
    *,
    pack_revision: str = "download-v1",
    payloads: dict[ModelRole, bytes] | None = None,
) -> tuple[LocalAIManifest, dict[str, bytes]]:
    values = payloads or {
        ModelRole.OCR: b"ocr",
        ModelRole.EXTRACTION: b"extract",
        ModelRole.SUMMARY: b"summary",
    }
    responses: dict[str, bytes] = {}
    artifacts: list[ManifestArtifact] = []
    for index, role in enumerate(ModelRole, start=1):
        path = f"nested/{role.value}.safetensors"
        repository = f"owner/{role.value}"
        revision = str(index) * 40
        content = values[role]
        responses[f"https://huggingface.co/{repository}/resolve/{revision}/{path}"] = (
            content
        )
        artifacts.append(
            ManifestArtifact(
                role=role,
                repository=repository,
                revision=revision,
                quantization="4bit",
                license="apache-2.0",
                attribution=f"https://huggingface.co/{repository}",
                decode_limits={"max_input_tokens": 4096, "max_output_tokens": 1024},
                files=(
                    ManifestFile(
                        path=path,
                        sha256=hashlib.sha256(content).hexdigest(),
                        size=len(content),
                    ),
                ),
            )
        )
    return (
        LocalAIManifest(
            schema_version=1,
            pack_revision=pack_revision,
            platform="apple_silicon",
            runtime={"name": "mlx-vlm", "version": "0.5.0"},
            validation_suite_version="fixtures-v1",
            artifacts=tuple(artifacts),
        ),
        responses,
    )


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, bytes],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    seen: list[httpx.Request] | None = None,
) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(
            status_code,
            headers=headers,
            content=responses.get(str(request.url), b"missing"),
            request=request,
        )

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(
        "app.services.local_ai.downloader.httpx.AsyncClient", client_factory
    )


def _install_handler(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
    *,
    addresses: list[str] | None = None,
) -> None:
    real_client = httpx.AsyncClient
    resolved = addresses or ["93.184.216.34"]
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
            for address in resolved
        ],
    )

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(
        "app.services.local_ai.downloader.httpx.AsyncClient", client_factory
    )


@pytest.mark.asyncio
async def test_downloader_is_manifest_only_and_progress_has_exact_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    seen: list[httpx.Request] = []
    _install_transport(monkeypatch, responses, seen=seen)
    progress: list[dict[str, int | str]] = []

    await download_manifest(manifest, ArtifactStore(tmp_path), progress.append)

    assert progress
    assert all(set(item) == {"role", "bytes_done", "bytes_total"} for item in progress)
    assert {request.url.host for request in seen} == {"huggingface.co"}
    assert all("document" not in str(request.url).lower() for request in seen)
    assert ArtifactStore(tmp_path).active_revision() == "download-v1"


def test_download_progress_serializes_to_exact_model_pack_fields() -> None:
    progress = DownloadProgress(role="ocr", bytes_done=3, bytes_total=10)

    assert asdict(progress) == {"role": "ocr", "bytes_done": 3, "bytes_total": 10}


@pytest.mark.asyncio
async def test_downloader_derives_encoded_urls_from_immutable_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    first = manifest.artifacts[0]
    content = next(iter(responses.values()))
    encoded_file = replace(first.files[0], path="weights/model name.safetensors")
    changed = replace(first, files=(encoded_file,))
    manifest = replace(manifest, artifacts=(changed, *manifest.artifacts[1:]))
    old_url = next(url for url in responses if "/owner/ocr/" in url)
    responses[
        old_url.replace("nested/ocr.safetensors", "weights/model%20name.safetensors")
    ] = content
    responses.pop(old_url)
    seen: list[httpx.Request] = []
    _install_transport(monkeypatch, responses, seen=seen)

    await download_manifest(manifest, ArtifactStore(tmp_path), None)

    assert any(
        "weights/model%20name.safetensors" in str(request.url) for request in seen
    )


@pytest.mark.asyncio
async def test_downloader_follows_validated_allowlisted_public_redirect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    first_url = next(iter(responses))
    redirected_url = "https://cdn-lfs.huggingface.co/model.safetensors"
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if url == first_url:
            return httpx.Response(302, headers={"location": redirected_url})
        if url == redirected_url:
            return httpx.Response(200, content=responses[first_url])
        return httpx.Response(200, content=responses[url])

    _install_handler(monkeypatch, handler)

    await download_manifest(manifest, ArtifactStore(tmp_path), None)

    assert seen[:2] == [first_url, redirected_url]


@pytest.mark.parametrize(
    "destination",
    [
        "http://cdn-lfs.huggingface.co/model.safetensors",
        "https://cdn-lfs.huggingface.co:444/model.safetensors",
        "https://user:secret@cdn-lfs.huggingface.co/model.safetensors",
        "https://example.com/model.safetensors",
        "https://127.0.0.1/model.safetensors",
    ],
)
@pytest.mark.asyncio
async def test_downloader_rejects_unsafe_redirect_before_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
) -> None:
    manifest, responses = _manifest()
    first_url = next(iter(responses))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": destination})

    _install_handler(monkeypatch, handler)

    with pytest.raises(LocalAIError, match="redirect"):
        await download_manifest(manifest, ArtifactStore(tmp_path), None)

    assert seen == [first_url]


@pytest.mark.asyncio
async def test_downloader_rejects_private_resolution_before_redirect_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    first_url = next(iter(responses))
    destination = "https://cdn-lfs.huggingface.co/model.safetensors"
    seen: list[str] = []
    resolutions = {
        "huggingface.co": "93.184.216.34",
        "cdn-lfs.huggingface.co": "10.0.0.8",
    }
    real_client = httpx.AsyncClient

    def resolve(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (resolutions[host], 443),
            )
        ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": destination})

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        "app.services.local_ai.downloader.httpx.AsyncClient", client_factory
    )

    with pytest.raises(LocalAIError, match="redirect"):
        await download_manifest(manifest, ArtifactStore(tmp_path), None)

    assert seen == [first_url]


@pytest.mark.asyncio
async def test_downloader_rejects_redirect_loop_at_bounded_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _manifest()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": str(request.url)})

    _install_handler(monkeypatch, handler)

    with pytest.raises(LocalAIError, match="redirect"):
        await download_manifest(manifest, ArtifactStore(tmp_path), None)

    assert len(seen) == 6


@pytest.mark.asyncio
async def test_downloader_rejects_observed_file_and_content_length_overruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    first_url = next(iter(responses))
    responses[first_url] += b"overrun"
    _install_transport(monkeypatch, responses)

    with pytest.raises(LocalValidationError, match="byte limit"):
        await download_manifest(manifest, ArtifactStore(tmp_path), None)

    assert list((tmp_path / ".staging").iterdir()) == []
    assert not list(tmp_path.rglob("*.partial"))

    second_root = tmp_path / "content-length"
    valid_manifest, valid_responses = _manifest(pack_revision="content-length")
    _install_transport(
        monkeypatch, valid_responses, headers={"content-length": "999999"}
    )
    with pytest.raises(LocalValidationError, match="byte limit"):
        await download_manifest(valid_manifest, ArtifactStore(second_root), None)


@pytest.mark.asyncio
async def test_downloader_rejects_hash_mismatch_and_cleans_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    first_url = next(iter(responses))
    responses[first_url] = b"bad"
    first = manifest.artifacts[0]
    manifest = replace(
        manifest,
        artifacts=(
            replace(first, files=(replace(first.files[0], size=3),)),
            *manifest.artifacts[1:],
        ),
    )
    _install_transport(monkeypatch, responses)

    with pytest.raises(LocalValidationError, match="SHA-256"):
        await download_manifest(manifest, ArtifactStore(tmp_path), None)

    assert list((tmp_path / ".staging").iterdir()) == []
    assert not (tmp_path / "activation-state.json").exists()


@pytest.mark.asyncio
async def test_downloader_normalizes_http_and_callback_errors_without_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    _install_transport(monkeypatch, responses, status_code=403)

    with pytest.raises(LocalAIError) as http_error:
        await download_manifest(manifest, ArtifactStore(tmp_path / "http"), None)

    assert "http" not in str(http_error.value).lower()
    assert "huggingface" not in str(http_error.value).lower()

    _install_transport(monkeypatch, responses)

    def bad_callback(_: dict[str, int | str]) -> None:
        raise RuntimeError("patient-secret")

    with pytest.raises(LocalAIError) as callback_error:
        await download_manifest(
            replace(manifest, pack_revision="callback"),
            ArtifactStore(tmp_path / "callback"),
            bad_callback,
        )

    assert "patient-secret" not in str(callback_error.value)
    assert list((tmp_path / "callback" / ".staging").iterdir()) == []


@pytest.mark.asyncio
async def test_downloader_cancellation_cleans_partial_and_staging_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    _install_transport(monkeypatch, responses)

    def cancel(_: dict[str, int | str]) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await download_manifest(manifest, ArtifactStore(tmp_path), cancel)

    assert list((tmp_path / ".staging").iterdir()) == []
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.asyncio
async def test_downloader_normalizes_malformed_manifest_without_path_or_hash_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    secret = "../private/patient-name.json"
    first = manifest.artifacts[0]
    malformed = replace(
        manifest,
        artifacts=(
            replace(first, files=(replace(first.files[0], path=secret),)),
            *manifest.artifacts[1:],
        ),
    )
    _install_transport(monkeypatch, responses)

    with pytest.raises(LocalValidationError) as error:
        await download_manifest(malformed, ArtifactStore(tmp_path), None)

    assert secret not in str(error.value)
    assert first.files[0].sha256 not in str(error.value)


@pytest.mark.asyncio
async def test_partial_and_operation_files_are_owner_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, responses = _manifest()
    observed_modes: list[int] = []

    async def callback(_: dict[str, int | str]) -> None:
        partials = list(tmp_path.rglob("*.partial"))
        if partials:
            observed_modes.append(partials[0].stat().st_mode & 0o777)

    _install_transport(monkeypatch, responses)
    await download_manifest(manifest, ArtifactStore(tmp_path), callback)

    assert observed_modes and set(observed_modes) == {0o600}
    operation_files = list((tmp_path / "operations").glob("*.json"))
    assert operation_files
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in operation_files)
    assert set(json.loads(operation_files[0].read_text())) == {
        "role",
        "bytes_done",
        "bytes_total",
    }
