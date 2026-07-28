"""Document-free streaming downloader for immutable local-AI manifests."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import inspect
import os
import socket
import stat
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import httpx

from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalAIError, LocalValidationError
from app.services.local_ai.manifest import (
    MAX_MANIFEST_FILE_BYTES,
    MAX_MANIFEST_PACK_BYTES,
    LocalAIManifest,
)

_DOWNLOAD_ORIGIN = "https://huggingface.co"
_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_REDIRECTS = 5
_MAX_STREAM_RESUME_RETRIES = 2
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DOWNLOAD_HOSTS = frozenset(
    {
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
        "cas-bridge.xethub.hf.co",
        "us.aws.cdn.hf.co",
    }
)
ProgressCallback = Callable[
    [dict[str, int | str]],
    None | Awaitable[None],
]


@dataclass(frozen=True)
class DownloadProgress:
    """Minimal model-only progress state safe for logs and APIs."""

    role: str
    bytes_done: int
    bytes_total: int


def progress_payload(value: DownloadProgress) -> dict[str, int | str]:
    """Serialize progress to the exact allowlisted model-pack fields."""

    return asdict(value)


def _download_url(repository: str, revision: str, path: str) -> str:
    encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"{_DOWNLOAD_ORIGIN}/{repository}/resolve/{revision}/{encoded_path}"


def _make_owner_only_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
    except OSError as exc:
        raise LocalAIError("Model artifact download failed") from exc


def _open_partial(path: Path) -> Any:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        return os.fdopen(descriptor, "wb")
    except OSError as exc:
        raise LocalAIError("Model artifact download failed") from exc


def _open_partial_append(path: Path, expected_size: int) -> Any:
    """Open an existing owner-only regular partial without following links."""

    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        )
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_size != expected_size
            or details.st_uid != os.getuid()
            or details.st_mode & 0o077
        ):
            raise LocalAIError("Model artifact download failed")
        return os.fdopen(descriptor, "ab")
    except LocalAIError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise LocalAIError("Model artifact download failed") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        raise LocalAIError("Model artifact download failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


async def _publish_progress(
    *,
    store: ArtifactStore,
    operation_id: str,
    progress: DownloadProgress,
    callback: ProgressCallback | None,
) -> None:
    payload = progress_payload(progress)
    store.write_progress(operation_id, payload)
    if callback is None:
        return
    try:
        result = callback(payload)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        raise LocalAIError("Model download progress callback failed") from exc


def _declared_content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise LocalValidationError("Model download byte limit is invalid") from exc
    if value < 0:
        raise LocalValidationError("Model download byte limit is invalid")
    return value


def _validate_download_target(url: httpx.URL) -> None:
    try:
        if (
            url.scheme != "https"
            or url.host not in _DOWNLOAD_HOSTS
            or url.port not in {None, 443}
            or bool(url.userinfo)
        ):
            raise LocalAIError("Model download redirect rejected")
        addresses = socket.getaddrinfo(
            url.host,
            443,
            type=socket.SOCK_STREAM,
        )
        if not addresses:
            raise LocalAIError("Model download redirect rejected")
        for address in addresses:
            parsed = ipaddress.ip_address(address[4][0].split("%", 1)[0])
            if not parsed.is_global:
                raise LocalAIError("Model download redirect rejected")
    except LocalAIError:
        raise
    except (OSError, TypeError, ValueError, IndexError) as exc:
        raise LocalAIError("Model download redirect rejected") from exc


@asynccontextmanager
async def _validated_stream(
    client: httpx.AsyncClient,
    initial_url: str,
    headers: dict[str, str] | None = None,
) -> Any:
    try:
        current = httpx.URL(initial_url)
    except (TypeError, ValueError) as exc:
        raise LocalAIError("Model download redirect rejected") from exc

    for redirect_count in range(_MAX_REDIRECTS + 1):
        _validate_download_target(current)
        response: httpx.Response | None = None
        try:
            request = client.build_request("GET", current, headers=headers)
            response = await client.send(request, stream=True, follow_redirects=False)
            if response.status_code not in _REDIRECT_STATUSES:
                try:
                    yield response
                finally:
                    await response.aclose()
                return

            location = response.headers.get("location")
            if location is None or redirect_count >= _MAX_REDIRECTS:
                raise LocalAIError("Model download redirect rejected")
            try:
                current = response.url.join(location)
            except (TypeError, ValueError) as exc:
                raise LocalAIError("Model download redirect rejected") from exc
        finally:
            if response is not None and response.status_code in _REDIRECT_STATUSES:
                await response.aclose()

    raise LocalAIError("Model download redirect rejected")


def _validate_resume_response(
    response: httpx.Response,
    *,
    start: int,
    expected_size: int,
) -> None:
    """Require an exact, bounded byte-range response before appending a partial."""

    content_range = response.headers.get("content-range")
    if response.status_code != 206 or content_range is None:
        raise LocalValidationError("Model download resume range is invalid")
    try:
        unit, values = content_range.split(" ", maxsplit=1)
        byte_range, raw_total = values.split("/", maxsplit=1)
        raw_start, raw_end = byte_range.split("-", maxsplit=1)
        range_start, range_end, total = int(raw_start), int(raw_end), int(raw_total)
    except (TypeError, ValueError) as exc:
        raise LocalValidationError("Model download resume range is invalid") from exc
    if (
        unit != "bytes"
        or range_start != start
        or range_end != expected_size - 1
        or total != expected_size
        or range_end < range_start
    ):
        raise LocalValidationError("Model download resume range is invalid")

    content_length = _declared_content_length(response)
    if content_length is not None and content_length != expected_size - start:
        raise LocalValidationError("Model download resume range is invalid")


async def download_manifest_to_stage(
    manifest: LocalAIManifest,
    store: ArtifactStore,
    progress_callback: ProgressCallback | None,
) -> Path:
    """Download and hash-verify a manifest without activating it."""

    staging: Path | None = None
    try:
        store.validate_manifest(manifest)
        declared_total = sum(
            file.size for artifact in manifest.artifacts for file in artifact.files
        )
        if declared_total > MAX_MANIFEST_PACK_BYTES:
            raise LocalValidationError("Model download aggregate byte limit exceeded")

        staging = store.stage(manifest.pack_revision)
        operation_id = store.operation_id(staging)
        observed_total = 0

        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(60.0, read=300.0),
        ) as client:
            for artifact in manifest.artifacts:
                role_total = sum(file.size for file in artifact.files)
                role_done = 0
                for file in artifact.files:
                    if file.size > MAX_MANIFEST_FILE_BYTES:
                        raise LocalValidationError(
                            "Model download file byte limit exceeded"
                        )
                    destination = staging / artifact.role.value / file.path
                    _make_owner_only_directory(destination.parent)
                    partial = destination.with_name(f"{destination.name}.partial")
                    digest = hashlib.sha256()
                    file_done = 0
                    url = _download_url(
                        artifact.repository,
                        artifact.revision,
                        file.path,
                    )

                    try:
                        for attempt in range(_MAX_STREAM_RESUME_RETRIES + 1):
                            resuming = attempt > 0 and partial.exists()
                            headers = (
                                {"Range": f"bytes={file_done}-"} if resuming else None
                            )
                            try:
                                async with _validated_stream(
                                    client, url, headers=headers
                                ) as response:
                                    if resuming:
                                        _validate_resume_response(
                                            response,
                                            start=file_done,
                                            expected_size=file.size,
                                        )
                                    else:
                                        response.raise_for_status()
                                        content_length = _declared_content_length(
                                            response
                                        )
                                        if (
                                            content_length is not None
                                            and content_length > file.size
                                        ):
                                            raise LocalValidationError(
                                                "Model download file byte limit exceeded"
                                            )
                                    stream = (
                                        _open_partial_append(partial, file_done)
                                        if resuming
                                        else _open_partial(partial)
                                    )
                                    with stream:
                                        # Avoid a fixed coalescing size: a buffered short
                                        # prefix must remain durable after a stream failure.
                                        async for chunk in response.aiter_bytes():
                                            file_done += len(chunk)
                                            observed_total += len(chunk)
                                            if (
                                                file_done > file.size
                                                or observed_total > declared_total
                                                or observed_total
                                                > MAX_MANIFEST_PACK_BYTES
                                            ):
                                                raise LocalValidationError(
                                                    "Model download byte limit exceeded"
                                                )
                                            stream.write(chunk)
                                            digest.update(chunk)
                                            await _publish_progress(
                                                store=store,
                                                operation_id=operation_id,
                                                progress=DownloadProgress(
                                                    role=artifact.role.value,
                                                    bytes_done=role_done + file_done,
                                                    bytes_total=role_total,
                                                ),
                                                callback=progress_callback,
                                            )
                                        stream.flush()
                                        os.fsync(stream.fileno())
                                break
                            except httpx.TransportError as exc:
                                if attempt >= _MAX_STREAM_RESUME_RETRIES:
                                    raise LocalAIError(
                                        "Model artifact download failed"
                                    ) from exc

                        if file_done != file.size:
                            raise LocalValidationError(
                                "Model artifact size does not match manifest"
                            )
                        if digest.hexdigest() != file.sha256:
                            raise LocalValidationError(
                                "Model artifact SHA-256 does not match manifest"
                            )
                        os.replace(partial, destination)
                        _fsync_directory(destination.parent)
                        role_done += file_done
                    except Exception:
                        try:
                            partial.unlink(missing_ok=True)
                        except OSError:
                            pass
                        raise

        if observed_total != declared_total:
            raise LocalValidationError(
                "Model download aggregate size does not match manifest"
            )
        store.verify(staging, manifest)
        result = staging
        staging = None
        return result
    except BaseException as exc:
        if staging is not None:
            try:
                store.discard_staging(staging)
            except LocalAIError:
                pass
        if isinstance(exc, (asyncio.CancelledError, LocalAIError)):
            raise
        if not isinstance(exc, Exception):
            raise
        raise LocalAIError("Model artifact download failed") from exc
