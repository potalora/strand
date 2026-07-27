"""Document-free streaming downloader for immutable local-AI manifests."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
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


async def download_manifest(
    manifest: LocalAIManifest,
    store: ArtifactStore,
    progress_callback: ProgressCallback | None,
) -> None:
    """Download, verify, and activate only files named by an immutable manifest."""

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
            follow_redirects=True,
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
                        async with client.stream("GET", url) as response:
                            response.raise_for_status()
                            content_length = _declared_content_length(response)
                            if (
                                content_length is not None
                                and content_length > file.size
                            ):
                                raise LocalValidationError(
                                    "Model download file byte limit exceeded"
                                )
                            with _open_partial(partial) as stream:
                                async for chunk in response.aiter_bytes(
                                    _STREAM_CHUNK_BYTES
                                ):
                                    file_done += len(chunk)
                                    observed_total += len(chunk)
                                    if (
                                        file_done > file.size
                                        or observed_total > declared_total
                                        or observed_total > MAX_MANIFEST_PACK_BYTES
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
        store.activate(staging, manifest)
        staging = None
    except asyncio.CancelledError:
        if staging is not None:
            try:
                store.discard_staging(staging)
            except LocalAIError:
                pass
        raise
    except LocalAIError:
        if staging is not None:
            try:
                store.discard_staging(staging)
            except LocalAIError:
                pass
        raise
    except httpx.HTTPError as exc:
        if staging is not None:
            try:
                store.discard_staging(staging)
            except LocalAIError:
                pass
        raise LocalAIError("Model artifact download failed") from exc
    except Exception as exc:
        if staging is not None:
            try:
                store.discard_staging(staging)
            except LocalAIError:
                pass
        raise LocalAIError("Model artifact download failed") from exc
