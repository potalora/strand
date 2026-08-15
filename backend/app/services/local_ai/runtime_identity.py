"""Resolve a portable, bounded identity for the effective local-AI worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import tomllib
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from app.services.local_ai.errors import LocalWorkerError

WORKER_IDENTITY_SCHEME = "local-ai-worker-bundle.v1"
WORKER_ENTRY_POINT = "local-ai-mlx-worker=local_ai_mlx_worker.__main__:main"
_MAX_IDENTITY_FILES = 64
_MAX_IDENTITY_FILE_BYTES = 8 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_ALLOWED_WORKER_ARGUMENTS: tuple[str, ...] = ()
_ERROR = "Local worker runtime identity is unavailable."
_PYVENV_KEYS = frozenset(
    {
        "home",
        "implementation",
        "uv",
        "version_info",
        "include-system-site-packages",
        "prompt",
    }
)
_PYTHON_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?$")
_SUPPORTED_PYTHON_SERIES = frozenset({"3.11", "3.12"})
_EXTENSION_SUFFIXES_CASEFOLDED = tuple(
    suffix.casefold() for suffix in EXTENSION_SUFFIXES
)
_EXPECTED_LAUNCHER_BODY = b"""# -*- coding: utf-8 -*-
import sys
from local_ai_mlx_worker.__main__ import main
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
"""
_STABLE_FILE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


@dataclass(frozen=True)
class WorkerRuntimeIdentity:
    """Portable identity of the code and locks used by the worker process."""

    scheme: str
    bundle_sha256: str


def _unavailable() -> LocalWorkerError:
    return LocalWorkerError(_ERROR)


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def normalize_worker_runtime_binding(
    command: str | Sequence[str],
    worker_project_dir: str | Path,
    *,
    cwd: str | Path | None = None,
) -> tuple[tuple[str, ...], Path]:
    """Normalize the exact command/project pair used for attestation and spawn."""

    base = Path(cwd) if cwd is not None else Path.cwd()
    base = _absolute_without_symlink_resolution(base.expanduser())
    if isinstance(command, str):
        try:
            parts = tuple(shlex.split(command))
        except ValueError:
            raise _unavailable() from None
    else:
        try:
            parts = tuple(os.fspath(item) for item in command)
        except TypeError:
            parts = ()
    if not parts or any(not part for part in parts):
        raise _unavailable()
    executable = Path(parts[0]).expanduser()
    if not executable.is_absolute():
        if executable.parent != Path("."):
            executable = base / executable
        else:
            located = shutil.which(parts[0], path=os.defpath)
            if located is None:
                raise _unavailable()
            executable = Path(located)
    executable = _absolute_without_symlink_resolution(executable)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise _unavailable()

    project = Path(worker_project_dir).expanduser()
    if not project.is_absolute():
        project = base / project
    return (str(executable), *parts[1:]), _absolute_without_symlink_resolution(project)


def _regular_real_directory(path: Path) -> Path:
    candidate = _absolute_without_symlink_resolution(path)
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise _unavailable() from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise _unavailable()
    if not metadata.st_mode & stat.S_IRUSR or not metadata.st_mode & stat.S_IXUSR:
        raise _unavailable()
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise _unavailable() from exc


def _resolve_worker_executable(command: str | Sequence[str]) -> Path:
    if isinstance(command, (str, bytes, bytearray)) or not isinstance(
        command, Sequence
    ):
        raise _unavailable()
    if len(command) != 1 + len(_ALLOWED_WORKER_ARGUMENTS):
        raise _unavailable()
    if tuple(command[1:]) != _ALLOWED_WORKER_ARGUMENTS:
        raise _unavailable()
    executable_value = command[0]
    if not isinstance(executable_value, str) or not executable_value:
        raise _unavailable()
    executable = _absolute_without_symlink_resolution(Path(executable_value))
    try:
        metadata = os.lstat(executable)
    except OSError as exc:
        raise _unavailable() from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise _unavailable()
    if not metadata.st_mode & stat.S_IRUSR or not metadata.st_mode & stat.S_IXUSR:
        raise _unavailable()
    return executable


def _require_launcher_binding(executable: Path, expected_bin: Path) -> None:
    launcher_bytes = _secure_file_bytes(executable)
    expected_interpreter = expected_bin / "python"
    direct_launcher = (
        f"#!{expected_interpreter}\n".encode("utf-8") + _EXPECTED_LAUNCHER_BODY
    )
    uv_macos_launcher = (
        f"#!/bin/sh\n'''exec' '{expected_interpreter}' \"$0\" \"$@\"\n' '''\n"
    ).encode("utf-8") + _EXPECTED_LAUNCHER_BODY
    if launcher_bytes not in {direct_launcher, uv_macos_launcher}:
        raise _unavailable()
    try:
        interpreter = os.stat(expected_interpreter)
    except OSError as exc:
        raise _unavailable() from exc
    if (
        not stat.S_ISREG(interpreter.st_mode)
        or not interpreter.st_mode & stat.S_IRUSR
        or not interpreter.st_mode & stat.S_IXUSR
    ):
        raise _unavailable()


def _require_no_import_candidates(
    directory: Path,
    stems: tuple[str, ...],
    *,
    allowed_names: frozenset[str] = frozenset(),
) -> None:
    try:
        members = os.scandir(directory)
    except OSError as exc:
        raise _unavailable() from exc
    try:
        for member in members:
            candidate = member.name.casefold()
            if candidate in allowed_names:
                continue
            if any(
                candidate == stem or candidate.startswith(f"{stem}.") for stem in stems
            ):
                raise _unavailable()
    finally:
        members.close()


def _require_no_launcher_import_shadow(expected_bin: Path) -> None:
    _require_no_import_candidates(
        expected_bin,
        (
            "local_ai_mlx_worker",
            "_virtualenv",
            "sitecustomize",
            "usercustomize",
        ),
    )


def _secure_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = os.lstat(path)
        if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
            raise _unavailable()
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _unavailable() from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not before.st_mode & stat.S_IRUSR:
            raise _unavailable()
        if before.st_size < 0 or before.st_size > _MAX_IDENTITY_FILE_BYTES:
            raise _unavailable()
        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > _MAX_IDENTITY_FILE_BYTES:
                raise _unavailable()
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise _unavailable() from exc
    finally:
        os.close(descriptor)
    if bytes_read != before.st_size or any(
        getattr(before, field) != getattr(after, field)
        for field in _STABLE_FILE_STAT_FIELDS
    ):
        raise _unavailable()
    return b"".join(chunks)


def _secure_file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = os.lstat(path)
        if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
            raise _unavailable()
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _unavailable() from exc
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not before.st_mode & stat.S_IRUSR:
            raise _unavailable()
        if before.st_size < 0 or before.st_size > _MAX_IDENTITY_FILE_BYTES:
            raise _unavailable()
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > _MAX_IDENTITY_FILE_BYTES:
                raise _unavailable()
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise _unavailable() from exc
    finally:
        os.close(descriptor)
    if bytes_read != before.st_size or any(
        getattr(before, field) != getattr(after, field)
        for field in _STABLE_FILE_STAT_FIELDS
    ):
        raise _unavailable()
    return digest.hexdigest()


def _require_exact_console_script(pyproject_path: Path, expected: str) -> None:
    try:
        raw = tomllib.loads(_secure_file_bytes(pyproject_path).decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise _unavailable() from exc
    name, target = expected.split("=", 1)
    project = raw.get("project")
    scripts = project.get("scripts") if isinstance(project, dict) else None
    if not isinstance(scripts, dict) or scripts.get(name) != target:
        raise _unavailable()


def _validated_python_runtime(
    environment: Path,
    expected_bin: Path,
) -> str:
    try:
        lines = (
            _secure_file_bytes(environment / "pyvenv.cfg").decode("utf-8").splitlines()
        )
    except UnicodeError as exc:
        raise _unavailable() from exc
    config: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition(" = ")
        if not separator or not key or not value or key in config:
            raise _unavailable()
        config[key] = value
    if set(config) != _PYVENV_KEYS:
        raise _unavailable()
    if (
        config["implementation"] != "CPython"
        or config["include-system-site-packages"] != "false"
        or config["prompt"] != "local-ai-mlx-worker"
        or _PYTHON_VERSION_RE.fullmatch(config["version_info"]) is None
        or not config["uv"]
        or len(config["uv"]) > 128
        or any(ord(character) < 32 for character in config["uv"])
    ):
        raise _unavailable()
    home_value = Path(config["home"])
    if not home_value.is_absolute():
        raise _unavailable()
    home = _regular_real_directory(home_value)
    version_parts = config["version_info"].split(".")
    python_series = ".".join(version_parts[:2])
    if python_series not in _SUPPORTED_PYTHON_SERIES:
        raise _unavailable()
    base_interpreter = home / f"python{python_series}"
    try:
        base_metadata = os.lstat(base_interpreter)
        interpreter_metadata = os.lstat(expected_bin / "python")
        resolved_interpreter = (expected_bin / "python").resolve(strict=True)
        resolved_base = base_interpreter.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _unavailable() from exc
    if (
        stat.S_ISLNK(base_metadata.st_mode)
        or not stat.S_ISREG(base_metadata.st_mode)
        or not base_metadata.st_mode & stat.S_IRUSR
        or not base_metadata.st_mode & stat.S_IXUSR
        or not stat.S_ISLNK(interpreter_metadata.st_mode)
        or resolved_interpreter != resolved_base
    ):
        raise _unavailable()
    return python_series


def _site_packages_directory(environment: Path, python_series: str) -> Path:
    lib = _regular_real_directory(environment / "lib")
    python_directory = _regular_real_directory(lib / f"python{python_series}")
    return _regular_real_directory(python_directory / "site-packages")


def _worker_direct_url(site_packages: Path) -> dict[str, Any] | None:
    try:
        candidates = sorted(
            site_packages.glob("local_ai_mlx_worker-*.dist-info/direct_url.json")
        )
    except OSError as exc:
        raise _unavailable() from exc
    if not candidates:
        return None
    if len(candidates) != 1:
        raise _unavailable()
    _regular_real_directory(candidates[0].parent)
    try:
        value = json.loads(_secure_file_bytes(candidates[0]))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise _unavailable() from exc
    if not isinstance(value, dict):
        raise _unavailable()
    return value


def _require_no_virtualenv_bootstrap(site_packages: Path) -> None:
    _require_no_import_candidates(site_packages, ("_virtualenv",))


def _require_no_customization_import_surface(directory: Path) -> None:
    _require_no_import_candidates(directory, ("sitecustomize", "usercustomize"))


def _require_pth_inventory(
    site_packages: Path,
    *,
    editable_pth: Path | None = None,
    editable_content: bytes | None = None,
) -> None:
    try:
        pth_files = sorted(site_packages.glob("*.pth"))
    except OSError as exc:
        raise _unavailable() from exc
    allowed: set[Path] = set()
    if editable_pth is not None:
        if editable_content is None:
            raise _unavailable()
        allowed.add(editable_pth)
    if set(pth_files) != allowed:
        raise _unavailable()
    for pth_file in pth_files:
        contents = _secure_file_bytes(pth_file)
        if pth_file == editable_pth and contents not in {
            editable_content,
            editable_content.removesuffix(b"\n"),
        }:
            raise _unavailable()


def _editable_source_from_metadata(
    project: Path,
    site_packages: Path,
    direct_url: dict[str, Any],
) -> Path | None:
    directory_info = direct_url.get("dir_info")
    if (
        not isinstance(directory_info, dict)
        or directory_info.get("editable") is not True
    ):
        return None
    direct_url_value = direct_url.get("url")
    parsed = urlparse(direct_url_value if isinstance(direct_url_value, str) else "")
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise _unavailable()
    direct_project = _absolute_without_symlink_resolution(Path(unquote(parsed.path)))
    if direct_project != project:
        raise _unavailable()

    try:
        pth_candidates = sorted(
            path
            for path in site_packages.glob("*.pth")
            if "local_ai_mlx_worker" in path.name.lower()
        )
    except OSError as exc:
        raise _unavailable() from exc
    if len(pth_candidates) != 1:
        raise _unavailable()
    expected_source_root = project / "src"
    expected_pth_bytes = f"{expected_source_root}\n".encode("utf-8")
    _require_pth_inventory(
        site_packages,
        editable_pth=pth_candidates[0],
        editable_content=expected_pth_bytes,
    )
    source_root = _absolute_without_symlink_resolution(expected_source_root)
    if source_root != expected_source_root:
        raise _unavailable()
    source_root_directory = _regular_real_directory(expected_source_root)
    _require_no_import_candidates(
        source_root_directory,
        ("_virtualenv", "sitecustomize", "usercustomize"),
    )
    expected_package = _regular_real_directory(
        source_root_directory / "local_ai_mlx_worker"
    )
    _require_no_import_candidates(site_packages, ("local_ai_mlx_worker",))
    return expected_package


def _resolve_effective_worker_package(
    project: Path,
    site_packages: Path,
) -> Path:
    direct_url = _worker_direct_url(site_packages)
    if direct_url is not None:
        editable_package = _editable_source_from_metadata(
            project, site_packages, direct_url
        )
        if editable_package is not None:
            return editable_package
    _require_pth_inventory(site_packages)
    _require_no_import_candidates(
        site_packages,
        ("local_ai_mlx_worker",),
        allowed_names=frozenset({"local_ai_mlx_worker"}),
    )
    return _regular_real_directory(site_packages / "local_ai_mlx_worker")


def _regular_python_tree(package_root: Path) -> list[Path]:
    root = _regular_real_directory(package_root)
    selected: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            members = sorted(os.scandir(directory), key=lambda member: member.name)
        except OSError as exc:
            raise _unavailable() from exc
        for member in members:
            if member.name == "__pycache__":
                continue
            member_path = Path(member.path)
            member_name_casefolded = member.name.casefold()
            try:
                metadata = member.stat(follow_symlinks=False)
            except OSError as exc:
                raise _unavailable() from exc
            if stat.S_ISDIR(metadata.st_mode):
                if member.is_symlink():
                    raise _unavailable()
                pending.append(_regular_real_directory(member_path))
            elif member_name_casefolded.endswith(".py"):
                if member.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                    raise _unavailable()
                selected.append(member_path)
            elif member_name_casefolded.endswith(
                ".pyc"
            ) or member_name_casefolded.endswith(_EXTENSION_SUFFIXES_CASEFOLDED):
                raise _unavailable()
    if not selected:
        raise _unavailable()
    return selected


def _canonical_identity_path(project: Path, path: Path) -> str:
    try:
        relative = path.relative_to(project)
    except ValueError as exc:
        raise _unavailable() from exc
    raw_parts = relative.parts
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise _unavailable()
    if relative.as_posix() != str(relative) or "\\" in relative.as_posix():
        raise _unavailable()
    if raw_parts in {("pyproject.toml",), ("uv.lock",)}:
        canonical = relative.as_posix()
    elif len(raw_parts) >= 3 and raw_parts[:2] == (
        "src",
        "local_ai_mlx_worker",
    ):
        canonical = PurePosixPath("effective_package", *raw_parts[1:]).as_posix()
    elif (
        len(raw_parts) >= 6
        and raw_parts[:2] == (".venv", "lib")
        and raw_parts[2].startswith("python")
        and raw_parts[3:5] == ("site-packages", "local_ai_mlx_worker")
    ):
        canonical = PurePosixPath("effective_package", *raw_parts[4:]).as_posix()
    else:
        raise _unavailable()
    canonical_path = PurePosixPath(canonical)
    if (
        canonical_path.is_absolute()
        or any(part in {"", ".", ".."} for part in canonical_path.parts)
        or canonical_path.as_posix() != canonical
    ):
        raise _unavailable()
    return canonical


def _hash_identity_file(project: Path, path: Path) -> dict[str, str]:
    canonical_path = _canonical_identity_path(project, path)
    return {
        "path": canonical_path,
        "sha256": _secure_file_sha256(path),
    }


def resolve_worker_runtime_identity(
    command: str | Sequence[str],
    worker_project_dir: str | Path,
) -> WorkerRuntimeIdentity:
    """Return the canonical identity of the effective installed worker bundle."""

    project = _regular_real_directory(Path(worker_project_dir))
    executable = _resolve_worker_executable(command)
    environment = _regular_real_directory(project / ".venv")
    expected_bin = _regular_real_directory(environment / "bin")
    python_series = _validated_python_runtime(
        environment,
        expected_bin,
    )
    expected_launcher = expected_bin / "local-ai-mlx-worker"
    if executable != expected_launcher or executable.is_symlink():
        raise _unavailable()
    _require_launcher_binding(executable, expected_bin)
    _require_no_launcher_import_shadow(expected_bin)

    files = [project / "pyproject.toml", project / "uv.lock"]
    _require_exact_console_script(project / "pyproject.toml", WORKER_ENTRY_POINT)
    site_packages = _site_packages_directory(environment, python_series)
    _require_no_virtualenv_bootstrap(site_packages)
    _require_no_customization_import_surface(site_packages)
    package_root = _resolve_effective_worker_package(project, site_packages)
    files.extend(_regular_python_tree(package_root))
    if len(files) > _MAX_IDENTITY_FILES:
        raise _unavailable()
    entries = sorted(
        [_hash_identity_file(project, path) for path in files],
        key=lambda entry: entry["path"],
    )
    canonical_paths = [entry["path"] for entry in entries]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise _unavailable()
    payload = {
        "scheme": WORKER_IDENTITY_SCHEME,
        "entry_point": WORKER_ENTRY_POINT,
        "files": entries,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return WorkerRuntimeIdentity(
        scheme=WORKER_IDENTITY_SCHEME,
        bundle_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def require_manifest_runtime_identity(
    manifest: Any,
    observed: WorkerRuntimeIdentity,
) -> None:
    """Fail closed unless a manifest is bound to the observed worker bundle."""

    runtime = getattr(manifest, "runtime", None)
    if not isinstance(runtime, dict) or (
        runtime.get("worker_identity_scheme") != observed.scheme
        or runtime.get("worker_bundle_sha256") != observed.bundle_sha256
    ):
        raise LocalWorkerError(
            "Local worker runtime identity does not match the manifest."
        )
