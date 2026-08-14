"""Portable, fail-closed identity tests for the strict-local worker bundle."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tracemalloc
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest

from app.services.local_ai import runtime_identity
from app.services.local_ai.errors import LocalWorkerError
from app.services.local_ai.runtime_identity import (
    WORKER_IDENTITY_SCHEME,
    WorkerRuntimeIdentity,
    require_manifest_runtime_identity,
    resolve_worker_runtime_identity,
)


@dataclass(frozen=True)
class WorkerProject:
    project_dir: Path
    command: tuple[str, ...]
    effective_package: Path
    source_package: Path


_CONSOLE_SCRIPT_BODY = """# -*- coding: utf-8 -*-
import sys
from local_ai_mlx_worker.__main__ import main
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
"""


def _write_worker_package(package: Path) -> None:
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (package / "__main__.py").write_text(
        "def main() -> int:\n    return 0\n",
        encoding="utf-8",
    )
    (package / "nuextract3.py").write_text(
        'MODEL_ROLE = "extraction"\n',
        encoding="utf-8",
    )


def _write_launcher(launcher: Path, interpreter: Path) -> None:
    launcher.write_text(
        f"#!{interpreter}\n{_CONSOLE_SCRIPT_BODY}",
        encoding="utf-8",
    )
    launcher.chmod(0o700)


def _write_uv_macos_launcher(launcher: Path, interpreter: Path) -> None:
    launcher.write_text(
        "#!/bin/sh\n"
        f"'''exec' '{interpreter}' \"$0\" \"$@\"\n"
        "' '''\n"
        f"{_CONSOLE_SCRIPT_BODY}",
        encoding="utf-8",
    )
    launcher.chmod(0o700)


def build_worker_project(root: Path, *, editable: bool = True) -> WorkerProject:
    project = root
    launcher = project / ".venv/bin/local-ai-mlx-worker"
    launcher.parent.mkdir(parents=True)
    python_home = project / "fixture-python-home/bin"
    python_home.mkdir(parents=True)
    base_interpreter = python_home / "python3.11"
    base_interpreter.write_bytes(b"fixture base interpreter\n")
    base_interpreter.chmod(0o700)
    interpreter = launcher.parent / "python"
    interpreter.symlink_to(base_interpreter)
    _write_launcher(launcher, interpreter)
    (project / ".venv/pyvenv.cfg").write_text(
        f"""home = {python_home}
implementation = CPython
uv = 0.10.2
version_info = 3.11.9
include-system-site-packages = false
prompt = local-ai-mlx-worker
""",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        """[project]
name = "local-ai-mlx-worker"
version = "0.1.0"

[project.scripts]
local-ai-mlx-worker = "local_ai_mlx_worker.__main__:main"
""",
        encoding="utf-8",
    )
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    source_package = project / "src/local_ai_mlx_worker"
    _write_worker_package(source_package)
    site_packages = project / ".venv/lib/python3.11/site-packages"
    site_packages.mkdir(parents=True)
    dist_info = site_packages / "local_ai_mlx_worker-0.1.0.dist-info"
    dist_info.mkdir()
    if editable:
        (site_packages / "_local_ai_mlx_worker.pth").write_text(
            f"{project / 'src'}\n",
            encoding="utf-8",
        )
        (dist_info / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": project.as_uri(),
                    "dir_info": {"editable": True},
                }
            ),
            encoding="utf-8",
        )
        effective_package = source_package
    else:
        effective_package = site_packages / "local_ai_mlx_worker"
        _write_worker_package(effective_package)
        (dist_info / "direct_url.json").write_text(
            json.dumps({"url": project.as_uri(), "dir_info": {}}),
            encoding="utf-8",
        )
    return WorkerProject(
        project_dir=project,
        command=(str(launcher),),
        effective_package=effective_package,
        source_package=source_package,
    )


@pytest.mark.parametrize("editable", [True, False])
def test_worker_bundle_identity_is_stable_and_path_independent(
    tmp_path: Path,
    editable: bool,
) -> None:
    left = build_worker_project(tmp_path / "left", editable=editable)
    right = build_worker_project(tmp_path / "right", editable=editable)

    left_identity = resolve_worker_runtime_identity(left.command, left.project_dir)
    right_identity = resolve_worker_runtime_identity(right.command, right.project_dir)

    assert left_identity == right_identity
    assert left_identity.scheme == WORKER_IDENTITY_SCHEME
    assert re.fullmatch(r"[0-9a-f]{64}", left_identity.bundle_sha256)


def test_worker_bundle_identity_matches_across_editable_and_installed_layouts(
    tmp_path: Path,
) -> None:
    editable = build_worker_project(tmp_path / "editable", editable=True)
    installed = build_worker_project(tmp_path / "installed", editable=False)

    assert resolve_worker_runtime_identity(
        editable.command, editable.project_dir
    ) == resolve_worker_runtime_identity(installed.command, installed.project_dir)


def test_worker_bundle_identity_accepts_exact_uv_macos_polyglot_launcher(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    _write_uv_macos_launcher(
        Path(project.command[0]),
        project.project_dir / ".venv/bin/python",
    )

    identity = resolve_worker_runtime_identity(project.command, project.project_dir)

    assert identity.scheme == WORKER_IDENTITY_SCHEME


def test_worker_bundle_identity_accepts_direct_python_shebang_launcher(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")

    identity = resolve_worker_runtime_identity(project.command, project.project_dir)

    assert identity.scheme == WORKER_IDENTITY_SCHEME


@pytest.mark.parametrize("mutation", ["interpreter", "body", "extra"])
def test_worker_bundle_identity_rejects_changed_uv_macos_polyglot_launcher(
    tmp_path: Path,
    mutation: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    interpreter = project.project_dir / ".venv/bin/python"
    if mutation == "interpreter":
        other = build_worker_project(tmp_path / "other")
        interpreter = other.project_dir / ".venv/bin/python"
    launcher = Path(project.command[0])
    _write_uv_macos_launcher(launcher, interpreter)
    if mutation == "body":
        launcher.write_text(
            launcher.read_text(encoding="utf-8").replace(
                "from local_ai_mlx_worker.__main__ import main",
                "from local_ai_mlx_worker.__main__ import other",
            ),
            encoding="utf-8",
        )
    elif mutation == "extra":
        launcher.write_bytes(launcher.read_bytes() + b"# unexpected\n")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize("relative_path", ["nuextract3.py", "pyproject.toml", "uv.lock"])
def test_worker_bundle_identity_changes_with_runtime_input(
    tmp_path: Path,
    relative_path: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    before = resolve_worker_runtime_identity(project.command, project.project_dir)
    target = (
        project.effective_package / relative_path
        if relative_path.endswith(".py")
        else project.project_dir / relative_path
    )
    target.write_bytes(target.read_bytes() + b"\n# changed\n")

    after = resolve_worker_runtime_identity(project.command, project.project_dir)

    assert after.bundle_sha256 != before.bundle_sha256


def test_installed_identity_uses_installed_package_not_source_tree(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker", editable=False)
    before = resolve_worker_runtime_identity(project.command, project.project_dir)

    source_file = project.source_package / "nuextract3.py"
    source_file.write_bytes(source_file.read_bytes() + b"\n# source-only change\n")
    after_source_change = resolve_worker_runtime_identity(
        project.command, project.project_dir
    )
    installed_file = project.effective_package / "nuextract3.py"
    installed_file.write_bytes(installed_file.read_bytes() + b"\n# installed change\n")
    after_installed_change = resolve_worker_runtime_identity(
        project.command, project.project_dir
    )

    assert after_source_change == before
    assert after_installed_change.bundle_sha256 != before.bundle_sha256


def test_worker_bundle_identity_ignores_python_bytecode_cache(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker")
    before = resolve_worker_runtime_identity(project.command, project.project_dir)
    cache = project.effective_package / "__pycache__"
    cache.mkdir()
    (cache / "nuextract3.cpython-311.pyc").write_bytes(b"normal bytecode cache")

    after = resolve_worker_runtime_identity(project.command, project.project_dir)

    assert after == before


def test_worker_bundle_identity_rejects_launcher_bound_to_other_environment(
    tmp_path: Path,
) -> None:
    first = build_worker_project(tmp_path / "first")
    second = build_worker_project(tmp_path / "second")
    (second.effective_package / "nuextract3.py").write_text(
        'MODEL_ROLE = "different-second-environment"\n',
        encoding="utf-8",
    )
    _write_launcher(
        Path(first.command[0]),
        second.project_dir / ".venv/bin/python",
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(first.command, first.project_dir)


def test_worker_bundle_identity_rejects_earlier_pth_package_shadow(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    shadow_root = tmp_path / "shadow"
    shadow_package = shadow_root / "local_ai_mlx_worker"
    _write_worker_package(shadow_package)
    (shadow_package / "nuextract3.py").write_text(
        'MODEL_ROLE = "shadow"\n',
        encoding="utf-8",
    )
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    (site_packages / "00-shadow.pth").write_text(
        f"{shadow_root}\n",
        encoding="utf-8",
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_installed_package_shadowing_editable_source(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    installed_shadow = site_packages / "local_ai_mlx_worker"
    _write_worker_package(installed_shadow)
    (installed_shadow / "nuextract3.py").write_text(
        'MODEL_ROLE = "installed-shadow"\n',
        encoding="utf-8",
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize("shadow_kind", ["package", "module"])
def test_worker_bundle_identity_rejects_launcher_directory_import_shadow(
    tmp_path: Path,
    shadow_kind: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    launcher_bin = project.project_dir / ".venv/bin"
    if shadow_kind == "package":
        _write_worker_package(launcher_bin / "local_ai_mlx_worker")
    else:
        (launcher_bin / "local_ai_mlx_worker.py").write_text(
            'MODEL_ROLE = "launcher-shadow"\n',
            encoding="utf-8",
        )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_executable_import_tab_pth(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    (site_packages / "00-shadow.pth").write_text(
        "import\tshadow_bootstrap\n",
        encoding="utf-8",
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_system_site_packages(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    config = project.project_dir / ".venv/pyvenv.cfg"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "include-system-site-packages = false",
            "include-system-site-packages = true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_interpreter_outside_declared_python_home(
    tmp_path: Path,
) -> None:
    first = build_worker_project(tmp_path / "first")
    second = build_worker_project(tmp_path / "second")
    interpreter = first.project_dir / ".venv/bin/python"
    interpreter.unlink()
    interpreter.symlink_to(second.project_dir / "fixture-python-home/bin/python3.11")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(first.command, first.project_dir)


@pytest.mark.parametrize("python_version", ["3.9.19", "3.13.1"])
def test_worker_bundle_identity_rejects_unsupported_python_series(
    tmp_path: Path,
    python_version: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    python_series = ".".join(python_version.split(".")[:2])
    base_interpreter = (
        project.project_dir / f"fixture-python-home/bin/python{python_series}"
    )
    base_interpreter.write_bytes(b"fixture base interpreter\n")
    base_interpreter.chmod(0o700)
    interpreter = project.project_dir / ".venv/bin/python"
    interpreter.unlink()
    interpreter.symlink_to(base_interpreter)
    python_directory = project.project_dir / ".venv/lib/python3.11"
    python_directory.rename(python_directory.with_name(f"python{python_series}"))
    config = project.project_dir / ".venv/pyvenv.cfg"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "version_info = 3.11.9",
            f"version_info = {python_version}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_ignores_python_patch_and_uv_builder_drift(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    before = resolve_worker_runtime_identity(project.command, project.project_dir)
    config = project.project_dir / ".venv/pyvenv.cfg"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "version_info = 3.11.9",
            "version_info = 3.11.10",
        ).replace("uv = 0.10.2", "uv = 0.11.0"),
        encoding="utf-8",
    )

    after = resolve_worker_runtime_identity(project.command, project.project_dir)

    assert after == before


@pytest.mark.parametrize("bootstrap_members", [("pth",), ("py",), ("pth", "py")])
def test_worker_bundle_identity_rejects_virtualenv_bootstrap(
    tmp_path: Path,
    bootstrap_members: tuple[str, ...],
) -> None:
    project = build_worker_project(tmp_path / "worker")
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    if "pth" in bootstrap_members:
        (site_packages / "_virtualenv.pth").write_text(
            "import _virtualenv\n",
            encoding="utf-8",
        )
    if "py" in bootstrap_members:
        (site_packages / "_virtualenv.py").write_text(
            "BOOTSTRAP = True\n",
            encoding="utf-8",
        )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_incomplete_virtualenv_bootstrap(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    (site_packages / "_virtualenv.pth").write_text(
        "import _virtualenv\n",
        encoding="utf-8",
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize(
    "shadow_kind",
    [
        "launcher-package",
        "launcher-module",
        "site-package",
        "site-extension",
    ],
)
def test_worker_bundle_identity_rejects_virtualenv_bootstrap_import_shadow(
    tmp_path: Path,
    shadow_kind: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    launcher_bin = project.project_dir / ".venv/bin"
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    shadow_root = launcher_bin if shadow_kind.startswith("launcher") else site_packages
    if shadow_kind.endswith("package"):
        package = shadow_root / "_virtualenv"
        package.mkdir()
        (package / "__init__.py").write_text("SHADOW = True\n", encoding="utf-8")
    elif shadow_kind == "launcher-module":
        (shadow_root / "_virtualenv.py").write_text(
            "SHADOW = True\n",
            encoding="utf-8",
        )
    else:
        (shadow_root / "_virtualenv.so").write_bytes(b"not an extension")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize("editable", [True, False])
@pytest.mark.parametrize("pth_kind", ["executable", "path", "zip"])
def test_worker_bundle_identity_rejects_unattested_pth_inventory(
    tmp_path: Path,
    editable: bool,
    pth_kind: str,
) -> None:
    project = build_worker_project(tmp_path / "worker", editable=editable)
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    extra_path = tmp_path / "extra-imports"
    extra_path.mkdir()
    zip_path = tmp_path / "extra-imports.zip"
    zip_path.write_bytes(b"not a trusted zip")
    content = {
        "executable": "import unexpected_bootstrap\n",
        "path": f"{extra_path}\n",
        "zip": f"{zip_path}\n",
    }[pth_kind]
    (site_packages / "99-unattested.pth").write_text(content, encoding="utf-8")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize("customization", ["sitecustomize", "usercustomize"])
@pytest.mark.parametrize("location", ["launcher", "site-packages"])
@pytest.mark.parametrize("shadow_kind", ["package", "module", "pyc", "extension"])
def test_worker_bundle_identity_rejects_customization_import_surface(
    tmp_path: Path,
    customization: str,
    location: str,
    shadow_kind: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    root = (
        project.project_dir / ".venv/bin"
        if location == "launcher"
        else project.project_dir / ".venv/lib/python3.11/site-packages"
    )
    if shadow_kind == "package":
        package = root / customization
        package.mkdir()
        (package / "__init__.py").write_text("SHADOW = True\n", encoding="utf-8")
    elif shadow_kind == "module":
        (root / f"{customization}.py").write_text(
            "SHADOW = True\n",
            encoding="utf-8",
        )
    elif shadow_kind == "pyc":
        (root / f"{customization}.pyc").write_bytes(b"not bytecode")
    else:
        (root / f"{customization}{EXTENSION_SUFFIXES[0]}").write_bytes(
            b"not an extension"
        )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize(
    "relative_shadow",
    [
        f"nuextract3{EXTENSION_SUFFIXES[0]}",
        "nuextract3.pyc",
    ],
)
def test_worker_bundle_identity_rejects_importable_non_source_package_member(
    tmp_path: Path,
    relative_shadow: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    (project.effective_package / relative_shadow).write_bytes(b"unattested import")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_hashes_mixed_case_python_source(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker")
    before = resolve_worker_runtime_identity(project.command, project.project_dir)
    (project.effective_package / "mixed_case.PY").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    after = resolve_worker_runtime_identity(project.command, project.project_dir)

    assert after.bundle_sha256 != before.bundle_sha256


@pytest.mark.parametrize(
    "candidate_name",
    ["nuextract3.PYC", f"nuextract3{EXTENSION_SUFFIXES[0].upper()}"],
)
def test_worker_bundle_identity_rejects_mixed_case_importable_package_member(
    tmp_path: Path,
    candidate_name: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    (project.effective_package / candidate_name).write_bytes(b"unattested import")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize(
    "startup_name",
    ["_virtualenv", "sitecustomize", "usercustomize"],
)
@pytest.mark.parametrize("candidate_kind", ["package", "module", "pyc", "extension"])
def test_worker_bundle_identity_rejects_editable_source_startup_import_surface(
    tmp_path: Path,
    startup_name: str,
    candidate_kind: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    source_root = project.project_dir / "src"
    if candidate_kind == "package":
        package = source_root / startup_name
        package.mkdir()
        (package / "__init__.py").write_text("SHADOW = True\n", encoding="utf-8")
    elif candidate_kind == "module":
        (source_root / f"{startup_name}.py").write_text(
            "SHADOW = True\n",
            encoding="utf-8",
        )
    elif candidate_kind == "pyc":
        (source_root / f"{startup_name}.pyc").write_bytes(b"not bytecode")
    else:
        (source_root / f"{startup_name}{EXTENSION_SUFFIXES[0]}").write_bytes(
            b"not an extension"
        )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize(
    "candidate_suffix",
    [".py", ".pyc", EXTENSION_SUFFIXES[0]],
)
def test_worker_bundle_identity_rejects_site_package_worker_module_alternative(
    tmp_path: Path,
    candidate_suffix: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    (site_packages / f"local_ai_mlx_worker{candidate_suffix}").write_bytes(
        b"unattested worker alternative"
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize(
    "candidate_name",
    ["UserCustomize.py", "Local_AI_MLX_Worker.py"],
)
def test_worker_bundle_identity_rejects_case_insensitive_import_shadow(
    tmp_path: Path,
    candidate_name: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    (site_packages / candidate_name).write_text("SHADOW = True\n", encoding="utf-8")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize("editable", [True, False])
def test_worker_console_script_imports_attested_package_under_sanitized_environment(
    tmp_path: Path,
    editable: bool,
) -> None:
    project = build_worker_project(tmp_path / "worker", editable=editable)
    host_executable = Path(sys.executable).resolve(strict=True)
    host_home = host_executable.parent
    version_info = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    fixture_interpreter = project.project_dir / ".venv/bin/python"
    fixture_interpreter.unlink()
    fixture_interpreter.symlink_to(host_executable)
    (project.project_dir / ".venv/pyvenv.cfg").write_text(
        f"""home = {host_home}
implementation = CPython
uv = 0.10.2
version_info = {version_info}
include-system-site-packages = false
prompt = local-ai-mlx-worker
""",
        encoding="utf-8",
    )
    (project.effective_package / "__main__.py").write_text(
        "def main() -> int:\n"
        "    print(__file__)\n"
        "    return 0\n",
        encoding="utf-8",
    )
    expected_identity = resolve_worker_runtime_identity(
        project.command,
        project.project_dir,
    )
    worker_home = tmp_path / "worker-home"
    worker_home.mkdir(mode=0o700)
    pycache = tmp_path / "worker-pycache"
    pycache.mkdir(mode=0o700)
    environment = {
        "PATH": os.defpath,
        "HOME": str(worker_home),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONPYCACHEPREFIX": str(pycache),
    }

    result = subprocess.run(
        project.command,
        cwd=worker_home,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert expected_identity.scheme == WORKER_IDENTITY_SCHEME
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (
        project.effective_package / "__main__.py"
    ).resolve()
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert not (project.effective_package / "__pycache__").exists()


@pytest.mark.parametrize(
    "relative_ancestor",
    [
        ".venv",
        ".venv/lib",
        ".venv/lib/python3.11",
        ".venv/lib/python3.11/site-packages",
        ".venv/lib/python3.11/site-packages/local_ai_mlx_worker-0.1.0.dist-info",
    ],
)
def test_worker_bundle_identity_rejects_symlink_runtime_ancestor_without_path_leak(
    tmp_path: Path,
    relative_ancestor: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    ancestor = project.project_dir / relative_ancestor
    moved = tmp_path / f"{ancestor.name}-real-sensitive-canary"
    ancestor.rename(moved)
    ancestor.symlink_to(moved, target_is_directory=True)

    with pytest.raises(LocalWorkerError) as exc_info:
        resolve_worker_runtime_identity(project.command, project.project_dir)

    assert "sensitive-canary" not in str(exc_info.value)
    assert str(project.project_dir) not in str(exc_info.value)


def test_worker_bundle_identity_hashing_has_bounded_peak_memory(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker")
    source_bytes = b"#" * (1024 * 1024)
    for index in range(59):
        (project.effective_package / f"source_{index:02d}.py").write_bytes(source_bytes)

    tracemalloc.start()
    try:
        identity = resolve_worker_runtime_identity(project.command, project.project_dir)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert re.fullmatch(r"[0-9a-f]{64}", identity.bundle_sha256)
    assert peak_bytes < 16 * 1024 * 1024


def test_worker_bundle_identity_rejects_symlink_launcher(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker")
    launcher = Path(project.command[0])
    real_launcher = launcher.with_name("real-launcher")
    launcher.rename(real_launcher)
    launcher.symlink_to(real_launcher)

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


@pytest.mark.parametrize("bad_name", ["other-worker", "local-ai-mlx-worker-copy"])
def test_worker_bundle_identity_rejects_different_launcher_basename(
    tmp_path: Path,
    bad_name: str,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    bad_launcher = project.project_dir / ".venv/bin" / bad_name
    bad_launcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity((str(bad_launcher),), project.project_dir)


def test_worker_bundle_identity_rejects_launcher_outside_project_venv(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    external = tmp_path / "local-ai-mlx-worker"
    external.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity((str(external),), project.project_dir)


@pytest.mark.parametrize(
    "command",
    [
        "local-ai-mlx-worker",
        ("local-ai-mlx-worker",),
        ("/tmp/local-ai-mlx-worker", "--unexpected"),
    ],
)
def test_worker_bundle_identity_rejects_shell_path_and_arguments(
    tmp_path: Path,
    command: str | tuple[str, ...],
) -> None:
    project = build_worker_project(tmp_path / "worker")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(command, project.project_dir)


def test_worker_bundle_identity_rejects_inexact_entry_point(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker")
    pyproject = project.project_dir / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            "local_ai_mlx_worker.__main__:main",
            "local_ai_mlx_worker.__main__:other",
        ),
        encoding="utf-8",
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_missing_lock_file(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker")
    (project.project_dir / "uv.lock").unlink()

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_symlink_python_file(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker")
    target = project.effective_package / "nuextract3.py"
    real_target = target.with_suffix(".real")
    target.rename(real_target)
    target.symlink_to(real_target)

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_nonregular_python_file(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker")
    target = project.effective_package / "nuextract3.py"
    target.unlink()
    os.mkfifo(target)

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_symlink_package_directory(tmp_path: Path) -> None:
    project = build_worker_project(tmp_path / "worker", editable=False)
    package = project.effective_package
    real_package = package.with_name("real_package")
    package.rename(real_package)
    package.symlink_to(real_package, target_is_directory=True)

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_editable_metadata_for_other_checkout(
    tmp_path: Path,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    site_packages = project.project_dir / ".venv/lib/python3.11/site-packages"
    pth = site_packages / "_local_ai_mlx_worker.pth"
    pth.write_text(f"{tmp_path / 'other/src'}\n", encoding="utf-8")

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_duplicate_selected_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    selected = project.effective_package / "nuextract3.py"
    monkeypatch.setattr(
        runtime_identity,
        "_regular_python_tree",
        lambda _: [selected, selected],
    )

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_worker_bundle_identity_rejects_noncanonical_selected_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = build_worker_project(tmp_path / "worker")
    selected = project.effective_package / "nested/../nuextract3.py"
    monkeypatch.setattr(runtime_identity, "_regular_python_tree", lambda _: [selected])

    with pytest.raises(LocalWorkerError, match="identity is unavailable"):
        resolve_worker_runtime_identity(project.command, project.project_dir)


def test_manifest_runtime_identity_must_match_observed_bundle() -> None:
    observed = WorkerRuntimeIdentity(
        scheme=WORKER_IDENTITY_SCHEME,
        bundle_sha256="a" * 64,
    )
    manifest = type(
        "Manifest",
        (),
        {
            "runtime": {
                "name": "mlx-vlm",
                "version": "0.5.0",
                "worker_identity_scheme": WORKER_IDENTITY_SCHEME,
                "worker_bundle_sha256": "a" * 64,
            }
        },
    )()

    require_manifest_runtime_identity(manifest, observed)
    manifest.runtime["worker_bundle_sha256"] = "b" * 64

    with pytest.raises(LocalWorkerError, match="identity does not match"):
        require_manifest_runtime_identity(manifest, observed)
