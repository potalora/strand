"""Deterministic contract tests for the macOS local-worker setup script."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_setup_repository(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    source_script = Path(__file__).parents[2] / "scripts/setup-local-ai-macos.sh"
    setup_script = scripts / source_script.name
    shutil.copy2(source_script, setup_script)
    (repository / "workers/local_ai/apple_mlx").mkdir(parents=True)

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"
    _write_executable(
        fake_bin / "uname",
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  -s) printf 'Darwin\\n' ;;\n"
        "  -m) printf 'arm64\\n' ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
    )
    _write_executable(
        fake_bin / "sysctl",
        "#!/bin/sh\nprintf '34359738368\\n'\n",
    )
    _write_executable(
        fake_bin / "df",
        "#!/bin/sh\n"
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf 'fixture 99999999 1 99999998 1%% /\\n'\n",
    )
    _write_executable(
        fake_bin / "fixture-python",
        "#!/bin/sh\n"
        "set -eu\n"
        "test -d \"$PYTHONPYCACHEPREFIX\"\n"
        "mode=$(stat -f '%Lp' \"$PYTHONPYCACHEPREFIX\")\n"
        "printf 'python %s\\n' \"$*\" >> \"$FAKE_COMMAND_LOG\"\n"
        "printf 'python-env %s|%s|%s|%s|%s|%s\\n' "
        "\"${PYTHONPATH-unset}\" \"${PYTHONHOME-unset}\" "
        "\"${PYTHONNOUSERSITE-unset}\" \"${PYTHONSAFEPATH-unset}\" "
        "\"$PYTHONPYCACHEPREFIX\" \"$mode\" >> \"$FAKE_COMMAND_LOG\"\n"
        "touch \"$PYTHONPYCACHEPREFIX/imported.pyc\"\n",
    )
    _write_executable(
        fake_bin / "uv",
        "#!/bin/sh\n"
        "set -eu\n"
        "printf 'uv %s\\n' \"$*\" >> \"$FAKE_COMMAND_LOG\"\n"
        "test \"$1\" = sync\n"
        "test \"$2\" = --frozen\n"
        "test \"$3\" = --project\n"
        "worker=$4\n"
        "site=$worker/.venv/lib/python3.11/site-packages\n"
        "case \"${FAKE_SYMLINK_COMPONENT:-}\" in\n"
        "  .venv)\n"
        "    external=$worker/outside-venv\n"
        "    mkdir -p \"$external\"\n"
        "    ln -s \"$external\" \"$worker/.venv\"\n"
        "    ;;\n"
        "  lib)\n"
        "    external=$worker/outside-lib\n"
        "    mkdir -p \"$worker/.venv\" \"$external\"\n"
        "    ln -s \"$external\" \"$worker/.venv/lib\"\n"
        "    ;;\n"
        "  python3.11)\n"
        "    external=$worker/outside-python\n"
        "    mkdir -p \"$worker/.venv/lib\" \"$external\"\n"
        "    ln -s \"$external\" \"$worker/.venv/lib/python3.11\"\n"
        "    ;;\n"
        "  site-packages)\n"
        "    external=$worker/outside-site-packages\n"
        "    mkdir -p \"$worker/.venv/lib/python3.11\" \"$external\"\n"
        "    ln -s \"$external\" \"$site\"\n"
        "    ;;\n"
        "esac\n"
        "mkdir -p \"$worker/.venv/lib/python3.11\" \"$worker/.venv/bin\" \"$site\"\n"
        "if test \"${FAKE_AMBIGUOUS:-0}\" = 1; then\n"
        "  mkdir -p \"$worker/.venv/lib/python3.12/site-packages\"\n"
        "fi\n"
        "ln -s \"$FAKE_FIXTURE_PYTHON\" \"$worker/.venv/bin/python\"\n"
        "launcher=$worker/.venv/bin/local-ai-mlx-worker\n"
        "embedded=$worker/.venv/bin/python\n"
        "entry=main\n"
        "if test \"${FAKE_LAUNCHER_MUTATION:-}\" = interpreter; then\n"
        "  embedded=$FAKE_FIXTURE_PYTHON\n"
        "elif test \"${FAKE_LAUNCHER_MUTATION:-}\" = body; then\n"
        "  entry=other\n"
        "fi\n"
        "{\n"
        "  if test \"${FAKE_LAUNCHER_STYLE:-polyglot}\" = direct; then\n"
        "    printf '#!%s\\n' \"$embedded\"\n"
        "  else\n"
        "    printf '#!/bin/sh\\n'\n"
        "    printf \"'''exec' '%s' \\\"\\$0\\\" \\\"\\$@\\\"\\n\" \"$embedded\"\n"
        "    printf \"' '''\\n\"\n"
        "  fi\n"
        "  printf '# -*- coding: utf-8 -*-\\n'\n"
        "  printf 'import sys\\n'\n"
        "  printf 'from local_ai_mlx_worker.__main__ import %s\\n' \"$entry\"\n"
        "  printf 'if __name__ == \"__main__\":\\n'\n"
        "  printf '    if sys.argv[0].endswith(\"-script.pyw\"):\\n'\n"
        "  printf '        sys.argv[0] = sys.argv[0][:-11]\\n'\n"
        "  printf '    elif sys.argv[0].endswith(\".exe\"):\\n'\n"
        "  printf '        sys.argv[0] = sys.argv[0][:-4]\\n'\n"
        "  printf '    sys.exit(main())\\n'\n"
        "} > \"$launcher\"\n"
        "if test \"${FAKE_LAUNCHER_MUTATION:-}\" = extra; then\n"
        "  printf '# unexpected\\n' >> \"$launcher\"\n"
        "fi\n"
        "chmod 700 \"$launcher\"\n"
        "case \"${FAKE_BOOTSTRAP_STATE:-pair}\" in\n"
        "  absent) ;;\n"
        "  missing-pth) printf 'BOOTSTRAP = True\\n' > \"$site/_virtualenv.py\" ;;\n"
        "  missing-py) printf 'import _virtualenv\\n' > \"$site/_virtualenv.pth\" ;;\n"
        "  symlink-pth)\n"
        "    printf 'outside\\n' > \"$worker/outside-bootstrap.pth\"\n"
        "    ln -s \"$worker/outside-bootstrap.pth\" \"$site/_virtualenv.pth\"\n"
        "    printf 'BOOTSTRAP = True\\n' > \"$site/_virtualenv.py\"\n"
        "    ;;\n"
        "  symlink-py)\n"
        "    printf 'outside\\n' > \"$worker/outside-bootstrap.py\"\n"
        "    ln -s \"$worker/outside-bootstrap.py\" \"$site/_virtualenv.py\"\n"
        "    printf 'import _virtualenv\\n' > \"$site/_virtualenv.pth\"\n"
        "    ;;\n"
        "  directory-pth)\n"
        "    mkdir \"$site/_virtualenv.pth\"\n"
        "    printf 'BOOTSTRAP = True\\n' > \"$site/_virtualenv.py\"\n"
        "    ;;\n"
        "  directory-py)\n"
        "    mkdir \"$site/_virtualenv.py\"\n"
        "    printf 'import _virtualenv\\n' > \"$site/_virtualenv.pth\"\n"
        "    ;;\n"
        "  pair)\n"
        "    printf 'import _virtualenv\\n' > \"$site/_virtualenv.pth\"\n"
        "    printf 'BOOTSTRAP = True\\n' > \"$site/_virtualenv.py\"\n"
        "    ;;\n"
        "esac\n"
        "printf 'preserve\\n' > \"$site/unrelated.txt\"\n",
    )
    environment = {
        "PATH": f"{fake_bin}:{os.defpath}",
        "FAKE_COMMAND_LOG": str(command_log),
        "FAKE_FIXTURE_PYTHON": str(fake_bin / "fixture-python"),
        "PYTHONPATH": "/hostile/parent/pythonpath",
        "PYTHONHOME": "/hostile/parent/pythonhome",
    }
    return repository, setup_script, command_log, environment


def test_setup_removes_only_generated_bootstrap_then_validates_exact_runtime(
    tmp_path: Path,
) -> None:
    repository, setup_script, command_log, environment = _fake_setup_repository(
        tmp_path
    )

    result = subprocess.run(
        ["bash", str(setup_script)],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    site_packages = (
        repository
        / "workers/local_ai/apple_mlx/.venv/lib/python3.11/site-packages"
    )
    assert result.returncode == 0, result.stderr
    assert not (site_packages / "_virtualenv.pth").exists()
    assert not (site_packages / "_virtualenv.py").exists()
    assert (site_packages / "unrelated.txt").read_text(encoding="utf-8") == "preserve\n"
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands[0].startswith("uv sync --frozen --project ")
    assert len([line for line in commands if line.startswith("uv ")]) == 1
    assert len([line for line in commands if line.startswith("python -c ")]) == 1
    environment_line = next(line for line in commands if line.startswith("python-env "))
    _, values = environment_line.split(" ", 1)
    pythonpath, pythonhome, no_user_site, safe_path, pycache, mode = values.split("|")
    assert (pythonpath, pythonhome) == ("unset", "unset")
    assert (no_user_site, safe_path, mode) == ("1", "1", "700")
    assert not Path(pycache).exists()


def test_setup_fails_before_removal_when_site_packages_is_ambiguous(
    tmp_path: Path,
) -> None:
    repository, setup_script, _command_log, environment = _fake_setup_repository(
        tmp_path
    )
    environment["FAKE_AMBIGUOUS"] = "1"

    result = subprocess.run(
        ["bash", str(setup_script)],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    site_packages = (
        repository
        / "workers/local_ai/apple_mlx/.venv/lib/python3.11/site-packages"
    )
    assert result.returncode == 1
    assert "exactly one worker site-packages directory" in result.stderr
    assert (site_packages / "_virtualenv.pth").is_file()
    assert (site_packages / "_virtualenv.py").is_file()


@pytest.mark.parametrize(
    "component",
    [".venv", "lib", "python3.11", "site-packages"],
)
def test_setup_rejects_symlinked_site_packages_topology_before_removal(
    tmp_path: Path,
    component: str,
) -> None:
    repository, setup_script, _command_log, environment = _fake_setup_repository(
        tmp_path
    )
    environment["FAKE_SYMLINK_COMPONENT"] = component

    result = subprocess.run(
        ["bash", str(setup_script)],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    site_packages = (
        repository
        / "workers/local_ai/apple_mlx/.venv/lib/python3.11/site-packages"
    )
    assert result.returncode == 1
    assert "site-packages must be a regular in-venv directory" in result.stderr
    assert (site_packages / "_virtualenv.pth").is_file()
    assert (site_packages / "_virtualenv.py").is_file()


def test_setup_accepts_already_absent_virtualenv_bootstrap(tmp_path: Path) -> None:
    repository, setup_script, _command_log, environment = _fake_setup_repository(
        tmp_path
    )
    environment["FAKE_BOOTSTRAP_STATE"] = "absent"

    result = subprocess.run(
        ["bash", str(setup_script)],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_setup_accepts_exact_direct_python_shebang_launcher(tmp_path: Path) -> None:
    repository, setup_script, _command_log, environment = _fake_setup_repository(
        tmp_path
    )
    environment["FAKE_LAUNCHER_STYLE"] = "direct"

    result = subprocess.run(
        ["bash", str(setup_script)],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("launcher_style", ["polyglot", "direct"])
@pytest.mark.parametrize("mutation", ["interpreter", "body", "extra"])
def test_setup_rejects_changed_supported_launcher(
    tmp_path: Path,
    mutation: str,
    launcher_style: str,
) -> None:
    repository, setup_script, _command_log, environment = _fake_setup_repository(
        tmp_path
    )
    environment["FAKE_LAUNCHER_MUTATION"] = mutation
    environment["FAKE_LAUNCHER_STYLE"] = launcher_style

    result = subprocess.run(
        ["bash", str(setup_script)],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "worker launcher" in result.stderr


@pytest.mark.parametrize("member", ["pth", "py"])
@pytest.mark.parametrize("state", ["missing", "symlink", "directory"])
def test_setup_rejects_nonregular_or_partial_bootstrap_without_removal(
    tmp_path: Path,
    member: str,
    state: str,
) -> None:
    repository, setup_script, _command_log, environment = _fake_setup_repository(
        tmp_path
    )
    environment["FAKE_BOOTSTRAP_STATE"] = f"{state}-{member}"

    result = subprocess.run(
        ["bash", str(setup_script)],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    site_packages = (
        repository
        / "workers/local_ai/apple_mlx/.venv/lib/python3.11/site-packages"
    )
    other = "py" if member == "pth" else "pth"
    assert result.returncode == 1
    assert "bootstrap must be absent or an exact regular-file pair" in result.stderr
    assert (site_packages / f"_virtualenv.{other}").is_file()
