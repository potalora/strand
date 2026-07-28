"""Packaging checks for the optional native Apple local-AI runtime."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def test_macos_runtime_setup_has_side_effect_free_help() -> None:
    script = REPOSITORY_ROOT / "scripts" / "setup-local-ai-macos.sh"

    result = subprocess.run(
        ["bash", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--download-pack" in result.stdout
    assert "does not download model files by default" in result.stdout


def test_main_native_setup_targets_supported_python_series() -> None:
    script = (REPOSITORY_ROOT / "scripts" / "setup-local.sh").read_text(encoding="utf-8")

    assert "python3.11" in script
    assert "python@3.11" in script
    assert "python3.12" not in script
    assert "python@3.12" not in script


def test_task_runner_keeps_local_ai_runtime_install_explicit() -> None:
    task_runner = (REPOSITORY_ROOT / "justfile").read_text(encoding="utf-8")

    assert "local-ai-runtime-install:" in task_runner
    assert "./scripts/setup-local-ai-macos.sh" in task_runner
    setup_recipe = task_runner.split("setup:", 1)[1].split("\n\n", 1)[0]
    assert "setup-local-ai-macos.sh" not in setup_recipe
