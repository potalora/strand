from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.local_model


def test_locked_worker_venv_exposes_json_schema_logits_processor_offline() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    worker_python = (
        repository_root
        / "workers"
        / "local_ai"
        / "apple_mlx"
        / ".venv"
        / "bin"
        / "python"
    )
    if not worker_python.is_file():
        pytest.skip(f"locked worker environment absent: {worker_python}")
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }

    result = subprocess.run(
        [
            str(worker_python),
            "-c",
            (
                "from mlx_vlm.structured import "
                "build_json_schema_logits_processor; "
                "assert callable(build_json_schema_logits_processor)"
            ),
        ],
        cwd=repository_root / "workers" / "local_ai" / "apple_mlx",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
