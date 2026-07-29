"""Launch the deterministic fake worker in pipeline-valid E2E mode."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.local_ai.fake_worker import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--pipeline-valid"]))
