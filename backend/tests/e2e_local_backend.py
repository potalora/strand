"""Run migrations and uvicorn with the local-only E2E socket guard installed."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main() -> None:
    """Start the E2E backend after denying non-loopback network access."""

    from tests.e2e_local_network_guard import install_loopback_socket_guard

    install_loopback_socket_guard()

    from alembic import command
    from alembic.config import Config
    import uvicorn

    command.upgrade(Config(str(BACKEND_ROOT / "alembic.ini")), "head")
    # uvloop performs networking in its native implementation, outside the
    # Python socket class patched above. Force asyncio so every backend client
    # connection crosses the E2E guard.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, loop="asyncio")


if __name__ == "__main__":
    main()
