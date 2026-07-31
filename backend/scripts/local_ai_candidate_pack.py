"""Install a runtime-verified candidate only for local release benchmarking."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.local_ai_pack import _run_lifecycle


async def execute(action: str) -> int:
    """Run a non-admitting candidate lifecycle operation for benchmark operators."""

    if action not in {"install", "verify"}:
        print("ERROR: unsupported candidate model pack action.", file=sys.stderr)
        return 2
    return await _run_lifecycle(
        action,
        stdout=sys.stdout,
        stderr=sys.stderr,
        require_release=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install a strict-local candidate for offline benchmark validation only.",
    )
    parser.add_argument("action", choices=("install", "verify"))
    try:
        return asyncio.run(execute(parser.parse_args(argv).action))
    except KeyboardInterrupt:
        print("ERROR: candidate model pack command interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
