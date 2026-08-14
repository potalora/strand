"""Install a runtime-verified candidate only for local release benchmarking."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TextIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.services.local_ai.artifact_store import resolve_retained_candidate_pack
from app.services.local_ai.errors import LocalAIError
from app.services.local_ai.pack_operations import platform_profile
from app.services.local_ai.pack_verifier import verify_pack_candidate
from scripts.local_ai_pack import _run_lifecycle


async def _verify_retained_candidate(*, stdout: TextIO, stderr: TextIO) -> int:
    """Validate retained artifact bytes against v2 without changing pack state."""

    platform_name, compatible = platform_profile()
    if platform_name != "apple_silicon" or not compatible:
        print(
            "ERROR: the candidate pack requires Apple Silicon with at least 16 GB.",
            file=stderr,
        )
        return 1
    try:
        candidate = resolve_retained_candidate_pack(
            manifest_path=Path(settings.local_ai_manifest_path),
            model_root=Path(settings.local_ai_model_dir),
        )
        await verify_pack_candidate(candidate.manifest, candidate.pack_path)
        candidate.revalidate()
    except LocalAIError:
        print("ERROR: candidate model pack verification failed.", file=stderr)
        return 1
    print(
        "Candidate model pack verified without activation: "
        f"{candidate.manifest.pack_revision}",
        file=stdout,
    )
    return 0


async def execute(action: str) -> int:
    """Run a non-admitting candidate lifecycle operation for benchmark operators."""

    if action not in {"install", "verify"}:
        print("ERROR: unsupported candidate model pack action.", file=sys.stderr)
        return 2
    if action == "verify":
        return await _verify_retained_candidate(stdout=sys.stdout, stderr=sys.stderr)
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
