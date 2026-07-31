"""Run the exact installed strict-local pack against fidelity goldens."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.services.local_ai.errors import LocalAIError
from app.services.local_ai.fidelity_metrics import FidelityGateError
from app.services.local_ai.fidelity_runner import (
    run_installed_fidelity_suite,
    write_fidelity_report,
)

_DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fidelity"
    / "local_ai"
    / "fixtures"
    / "synthetic"
    / "corpus-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the active manifest-verified local model pack through the "
            "content-free fidelity release gate."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(settings.local_ai_manifest_path),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(settings.local_ai_model_dir),
    )
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(settings.local_ai_scratch_dir) / "fidelity",
    )
    parser.add_argument(
        "--private-fixtures-dir",
        type=Path,
        default=(
            Path(os.environ["REAL_MEDICAL_FIXTURES_DIR"])
            if os.environ.get("REAL_MEDICAL_FIXTURES_DIR")
            else None
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/local-ai-fidelity.json"),
    )
    return parser


async def _execute(args: argparse.Namespace) -> int:
    try:
        report = await run_installed_fidelity_suite(
            corpus_path=args.corpus,
            manifest_path=args.manifest,
            model_root=args.model_root,
            scratch_root=args.scratch_root,
            private_fixtures_dir=args.private_fixtures_dir,
        )
        write_fidelity_report(args.output, report)
        report.assert_release_thresholds()
    except (FidelityGateError, LocalAIError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Local AI fidelity gate passed: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the real installed-pack release gate."""

    return asyncio.run(_execute(_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
