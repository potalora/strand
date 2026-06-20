"""Shadow-compare legacy vs. Presidio de-identification (WS-B validation).

Implements the project's flag-flip protocol ("compute old+new, log diffs, keep
old authoritative until validated"): runs BOTH engines over the same corpus and
reports, per string,

* whether the two engines produced identical output (parity),
* spans redacted by Presidio but NOT legacy (potential recall gains OR new
  false-positives — inspect for drug names), and
* spans redacted by legacy but NOT Presidio (potential recall regressions).

Data sources (in priority order, whichever are present in this checkout):
  1. ``tests/fixtures/user_provided_fhir.json``  (gitignored real bundle — absent
     in an isolated worktree; used if the integrator runs this on the main tree)
  2. ``tests/fixtures/sample_fhir_bundle.json``  (synthetic; always present)
  3. the crafted i2b2-style corpus in ``phi_recall_benchmark`` (always present)

NOTE: the gitignored real-PHI fixtures are intentionally NOT in the WS-B
worktree. Run this on the main tree (where they are retained) to shadow-compare
on real data before flipping ``PHI_ENGINE`` to ``presidio``.

Run:
    .venv-wsb/bin/python scripts/phi_shadow_compare.py
    .venv-wsb/bin/python scripts/phi_shadow_compare.py --ner   # include Layer-3 NER
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.disable(logging.CRITICAL)

_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _strings_from_json(path: Path) -> list[str]:
    out: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and len(o.strip()) >= 3:
            out.append(o)

    walk(json.loads(path.read_text()))
    return out


def _corpus() -> list[tuple[str, str]]:
    """Return (source_label, text) pairs from the best available data."""
    items: list[tuple[str, str]] = []
    real = _FIXTURES / "user_provided_fhir.json"
    synth = _FIXTURES / "sample_fhir_bundle.json"
    used_real = False
    if real.exists():
        for s in _strings_from_json(real):
            items.append(("real_fhir", s))
        used_real = True
    if synth.exists():
        for s in _strings_from_json(synth):
            items.append(("synthetic_fhir", s))
    # Always include the crafted clinical corpus (known PHI).
    from phi_recall_benchmark import CASES

    for c in CASES:
        items.append(("crafted", c.text))
    real_state = "YES" if used_real else "ABSENT (gitignored; run on main tree)"
    print(f"[corpus] real_fhir={real_state}; total strings={len(items)}")
    return items


_REDACT = re.compile(r"\[[A-Z_]+\]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ner", action="store_true", help="include Layer-3 person NER")
    ap.add_argument("--location", action="store_true", help="include LOCATION pass for presidio")
    args = ap.parse_args()

    from app.config import settings
    from app.services.ai.phi_scrubber import scrub_phi

    enable_ner = bool(args.ner)
    corpus = _corpus()

    identical = 0
    presidio_only: list[tuple[str, str, str]] = []  # (src, legacy_out, presidio_out)
    legacy_only: list[tuple[str, str, str]] = []
    changed_inputs = 0

    for src, text in corpus:
        settings.phi_engine = "legacy"
        settings.phi_location_ner_enabled = False
        legacy_out, _ = scrub_phi(text, enable_ner=enable_ner)

        settings.phi_engine = "presidio"
        settings.phi_location_ner_enabled = bool(args.location)
        presidio_out, _ = scrub_phi(text, enable_ner=enable_ner)

        if legacy_out == presidio_out:
            identical += 1
            continue
        changed_inputs += 1
        # Tokens present in one output but not the other indicate divergence.
        if _REDACT.search(presidio_out) and presidio_out != text and legacy_out == text:
            presidio_only.append((src, text, presidio_out))
        elif _REDACT.search(legacy_out) and legacy_out != text and presidio_out == text:
            legacy_only.append((src, text, presidio_out))
        else:
            # Both redacted but differently — record under presidio_only for review.
            presidio_only.append((src, legacy_out, presidio_out))

    total = len(corpus)
    print(f"\n=== shadow-compare (enable_ner={enable_ner}, location={bool(args.location)}) ===")
    print(f"strings compared:        {total}")
    print(f"identical output:        {identical}  ({identical / total:.1%})")
    print(f"diverged:                {changed_inputs}")
    print(f"\nPresidio redacted where legacy did not / differently ({len(presidio_only)}):")
    for src, a, b in presidio_only[:40]:
        print(f"  [{src}] legacy/in: {a!r}\n         presidio: {b!r}")
    print(f"\nLegacy redacted where Presidio did not ({len(legacy_only)}):")
    for src, a, b in legacy_only[:40]:
        print(f"  [{src}] in: {a!r}\n      presidio: {b!r}")
    print(
        "\n[review] inspect 'Presidio redacted where legacy did not' for NEW "
        "false-positives — especially drug/clinical terms wrongly redacted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
