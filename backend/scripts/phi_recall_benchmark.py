"""PHI-recall benchmark: legacy vs. Presidio de-identification (WS-B / B6).

Recall is THE metric for de-identification — a missed identifier is a breach, so
we measure the fraction of known PHI spans that each engine actually removes. We
also measure a drug-name **false-positive** guard (clinical terms that must
survive), because the whole reason the legacy scrubber leaves the city gap open
is that blanket location NER corrupts clinical content.

The corpus below is an i2b2-/Synthea-style **held-out** set of synthetic clinical
text with hand-labeled PHI. It is synthetic on purpose: real i2b2 notes are
license-restricted and not bundled, and Synthea fixtures (WS-E) are not merged
into this branch. Each record lists the exact PHI substrings that MUST disappear
and the clinical terms that MUST survive.

Run:
    DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_wsb \
      .venv-wsb/bin/python scripts/phi_recall_benchmark.py
    # add --location to include the clinical LOCATION pass (downloads the model)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make the backend package importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.disable(logging.CRITICAL)


@dataclass
class Case:
    text: str
    phi: list[str]  # substrings that MUST be removed (recall target)
    survive: list[str] = field(default_factory=list)  # must NOT be redacted
    category: str = "mixed"


# --- i2b2-style held-out corpus (synthetic; hand-labeled) ---------------------
CASES: list[Case] = [
    Case(
        "Patient Pedro Otalora (MRN: 88451239) seen on 07/31/1996. "
        "SSN 123-45-6789. Phone (203) 555-0147.",
        phi=["Pedro", "Otalora", "88451239", "07/31/1996", "123-45-6789", "555-0147"],
        survive=[],
        category="identifiers",
    ),
    Case(
        "Follow-up with Dr. Waseem Ahmad in Gastroenterology. Prescribed Rifaximin "
        "550mg twice daily for Crohn's disease.",
        phi=["Waseem", "Ahmad"],
        survive=["Gastroenterology", "Rifaximin", "Crohn's disease"],
        category="provider_name",
    ),
    Case(
        "Mother Susan Miller has a history of Hodgkin lymphoma. Father deceased.",
        phi=["Susan", "Miller"],
        survive=["Hodgkin lymphoma"],
        category="family_name",
    ),
    Case(
        "Contact: jane.doe@example.com, fax 555-123-4567. "
        "Account No: 235410324. Lab Accession: 87414853.",
        phi=["jane.doe@example.com", "555-123-4567", "235410324", "87414853"],
        survive=[],
        category="contact_account",
    ),
    Case(
        "Home address 275 Post Rd E, Ste 10, Westport, CT 06880. "
        "Seen January 5, 2026.",
        phi=["275 Post Rd E", "06880", "January 5, 2026"],
        survive=[],
        category="address_date",
    ),
    Case(
        "Patient resides in Springfield. Started on Rituximab infusion. "
        "BP 120/80, A1c 6.8%.",
        phi=["Springfield"],
        survive=["Rituximab", "120/80", "6.8%"],
        category="city_location",
    ),
    Case(
        "Reviewed by John Doe, RN. Device serial: ABC123-DEF456. "
        "Member number: HPN12345.",
        phi=["John", "Doe", "ABC123-DEF456", "HPN12345"],
        survive=[],
        category="identifiers",
    ),
    Case(
        "Maria Gonzalez presents with Parkinson's disease and Bell's palsy. "
        "DEA license BX1234563.",
        phi=["Maria", "Gonzalez", "BX1234563"],
        survive=["Parkinson's disease", "Bell's palsy"],
        category="provider_eponym",
    ),
    Case(
        "Encounter at Mercy General Hospital in Sacramento on 02/16/2026. "
        "Metformin 500mg continued.",
        phi=["Sacramento", "02/16/2026"],
        survive=["Metformin"],
        category="city_location",
    ),
    Case(
        "Seen by Dr. Jennifer Lee. IP 10.0.12.45 logged. "
        "URL https://portal.example.org/patient/9921.",
        phi=["Jennifer", "Lee", "10.0.12.45", "https://portal.example.org/patient/9921"],
        survive=[],
        category="identifiers",
    ),
]


def _recall(scrub, case: Case) -> tuple[int, int]:
    out, _ = scrub(case.text)
    hit = sum(1 for p in case.phi if p not in out)
    return hit, len(case.phi)


def _survive(scrub, case: Case) -> tuple[int, int]:
    out, _ = scrub(case.text)
    kept = sum(1 for s in case.survive if s in out)
    return kept, len(case.survive)


def run(scrub, label: str) -> None:
    by_cat: dict[str, list[int]] = {}
    tot_hit = tot = 0
    surv_kept = surv_tot = 0
    misses: list[str] = []
    false_pos: list[str] = []
    for case in CASES:
        out, _ = scrub(case.text)
        for p in case.phi:
            tot += 1
            if p not in out:
                tot_hit += 1
            else:
                misses.append(f"[{case.category}] {p!r}")
            by_cat.setdefault(case.category, [0, 0])
            by_cat[case.category][1] += 1
            if p not in out:
                by_cat[case.category][0] += 1
        for s in case.survive:
            surv_tot += 1
            if s in out:
                surv_kept += 1
            else:
                false_pos.append(f"[{case.category}] {s!r}")

    print(f"\n=== {label} ===")
    print(f"PHI recall:      {tot_hit}/{tot}  = {tot_hit / tot:.1%}")
    print(
        f"Clinical kept:   {surv_kept}/{surv_tot} = "
        f"{(surv_kept / surv_tot if surv_tot else 1):.1%}  "
        f"(false-positive redactions: {surv_tot - surv_kept})"
    )
    print("  by category:")
    for cat, (h, n) in sorted(by_cat.items()):
        print(f"    {cat:18s} {h}/{n} = {h / n:.0%}")
    if misses:
        print("  MISSES (PHI leaked):")
        for m in misses:
            print(f"    - {m}")
    if false_pos:
        print("  FALSE POSITIVES (clinical destroyed):")
        for f in false_pos:
            print(f"    - {f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", action="store_true", help="enable clinical LOCATION pass")
    args = ap.parse_args()

    from app.config import settings
    from app.services.ai.phi_scrubber import scrub_phi

    # Legacy
    settings.phi_engine = "legacy"
    settings.phi_location_ner_enabled = False
    run(scrub_phi, "LEGACY (regex + spaCy PERSON NER)")

    # Presidio (no location)
    settings.phi_engine = "presidio"
    settings.phi_location_ner_enabled = False
    run(scrub_phi, "PRESIDIO (no location pass)")

    if args.location:
        settings.phi_engine = "presidio"
        settings.phi_location_ner_enabled = True
        from app.services.ai import phi_location_ner

        if phi_location_ner.warm_load_location_ner():
            run(scrub_phi, "PRESIDIO + clinical LOCATION (Stanford)")
        else:
            print("\n[!] LOCATION model unavailable (offline/blocked); skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
