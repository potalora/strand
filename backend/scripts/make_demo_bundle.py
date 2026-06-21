"""Build a down-the-middle demo FHIR bundle for the README screenshots.

Synthea patients accumulate a realistic clinical history that includes content
that reads badly in a marketing screenshot taken out of context: smoking status,
social-determinant findings (employment, housing, criminal record, intimate-partner
abuse), mental-health / substance screening scores (PHQ, GAD, AUDIT, DAST, HARK),
an oncology thread (prostate cancer, PSA, biopsy, chemo, hospice), serious acute
cardiac disease, and reproductive content (pregnancy, contraception, STI panels).

This script takes one Synthea bundle and produces a *demo-safe* copy by:

1. Dropping every resource whose displayable text matches the sensitive-content
   exclusion list (see ``EXCLUDE``). Removal is resource-level only -- a dropped
   resource may leave a dangling reference inside a kept Claim / DiagnosticReport,
   but those internal links never surface in the UI, so no referential cleanup is
   needed. What survives is the mundane core: vitals, routine labs, common
   illnesses, a statin / cholesterol / prediabetes story, dental care, and
   flu / Td vaccines.
2. Stripping Synthea's numeric name suffixes (``Teddy976`` -> ``Teddy``,
   ``Dr. Shanita356 Wolff180`` -> ``Dr. Shanita Wolff``) so names read cleanly.
   Only capitalized-word-then-digits tokens are touched, so coded terms like
   ``COVID-19`` or ``Hemoglobin A1c`` are left alone.

The output is still fully synthetic (no real PHI). Source bundles live under the
gitignored ``tests/fixtures/synthea/`` tree, so this script -- not its output --
is what gets committed.

Usage::

    python backend/scripts/make_demo_bundle.py SOURCE.json [-o OUT.json] [--dry-run]

``--dry-run`` prints what would be removed (grouped by resource type) and the
surviving conditions / meds / care plans / vaccines / observation displays, so the
result can be eyeballed before anything is written or uploaded.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

# --- Sensitive-content exclusion list -------------------------------------------------
# (category, regex). A resource is dropped if ANY of its displayable strings matches
# ANY pattern. Patterns are deliberately specific so they never catch the mundane
# records we want to keep (e.g. "ischemic heart" not bare "heart", which would nuke
# "Heart rate"; "higher education" not bare "education", which would nuke "Oral health
# education"). Tested against the keep-list via --dry-run.
EXCLUDE: list[tuple[str, re.Pattern[str]]] = [
    ("smoking", re.compile(r"tobacco|smoking", re.I)),
    ("oncology", re.compile(
        r"prostate|prostatectomy|carcinoma|neoplasm|tumou?r|malignan|cancer|"
        r"leuprolide|docetaxel|chemotherap|antineoplastic|biopsy|hospice|"
        r"rectum|rectal|oncolog",
        re.I,
    )),
    ("cardiac", re.compile(
        r"isch(a)?emic heart|coronary|pulmonic valve|"
        r"abnormal findings diagnostic imaging heart|echocardiograph|transthoracic|"
        r"metoprolol|clopidogrel|nitroglycerin|myocardial|angina",
        re.I,
    )),
    ("screening", re.compile(
        r"\bphq\b|patient health questionnaire|gad-7|generalized anxiety|"
        r"assessment of anxiety|depression|audit-c|alcohol use disorder|"
        r"\bdast\b|drug abuse|substance use|\bhark\b",
        re.I,
    )),
    ("sdoh", re.compile(
        r"not in labor force|employment|housing unsatisfactory|higher education|"
        r"military service|social isolation|social contact|social care needs|"
        r"prapare|protocol for responding to and assessing|criminal record|"
        r"victim of|intimate partner|domestic abuse|stress \(finding\)|\bstress\b|"
        r"transport problem|lack of access to transport",
        re.I,
    )),
    ("reproductive", re.compile(
        r"pregnan|fetal|fetus|antenatal|antepartum|parturition|episiotom|cesarean|"
        r"amniotic|aneuploidy|fetoprotein|fundal|premature birth|gestation|obstetric|"
        r"augmentation of labor|contracept|intrauterine|chlamydia|gonorrhea|syphilis|"
        r"immunodeficiency virus|hepatitis|genital|cytopathology|"
        r"ethinyl estradiol|estradiol|levonorgestrel|norethindrone|dienogest",
        re.I,
    )),
    ("pain", re.compile(r"pain severity", re.I)),
    ("covid", re.compile(r"covid", re.I)),
]

# Suffix stripper: a capitalized word (incl. accented letters / apostrophes) directly
# followed by digits at a word boundary -> drop the digits. Leaves all-caps codes
# (COVID-19) and lower-case tokens (mcg) untouched.
_SUFFIX = re.compile(r"\b([A-Z][a-zà-ÿA-Za-zà-ÿ'’]*?)\d+\b")


def _coding_texts(cc: object) -> list[str]:
    """Pull .text and every .coding[].display out of a CodeableConcept-ish object."""
    out: list[str] = []
    if isinstance(cc, dict):
        if cc.get("text"):
            out.append(cc["text"])
        for c in cc.get("coding", []) or []:
            if isinstance(c, dict) and c.get("display"):
                out.append(c["display"])
    return out


def displayable_strings(resource: dict, med_index: dict[str, str]) -> list[str]:
    """Every human-readable label a resource could surface in the UI.

    Covers code, medication (inline concept or referenced Medication), vaccine,
    category, type, procedure/reason codes, care-plan activity, imaging-study
    description/body-site, and free-text name/description fields.
    """
    out: list[str] = []
    for key in ("code", "medicationCodeableConcept", "vaccineCode", "type"):
        out += _coding_texts(resource.get(key))
    for key in ("category", "reasonCode", "bodySite", "procedureCode"):
        val = resource.get(key)
        if isinstance(val, list):
            for cc in val:
                out += _coding_texts(cc)
        elif isinstance(val, dict):
            out += _coding_texts(val)
    # medicationReference -> resolved Medication display
    ref = (resource.get("medicationReference") or {}).get("reference", "")
    if ref:
        mid = ref.split(":")[-1].split("/")[-1]
        if med_index.get(mid):
            out.append(med_index[mid])
    # CarePlan.activity[].detail.code
    for act in resource.get("activity", []) or []:
        out += _coding_texts((act.get("detail") or {}).get("code"))
    # ImagingStudy / CareTeam / DiagnosticReport free text
    for key in ("description", "name"):
        v = resource.get(key)
        if isinstance(v, str):
            out.append(v)
    for series in resource.get("series", []) or []:
        out += _coding_texts(series.get("bodySite"))
        if series.get("description"):
            out.append(series["description"])
    return out


def match_category(resource: dict, med_index: dict[str, str]) -> str | None:
    """Return the exclusion category a resource matches, or None to keep it."""
    haystack = " ¦ ".join(displayable_strings(resource, med_index))
    if not haystack:
        return None
    for category, pattern in EXCLUDE:
        if pattern.search(haystack):
            return category
    return None


def strip_suffixes(node: object) -> None:
    """Recursively strip Synthea numeric name suffixes from name/display fields."""
    if isinstance(node, dict):
        for key, val in node.items():
            if key in ("family", "display") and isinstance(val, str):
                node[key] = _SUFFIX.sub(r"\1", val)
            elif key in ("given", "prefix", "suffix") and isinstance(val, list):
                node[key] = [_SUFFIX.sub(r"\1", v) if isinstance(v, str) else v for v in val]
            elif key == "text" and isinstance(val, str) and ("family" in node or "given" in node):
                node[key] = _SUFFIX.sub(r"\1", val)
            else:
                strip_suffixes(val)
    elif isinstance(node, list):
        for item in node:
            strip_suffixes(item)


def best_display(resource: dict, med_index: dict[str, str]) -> str:
    labels = displayable_strings(resource, med_index)
    return labels[0] if labels else resource.get("resourceType", "?")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a demo-safe FHIR bundle.")
    ap.add_argument("source", type=Path, help="source Synthea bundle (.json)")
    ap.add_argument("-o", "--out", type=Path, help="output path (default: <source>.demo.json)")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"Source not found: {args.source}")

    bundle = json.loads(args.source.read_text())
    entries = bundle.get("entry", [])

    # Index inline Medication resources so medicationReference can be resolved.
    med_index: dict[str, str] = {}
    for e in entries:
        r = e["resource"]
        if r["resourceType"] == "Medication":
            med_index[r.get("id", "")] = best_display(r, {})

    kept, removed = [], []
    removed_by_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    removed_by_cat: collections.Counter = collections.Counter()
    for e in entries:
        r = e["resource"]
        cat = match_category(r, med_index)
        if cat is None:
            kept.append(e)
        else:
            removed.append(e)
            removed_by_cat[cat] += 1
            removed_by_type[r["resourceType"]][f"[{cat}] {best_display(r, med_index)}"] += 1

    print(f"Source: {args.source.name}")
    print(f"  entries: {len(entries)}  kept: {len(kept)}  removed: {len(removed)}")
    print(f"  removed by category: {dict(removed_by_cat)}\n")

    print("=== REMOVED (grouped by resource type) ===")
    for rtype in sorted(removed_by_type):
        print(f"  {rtype}:")
        for label, n in sorted(removed_by_type[rtype].items()):
            print(f"      {n:3d}  {label}")

    bundle["entry"] = kept
    strip_suffixes(bundle)

    def surviving(rtype: str) -> collections.Counter:
        c: collections.Counter = collections.Counter()
        for e in kept:
            r = e["resource"]
            if r["resourceType"] == rtype:
                c[best_display(r, med_index)] += 1
        return c

    print("\n=== SURVIVING core (eyeball for anything still loud) ===")
    for rtype in ("Condition", "MedicationRequest", "CarePlan", "Immunization", "Observation", "Procedure"):
        c = surviving(rtype)
        print(f"  {rtype} ({sum(c.values())}):")
        for label, n in sorted(c.items()):
            print(f"      {n:3d}  {label}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    out = args.out or args.source.with_suffix(".demo.json")
    out.write_text(json.dumps(bundle, indent=2))
    print(f"\nWrote demo-safe bundle: {out}  ({len(kept)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
