"""Within-document (intra-doc) deduplication of AI-extracted records (A5).

A single unstructured document routinely spawns near-duplicate records that the
existing ``(entity_class, normalized_text)`` dedup in the worker does NOT catch:

* **Encounter over-fragmentation** — one clinical visit yields 2-4 ``encounter``
  records derived from section headers / facility names / boilerplate
  ("Today's Visit", "UCSF MyChart", "After Visit Summary", a bare practice name
  like "Peninsula Gastroenterology Group", a literal "Date of Visit:" field).
* **Brand + generic medication duplication** — the same drug extracted twice
  because both names appear (Lexapro + escitalopram; Seysara + sarecycline),
  resolving to the SAME RxNorm code.

This pass runs PER DOCUMENT, after terminology resolution (so medications carry
their ``code_value``/``code_system``) and before insert. It NEVER merges across
documents — cross-document dedup is a separate pipeline (``services/dedup``).

PRIME DIRECTIVE: reduce duplication WITHOUT regressing recall. When in doubt,
keep both records. Specifically:

* Encounters on clearly DIFFERENT dates are different visits — never collapsed.
* Medications with different RxNorm codes are different drugs — never merged.
* Uncoded medications only collapse on a conservative fuzzy-name match, and
  two clearly-different drug names (e.g. "insulin glargine" vs "insulin aspart")
  are never merged.

The public entry point operates on the worker's ``built_records`` list — a list
of ``(ExtractedEntity, record_dict | None)`` pairs as produced by
``_build_record_dicts`` — and returns a filtered list preserving input order.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# A record is RxNorm-coded when its code_system carries this marker.
_RXNORM_HINT = "rxnorm"

# Normalized substrings that mark an encounter "record" as document chrome
# (a header / boilerplate fragment), not a real clinical visit.
_ENCOUNTER_BOILERPLATE_SUBSTRINGS = (
    "mychart",
    "after visit summary",
    "avs",
    "date of visit",
    "todays visit",
    "today visit",
    "visit summary",
    "patient instructions",
    "clinical summary",
    "discharge instructions",
    "printed on",
)

# Tokens that mark a *bare facility / practice name* (an org, not a visit).
_FACILITY_TOKENS = (
    "group",
    "clinic",
    "hospital",
    "medical center",
    "med center",
    "associates",
    "physicians",
    "health system",
    "healthcare",
    "practice",
    "department",
    "institute",
    "gastroenterology",
    "cardiology",
    "dermatology",
    "orthopedics",
    "radiology",
    "oncology",
    "urology",
    "neurology",
    "pediatrics",
    "ophthalmology",
)

# Words that indicate genuine visit content; their presence vetoes the
# "bare facility name" classification (so "Cardiology Follow-up Visit" survives).
_CLINICAL_VISIT_WORDS = frozenset(
    {
        "visit",
        "follow",
        "followup",
        "consult",
        "consultation",
        "exam",
        "examination",
        "admission",
        "encounter",
        "appointment",
        "telehealth",
        "office",
        "established",
    }
)

# Conservative fuzzy thresholds for the BOTH-UNCODED medication fallback.
_MED_TOKEN_SET_CUTOFF = 90
_MED_TOKEN_SORT_CUTOFF = 80
_MED_LENGTH_RATIO_FLOOR = 0.6

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def dedup_within_document(built_records):
    """Collapse intra-document duplicate encounters and medications.

    Args:
        built_records: list of ``(ExtractedEntity, record_dict | None)`` pairs
            (the worker's ``built_records``). ``record_dict`` is ``None`` for
            entity classes that do not map to a storable record; those pass
            through untouched.

    Returns:
        A filtered list of the same shape, in the original order, with
        duplicate encounters/medications removed. The input is not mutated.
    """
    if not built_records:
        return built_records

    indexed = [
        (i, ent, rec) for i, (ent, rec) in enumerate(built_records)
    ]
    drop_ids: set[int] = set()

    encounters = [
        item for item in indexed
        if item[2] is not None and item[2].get("record_type") == "encounter"
    ]
    drop_ids |= _encounter_drops(encounters)

    medications = [
        item for item in indexed
        if item[2] is not None and item[2].get("record_type") == "medication"
    ]
    drop_ids |= _medication_drops(medications)

    if not drop_ids:
        return built_records

    result = [
        (ent, rec)
        for (ent, rec) in built_records
        if rec is None or id(rec) not in drop_ids
    ]
    removed = len(built_records) - len(result)
    if removed:
        logger.info("intra-doc dedup removed %d duplicate record(s)", removed)
    return result


# --- Encounters --------------------------------------------------------------


def _encounter_drops(encounters: list[tuple[int, object, dict]]) -> set[int]:
    """Return ``id(record_dict)`` of encounter records to drop.

    Encounters within one document that share a date — or have no date — are the
    same visit and collapse to a single, most-informative survivor. Encounters on
    clearly different dates are kept (different visits). In a multi-date document,
    a dateless encounter is ambiguous: it is kept as its own visit unless it is
    clearly document chrome (boilerplate / bare facility name), which is dropped.
    """
    if len(encounters) <= 1:
        return set()

    dated: dict[object, list] = defaultdict(list)
    dateless: list = []
    for item in encounters:
        key = _date_key(item[2])
        if key is None:
            dateless.append(item)
        else:
            dated[key].append(item)

    distinct_dates = list(dated.keys())
    drops: set[int] = set()
    groups: list[list] = []

    if not distinct_dates:
        # All encounters are dateless → treat as one (same-or-missing-date) visit.
        groups.append(dateless)
    elif len(distinct_dates) == 1:
        # A single visit date → dateless encounters belong to that visit.
        groups.append(dated[distinct_dates[0]] + dateless)
    else:
        # Multiple distinct dates → each date is its own visit. Dateless ones are
        # ambiguous: keep real-looking ones, drop clear document chrome.
        for key in distinct_dates:
            groups.append(dated[key])
        for item in dateless:
            if _is_boilerplate_encounter(item):
                drops.add(id(item[2]))
            else:
                groups.append([item])

    for group in groups:
        if len(group) <= 1:
            continue
        survivor = _best_encounter(group)
        for item in group:
            if item[2] is not survivor[2]:
                drops.add(id(item[2]))
    return drops


def _date_key(rec: dict):
    """Day-precision key for an encounter's effective date, or ``None``."""
    dt = rec.get("effective_date")
    if dt is None:
        return None
    try:
        return dt.date()
    except AttributeError:
        return dt


def _best_encounter(group: list[tuple[int, object, dict]]):
    """Pick the most informative encounter in a same-visit group.

    Higher ``_encounter_score`` wins; ties keep the earliest-extracted record.
    """
    return max(group, key=lambda item: (_encounter_score(item), -item[0]))


def _encounter_score(item: tuple[int, object, dict]) -> float:
    """Score an encounter by how much real visit information it carries."""
    _, ent, rec = item
    fhir = rec.get("fhir_resource") or {}
    score = 0.0
    # A genuine visit must always outrank document chrome.
    if not _is_boilerplate_encounter(item):
        score += 1000.0
    if _encounter_has_provider(fhir):
        score += 100.0
    if rec.get("effective_date") is not None:
        score += 50.0
    score += _populated_field_count(fhir)
    for key, pts in (("text", 10), ("reasonCode", 10), ("type", 5), ("serviceProvider", 3)):
        if fhir.get(key):
            score += pts
    return score


def _encounter_has_provider(fhir: dict) -> bool:
    for participant in fhir.get("participant") or []:
        display = (participant.get("individual") or {}).get("display")
        if display and str(display).strip():
            return True
    return False


def _populated_field_count(fhir: dict) -> int:
    """Count populated, non-metadata top-level fields of a FHIR resource."""
    skip = {"resourceType", "_extraction_metadata"}
    return sum(1 for key, value in fhir.items() if key not in skip and value)


def _is_boilerplate_encounter(item: tuple[int, object, dict]) -> bool:
    """True when an encounter is document chrome rather than a clinical visit."""
    signal = _encounter_signal_text(item[1], item[2])
    if not signal:
        return False
    if any(sub in signal for sub in _ENCOUNTER_BOILERPLATE_SUBSTRINGS):
        return True
    return _looks_like_bare_facility(signal)


def _encounter_signal_text(ent: object, rec: dict) -> str:
    """Normalized text describing an encounter (entity text + FHIR type label)."""
    parts: list[str] = []
    text = getattr(ent, "text", None)
    if text:
        parts.append(str(text))
    fhir = rec.get("fhir_resource") or {}
    for type_concept in fhir.get("type") or []:
        label = type_concept.get("text") if isinstance(type_concept, dict) else None
        if label:
            parts.append(str(label))
    return _normalize(" ".join(parts))


def _looks_like_bare_facility(signal: str) -> bool:
    """True for a short string that names a facility/practice with no visit content."""
    tokens = signal.split()
    if not tokens or len(tokens) > 7:
        return False
    if any(word in _CLINICAL_VISIT_WORDS for word in tokens):
        return False
    return any(token in signal for token in _FACILITY_TOKENS)


# --- Medications -------------------------------------------------------------


def _medication_drops(medications: list[tuple[int, object, dict]]) -> set[int]:
    """Return ``id(record_dict)`` of medication records to drop.

    Coded medications collapse only when they share the SAME RxNorm code (the
    brand + generic case). Uncoded medications collapse only on a conservative
    fuzzy-name match. Different RxNorm codes / clearly-different names are kept.
    """
    if len(medications) <= 1:
        return set()

    drops: set[int] = set()
    by_code: dict[str, tuple] = {}
    uncoded: list = []

    for item in medications:
        rec = item[2]
        code = rec.get("code_value")
        system = rec.get("code_system") or ""
        if code and _RXNORM_HINT in system.lower():
            key = str(code)
            existing = by_code.get(key)
            if existing is None:
                by_code[key] = item
            else:
                winner, loser = _prefer_med(existing, item)
                by_code[key] = winner
                drops.add(id(loser[2]))
        else:
            uncoded.append(item)

    survivors: list = []
    for item in uncoded:
        name = _med_name(item[1], item[2])
        merged = False
        for idx, survivor in enumerate(survivors):
            if _med_names_match(name, _med_name(survivor[1], survivor[2])):
                winner, loser = _prefer_med(survivor, item)
                survivors[idx] = winner
                drops.add(id(loser[2]))
                merged = True
                break
        if not merged:
            survivors.append(item)
    return drops


def _med_name(ent: object, rec: dict) -> str:
    name = getattr(ent, "text", None) or rec.get("code_display") or ""
    return _normalize(str(name))


def _prefer_med(a: tuple, b: tuple) -> tuple:
    """Return ``(winner, loser)`` for two duplicate medications.

    Prefers the coded record, then the most populated (dosage etc.), then the
    earliest-extracted. The winner is the survivor.
    """
    score_a, score_b = _med_score(a), _med_score(b)
    if score_a > score_b:
        return a, b
    if score_b > score_a:
        return b, a
    return (a, b) if a[0] <= b[0] else (b, a)


def _med_score(item: tuple[int, object, dict]) -> float:
    _, _, rec = item
    fhir = rec.get("fhir_resource") or {}
    score = 0.0
    if rec.get("code_value"):
        score += 100.0
    if fhir.get("dosageInstruction"):
        score += 10.0
    score += _populated_field_count(fhir)
    return score


def _med_names_match(a: str, b: str) -> bool:
    """Conservative fuzzy match for two UNCODED medication names.

    Requires a high ``token_set_ratio`` AND a corroborating ``token_sort_ratio``
    (which is not fooled by the subset inflation ``token_set_ratio`` is prone to)
    plus a length-ratio floor, so a short name never absorbs a longer different
    one and two distinct drugs sharing a head token (e.g. "insulin glargine" vs
    "insulin aspart") are never merged.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((len(a), len(b)))
    if longer == 0 or shorter / longer < _MED_LENGTH_RATIO_FLOOR:
        return False
    return (
        fuzz.token_set_ratio(a, b) >= _MED_TOKEN_SET_CUTOFF
        and fuzz.token_sort_ratio(a, b) >= _MED_TOKEN_SORT_CUTOFF
    )


def _normalize(text: str) -> str:
    """Lowercase, drop apostrophes, collapse non-alphanumerics to single spaces."""
    lowered = text.lower().replace("'", "").replace("’", "")
    return _NON_ALNUM_RE.sub(" ", lowered).strip()
