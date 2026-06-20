"""Clinical terminology lookup backed by bundled, offline, public-domain indexes.

Why this exists
---------------
LangExtract emits clinical entities as free-text labels with no codes, and even
structured medications were 0% RxNorm-coded. This module post-maps clinical
terms to standard codes so downstream FHIR resources carry a ``coding`` (and the
``health_records.code_system``/``code_value`` columns get populated).

Design (bundled-offline, replaces the former ~94-entry hand-curated map)
-----------------------------------------------------------------------
* Lookups read **compact derived indexes** committed under
  ``terminology_data/`` (gzipped JSON). **No network call ever happens at
  runtime.** Indexes are loaded lazily on first lookup and cached, so import and
  startup stay cheap.
* Each index is built by ``backend/scripts/build_terminology_index.py`` from a
  free, public-domain source (see that script's header for provenance):
    - Conditions  -> **ICD-10-CM**  (public domain; full code+description index
      from the MIT ``simple-icd-10-cm`` package, plus a curated colloquial-alias
      overlay so "diabetes"/"htn"/"gerd" resolve).
    - Medications -> **RxNorm**     (public domain; ingredient/brand concepts via
      the no-login RxNav API), plus a small synonym overlay and a **local
      supplement marker** overlay for functional-medicine items that exist in no
      standard vocabulary.
    - Labs        -> **LOINC**      (curated common-lab subset; each code verified
      against the public NLM Clinical Tables service — see LOINC attribution in
      ``terminology_data/NOTICE.md``). Esoteric labs are intentionally uncoded.
    - Procedures  -> **local category markers** (CPT is AMA-proprietary and
      SNOMED CT is license-restricted, so neither is bundled; public-domain
      HCPCS/ICD-10-PCS do not cleanly cover common outpatient procedures).
* **Correctness over coverage**: an unknown/uncodable term returns ``None`` —
  the lookups never guess a wrong code.

Public API (unchanged — callers in ``entity_to_fhir`` and the Epic mappers rely
on it): :class:`Coding`, :func:`normalize_term`, :func:`lookup_condition`,
:func:`lookup_medication`, :func:`lookup_lab`, :func:`lookup_procedure`,
:func:`lookup`, :func:`parse_dosage`, the ``*_SYSTEM`` constants, and the
``CONDITION_INDEX``/``MEDICATION_INDEX``/``LAB_INDEX``/``PROCEDURE_INDEX``
module attributes (now lazily materialized).

How to extend / refresh
-----------------------
Edit the curated overlays (colloquial conditions, lab LOINC codes, procedures,
supplements, medication synonyms) in ``scripts/build_terminology_index.py`` and
re-run it; commit the regenerated ``terminology_data/*.json.gz``. Verify any new
code against its source terminology — correctness matters more than coverage.
"""
from __future__ import annotations

import asyncio
import gzip
import importlib.util
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from app.config import settings

logger = logging.getLogger(__name__)

# --- Code system canonical URIs (FHIR) -------------------------------------

ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
LOINC_SYSTEM = "http://loinc.org"
# Local marker systems for items with no permissively-licensed standard code.
SUPPLEMENT_SYSTEM = "https://medtimeline.local/CodeSystem/supplement"
PROCEDURE_SYSTEM = "https://medtimeline.local/CodeSystem/procedure"
# Kept for backward compatibility with callers/tests that reference the constant.
# NOTE: SNOMED CT is license-restricted and is no longer emitted; procedures now
# resolve to PROCEDURE_SYSTEM local markers.
SNOMED_SYSTEM = "http://snomed.info/sct"


@dataclass(frozen=True)
class Coding:
    """A single terminology coding (one row of a FHIR ``coding`` array)."""

    system: str
    code: str
    display: str

    def as_coding(self) -> dict:
        """Return the FHIR ``coding`` element dict."""
        return {"system": self.system, "code": self.code, "display": self.display}


# --- Normalization ----------------------------------------------------------

# A numeric strength/dose token, e.g. "500mg", "10 mg", "0.5 mcg", "1000 IU",
# "40 units", "20 mEq", "5%". Stripped so "Metformin 500mg" matches "metformin".
_DOSE_TOKEN = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:mg|mcg|µg|ug|g|kg|ml|l|iu|units?|unit|meq|mmol|mol|%|"
    r"tablets?|tabs?|caps?|capsules?)\b",
    re.IGNORECASE,
)
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize_term(text: str | None) -> str:
    """Normalize a clinical label for alias matching.

    Lower-cases, drops parenthetical asides and numeric dose tokens, removes
    punctuation (apostrophes too, so ``Crohn's`` -> ``crohns``) and collapses
    whitespace. Returns ``""`` for falsy input.
    """
    if not text:
        return ""
    s = str(text).lower()
    # Drop apostrophes (straight + curly) with no space, so possessives
    # collapse cleanly: "crohn's" -> "crohns", not "crohn s".
    s = s.replace("'", "").replace("’", "")
    s = _PARENTHETICAL.sub(" ", s)
    s = _DOSE_TOKEN.sub(" ", s)
    s = _NON_WORD.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


# --- Lazy index loading -----------------------------------------------------
# Indexes are compact gzipped JSON built offline by
# ``scripts/build_terminology_index.py``. Each file is:
#   {"codes": {key: [system, code, display]}, "index": {normalized_alias: key}}
# We materialize them to ``dict[str, Coding]`` lazily on first lookup and cache.

_DATA_DIR = Path(__file__).resolve().parent / "terminology_data"
_INDEX_FILES = {
    "condition": "conditions.json.gz",
    "medication": "medications.json.gz",
    "lab": "labs.json.gz",
    "procedure": "procedures.json.gz",
}
_INDEX_CACHE: dict[str, dict[str, Coding]] = {}

# Live, periodically-refreshed RxNorm medication index. Written under the
# gitignored ``backend/data/`` tree so refreshes never dirty the tracked
# baseline. Medication lookups prefer this when present; everything else (and
# the first-clone / offline case) falls back to the committed baseline.
_LIVE_MED_CACHE = Path(__file__).resolve().parents[3] / "data" / "terminology" / "medications.json.gz"
# The offline build script (reused by the live refresh to rebuild medications).
_BUILD_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_terminology_index.py"


def medication_live_cache_path() -> Path:
    """Path of the gitignored live medication index (may not exist yet)."""
    return _LIVE_MED_CACHE


def _resolve_index_path(category: str) -> Path:
    """Resolve the on-disk index path, preferring the live cache for medications."""
    if category == "medication" and _LIVE_MED_CACHE.exists():
        return _LIVE_MED_CACHE
    return _DATA_DIR / _INDEX_FILES[category]

# Bare/abbreviated medication synonyms applied at load time as a runtime overlay
# (so no index rebuild is needed). Each maps a human alias -> a canonical term
# already present in the RxNorm medication index whose Coding is reused. The same
# entries also live in the build script's ``MEDICATION_ALIASES`` so a future
# rebuild bakes them into the committed index. Example: bare "B12"/"b-12" -> the
# same RxNorm 11248 (vitamin B12) coding as "vitamin B12"/"cyanocobalamin".
_MEDICATION_SYNONYMS: dict[str, str] = {
    "B12": "vitamin B12",
    "b-12": "vitamin B12",
    "b 12": "vitamin B12",
}


def _apply_medication_synonyms(index: dict[str, Coding]) -> None:
    """Layer :data:`_MEDICATION_SYNONYMS` onto a loaded medication index.

    Each synonym reuses the :class:`Coding` of an existing canonical term; it is
    only added when (a) the canonical term is present and (b) the synonym key is
    not already a real concept (never overrides a genuine RxNorm entry).
    """
    for alias, target in _MEDICATION_SYNONYMS.items():
        alias_key = normalize_term(alias)
        if not alias_key or alias_key in index:
            continue
        coding = index.get(normalize_term(target))
        if coding is not None:
            index[alias_key] = coding


def _load_index(category: str) -> dict[str, Coding]:
    """Load (and cache) a category index as ``{normalized_alias: Coding}``.

    Missing/corrupt data files degrade gracefully to an empty index (lookups
    return ``None``) rather than crashing the extraction pipeline.
    """
    cached = _INDEX_CACHE.get(category)
    if cached is not None:
        return cached
    index: dict[str, Coding] = {}
    path = _resolve_index_path(category)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        codes = payload["codes"]
        for alias, key in payload["index"].items():
            entry = codes.get(key)
            if entry:
                system, code, display = entry
                index[alias] = Coding(system, code, display)
    except FileNotFoundError:
        logger.warning(
            "terminology index missing: %s — run scripts/build_terminology_index.py "
            "(lookups for %s will return None)", path, category,
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("failed to load terminology index %s: %s", path, exc)
    if category == "medication":
        _apply_medication_synonyms(index)
    _INDEX_CACHE[category] = index
    return index


# --- Live medication-index refresh (RxNorm) --------------------------------
# A periodic *bulk* refresh — NOT a per-lookup network call. Ingestion stays
# fully offline; this only rebuilds the medications index occasionally so RxNorm
# coverage stays current. Everything fails open: any error keeps the existing
# index (live cache or committed baseline) in service.


def _age_days(path: Path) -> float:
    """Age of ``path`` in days from its mtime (raises if it doesn't exist)."""
    return (time.time() - path.stat().st_mtime) / 86400.0


def _load_build_module():
    """Import the offline build script as a module (reuses its build logic)."""
    spec = importlib.util.spec_from_file_location("_terminology_build", _BUILD_SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load build script at {_BUILD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_medication_cache(out_path: Path) -> None:
    """Rebuild ONLY the medications index from RxNorm into ``out_path``.

    Reuses the offline build script's ``build_medications`` (RxNav pull + curated
    overlays). Network-bound; callers wrap this in fail-open handling.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _load_build_module().build_medications(offline=False, out_path=out_path)


def refresh_medication_index(max_age_days: int = 7) -> bool:
    """Refresh the live RxNorm medication index if missing/stale; fail-open.

    No-ops (returns ``False``, no network) when the live cache exists and is
    younger than ``max_age_days``. Otherwise rebuilds it from RxNorm, writes the
    gitignored live cache, hot-swaps the in-memory index, and returns ``True``.
    On ANY error (no connectivity, fetch/parse failure) it logs a warning, keeps
    whatever index is already in service, and returns ``False`` — never raises.
    """
    try:
        cache = _LIVE_MED_CACHE
        if cache.exists() and _age_days(cache) < max_age_days:
            return False
        _build_medication_cache(cache)
        _INDEX_CACHE.pop("medication", None)  # hot-swap on next lookup/load
        _load_index("medication")
        logger.info("medication index refreshed from RxNorm -> %s", cache)
        return True
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        logger.warning("medication index refresh skipped (keeping existing): %s", exc)
        return False


def schedule_medication_refresh(max_age_days: int = 7) -> "asyncio.Task":
    """Schedule a NON-BLOCKING background medication refresh; return the task.

    Runs the (blocking) refresh in a worker thread so the event loop / startup is
    never blocked, and the staleness gate makes it a millisecond no-op most boots.
    Errors are swallowed (defense in depth on top of the fail-open refresh).
    """
    async def _runner() -> None:
        try:
            await asyncio.to_thread(refresh_medication_index, max_age_days)
        except Exception:  # noqa: BLE001 — never let a background refresh surface
            logger.warning("background medication refresh raised", exc_info=True)

    return asyncio.create_task(_runner())


# --- High-threshold fuzzy fallback (WS-C, flagged off by default) -----------
# A last-resort RapidFuzz match for near-miss MISSPELLINGS of known terms. It
# runs only after exact/normalized/token lookups miss and only when
# ``settings.terminology_fuzzy_enabled`` is set (default off until validated).
# The cutoff is deliberately high so the "never emit a wrong code; unknown stays
# uncoded" guarantee holds.
#
# The gate intentionally pairs TWO scorers on the same cutoff:
#   * ``token_set_ratio`` — the decided primary scorer (order-insensitive).
#   * ``ratio`` — a char-level guard that defeats ``token_set_ratio``'s
#     subset-inflation: a generic token that is a SUBSET of a long alias scores
#     ``token_set_ratio`` = 100 (e.g. "deficiency" ⊂ "21 hydroxylase deficiency")
#     but has a low ``ratio``, so it is rejected instead of coded to an arbitrary
#     concept. True misspellings keep both scores high (≈90–96).
FUZZY_MATCH_CUTOFF = 88  # token_set_ratio & ratio are 0-100; >= this is accepted


def _fuzzy_lookup(index: dict[str, Coding], key: str) -> Coding | None:
    """Best high-confidence fuzzy match of ``key`` against an index's aliases.

    Returns the matched alias's :class:`Coding` only when BOTH the RapidFuzz
    ``token_set_ratio`` and the char-level ``ratio`` are at or above
    :data:`FUZZY_MATCH_CUTOFF`; otherwise ``None`` (a near-miss of nothing known,
    or a ``token_set_ratio`` subset-inflation false positive, stays uncoded —
    never a wrong code). ``key`` must already be normalized via
    :func:`normalize_term`.
    """
    if not key or not index:
        return None
    match = process.extractOne(
        key, index.keys(), scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_MATCH_CUTOFF
    )
    if match is None:
        return None
    alias = match[0]
    if fuzz.ratio(key, alias) < FUZZY_MATCH_CUTOFF:
        return None
    return index.get(alias)


def _lookup(
    category: str,
    text: str | None,
    *,
    first_token: bool,
    last_token: bool = False,
    fuzzy: bool | None = None,
) -> Coding | None:
    """Look up a normalized term in a category index.

    ``first_token`` enables a fallback: if the full normalized string misses,
    retry with just the first word (helps medications/labs that carry a trailing
    form/specimen word, e.g. "lisinopril tablet" -> "lisinopril"). ``last_token``
    additionally retries the last word (medications only — handles a leading
    qualifier, e.g. "daily b12" -> "b12", "low dose naltrexone" -> "naltrexone").

    When ``settings.terminology_fuzzy_enabled`` is set, a high-threshold RapidFuzz
    fallback (:func:`_fuzzy_lookup`) is consulted last — after every exact/token
    lookup misses — to catch misspellings of known terms; it stays default-off.
    """
    index = _load_index(category)
    key = normalize_term(text)
    if not key:
        return None
    hit = index.get(key)
    if hit is not None:
        return hit
    if first_token or last_token:
        tokens = key.split(" ")
        if len(tokens) > 1:
            if first_token:
                head = index.get(tokens[0])
                if head is not None:
                    return head
            if last_token:
                tail = index.get(tokens[-1])
                if tail is not None:
                    return tail
    use_fuzzy = settings.terminology_fuzzy_enabled if fuzzy is None else fuzzy
    if use_fuzzy:
        return _fuzzy_lookup(index, key)
    return None


def lookup_condition(text: str | None) -> Coding | None:
    """Map a condition label to an ICD-10-CM coding, or ``None``."""
    return _lookup("condition", text, first_token=False)


def lookup_medication(text: str | None, *, fuzzy: bool | None = None) -> Coding | None:
    """Map a medication/supplement label to an RxNorm (or local) coding, or ``None``.

    ``fuzzy`` overrides ``settings.terminology_fuzzy_enabled`` for this call — pass
    ``fuzzy=False`` for an exact/normalized-only lookup (used by the PHI de-id drug
    guard, which must be fast and must never protect a span via a fuzzy match)."""
    return _lookup("medication", text, first_token=True, last_token=True, fuzzy=fuzzy)


def lookup_lab(text: str | None) -> Coding | None:
    """Map a lab/analyte label to a LOINC coding, or ``None``."""
    return _lookup("lab", text, first_token=True)


def lookup_procedure(text: str | None) -> Coding | None:
    """Map a procedure label to a local procedure-marker coding, or ``None``."""
    return _lookup("procedure", text, first_token=False)


_CATEGORY_DISPATCH = {
    "condition": lookup_condition,
    "medication": lookup_medication,
    "lab": lookup_lab,
    "observation": lookup_lab,
    "procedure": lookup_procedure,
}


def lookup(category: str, text: str | None) -> Coding | None:
    """Dispatch a lookup by category name (condition/medication/lab/procedure)."""
    fn = _CATEGORY_DISPATCH.get(category)
    return fn(text) if fn else None


# Backward-compatible module attributes. The former implementation exposed
# ``CONDITION_INDEX``/``MEDICATION_INDEX``/``LAB_INDEX``/``PROCEDURE_INDEX`` as
# plain dicts; we keep them accessible but materialize lazily (PEP 562) so import
# stays cheap and the large indexes are only read when actually used.
_ATTR_TO_CATEGORY = {
    "CONDITION_INDEX": "condition",
    "MEDICATION_INDEX": "medication",
    "LAB_INDEX": "lab",
    "PROCEDURE_INDEX": "procedure",
}


def __getattr__(name: str) -> dict[str, Coding]:
    category = _ATTR_TO_CATEGORY.get(name)
    if category is not None:
        return _load_index(category)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Medication sig parsing (shared by AI + Epic structured paths) ----------
# Kept here (regex-only, no heavy deps) so the Epic mappers can reuse it
# without importing the LangExtract entity pipeline.

# Map an abbreviated/spelled route to a normalized route word.
_ROUTE_PATTERNS = (
    (re.compile(r"\b(po|by mouth|orally|oral)\b", re.IGNORECASE), "oral"),
    (re.compile(r"\b(iv|intravenous(?:ly)?)\b", re.IGNORECASE), "intravenous"),
    (re.compile(r"\b(im|intramuscular(?:ly)?)\b", re.IGNORECASE), "intramuscular"),
    (re.compile(r"\b(sc|sq|subq|subcutaneous(?:ly)?)\b", re.IGNORECASE), "subcutaneous"),
    (re.compile(r"\b(sl|sublingual(?:ly)?)\b", re.IGNORECASE), "sublingual"),
    (re.compile(r"\b(pr|rectal(?:ly)?)\b", re.IGNORECASE), "rectal"),
    (re.compile(r"\b(inhaled|inhalation|nebulized|neb)\b", re.IGNORECASE), "inhalation"),
    (re.compile(r"\b(topical(?:ly)?)\b", re.IGNORECASE), "topical"),
)
# Frequency phrases -> (frequency, period, periodUnit). Checked longest-first.
_FREQ_PATTERNS = (
    (re.compile(r"\b(qid|four times (?:a )?(?:day|daily)|q6h)\b", re.IGNORECASE), (4, 1, "d")),
    (re.compile(r"\b(tid|three times (?:a )?(?:day|daily)|q8h)\b", re.IGNORECASE), (3, 1, "d")),
    (re.compile(r"\b(bid|twice (?:a )?(?:day|daily)|two times (?:a )?(?:day|daily)|q12h)\b", re.IGNORECASE), (2, 1, "d")),
    (re.compile(r"\b(q(?:every)?\s*week(?:ly)?|once weekly|weekly|qweek)\b", re.IGNORECASE), (1, 1, "wk")),
    (re.compile(r"\b(qid|qhs|qam|qpm|at bedtime|nightly|every morning|once (?:a )?(?:day|daily)|once daily|daily|qd|qday|q24h|every day)\b", re.IGNORECASE), (1, 1, "d")),
)
_DOSE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g|ml|units?|unit|iu|meq|mmol|puffs?|%)\b",
    re.IGNORECASE,
)
_PRN_RE = re.compile(r"\b(prn|as needed|as required)\b", re.IGNORECASE)


def parse_dosage(text: str | None) -> dict:
    """Parse a medication sig into structured dose / route / frequency (B4).

    Returns dose_value/dose_unit, a normalized route word, a
    frequency/period/period_unit triple (for FHIR ``timing.repeat``), and an
    ``as_needed`` flag. Missing pieces stay ``None``/``False``.
    """
    result: dict = {
        "dose_value": None,
        "dose_unit": None,
        "route": None,
        "frequency": None,
        "period": None,
        "period_unit": None,
        "as_needed": False,
        "text": (str(text).strip() if text and str(text).strip() else None),
    }
    if not text:
        return result
    s = str(text)

    dm = _DOSE_RE.search(s)
    if dm:
        try:
            result["dose_value"] = float(dm.group(1))
        except (ValueError, TypeError):
            pass
        result["dose_unit"] = dm.group(2).lower()

    for pattern, route in _ROUTE_PATTERNS:
        if pattern.search(s):
            result["route"] = route
            break

    if _PRN_RE.search(s):
        result["as_needed"] = True

    for pattern, (freq, period, unit) in _FREQ_PATTERNS:
        if pattern.search(s):
            result["frequency"] = freq
            result["period"] = period
            result["period_unit"] = unit
            break

    return result
