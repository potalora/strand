"""Post-extraction precision guards for LangExtract output (remediation A1-A6).

LangExtract over-extracts: it invents procedures the patient never had, turns
bare measurements into observations, files lifestyle counseling as lab results,
and treats drug-class abbreviations / scrubber placeholders as content. This
module is a pure, defensive layer that drops or repairs those false entities
*after* extraction and *before* the entity→FHIR mapping.

It is intentionally conservative: a true clinical record should never be
dropped, only the obvious false positives. The extraction prompt + few-shot
examples (``clinical_examples.py``) are the first line of defence; this is the
backstop.

Public API:
    validate_entities(entities) -> list[ExtractedEntity]   # A1-A5
    normalize_entity_text(text) -> str                     # A6 dedup key
"""
from __future__ import annotations

import dataclasses
import logging
import re

from app.services.extraction.entity_extractor import ExtractedEntity

logger = logging.getLogger(__name__)

# --- A5: PHI scrubber placeholders -----------------------------------------
# The three-layer scrubber inserts bracketed all-caps tokens ([NAME], [DATE],
# [MRN], [LOCATION], …). Extraction runs AFTER scrubbing, so these leak in as
# "content". Match any bracketed all-caps token of >=2 chars (covers [IP] too),
# plus the generic [REDACTED].
_PHI_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]+\]")

# --- A1: procedure performed-vs-mentioned ----------------------------------
# Recommendation / planned / differential signals — a procedure carrying any of
# these was NOT performed, so it must not become a Procedure record. Checked
# against the entity text AND every attribute value.
_PROCEDURE_REJECT_SIGNALS = (
    "recommend",
    "consider",
    "due for",
    "due ",
    "screening option",
    "options include",
    "should ",
    "plan to",
    "planned",
    "schedule",  # scheduled / will schedule / schedule for
    "offered",
    "candidate for",
    "discuss",
    "advise",
    "elective pending",
    "to be done",
    "will need",
    "needs ",
    "differential",
)

# Evidence the procedure actually happened.
_PROCEDURE_PERFORMED_SIGNALS = (
    "s/p",
    "status post",
    "status-post",
    "underwent",
    "performed",
    "post-op",
    "postop",
    "h/o",
    "history of",
    "removed",
    "resection",
    "excision",
    "completed",
    "done on",
    "biopsy",
    "status: completed",
)

# Surgical suffixes that, on their own, imply a performed operation.
_SURGICAL_SUFFIX_RE = re.compile(r"\w+(ectomy|otomy|ostomy|oplasty|plasty)\b", re.IGNORECASE)

_PROCEDURE_PERFORMED_STATUSES = frozenset(
    {"completed", "performed", "done", "historical", "resolved", "finished"}
)

# --- A2: fragment detection -------------------------------------------------
# Unit / measurement words that do NOT count as an analyte or drug name. If the
# only alphabetic tokens in an observation are units and there is a number, the
# entity is a value-only fragment (e.g. "2mg", "120/80", "98.6", "5' 9\"").
_UNIT_WORDS = frozenset(
    {
        "mg", "mcg", "ug", "g", "kg", "lb", "lbs", "oz", "ml", "l", "dl",
        "mmhg", "bpm", "mm", "cm", "m", "in", "ft", "iu", "meq", "mmol",
        "mol", "mol/l", "mmol/l", "ng", "pg", "u", "units", "unit", "kcal",
        "cal", "bmi", "f", "c", "percent", "pct",
    }
)
_OBSERVATION_CLASSES = frozenset({"observation", "lab_result", "vital"})

# --- A3: lifestyle / counseling --------------------------------------------
# A leading "Word:" label that marks social/lifestyle content rather than a lab
# or vital observation.
# NOTE: "occupation" is deliberately NOT here. A job title is a drop category
# (finding A6), not legitimate social history, so it must fall through to
# ``_is_lifestyle_noise`` and be dropped rather than reclassified.
_LIFESTYLE_LABELS = frozenset(
    {
        "exercise", "diet", "alcohol", "tobacco", "smoking", "smoke", "caffeine",
        "sleep", "drugs", "substance", "activity", "nutrition",
        "recreational drugs", "physical activity",
    }
)
_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /]*?)\s*:\s*(.*)$", re.DOTALL)

# Imperative counseling verbs — when the *value* starts with one of these, the
# entity is a recommendation/directive, not the patient's recorded history.
_DIRECTIVE_VERBS = frozenset(
    {
        "avoid", "increase", "decrease", "reduce", "stop", "quit", "limit",
        "continue", "maintain", "start", "begin", "recommend", "recommended",
        "consider", "discuss", "encourage", "advise", "advised", "counsel",
        "cut", "eliminate", "abstain", "minimize", "follow",
    }
)

# --- A6: occupation / hobby / diet free-text leaking into observations ------
# Real-data regression: LangExtract files occupation ("business analyst"),
# hobbies/sports ("avid tennis player", "weight-lifting"), and free-text
# diet/food commentary ("low fodmap diet", "spicy and acidic foods", "Coffee is
# fine") as ``observation`` records. None carry a measured value, so they can
# never be a real lab or vital. The guard fires ONLY when the observation has
# NO numeric value (so real vitals/labs are untouched) AND its text matches one
# of these non-clinical keyword families. Matched as whole-word tokens (single
# words) or as substrings of the space-normalized text (phrases).
_OCCUPATION_WORDS = frozenset(
    {
        "occupation", "accountant", "attorney", "lawyer", "analyst", "engineer",
        "manager", "consultant", "teacher", "professor", "developer",
        "programmer", "technician", "administrator", "executive", "director",
        "clerk", "cashier", "secretary", "salesman", "saleswoman", "salesperson",
        "electrician", "plumber", "carpenter", "mechanic", "chef", "waiter",
        "waitress", "pilot", "architect", "scientist", "researcher", "contractor",
        "realtor", "banker", "broker", "designer", "writer", "editor",
        "journalist", "retired", "unemployed", "homemaker", "student", "employed",
        "businessman", "businesswoman", "business",
    }
)
_OCCUPATION_PHRASES = (
    "works as", "occupation is", "job title", "self employed", "employed as",
    "business owner", "stay at home",
)

_HOBBY_WORDS = frozenset(
    {
        "tennis", "golf", "golfer", "soccer", "basketball", "baseball",
        "football", "hockey", "yoga", "pilates", "crossfit", "marathon",
        "triathlon", "skiing", "snowboarding", "surfing", "gardening", "fishing",
        "hunting", "athlete", "weightlifting", "bodybuilding", "jogging",
        "cycling", "avid", "player", "hobby", "hobbies",
    }
)
_HOBBY_PHRASES = (
    "weight lifting", "weight training", "lifts weights", "plays tennis",
    "plays golf", "works out", "tennis player", "martial arts", "rock climbing",
)

_DIET_FOOD_WORDS = frozenset(
    {
        "diet", "fodmap", "vegetarian", "vegan", "pescatarian", "keto",
        "ketogenic", "paleo", "spicy", "coffee", "snacks", "snacking", "food",
        "foods",
    }
)
_DIET_FOOD_PHRASES = (
    "low fodmap", "gluten free", "dairy free", "low carb", "low fat",
    "acidic foods", "acidic food", "spicy foods", "fatty foods",
    "intermittent fasting", "food preference", "food preferences",
    "small frequent meals", "balanced diet",
)

# Substance use (smoking/tobacco/vaping/etc.) IS legitimate social history when
# expressed as a concise STATUS ("former smoker", "non-smoker", "tobacco use").
# Only verbose free-text commentary with no status qualifier and no measured
# value is noise ("Only vape no cigarette smoke"). Keep when in doubt.
_SUBSTANCE_WORDS = frozenset(
    {
        "smoke", "smoking", "smoker", "smokes", "tobacco", "cigarette",
        "cigarettes", "cigar", "cigars", "vape", "vapes", "vaping", "nicotine",
        "marijuana", "cannabis",
    }
)
_SUBSTANCE_STATUS_QUALIFIERS = frozenset(
    {
        "former", "never", "current", "currently", "non", "ex", "active",
        "social", "occasional", "occasionally", "heavy", "light", "daily",
        "denies", "denied", "quit", "none", "history", "past", "prior", "pack",
        "packs", "ppd", "year", "years", "use", "used", "status",
    }
)
# Verbose substance commentary needs at least this many words to be dropped, so
# concise statuses ("former smoker", "tobacco use", bare "smoker") always survive.
_SUBSTANCE_COMMENTARY_MIN_WORDS = 4

# --- A4: medication quality -------------------------------------------------
# Drug-class abbreviations and other non-drug tokens that are never a specific
# medication record. We deliberately do NOT build an RxNorm map here (that is
# Agent B's terminology work) — just reject the obvious non-drugs/garbage.
_NON_DRUG_TOKENS = frozenset(
    {
        "ppi", "ppis", "ldn", "nsaid", "nsaids", "ssri", "ssris", "snri",
        "snris", "ace", "acei", "aceis", "arb", "arbs", "tca", "tcas", "maoi",
        "maois", "bb", "ccb", "dmard", "dmards", "otc", "prn", "ic", "go",
        "iv", "po", "im", "sq", "bid", "tid", "qid", "qhs", "qd", "tab",
        "tabs", "cap", "caps", "rx", "med", "meds", "drug", "drugs",
    }
)
# Single letter + digits, e.g. "x7" — garbage per remediation A4. NOTE: many
# legitimate vitamins/supplements share this shape ("B12", "D3", "K2"), so the
# allowlist below short-circuits this rule before it fires.
_GARBAGE_LETTER_NUM_RE = re.compile(r"^[A-Za-z]\d+$")

# Vitamins / supplements are genuine medications. They are short and/or
# letter+digit shaped, so without an allowlist the garbage/min-length heuristics
# would wrongly drop them (recall loss). Matched on a normalized form
# (lowercased, non-alphanumerics stripped): "B-complex"→"bcomplex",
# "omega-3"→"omega3", "CoQ10"→"coq10", "vitamin D"→"vitamind".
#
# DELIBERATELY NO BARE SINGLE LETTERS. Earlier this set carried "a"/"c"/"d"/
# "e"/"k" so a stray "D" or "K" survived as a supplement (a recall-favoring
# choice). That misfired: a single stray letter misclassified as a med slipped
# through as a "supplement" false positive. A bare letter is now kept ONLY when
# it is vitamin-prefixed ("vitamin D"); the named/letter+digit forms below cover
# the genuine supplements ("D3", "B12", "folate", …).
_SUPPLEMENT_ALLOWLIST = frozenset(
    {
        # B vitamins (letter+digit forms — bare "b" is intentionally excluded)
        "b1", "b2", "b3", "b5", "b6", "b7", "b9", "b12", "bcomplex",
        "thiamine", "riboflavin", "niacin", "biotin", "folate", "folicacid",
        "cobalamin", "cyanocobalamin",
        # fat-soluble vitamins (letter+digit only; bare "a"/"d"/"e"/"k" dropped)
        "d2", "d3", "k1", "k2",
        # vitamin C (bare "c" dropped — only "vitamin C" or named form)
        "ascorbicacid",
        # minerals / other supplements
        "iron", "ferroussulfate", "ironsulfate", "magnesium", "magnesiumoxide",
        "zinc", "calcium", "potassium", "selenium", "chromium", "iodine",
        # compounds
        "coq10", "coenzymeq10", "omega3", "omega6", "fishoil", "melatonin",
        "probiotic", "multivitamin", "glucosamine", "creatine",
    }
)


def _is_supplement(token: str) -> bool:
    """True for vitamins/supplements (e.g. "B12", "vitamin D", "omega-3").

    A supplement is recognized ONLY when it is vitamin-prefixed ("vitamin D"),
    a letter+digit form ("D3", "B12"), or a multi-character named supplement in
    the allowlist ("folate", "omega-3"). Bare single letters ("D", "K", "A",
    "C", "E") are NOT supplements unless vitamin-prefixed.
    """
    norm = re.sub(r"[^a-z0-9]", "", (token or "").lower())
    if not norm:
        return False
    if norm in _SUPPLEMENT_ALLOWLIST:
        return True
    # "vitamin D", "vitamin B12", "vitamin C", … — the vitamin prefix is decisive.
    return norm.startswith("vitamin")


def _contains_phi_placeholder(text: str) -> bool:
    return bool(_PHI_PLACEHOLDER_RE.search(text or ""))


def _attr_values(entity: ExtractedEntity) -> list[str]:
    """All string attribute values (skipping the internal _source_section tag)."""
    out: list[str] = []
    for key, value in (entity.attributes or {}).items():
        if key.startswith("_"):
            continue
        if isinstance(value, str):
            out.append(value)
    return out


def _haystack(entity: ExtractedEntity) -> str:
    """Lowercased text + attribute values, for signal scanning."""
    parts = [entity.text or ""] + _attr_values(entity)
    return " ".join(parts).lower()


def _has_date_evidence(entity: ExtractedEntity) -> bool:
    """A 4-digit-year date in any attribute value or the text."""
    blob = " ".join([entity.text or ""] + _attr_values(entity))
    return bool(re.search(r"\b(?:19|20)\d{2}\b", blob))


# --- A1 --------------------------------------------------------------------


def _validate_procedure(entity: ExtractedEntity) -> ExtractedEntity | None:
    hay = _haystack(entity)

    # Rejection signals (recommended / planned / due) take priority — a future
    # date does not make a recommended procedure a performed one.
    if any(sig in hay for sig in _PROCEDURE_REJECT_SIGNALS):
        logger.debug("A1 drop (mentioned-not-performed): %r", entity.text)
        return None

    status = str(entity.attributes.get("status", "")).lower()
    has_evidence = (
        _has_date_evidence(entity)
        or status in _PROCEDURE_PERFORMED_STATUSES
        or any(sig in hay for sig in _PROCEDURE_PERFORMED_SIGNALS)
        or bool(_SURGICAL_SUFFIX_RE.search(entity.text or ""))
    )
    if not has_evidence:
        logger.debug("A1 drop (no evidence of performance): %r", entity.text)
        return None
    return entity


# --- A2 / A3 ---------------------------------------------------------------


def _split_label(text: str) -> tuple[str, str] | None:
    """Return (label, remainder) for a leading 'Word:' prefix, else None."""
    match = _LABEL_RE.match(text or "")
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _is_directive(value: str) -> bool:
    """True when the value reads as imperative counseling ('avoid alcohol')."""
    first = re.sub(r"[^A-Za-z]", "", (value or "").strip().split(" ")[0]).lower()
    return first in _DIRECTIVE_VERBS


def _reclassify_lifestyle(entity: ExtractedEntity) -> ExtractedEntity | None:
    """A3: route a lifestyle-labelled observation to social_history, or drop a
    directive. Returns the entity unchanged (None signal handled by caller) if
    it is not lifestyle-labelled."""
    split = _split_label(entity.text)
    label = None
    remainder = entity.text
    if split is not None:
        label, remainder = split
    if label is None or label.lower() not in _LIFESTYLE_LABELS:
        return entity  # not lifestyle — caller continues to A2

    if _is_directive(remainder):
        logger.debug("A3 drop (directive counseling): %r", entity.text)
        return None

    attrs = dict(entity.attributes)
    attrs["category"] = label.lower()
    attrs.setdefault("value", remainder or entity.text)
    return dataclasses.replace(entity, entity_class="social_history", attributes=attrs)


def _norm_tokens(text: str) -> tuple[str, set[str]]:
    """Lowercase + collapse non-alphanumerics to spaces; return (norm, token set)."""
    norm = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return norm, set(norm.split())


def _match_terms(norm: str, tokens: set[str], words: frozenset[str], phrases: tuple) -> bool:
    """True if any single-word keyword is a token, or any phrase is a substring."""
    if tokens & words:
        return True
    padded = f" {norm} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _is_lifestyle_noise(text: str) -> bool:
    """A6: occupation / hobby / diet free-text misfiled as an observation.

    Only fires for value-less text — any digit means it may be a real
    measurement (lab/vital), so we keep it (prime directive: when uncertain,
    KEEP).
    """
    if any(ch.isdigit() for ch in (text or "")):
        return False
    norm, tokens = _norm_tokens(text)
    if not norm:
        return False
    return (
        _match_terms(norm, tokens, _OCCUPATION_WORDS, _OCCUPATION_PHRASES)
        or _match_terms(norm, tokens, _HOBBY_WORDS, _HOBBY_PHRASES)
        or _match_terms(norm, tokens, _DIET_FOOD_WORDS, _DIET_FOOD_PHRASES)
    )


def _is_substance_commentary(text: str) -> bool:
    """A6: verbose substance free-text with no status qualifier and no value.

    Concise substance STATUS ("former smoker", "tobacco use") is legitimate
    social history and is kept; only multi-word commentary with no qualifier and
    no measured value ("Only vape no cigarette smoke") is dropped.
    """
    if any(ch.isdigit() for ch in (text or "")):
        return False
    norm, tokens = _norm_tokens(text)
    if not (tokens & _SUBSTANCE_WORDS):
        return False
    if tokens & _SUBSTANCE_STATUS_QUALIFIERS:
        return False
    return len(norm.split()) >= _SUBSTANCE_COMMENTARY_MIN_WORDS


def _is_value_only_fragment(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True
    has_digit = any(ch.isdigit() for ch in text)
    has_measure_punct = any(ch in text for ch in ("'", '"'))
    words = re.findall(r"[A-Za-z]+", text)
    non_unit = [w for w in words if w.lower() not in _UNIT_WORDS]
    if non_unit:
        return False  # has a real analyte/drug name → keep
    return has_digit or has_measure_punct


def _validate_observation(entity: ExtractedEntity) -> ExtractedEntity | None:
    # A3 first — a lifestyle label routes to social_history (or drops).
    reclassified = _reclassify_lifestyle(entity)
    if reclassified is None:
        return None
    if reclassified.entity_class == "social_history":
        return reclassified
    # A6 — occupation / hobby / diet free-text misfiled as an observation.
    if _is_lifestyle_noise(entity.text):
        logger.debug("A6 drop (occupation/hobby/diet noise): %r", entity.text)
        return None
    # A6 — verbose substance commentary (concise status is kept above/below).
    if _is_substance_commentary(entity.text):
        logger.debug("A6 drop (substance-use commentary): %r", entity.text)
        return None
    # A2 — value-only fragment with no analyte/drug name.
    if _is_value_only_fragment(entity.text):
        logger.debug("A2 drop (value-only fragment): %r", entity.text)
        return None
    return entity


# --- A3 directive guard for already-social_history entities ----------------


def _validate_social_history(entity: ExtractedEntity) -> ExtractedEntity | None:
    value = entity.attributes.get("value") or entity.text
    split = _split_label(entity.text)
    if split is not None:
        value = entity.attributes.get("value") or split[1]
    if _is_directive(value):
        logger.debug("A3 drop (directive social_history): %r", entity.text)
        return None
    return entity


# --- A4 --------------------------------------------------------------------


def _validate_medication(entity: ExtractedEntity) -> ExtractedEntity | None:
    token = (entity.text or "").strip()
    if not token:
        return None
    low = token.lower()
    if low in _NON_DRUG_TOKENS:
        logger.debug("A4 drop (non-drug abbreviation): %r", entity.text)
        return None
    # Vitamins/supplements are legitimate meds — keep them before the garbage /
    # min-length heuristics that would otherwise drop "B12", "D3", "D", etc.
    if _is_supplement(token):
        return entity
    if _GARBAGE_LETTER_NUM_RE.match(token):
        logger.debug("A4 drop (letter+digit garbage): %r", entity.text)
        return None
    # Require a recognizable drug name: at least 3 chars with an alphabetic core.
    alpha = re.sub(r"[^A-Za-z]", "", token)
    if len(alpha) < 3:
        logger.debug("A4 drop (too short / no drug name): %r", entity.text)
        return None
    return entity


def _validate_one(entity: ExtractedEntity) -> ExtractedEntity | None:
    # A5 — placeholder filter applies to every class first.
    if _contains_phi_placeholder(entity.text):
        logger.debug("A5 drop (PHI placeholder): %r", entity.text)
        return None

    cls = entity.entity_class
    if cls == "procedure":
        return _validate_procedure(entity)
    if cls == "medication":
        return _validate_medication(entity)
    if cls in _OBSERVATION_CLASSES:
        return _validate_observation(entity)
    if cls == "social_history":
        return _validate_social_history(entity)
    return entity


def validate_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Drop/repair false LangExtract entities (precision guards A1-A5).

    Order-preserving. Returns a new list; reclassified entities are copies so
    the input objects are not mutated.
    """
    result: list[ExtractedEntity] = []
    for entity in entities:
        validated = _validate_one(entity)
        if validated is not None:
            result.append(validated)
    return result


# ---------------------------------------------------------------------------
# A6 — within-document duplicate normalization
# ---------------------------------------------------------------------------

_PARENTHETICAL_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_DATE_TOKEN_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{4}|(?:19|20)\d{2}-\d{1,2}-\d{1,2}|(?:19|20)\d{2})\b"
)
_WS_RE = re.compile(r"\s+")


def normalize_entity_text(text: str) -> str:
    """Normalize entity text into a stable within-document dedup key.

    Collapses whitespace, strips parenthetical/bracketed groups and bare date
    tokens, and casefolds — so ``Cystectomy``, ``Cystectomy (2020)`` and
    ``Cystectomy  (Jan 2020)`` all collapse to a single key. Without this the
    ``(entity_class, text.lower())`` dedup let parenthetical-date variants slip
    through (``Cystectomy`` ×9 from one document).
    """
    if not text:
        return ""
    out = _PARENTHETICAL_RE.sub(" ", text)
    out = _DATE_TOKEN_RE.sub(" ", out)
    out = _WS_RE.sub(" ", out)
    return out.strip().lower()
