"""NER-based person-name redaction for free clinical text.

The regex PHI scrubber catches structured identifiers; `patient_phi` strips the
*known* patient's identifiers. Neither catches arbitrary free-text names —
ordering providers, family members, anyone not in the patient record. This
module uses spaCy's PERSON NER to redact those before text is sent to Gemini.

Two safeguards keep it from destroying clinical meaning:

1. **Token-level redaction.** spaCy PERSON spans are noisy and can swallow
   adjacent verbs (e.g. "Pedro Otalora prescribed"). We redact only the
   proper-noun / Title-Case name tokens inside a span, leaving ordinary words.
2. **Clinical eponym allowlist.** Diseases, signs, and devices named after
   people ("Crohn", "Parkinson", "Hodgkin", "Bell", "Foley") must survive even
   when the model tags them PERSON.

The spaCy model is loaded lazily as a process-wide singleton. If it is missing
or fails to load, redaction is skipped (fail-open) — de-identification of the
patient's own identifiers and structured PHI still happened upstream.
"""

from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Surnames embedded in common eponymous diseases / signs / devices. These must
# never be redacted as names, or clinical meaning is lost. Lowercased.
CLINICAL_EPONYMS: frozenset[str] = frozenset(
    {
        "crohn", "parkinson", "alzheimer", "hodgkin", "bell", "foley",
        "graves", "cushing", "addison", "paget", "wilson", "huntington",
        "marfan", "raynaud", "sjogren", "guillain", "barre", "hashimoto",
        "kawasaki", "asperger", "tourette", "down", "charcot", "babinski",
        "romberg", "murphy", "mcburney", "homans", "tinel", "phalen", "apgar",
        "glasgow", "whipple", "meniere", "barrett", "dupuytren", "wegener",
        "behcet", "hirschsprung", "ehlers", "danlos", "klinefelter", "turner",
        "gilbert", "mallory", "weiss", "swan", "ganz", "hickman", "kaposi",
        "burkitt", "wernicke", "korsakoff", "lewy", "pick", "broca",
        "brudzinski", "kernig", "wilms", "ewing", "reye", "still", "felty",
    }
)

# Suffixes that mark a Title-Case token as a clinical term, not a name — spaCy
# routinely mistags specialties ("Gastroenterology"), conditions, and procedures
# as PERSON. No real surname ends this way, so the false-negative risk is nil.
CLINICAL_SUFFIXES: tuple[str, ...] = (
    "ology", "ologist", "iatry", "iatrist", "itis", "emia", "aemia",
    "opathy", "pathy", "ectomy", "ostomy", "otomy", "plasty", "oma",
    "osis", "iasis", "algia", "ostosis", "plegia", "trophy",
)

_NAME = "[NAME]"

# NOTE on locations: the general-purpose model mislabels DRUG names as GPE/ORG
# (e.g. "Rifaximin" -> GPE), so redacting GPE/LOC/FAC here destroys clinical
# content. Street addresses are handled by a regex in phi_scrubber instead;
# safely redacting cities/facilities would need a clinical-aware NER model.

_nlp = None
_warned = False


def _get_nlp():
    """Load the spaCy model lazily, caching the loaded model as a singleton.

    A load *failure* is NOT latched permanently: earlier behavior set a
    process-wide flag on the first exception, so a single transient failure
    (e.g. memory pressure while a PDF was being OCR'd concurrently) silently
    disabled name redaction for the entire life of the worker — a PHI hazard.
    Now each call re-attempts the load until it succeeds; we only log the
    warning once to avoid spam.
    """
    global _nlp, _warned
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        _nlp = spacy.load(settings.phi_ner_spacy_model)
        _warned = False
        return _nlp
    except Exception:  # noqa: BLE001 - missing model must not break scrubbing
        if not _warned:
            logger.warning(
                "spaCy model %r unavailable; skipping NER name redaction this "
                "call (will retry; structured + known-identifier scrubbing still "
                "applied)",
                settings.phi_ner_spacy_model,
                exc_info=True,
            )
            _warned = True
        return None


def warm_load_ner() -> bool:
    """Eagerly load the spaCy model (call at app startup, before heavy work).

    Loading once at boot — when memory is free and nothing competes for the GIL
    — makes the steady-state NER path a cached singleton and avoids first-load
    failures during concurrent extraction. Returns True if the model is ready.
    """
    return _get_nlp() is not None


def _is_known_medication(word: str) -> bool:
    """True when the token codes to a known drug — clinical content the general NER
    model routinely mistags as PERSON (e.g. "Rifaximin", "Rituximab"). Legacy
    preserved such drugs only by accident of redaction order; this terminology-
    backed check makes it deterministic. Uses an EXACT (non-fuzzy) lookup so it is
    O(1) per token and can never protect a real name via a fuzzy match. Fail-open:
    any error means "not a known drug" (token stays eligible for redaction), so it
    can never cause under-redaction by raising."""
    if len(word) < 4:
        return False
    try:
        from app.services.extraction import terminology

        return terminology.lookup_medication(word, fuzzy=False) is not None
    except Exception:  # noqa: BLE001 - guard must never break scrubbing
        return False


def _is_name_token(token) -> bool:
    """A token worth redacting: an alphabetic proper noun / Title-Case word
    that is not a protected clinical eponym or known drug name."""
    if not token.is_alpha:
        return False
    if not (token.pos_ == "PROPN" or token.is_title):
        return False
    lower = token.text.lower()
    if lower in CLINICAL_EPONYMS:
        return False
    if lower.endswith(CLINICAL_SUFFIXES):
        return False
    if _is_known_medication(token.text):
        return False
    return True


def redact_named_entities(text: str) -> tuple[str, dict[str, int]]:
    """Redact free-text PERSON names, preserving clinical content.

    Token-level: only proper-noun / Title-Case name tokens inside a PERSON span
    are redacted to ``[NAME]``, so verbs spaCy swallows into a span ("Pedro
    Otalora prescribed") survive, and clinical eponyms/specialties are protected.

    Locations are intentionally NOT redacted here — the general model mislabels
    drugs as GPE (see module note). Street addresses are handled by a regex in
    phi_scrubber.

    Returns ``(redacted_text, {"names": n})``; the input unchanged with an empty
    report when nothing matches or the model is unavailable.
    """
    if not text or not text.strip():
        return text, {}

    nlp = _get_nlp()
    if nlp is None:
        return text, {}

    doc = nlp(text)

    # Collect char spans of name tokens inside PERSON entities.
    spans: list[tuple[int, int]] = []
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        for token in ent:
            if _is_name_token(token):
                spans.append((token.idx, token.idx + len(token.text)))

    if not spans:
        return text, {}

    # Merge spans separated only by whitespace so "Pedro Otalora" -> one [NAME].
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and text[merged[-1][1] : start].strip() == "":
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    redacted = text
    for start, end in reversed(merged):
        redacted = redacted[:start] + _NAME + redacted[end:]

    return redacted, {"names": len(merged)}
