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
_load_failed = False


def _get_nlp():
    """Lazily load the spaCy model once; cache the result (and any failure)."""
    global _nlp, _load_failed
    if _nlp is not None:
        return _nlp
    if _load_failed:
        return None
    try:
        import spacy

        _nlp = spacy.load(settings.phi_ner_spacy_model)
    except Exception:  # noqa: BLE001 - missing model must not break scrubbing
        _load_failed = True
        logger.warning(
            "spaCy model %r unavailable; skipping NER name redaction "
            "(structured + known-identifier scrubbing still applied)",
            settings.phi_ner_spacy_model,
            exc_info=True,
        )
        return None
    return _nlp


def _is_name_token(token) -> bool:
    """A token worth redacting: an alphabetic proper noun / Title-Case word
    that is not a protected clinical eponym."""
    if not token.is_alpha:
        return False
    if not (token.pos_ == "PROPN" or token.is_title):
        return False
    lower = token.text.lower()
    if lower in CLINICAL_EPONYMS:
        return False
    if lower.endswith(CLINICAL_SUFFIXES):
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
