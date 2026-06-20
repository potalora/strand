"""Microsoft Presidio de-identification path (``PHI_ENGINE=presidio``).

This is the flag-gated alternative to the hand-rolled regex scrubber in
``phi_scrubber.py``. It re-homes the same four de-identification layers onto
Presidio's maintained analyzer framework, and adds a clinical LOCATION pass that
closes the documented city/geography Safe Harbor gap:

* **Layer 1 — structured identifiers.** Presidio predefined recognizers
  (SSN/email/IP/URL/phone/credit-card) for recall, plus **custom
  ``PatternRecognizer``s ported verbatim from the legacy regexes** (fax, MRN,
  ZIP, street address, account/accession, license, VIN, device, biometric,
  health-plan, slash-date + month-name-date generalization) so every existing
  scrubber regression holds exactly.
* **Layer 2 — known patient identity.** The decrypt-name/MRN/DOB defense (against
  the ``name_encrypted``-NULL leak) re-homed as per-call Presidio **deny-list /
  ad-hoc recognizers**. Score 1.0 so it always wins overlaps.
* **Layer 3 — person names + clinical eponyms.** Re-homed by reusing the tested,
  eponym-aware ``phi_ner.redact_named_entities`` (same spaCy model + clinical
  allowlist/suffix guard); fail-open **non-latching**.
* **Layer 4 — clinical LOCATION.** ``phi_location_ner.redact_locations`` via the
  Stanford clinical de-identifier, behind ``PHI_LOCATION_NER_ENABLED``; drug
  names survive. Fail-open non-latching.

The legacy path stays the default and is untouched. ``phi_scrubber.scrub_phi``
dispatches here when ``settings.phi_engine == "presidio"`` and falls back to
legacy if Presidio errors (de-identification must never crash a caller).
"""

from __future__ import annotations

import logging
import re
import threading

from app.config import settings

logger = logging.getLogger(__name__)

# --- Replacement tokens (identical to the legacy scrubber's tokens) -----------
_TOKEN: dict[str, str] = {
    # Presidio predefined entity types
    "US_SSN": "[SSN]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "IP_ADDRESS": "[IP]",
    "URL": "[URL]",
    "CREDIT_CARD": "[ACCOUNT]",
    "US_DRIVER_LICENSE": "[LICENSE]",
    # Custom (ported-regex) entity types
    "MT_SSN": "[SSN]",
    "MT_PHONE": "[PHONE]",
    "MT_FAX": "[FAX]",
    "MT_EMAIL": "[EMAIL]",
    "MT_MRN": "[MRN]",
    "MT_IP": "[IP]",
    "MT_URL": "[URL]",
    "MT_ZIP": "[ZIP]",
    "MT_STREET_ADDRESS": "[LOCATION]",
    "MT_ACCOUNT": "[ACCOUNT]",
    "MT_LICENSE": "[LICENSE]",
    "MT_VIN": "[VIN]",
    "MT_DEVICE_ID": "[DEVICE_ID]",
    "MT_BIOMETRIC": "[BIOMETRIC]",
    "MT_HEALTH_PLAN": "[HEALTH_PLAN]",
    # Known-identity (deny-list) entity types
    "MT_PATIENT_IDENTITY": "[PATIENT]",
    "MT_PATIENT_MRN": "[MRN]",
    "MT_PATIENT_DOB": "[DATE]",
    "MT_PATIENT_ADDRESS": "[LOCATION]",
}

# entity_type -> de-identification report key (mirrors legacy report keys so
# callers/tests that read report["dates_generalized"] / ["names_scrubbed"] work).
_REPORT_KEY: dict[str, str] = {
    "US_SSN": "ssn_scrubbed",
    "MT_SSN": "ssn_scrubbed",
    "PHONE_NUMBER": "phone_scrubbed",
    "MT_PHONE": "phone_scrubbed",
    "MT_FAX": "fax_scrubbed",
    "EMAIL_ADDRESS": "email_scrubbed",
    "MT_EMAIL": "email_scrubbed",
    "MT_MRN": "mrn_scrubbed",
    "IP_ADDRESS": "ip_address_scrubbed",
    "MT_IP": "ip_address_scrubbed",
    "URL": "url_scrubbed",
    "MT_URL": "url_scrubbed",
    "MT_ZIP": "zip_code_scrubbed",
    "MT_STREET_ADDRESS": "street_address_scrubbed",
    "MT_ACCOUNT": "account_scrubbed",
    "CREDIT_CARD": "account_scrubbed",
    "MT_LICENSE": "license_scrubbed",
    "US_DRIVER_LICENSE": "license_scrubbed",
    "MT_VIN": "vehicle_id_scrubbed",
    "MT_DEVICE_ID": "device_id_scrubbed",
    "MT_BIOMETRIC": "biometric_id_scrubbed",
    "MT_HEALTH_PLAN": "health_plan_number_scrubbed",
    "MT_DATE_GENERALIZE": "dates_generalized",
    "MT_PATIENT_IDENTITY": "names_scrubbed",
    "MT_PATIENT_MRN": "mrns_removed",
    "MT_PATIENT_DOB": "dobs_removed",
    "MT_PATIENT_ADDRESS": "addresses_removed",
}

_DATE_ENTITY = "MT_DATE_GENERALIZE"

# Minimum analyzer score to act on. Presidio's predefined recognizers include
# very-weak numeric heuristics (e.g. UsSsnRecognizer tags any bare 9-digit run as
# US_SSN at score 0.05) that would corrupt clinical CODES (SNOMED/LOINC/ICD are
# numeric). Our ported custom recognizers all score 0.85 and real predefined hits
# (email 1.0, phone 0.4, ip/url 0.5+) clear this floor, so 0.4 drops the junk
# heuristics WITHOUT losing any parity match. (Shadow-compare regression: a
# 9-digit SNOMED code must NOT become [SSN].)
_SCORE_THRESHOLD = 0.4

# --- Ported regexes (verbatim from phi_scrubber.PATTERNS) ---------------------
_CUSTOM_PATTERNS: list[tuple[str, str, str]] = [
    # (entity_type, pattern_name, regex)
    ("MT_SSN", "ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("MT_PHONE", "phone", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    (
        "MT_FAX",
        "fax",
        r"\b(?:fax|facsimile)[:\s]*(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    ),
    ("MT_EMAIL", "email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    ("MT_MRN", "mrn", r"\b(?:MRN|mrn|Medical Record Number)[:\s]*\d+\b"),
    ("MT_IP", "ip_address", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    ("MT_URL", "url", r"https?://[^\s<>\"]+"),
    ("MT_ZIP", "zip_code", r"\b\d{5}(?:-\d{4})?\b"),
    (
        "MT_STREET_ADDRESS",
        "street_address",
        r"\b\d{1,6}\s+(?:[A-Za-z0-9'.\-]+\s+){0,4}"
        r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
        r"Court|Ct|Place|Pl|Way|Circle|Cir|Terrace|Ter|Parkway|Pkwy|"
        r"Highway|Hwy|Square|Sq|Trail|Trl)\b\.?"
        r"(?:\s+[NSEW]{1,2}\b)?"
        r"(?:\s*,?\s*(?:Ste|Suite|Unit|Apt|Apartment|Fl|Floor|Rm|Room|Bldg|Building)"
        r"\.?\s*#?\s*[\w\-]+)*",
    ),
    (
        "MT_ACCOUNT",
        "account",
        r"\b(?:account|acct|accession)\s*(?:no\.?|number|num|#|id)?[:\s#*]*\d+\b",
    ),
    ("MT_LICENSE", "license", r"\b(?:license|certificate|DEA)[:\s#]*[A-Z0-9]+\b"),
    ("MT_VIN", "vehicle_id", r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    (
        "MT_DEVICE_ID",
        "device_id",
        r"\b(?:serial|UDI|device\s*(?:id|identifier))[:\s#]*[A-Za-z0-9\-]+\b",
    ),
    (
        "MT_BIOMETRIC",
        "biometric_id",
        r"\b(?:biometric|fingerprint|retina|voiceprint)[:\s#]*[A-Za-z0-9\-]+\b",
    ),
    (
        "MT_HEALTH_PLAN",
        "health_plan_number",
        r"\b(?:plan|policy|member|group|subscriber|beneficiary)\s*"
        r"(?:number|no|#|id)[:\s#]*[A-Za-z0-9\-]+\b",
    ),
    # Date generalization is handled by a custom anonymizer operator (below); these
    # two recognizers only LOCATE the dates so the operator can shorten them.
    (
        _DATE_ENTITY,
        "month_name_date",
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    ),
    (
        _DATE_ENTITY,
        "slash_date",
        r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/\d{4}\b",
    ),
]

# Build cached, case-insensitive matchers for date generalization.
_SLASH_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(\d{4})\b")
_MONTH_NAME_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)


def _generalize_date(matched: str) -> str:
    """Generalize a detected date to month+year (drop the most-identifying day).

    ``07/31/1996`` -> ``07/1996``; ``January 5, 2026`` -> ``January 2026``.
    Mirrors the legacy scrubber's date handling exactly. Unrecognized formats are
    returned unchanged (the recognizer only matches the two supported forms).
    """
    m = _SLASH_DATE_RE.fullmatch(matched) or _SLASH_DATE_RE.match(matched)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if _MONTH_NAME_DATE_RE.match(matched):
        parts = re.split(r"[\s,]+", matched.strip())
        if len(parts) >= 3:
            return f"{parts[0]} {parts[-1]}"
    return matched


# --- Lazy, thread-safe singleton analyzer/anonymizer --------------------------
_analyzer = None
_anonymizer = None
_lock = threading.Lock()
_warned = False


def _build_engines():
    """Construct the Presidio analyzer + anonymizer once (expensive: loads spaCy).

    Uses an explicit ``en_core_web_md`` NLP engine (the model the project already
    ships) and an empty registry seeded only with the recognizers we want — no
    default spaCy ``PERSON``/``GPE``/``DATE_TIME`` recognizer is registered, so
    person/location/date handling stays under our explicit, tested control
    (Layers 3 & 4 below; date generalization via the custom operator).
    """
    from presidio_analyzer import (
        AnalyzerEngine,
        Pattern,
        PatternRecognizer,
        RecognizerRegistry,
    )
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        EmailRecognizer,
        IpRecognizer,
        PhoneRecognizer,
        UrlRecognizer,
        UsSsnRecognizer,
    )
    from presidio_anonymizer import AnonymizerEngine

    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": settings.phi_ner_spacy_model}
            ],
        }
    ).create_engine()

    registry = RecognizerRegistry()

    # Presidio predefined recognizers (maintained; recall boosters).
    for rec in (
        EmailRecognizer(),
        UsSsnRecognizer(),
        IpRecognizer(),
        UrlRecognizer(),
        CreditCardRecognizer(),
        PhoneRecognizer(supported_regions=["US"]),
    ):
        registry.add_recognizer(rec)

    # Custom recognizers ported verbatim from the legacy regexes (parity).
    flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
    for entity, name, regex in _CUSTOM_PATTERNS:
        registry.add_recognizer(
            PatternRecognizer(
                supported_entity=entity,
                patterns=[Pattern(name=name, regex=regex, score=0.85)],
                global_regex_flags=flags,
            )
        )

    analyzer = AnalyzerEngine(
        registry=registry, nlp_engine=nlp_engine, supported_languages=["en"]
    )
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def _get_engines():
    global _analyzer, _anonymizer, _warned
    if _analyzer is not None and _anonymizer is not None:
        return _analyzer, _anonymizer
    with _lock:
        if _analyzer is None or _anonymizer is None:
            # NOT latched: a failed build retries on the next call.
            built = _build_engines()
            _analyzer, _anonymizer = built
            _warned = False
    return _analyzer, _anonymizer


def warm_load_presidio() -> bool:
    """Eagerly build the Presidio engines (call at startup when engine=presidio)."""
    try:
        _get_engines()
        return True
    except Exception:  # noqa: BLE001 - build failure must not crash startup
        logger.warning("Presidio engine warm-load failed (will retry per call)", exc_info=True)
        return False


def _build_deny_recognizers(
    patient_names: list[str] | None,
    patient_mrn: str | None,
    patient_dob: str | None,
    patient_address: str | None,
):
    """Layer 2: per-call deny-list recognizers from decrypted patient identity.

    Word-boundary deny-list matching (Presidio default) preserves the legacy
    short-name guard — ``Li`` redacts the patient but never ``lipid``. Score 1.0
    so known identity always wins overlap resolution.
    """
    from presidio_analyzer import PatternRecognizer

    recs = []
    if patient_names:
        tokens = [
            part
            for name in patient_names
            if name
            for part in name.split()
            if len(part) >= 2
        ]
        if tokens:
            recs.append(
                PatternRecognizer(
                    supported_entity="MT_PATIENT_IDENTITY",
                    deny_list=tokens,
                    deny_list_score=1.0,
                )
            )
    if patient_mrn:
        recs.append(
            PatternRecognizer(
                supported_entity="MT_PATIENT_MRN",
                deny_list=[patient_mrn],
                deny_list_score=1.0,
            )
        )
    if patient_dob:
        recs.append(
            PatternRecognizer(
                supported_entity="MT_PATIENT_DOB",
                deny_list=[patient_dob],
                deny_list_score=1.0,
            )
        )
    if patient_address:
        parts = [p.strip() for p in patient_address.split(",") if len(p.strip()) > 3]
        if parts:
            recs.append(
                PatternRecognizer(
                    supported_entity="MT_PATIENT_ADDRESS",
                    deny_list=parts,
                    deny_list_score=1.0,
                )
            )
    return recs


def _dedupe_overlaps(results):
    """Resolve overlapping analyzer hits (keep highest score, then longest span).

    Used for the de-identification REPORT counts: the anonymizer merges adjacent
    same-entity spans into one token (so ``PEDRO OTALORA`` -> a single ``[PATIENT]``),
    which would under-count distinct identifiers. Counting from overlap-deduped —
    but NOT adjacency-merged — results restores the legacy per-occurrence counts
    (and de-dupes the case where two recognizers, e.g. ``US_SSN`` + ``MT_SSN``,
    hit the same span).
    """
    ordered = sorted(results, key=lambda r: (r.start, -r.score, -(r.end - r.start)))
    kept: list = []
    for r in ordered:
        if any(not (r.end <= k.start or r.start >= k.end) for k in kept):
            continue
        kept.append(r)
    return kept


def _operators():
    from presidio_anonymizer.entities import OperatorConfig

    ops: dict[str, OperatorConfig] = {}
    for entity, token in _TOKEN.items():
        ops[entity] = OperatorConfig("replace", {"new_value": token})
    ops[_DATE_ENTITY] = OperatorConfig("custom", {"lambda": _generalize_date})
    ops["DEFAULT"] = OperatorConfig("replace", {"new_value": "[REDACTED]"})
    return ops


def scrub_phi_presidio(
    text: str,
    patient_names: list[str] | None = None,
    patient_dob: str | None = None,
    patient_address: str | None = None,
    patient_mrn: str | None = None,
    enable_ner: bool | None = None,
) -> tuple[str, dict[str, int]]:
    """Presidio implementation of ``scrub_phi`` (see module docstring).

    Same signature and return contract as ``phi_scrubber.scrub_phi``: returns
    ``(scrubbed_text, report)``. Runs Layers 1+2 via Presidio analyze/anonymize,
    then the clinical LOCATION pass (Layer 4, flagged) and the eponym-aware
    person-name pass (Layer 3) on the result.
    """
    if not text:
        return text, {}

    report: dict[str, int] = {}
    analyzer, anonymizer = _get_engines()

    ad_hoc = _build_deny_recognizers(
        patient_names, patient_mrn, patient_dob, patient_address
    )

    results = analyzer.analyze(
        text=text,
        language="en",
        ad_hoc_recognizers=ad_hoc or None,
        score_threshold=_SCORE_THRESHOLD,
        return_decision_process=False,
    )

    if results:
        anonymized = anonymizer.anonymize(
            text=text, analyzer_results=results, operators=_operators()
        )
        scrubbed = anonymized.text
        # Build report from overlap-deduped (not adjacency-merged) detections so
        # counts mirror the legacy per-occurrence semantics.
        for r in _dedupe_overlaps(results):
            key = _REPORT_KEY.get(r.entity_type)
            if key:
                report[key] = report.get(key, 0) + 1
    else:
        scrubbed = text

    # Layer 4: clinical LOCATION pass (cities/facilities) — closes the Safe
    # Harbor gap. Behind PHI_LOCATION_NER_ENABLED; fail-open non-latching.
    if settings.phi_location_ner_enabled:
        from app.services.ai.phi_location_ner import redact_locations

        scrubbed, loc_report = redact_locations(scrubbed)
        if loc_report.get("locations"):
            report["locations_scrubbed"] = loc_report["locations"]

    # Layer 3: eponym-aware free-text person-name redaction (shared with legacy).
    use_ner = settings.phi_ner_enabled if enable_ner is None else enable_ner
    if use_ner:
        from app.services.ai.phi_ner import redact_named_entities

        scrubbed, ner_report = redact_named_entities(scrubbed)
        if ner_report.get("names"):
            report["ner_names_scrubbed"] = ner_report["names"]

    return scrubbed, report
