from __future__ import annotations

import re
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Regex patterns for all 18 HIPAA identifiers
PATTERNS = {
    "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    "phone": (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    "fax": (re.compile(r"\b(?:fax|facsimile)[:\s]*(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", re.IGNORECASE), "[FAX]"),
    "email": (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    "mrn": (re.compile(r"\b(?:MRN|mrn|Medical Record Number)[:\s]*\d+\b"), "[MRN]"),
    "mrn_numeric": (re.compile(r"\b\d{8,12}\b"), None),  # Only scrub in context
    "ip_address": (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP]"),
    "url": (re.compile(r"https?://[^\s<>\"]+"), "[URL]"),
    "zip_code": (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[ZIP]"),
    # Street addresses: <number> <name words> <street-type> [direction] [unit].
    # spaCy fragments these (e.g. "275 Post Rd E" -> CARDINAL + PERSON), so a
    # regex is more reliable than NER for the street line. Requires a street-type
    # keyword after a leading number, keeping false positives on clinical text low.
    "street_address": (
        re.compile(
            r"\b\d{1,6}\s+(?:[A-Za-z0-9'.\-]+\s+){0,4}"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
            r"Court|Ct|Place|Pl|Way|Circle|Cir|Terrace|Ter|Parkway|Pkwy|"
            r"Highway|Hwy|Square|Sq|Trail|Trl)\b\.?"
            r"(?:\s+[NSEW]{1,2}\b)?"
            r"(?:\s*,?\s*(?:Ste|Suite|Unit|Apt|Apartment|Fl|Floor|Rm|Room|Bldg|Building)"
            r"\.?\s*#?\s*[\w\-]+)*",
            re.IGNORECASE,
        ),
        "[LOCATION]",
    ),
    # Slash dates (MM/DD/YYYY) are generalized to MM/YYYY below (not a fixed
    # token), so they are handled separately from these replacement patterns.
    "account": (
        re.compile(
            # The trailing [:\s#*]* tolerates markdown the OCR sometimes emits
            # ("**Account No:** 235410324") between the label and the number.
            r"\b(?:account|acct|accession)\s*(?:no\.?|number|num|#|id)?[:\s#*]*\d+\b",
            re.IGNORECASE,
        ),
        "[ACCOUNT]",
    ),
    "license": (
        re.compile(r"\b(?:license|certificate|DEA)[:\s#]*[A-Z0-9]+\b", re.IGNORECASE),
        "[LICENSE]",
    ),
    "vehicle_id": (re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"), "[VIN]"),
    "device_id": (
        re.compile(r"\b(?:serial|UDI|device\s*(?:id|identifier))[:\s#]*[A-Za-z0-9\-]+\b", re.IGNORECASE),
        "[DEVICE_ID]",
    ),
    "biometric_id": (
        re.compile(r"\b(?:biometric|fingerprint|retina|voiceprint)[:\s#]*[A-Za-z0-9\-]+\b", re.IGNORECASE),
        "[BIOMETRIC]",
    ),
    "health_plan_number": (
        re.compile(r"\b(?:plan|policy|member|group|subscriber|beneficiary)\s*(?:number|no|#|id)[:\s#]*[A-Za-z0-9\-]+\b", re.IGNORECASE),
        "[HEALTH_PLAN]",
    ),
}


def scrub_phi(
    text: str,
    patient_names: list[str] | None = None,
    patient_dob: str | None = None,
    patient_address: str | None = None,
    patient_mrn: str | None = None,
    enable_ner: bool | None = None,
) -> tuple[str, dict[str, int]]:
    """Remove PHI from text and return scrubbed text + de-identification report.

    Dispatches to the de-identification engine selected by ``settings.phi_engine``:

    * ``"legacy"`` (default) — the hand-rolled regex + known-identity + spaCy NER
      layers below.
    * ``"presidio"`` — Microsoft Presidio analyzer with the same four layers
      re-homed, plus an optional clinical LOCATION pass (``phi_presidio``). If the
      Presidio path raises, this falls back to the legacy path: de-identification
      must never crash a caller before any scrubbing happens.

    Args:
        enable_ner: Run the spaCy NER pass for free-text person names (providers,
            family, anyone not in the patient record). Defaults to
            ``settings.phi_ner_enabled`` when None. The pass fails open if the
            model is unavailable.

    Returns:
        tuple: (scrubbed_text, report_dict)
    """
    if settings.phi_engine == "presidio":
        try:
            from app.services.ai.phi_presidio import scrub_phi_presidio

            return scrub_phi_presidio(
                text,
                patient_names=patient_names,
                patient_dob=patient_dob,
                patient_address=patient_address,
                patient_mrn=patient_mrn,
                enable_ner=enable_ner,
            )
        except Exception:  # noqa: BLE001 - never let de-id crash the caller
            logger.warning(
                "Presidio de-identification path failed; falling back to the "
                "legacy regex scrubber for this call",
                exc_info=True,
            )

    return _scrub_phi_legacy(
        text,
        patient_names=patient_names,
        patient_dob=patient_dob,
        patient_address=patient_address,
        patient_mrn=patient_mrn,
        enable_ner=enable_ner,
    )


def _scrub_phi_legacy(
    text: str,
    patient_names: list[str] | None = None,
    patient_dob: str | None = None,
    patient_address: str | None = None,
    patient_mrn: str | None = None,
    enable_ner: bool | None = None,
) -> tuple[str, dict[str, int]]:
    """Legacy hand-rolled regex + known-identity + spaCy NER scrubber.

    Default de-identification path (``PHI_ENGINE=legacy``). Unchanged behavior;
    kept authoritative until the Presidio path is validated on real data.
    """
    report: dict[str, int] = {}
    scrubbed = text

    # Scrub known patient names first (targeted)
    if patient_names:
        for name in patient_names:
            if not name:
                continue
            for part in name.split():
                if len(part) < 2:
                    continue
                # Use word boundaries for short names to reduce false positives
                if len(part) <= 3:
                    pattern = re.compile(r"\b" + re.escape(part) + r"\b", re.IGNORECASE)
                else:
                    pattern = re.compile(re.escape(part), re.IGNORECASE)
                matches = pattern.findall(scrubbed)
                if matches:
                    report["names_scrubbed"] = report.get("names_scrubbed", 0) + len(matches)
                    scrubbed = pattern.sub("[PATIENT]", scrubbed)

    # Scrub known MRN
    if patient_mrn:
        pattern = re.compile(re.escape(patient_mrn))
        matches = pattern.findall(scrubbed)
        if matches:
            report["mrns_removed"] = len(matches)
            scrubbed = pattern.sub("[MRN]", scrubbed)

    # Scrub known address
    if patient_address:
        for part in patient_address.split(","):
            part = part.strip()
            if len(part) > 3:
                pattern = re.compile(re.escape(part), re.IGNORECASE)
                matches = pattern.findall(scrubbed)
                if matches:
                    report["addresses_removed"] = report.get("addresses_removed", 0) + len(matches)
                    scrubbed = pattern.sub("[LOCATION]", scrubbed)

    # Scrub known DOB
    if patient_dob:
        pattern = re.compile(re.escape(patient_dob))
        matches = pattern.findall(scrubbed)
        if matches:
            report["dobs_removed"] = len(matches)
            scrubbed = pattern.sub("[DATE]", scrubbed)

    # Apply regex patterns
    for key, (pattern, replacement) in PATTERNS.items():
        if replacement is None:
            continue
        matches = pattern.findall(scrubbed)
        if matches:
            report_key = f"{key}_scrubbed"
            report[report_key] = len(matches)
            scrubbed = pattern.sub(replacement, scrubbed)

    # Generalize specific dates to month/year
    date_pattern = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    )
    date_matches = date_pattern.findall(scrubbed)
    if date_matches:
        report["dates_generalized"] = len(date_matches)
        for m in date_matches:
            parts = re.split(r"[\s,]+", m)
            if len(parts) >= 3:
                scrubbed = scrubbed.replace(m, f"{parts[0]} {parts[-1]}")

    # Generalize slash dates MM/DD/YYYY -> MM/YYYY (drop the most-identifying
    # day), mirroring the month-name handling above. Lab reports carry DOB and
    # collection/report dates in this format; the day is a HIPAA date element.
    slash_date = re.compile(
        r"\b(0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(\d{4})\b"
    )
    scrubbed, n_slash = slash_date.subn(r"\1/\2", scrubbed)
    if n_slash:
        report["dates_generalized"] = report.get("dates_generalized", 0) + n_slash

    # NER pass: redact free-text person names the patterns/known-identifier
    # passes can't catch (providers, family members). Runs last, after known
    # patient names are replaced. Fails open if the model is missing. (Street
    # addresses are handled by the regex above; location NER is unsafe with the
    # general model — it mislabels drugs as places.)
    use_ner = settings.phi_ner_enabled if enable_ner is None else enable_ner
    if use_ner:
        from app.services.ai.phi_ner import redact_named_entities

        scrubbed, ner_report = redact_named_entities(scrubbed)
        if ner_report.get("names"):
            report["ner_names_scrubbed"] = ner_report["names"]

    return scrubbed, report
