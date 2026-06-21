from __future__ import annotations

import asyncio
import re
import logging
from typing import Any

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
    # Slash dates (MM/DD/YYYY) are generalized to the year only below (not a
    # fixed token), so they are handled separately from these replacement patterns.
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

# --- Date generalization (HIPAA Safe Harbor: keep YEAR ONLY) -----------------
# Safe Harbor permits the four-digit year of a date related to an individual but
# NOT the month or day. Every recognized date format below collapses to its year
# only (month + day dropped). Dates are handled separately from PATTERNS because
# the replacement is data-dependent (the captured year), not a fixed token.
_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)"
)
# "July 14, 2023" / "July 14 2023" -> year (group 1)
_MONTH_FIRST_DATE = re.compile(
    rf"\b{_MONTHS}\s+\d{{1,2}},?\s+(\d{{4}})\b", re.IGNORECASE
)
# "14 July 2023" -> year (group 1)
_DAY_FIRST_DATE = re.compile(
    rf"\b\d{{1,2}}\s+{_MONTHS}\s+(\d{{4}})\b", re.IGNORECASE
)
# ISO "2023-07-14" / "2023/07/14" -> year (group 1)
_ISO_DATE = re.compile(
    r"\b(\d{4})[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"
)
# US slash "07/14/2023" / "7/4/2023" -> year (group 1, the 4-digit year)
_SLASH_DATE = re.compile(
    r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(\d{4})\b"
)

# --- Age generalization (HIPAA Safe Harbor: ages > 89 aggregate to "90+") ----
# "95-year-old" / "95 year old" — the leading number is group 1.
_AGE_YEAROLD = re.compile(r"\b(\d{1,3})[\s-]year[\s-]old\b", re.IGNORECASE)
# "age 95" / "aged 95" / "age: 95" — prefix is group 1, the number is group 2.
_AGE_PHRASE = re.compile(r"\b(aged?[:\s]+)(\d{1,3})\b", re.IGNORECASE)


def scrub_phi(
    text: str,
    patient_names: list[str] | None = None,
    patient_dob: str | None = None,
    patient_address: str | None = None,
    patient_mrn: str | None = None,
    enable_ner: bool | None = None,
) -> tuple[str, dict[str, int]]:
    """Remove PHI from text and return scrubbed text + de-identification report.

    Args:
        enable_ner: Run the spaCy NER pass for free-text person names (providers,
            family, anyone not in the patient record). Defaults to
            ``settings.phi_ner_enabled`` when None. The pass fails open if the
            model is unavailable.

    Returns:
        tuple: (scrubbed_text, report_dict)
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

    # Generalize specific dates to YEAR ONLY. HIPAA Safe Harbor permits the
    # four-digit year but NOT the month or day for dates related to an
    # individual, so we drop everything but the year. Covers month-name dates in
    # both orders, ISO (YYYY-MM-DD / YYYY/MM/DD), and US slash (MM/DD/YYYY).
    # Order matters: ISO runs before US slash so "2023/07/14" is taken as ISO.
    n_dates = 0
    scrubbed, n = _MONTH_FIRST_DATE.subn(lambda m: m.group(1), scrubbed)
    n_dates += n
    scrubbed, n = _DAY_FIRST_DATE.subn(lambda m: m.group(1), scrubbed)
    n_dates += n
    scrubbed, n = _ISO_DATE.subn(lambda m: m.group(1), scrubbed)
    n_dates += n
    scrubbed, n = _SLASH_DATE.subn(lambda m: m.group(1), scrubbed)
    n_dates += n
    if n_dates:
        report["dates_generalized"] = report.get("dates_generalized", 0) + n_dates

    # Generalize ages over 89 to "90+" (Safe Harbor aggregates all ages >89 into
    # a single category; the exact age is otherwise an identifier).
    ages_capped = 0

    def _cap_yearold(m: re.Match) -> str:
        nonlocal ages_capped
        if int(m.group(1)) > 89:
            ages_capped += 1
            return m.group(0).replace(m.group(1), "90+", 1)
        return m.group(0)

    def _cap_phrase(m: re.Match) -> str:
        nonlocal ages_capped
        if int(m.group(2)) > 89:
            ages_capped += 1
            return f"{m.group(1)}90+"
        return m.group(0)

    scrubbed = _AGE_YEAROLD.sub(_cap_yearold, scrubbed)
    scrubbed = _AGE_PHRASE.sub(_cap_phrase, scrubbed)
    if ages_capped:
        report["ages_generalized"] = report.get("ages_generalized", 0) + ages_capped

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


async def scrub_phi_async(
    text: str,
    patient_names: list[str] | None = None,
    patient_dob: str | None = None,
    patient_address: str | None = None,
    patient_mrn: str | None = None,
    enable_ner: bool | None = None,
) -> tuple[str, dict[str, int]]:
    """Async wrapper around :func:`scrub_phi` that runs it off the event loop.

    ``scrub_phi`` is CPU-bound — the spaCy PERSON-NER pass in particular pins the
    interpreter for hundreds of milliseconds to seconds on large documents. When
    that runs directly inside the background extraction worker's coroutine it
    blocks the single asyncio event-loop thread, so every concurrent API request
    stalls until the scrub finishes (perf divergence D3).

    Offloading to a worker thread keeps the event loop responsive: CPython
    releases the GIL periodically and spaCy/BLAS release it during their heavy C
    sections, so request handling interleaves with the scrub. The output is
    byte-identical to calling :func:`scrub_phi` directly — only WHERE the work
    runs changes, never WHAT it produces.

    This does NOT speed the scrub up (the work is still GIL-bound); it only stops
    it from freezing the loop. Use this from any ``async def`` on a hot path; the
    plain ``scrub_phi`` stays for sync callers and tests.
    """
    return await asyncio.to_thread(
        scrub_phi,
        text,
        patient_names=patient_names,
        patient_dob=patient_dob,
        patient_address=patient_address,
        patient_mrn=patient_mrn,
        enable_ner=enable_ner,
    )
