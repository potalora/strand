"""NuExtract3 grounded JSON extraction with one syntax-only retry."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping

from .common import (
    Generate,
    GenerationError,
    LoadedRole,
    WorkerInputError,
    bounded_json,
    generate_content,
    load_role_from_payload,
    parse_json_object,
    requested_output_tokens,
    validate_scratch_image,
    validate_token_budget,
)

EXTRACTION_OUTPUT_CAP = 8192
MAX_EXTRACTION_INPUT_BYTES = 4 * 1024 * 1024
MAX_SELECTED_IMAGES = 8
_EXTRACTION_INPUT_KEYS = frozenset({"page_markdown", "scratch_dir", "image_paths", "schema"})
_EXTRACTION_TRANSPORT_KEYS = frozenset(
    {
        "job_id",
        "manifest_path",
        "model_dir",
        "manifest_identity",
        "max_output_tokens",
    }
)
_NEGATION_MARKER = (
    r"(?:no(?:\s+evidence\s+of|\s+recurrence\s+of)?|denies?|denles?|"
    r"negative\s+for|ruled?\s+out)"
)
_NEGATED_AFTER_MARKER = r"(?:absent|not\s+present|negative|ruled?\s+out)"
_FAMILY_MARKER = (
    r"(?:family\s+(?:history|hx)|fhx|mother|father|sister|brother|"
    r"maternal|paternal)"
)
_NOT_PERFORMED_MARKER = r"(?:planned|cancelled|canceled|deferred|not\s+performed|not\s+done)"
_UNCERTAIN_MARKER = (
    r"(?:possible|possibly|probable|probably|suspected?|question\s+of|"
    r"concern\s+for|cannot\s+rule\s+out|can(?:not|'t)\s+exclude|"
    r"may\s+have|might\s+have|uncertain|unclear|equivocal|unlikely)"
)
_ASSERTION_CATEGORIES = (
    "conditions",
    "procedures",
    "allergies",
    "diagnostic_reports",
)
_FACT_SUBJECT_FIELDS = {
    "medications": "name",
    "conditions": "name",
    "procedures": "name",
    "labs": "name",
    "allergies": "substance",
    "encounters": "name",
    "immunizations": "name",
    "vital_signs": "name",
    "diagnostic_reports": "name",
    "care_plans": "title",
}
_MAX_BOUND_EVIDENCE_CHARS = 4096
_TABLE_ROW_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NUMERIC_UNIT_FIELDS = {
    "medications": ("dose_value", "dose_unit"),
    "labs": ("value", "unit"),
    "immunizations": ("dose", "dose_unit"),
    "vital_signs": ("value", "unit"),
}
_ADJACENT_UNIT_RE = re.compile(
    r"\s*(?:\|\s*)?(?P<unit>%|[^\W\d_][\w%./\u00b5\u03bc\u00b7^+-]*)",
    re.UNICODE,
)
_NON_UNIT_WORDS = frozenset(
    {
        "abnormal",
        "and",
        "at",
        "high",
        "is",
        "low",
        "normal",
        "on",
        "was",
        "were",
    }
)
_NON_PROMOTABLE_ASSERTIONS = frozenset({"negated", "uncertain", "mentioned_not_performed"})
_DIAGNOSTIC_ANCHOR_RE = re.compile(
    r"\b(?:report|study|imaging|radiology|pathology|impression|findings?|"
    r"interpretation|ct|mri|ultrasound|x[- ]?ray)\b",
    re.IGNORECASE,
)
_STATUS_SIGNALS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "medications": (
        (
            "stopped",
            re.compile(
                r"\b(?:stopped|discontinued|no longer taking|not taking|held|ceased)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "historical",
            re.compile(
                r"\b(?:historical|previously took|prior medication|past medication)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "active",
            re.compile(
                r"\b(?:active|current|taking|takes|continue(?:d)?|daily|weekly|"
                r"monthly|nightly|bid|tid|qid)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    "allergies": (
        ("resolved", re.compile(r"\bresolved\b", re.IGNORECASE)),
        (
            "inactive",
            re.compile(r"\b(?:inactive|not active|no longer active)\b", re.IGNORECASE),
        ),
        (
            "historical",
            re.compile(r"\b(?:historical|past allergy)\b", re.IGNORECASE),
        ),
        ("active", re.compile(r"\b(?:active|allergy|allergic)\b", re.IGNORECASE)),
    ),
    "encounters": (
        ("planned", re.compile(r"\b(?:planned|scheduled|upcoming)\b", re.IGNORECASE)),
        (
            "in_progress",
            re.compile(r"\b(?:in[_ ]progress|ongoing)\b", re.IGNORECASE),
        ),
        (
            "finished",
            re.compile(
                r"\b(?:finished|completed|visited|seen|encounter|visit)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    "immunizations": (
        (
            "entered_in_error",
            re.compile(r"\b(?:entered|recorded) in error\b", re.IGNORECASE),
        ),
        (
            "not_done",
            re.compile(
                r"\b(?:not done|not administered|not received|not given|refused|"
                r"declined)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "completed",
            re.compile(
                r"\b(?:administered|received|given|completed|vaccinated)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    "diagnostic_reports": (
        ("preliminary", re.compile(r"\bpreliminary\b", re.IGNORECASE)),
        ("amended", re.compile(r"\bamended\b", re.IGNORECASE)),
        (
            "final",
            re.compile(r"\b(?:final|finalized|reported|showed)\b", re.IGNORECASE),
        ),
    ),
    "care_plans": (
        ("resolved", re.compile(r"\bresolved\b", re.IGNORECASE)),
        (
            "inactive",
            re.compile(
                r"\b(?:inactive|not active|no longer active|closed|ended|discontinued)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "historical",
            re.compile(r"\b(?:historical|past plan)\b", re.IGNORECASE),
        ),
        (
            "active",
            re.compile(r"\b(?:active|current|continue(?:d)?)\b", re.IGNORECASE),
        ),
    ),
}


def _source_pages(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise WorkerInputError("Extraction OCR input is invalid.")
    pages: list[dict[str, object]] = []
    seen_pages: set[int] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"page_number", "markdown"}:
            raise WorkerInputError("Extraction OCR input is invalid.")
        page_number = item.get("page_number")
        markdown = item.get("markdown")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number <= 0
            or page_number in seen_pages
            or not isinstance(markdown, str)
        ):
            raise WorkerInputError("Extraction OCR input is invalid.")
        seen_pages.add(page_number)
        pages.append({"page_number": page_number, "markdown": markdown})
    return pages


def _selected_images(
    value: object,
    page_numbers: set[int],
    scratch_dir: object,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict) or len(value) > MAX_SELECTED_IMAGES:
        raise WorkerInputError("Extraction image selection is invalid.")
    selected: list[tuple[int, str]] = []
    for raw_page, raw_path in value.items():
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            raise WorkerInputError("Extraction image selection is invalid.") from None
        if str(page) != str(raw_page) or page not in page_numbers:
            raise WorkerInputError("Extraction image selection is invalid.")
        selected.append((page, validate_scratch_image(raw_path, scratch_dir)))
    return [path for _page, path in sorted(selected)]


def _prompt(pages: list[dict[str, object]], *, retry: bool) -> str:
    instruction = "Use the supplied OCR Markdown as the document source."
    if retry:
        instruction += (
            " The prior response violated JSON syntax or assertion grounding; return "
            "valid JSON only. Explicit negation requires assertion=negated, explicit "
            "family history requires assertion=family_history, and a planned, cancelled, "
            "deferred, or not-done procedure requires assertion=mentioned_not_performed."
        )
    serialized = bounded_json(
        {"source_pages": pages},
        max_bytes=MAX_EXTRACTION_INPUT_BYTES,
    )
    return f"{instruction}\nINPUT_JSON={serialized}"


def _ground_explicit_assertions(value: dict[str, object]) -> dict[str, object]:
    """Apply narrow deterministic assertions for explicit source phrases."""

    for category in _ASSERTION_CATEGORIES:
        facts = value.get(category)
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            context = " ".join(
                item
                for item in (
                    fact.get("verbatim"),
                    fact.get("evidence_excerpt"),
                )
                if isinstance(item, str)
            )
            subject_field = _FACT_SUBJECT_FIELDS[category]
            subject = fact.get(subject_field)
            if not isinstance(subject, str) or not subject.strip():
                continue
            subject_pattern = re.escape(subject.strip()).replace(r"\ ", r"\s+")
            before_subject = rf"(?<!\w){subject_pattern}(?!\w)"
            bounded_gap = r"(?:(?![.;|\n]).){0,80}"
            expected: str | None = None
            if re.search(
                rf"\b{_NEGATION_MARKER}\s+(?:the\s+)?{before_subject}",
                context,
                re.IGNORECASE,
            ) or re.search(
                rf"{before_subject}{bounded_gap}\b{_NEGATED_AFTER_MARKER}\b",
                context,
                re.IGNORECASE,
            ):
                expected = "negated"
            elif category == "procedures" and (
                re.search(
                    rf"\b{_NOT_PERFORMED_MARKER}\b{bounded_gap}{before_subject}",
                    context,
                    re.IGNORECASE,
                )
                or re.search(
                    rf"{before_subject}{bounded_gap}\b{_NOT_PERFORMED_MARKER}\b",
                    context,
                    re.IGNORECASE,
                )
            ):
                expected = "mentioned_not_performed"
            elif re.search(
                rf"\b{_UNCERTAIN_MARKER}\b{bounded_gap}{before_subject}",
                context,
                re.IGNORECASE,
            ) or re.search(
                rf"{before_subject}{bounded_gap}\b{_UNCERTAIN_MARKER}\b",
                context,
                re.IGNORECASE,
            ):
                expected = "uncertain"
            elif re.search(
                rf"\b{_FAMILY_MARKER}\b{bounded_gap}{before_subject}",
                context,
                re.IGNORECASE,
            ) or re.search(
                rf"{before_subject}{bounded_gap}\b{_FAMILY_MARKER}\b",
                context,
                re.IGNORECASE,
            ):
                expected = "family_history"
            else:
                expected = "present"
            fact["assertion"] = expected
    return value


def _ground_lifecycle_statuses(value: dict[str, object]) -> dict[str, object]:
    """Replace model lifecycle claims with conservative source-derived states."""

    for category, signals in _STATUS_SIGNALS.items():
        facts = value.get(category)
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            context = fact.get("verbatim")
            if not isinstance(context, str) or not context.strip():
                context = fact.get("evidence_excerpt")
            if not isinstance(context, str):
                context = ""
            fact["status"] = next(
                (status for status, pattern in signals if pattern.search(context)),
                "unknown",
            )
    return value


def _discard_missing_fact_subjects(value: dict[str, object]) -> dict[str, object]:
    """Discard structurally incomplete facts without inventing their subject."""

    raw_rejected = value.get("rejected_fields")
    rejected = (
        [item for item in raw_rejected if isinstance(item, str) and item.strip()]
        if isinstance(raw_rejected, list)
        else []
    )
    for category, subject_field in _FACT_SUBJECT_FIELDS.items():
        facts = value.get(category)
        if not isinstance(facts, list):
            continue
        accepted: list[object] = []
        for index, fact in enumerate(facts):
            subject = fact.get(subject_field) if isinstance(fact, dict) else None
            if not isinstance(subject, str) or not subject.strip():
                path = f"{category}[{index}].{subject_field}"
                if path not in rejected:
                    rejected.append(path)
                continue
            accepted.append(fact)
        value[category] = accepted
    if raw_rejected is not None or rejected:
        value["rejected_fields"] = rejected
    return value


def _semantic_evidence(value: str) -> str:
    """Return readable evidence text while preserving table cell boundaries."""

    decoded = html.unescape(_HTML_TAG_RE.sub(" | ", value))
    return re.sub(r"(?:\s*\|\s*)+", " | ", decoded)


def _bind_missing_adjacent_units(value: dict[str, object]) -> dict[str, object]:
    """Copy one unambiguous unit adjacent to an extracted numeric value."""

    for category, (value_field, unit_field) in _NUMERIC_UNIT_FIELDS.items():
        facts = value.get(category)
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            unit = fact.get(unit_field)
            numeric = fact.get(value_field)
            evidence = fact.get("verbatim")
            if (
                (isinstance(unit, str) and unit.strip())
                or not isinstance(numeric, str)
                or not numeric.strip()
                or not isinstance(evidence, str)
            ):
                continue
            source = _semantic_evidence(evidence)
            numeric_pattern = re.escape(numeric.strip()).replace(r"\ ", r"\s*")
            matches = list(
                re.finditer(
                    rf"(?<![\w.,]){numeric_pattern}(?![\w.,])",
                    source,
                    re.IGNORECASE,
                )
            )
            if len(matches) != 1:
                continue
            suffix = _ADJACENT_UNIT_RE.match(source, matches[0].end())
            if suffix is None:
                continue
            candidate = suffix.group("unit")
            if candidate.casefold() not in _NON_UNIT_WORDS:
                fact[unit_field] = candidate
    return value


def _searchable_source(value: str) -> str:
    """Normalize a source fragment for conservative term matching only."""

    without_tags = _HTML_TAG_RE.sub(" ", value)
    return " ".join(html.unescape(without_tags).casefold().split())


def _source_segments(source: str) -> list[str]:
    """Return bounded exact source fragments, preferring individual table rows."""

    candidates = [match.group(0) for match in _TABLE_ROW_RE.finditer(source)]
    candidates.extend(line for line in source.splitlines() if line.strip())
    seen: set[str] = set()
    return [
        candidate
        for candidate in candidates
        if candidate not in seen
        and not seen.add(candidate)
        and len(candidate) <= _MAX_BOUND_EVIDENCE_CHARS
    ]


def _find_exact_evidence(source: str, fact: Mapping[str, object], subject_field: str) -> str | None:
    """Find the shortest exact source fragment supporting an extracted fact."""

    subject = fact.get(subject_field)
    if not isinstance(subject, str) or not subject.strip():
        return None
    required = [_searchable_source(subject)]
    for field in ("value", "dose_value", "dose"):
        value = fact.get(field)
        if isinstance(value, str) and value.strip():
            required.append(_searchable_source(value))
            break

    matches = [
        segment
        for segment in _source_segments(source)
        if all(term in _searchable_source(segment) for term in required)
    ]
    return min(matches, key=len) if matches else None


def _bind_missing_evidence(
    value: dict[str, object],
    pages: list[dict[str, object]],
) -> dict[str, object]:
    """Bind missing evidence fields to an exact fragment on the claimed page."""

    page_text = {
        page["page_number"]: page["markdown"]
        for page in pages
        if isinstance(page["page_number"], int) and isinstance(page["markdown"], str)
    }
    for category, subject_field in _FACT_SUBJECT_FIELDS.items():
        facts = value.get(category)
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            verbatim = fact.get("verbatim")
            excerpt = fact.get("evidence_excerpt")
            if (
                isinstance(verbatim, str)
                and verbatim.strip()
                and isinstance(excerpt, str)
                and excerpt.strip()
            ):
                continue
            source = page_text.get(fact.get("page_number"))
            if not isinstance(source, str):
                continue
            evidence = _find_exact_evidence(source, fact, subject_field)
            if evidence is None:
                continue
            if not isinstance(verbatim, str) or not verbatim.strip():
                fact["verbatim"] = evidence
            if not isinstance(excerpt, str) or not excerpt.strip():
                fact["evidence_excerpt"] = evidence
    return value


def _append_rejected_path(value: dict[str, object], path: str) -> None:
    rejected = value.get("rejected_fields")
    if not isinstance(rejected, list):
        rejected = []
        value["rejected_fields"] = rejected
    if path not in rejected:
        rejected.append(path)


def _filter_promotable_facts(value: dict[str, object]) -> dict[str, object]:
    """Keep only record-promotable facts and audit deterministic rejections."""

    for category in _ASSERTION_CATEGORIES:
        facts = value.get(category)
        if not isinstance(facts, list):
            continue
        accepted: list[object] = []
        for index, fact in enumerate(facts):
            assertion = fact.get("assertion") if isinstance(fact, dict) else None
            if assertion in _NON_PROMOTABLE_ASSERTIONS:
                _append_rejected_path(value, f"{category}[{index}]")
                continue
            accepted.append(fact)
        value[category] = accepted

    vital_facts = value.get("vital_signs")
    vital_keys: set[tuple[str, str, str, object]] = set()
    if isinstance(vital_facts, list):
        for fact in vital_facts:
            if not isinstance(fact, dict):
                continue
            vital_keys.add(
                (
                    _searchable_source(str(fact.get("name", ""))),
                    _searchable_source(str(fact.get("value", ""))),
                    _searchable_source(str(fact.get("unit", ""))),
                    fact.get("page_number"),
                )
            )
    lab_facts = value.get("labs")
    if isinstance(lab_facts, list):
        accepted_labs: list[object] = []
        for index, fact in enumerate(lab_facts):
            key = (
                (
                    _searchable_source(str(fact.get("name", ""))),
                    _searchable_source(str(fact.get("value", ""))),
                    _searchable_source(str(fact.get("unit", ""))),
                    fact.get("page_number"),
                )
                if isinstance(fact, dict)
                else None
            )
            if key is not None and key in vital_keys:
                _append_rejected_path(value, f"labs[{index}]")
                continue
            accepted_labs.append(fact)
        value["labs"] = accepted_labs

    report_facts = value.get("diagnostic_reports")
    if isinstance(report_facts, list):
        accepted_reports: list[object] = []
        for index, fact in enumerate(report_facts):
            if not isinstance(fact, dict):
                _append_rejected_path(value, f"diagnostic_reports[{index}]")
                continue
            details = (
                fact.get("findings"),
                fact.get("interpretation"),
            )
            context = " ".join(
                item
                for item in (
                    fact.get("name"),
                    fact.get("verbatim"),
                    fact.get("evidence_excerpt"),
                )
                if isinstance(item, str)
            )
            if (
                not any(isinstance(item, str) and item.strip() for item in details)
                or _DIAGNOSTIC_ANCHOR_RE.search(context) is None
            ):
                _append_rejected_path(value, f"diagnostic_reports[{index}]")
                continue
            accepted_reports.append(fact)
        value["diagnostic_reports"] = accepted_reports
    return value


def run_extraction(
    payload: Mapping[str, object],
    *,
    loaded: LoadedRole | None = None,
    generate_fn: Generate = generate_content,
) -> dict[str, object]:
    """Extract one schema-bounded clinical JSON document."""

    if (
        not _EXTRACTION_INPUT_KEYS.issubset(payload)
        or set(payload) - _EXTRACTION_INPUT_KEYS - _EXTRACTION_TRANSPORT_KEYS
    ):
        raise WorkerInputError("Extraction request is invalid.")
    schema = payload.get("schema")
    if not isinstance(schema, dict) or not schema:
        raise WorkerInputError("Extraction schema is invalid.")
    pages = _source_pages(payload.get("page_markdown"))
    images = _selected_images(
        payload.get("image_paths"),
        {int(page["page_number"]) for page in pages},
        payload.get("scratch_dir"),
    )
    selected = loaded or load_role_from_payload("extraction", payload)
    max_tokens = requested_output_tokens(payload, selected, role_cap=EXTRACTION_OUTPUT_CAP)
    template = bounded_json(schema, max_bytes=MAX_EXTRACTION_INPUT_BYTES)
    instructions = (
        "Extract only values present in the document. Use JSON null only for optional "
        "text fields without evidence, and use empty lists for absent categories. Never "
        "use JSON null for enum-valued fields. Use unknown for status enums, uncertain "
        "for assertion enums, and other for an unestablished category or visit type. "
        "Set assertion to negated for explicit no, denies, absent, or negative evidence. "
        "Set assertion to family_history only for explicit family-history context. Set "
        "procedure assertion to mentioned_not_performed for planned, cancelled, deferred, "
        "or not-done procedures. Use uncertain for explicit possible, suspected, or unclear "
        "evidence; otherwise use present only when the source affirms the fact. "
        "Preserve verbatim clinical values."
    )

    for retry in (False, True):
        prompt = _prompt(pages, retry=retry)
        validate_token_budget(
            selected,
            [prompt, template, instructions],
            max_output_tokens=max_tokens,
        )
        raw = generate_fn(
            model=selected.model,
            processor=selected.processor,
            prompt=prompt,
            images=images,
            max_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            input_token_limit=selected.decode_limits["max_input_tokens"],
            enable_thinking=False,
            template=template,
            mode="structured",
            instructions=instructions,
        )
        try:
            value = parse_json_object(raw)
        except GenerationError:
            continue
        value = _discard_missing_fact_subjects(value)
        value = _bind_missing_evidence(value, pages)
        value = _bind_missing_adjacent_units(value)
        value = _ground_explicit_assertions(value)
        value = _ground_lifecycle_statuses(value)
        return _filter_promotable_facts(value)
    raise GenerationError("Local extraction returned invalid JSON.")
