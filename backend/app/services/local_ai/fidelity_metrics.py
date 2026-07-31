"""Exact, content-free fidelity scoring for strict-local release gates."""

from __future__ import annotations

import html
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

_REPORT_KEYS = frozenset(
    {
        "critical_numeric_exact",
        "critical_precision",
        "critical_recall",
        "accepted_output_schema_validity",
        "forbidden_extraction_facts",
        "unsupported_summary_facts",
        "accepted_facts_without_evidence",
        "summary_fact_recall",
        "summary_typed_field_recall",
    }
)
_RELEASE_THRESHOLDS = {
    "critical_numeric_exact_min": 0.99,
    "critical_precision_min": 0.98,
    "critical_recall_min": 0.95,
    "accepted_output_schema_validity_min": 1.0,
    "forbidden_extraction_facts_max": 0,
    "unsupported_summary_facts_max": 0,
    "accepted_facts_without_evidence_max": 0,
    "summary_fact_recall_min": 1.0,
    "summary_typed_field_recall_min": 1.0,
}


def release_thresholds() -> dict[str, int | float]:
    """Return the exact fidelity policy that release evidence must bind."""

    return dict(_RELEASE_THRESHOLDS)


class FidelityGateError(RuntimeError):
    """A local-model fixture run missed a clinical fidelity hard gate."""


@dataclass(frozen=True)
class FidelityFactObservation:
    """Expected, prohibited, and accepted facts for one source document."""

    expected: tuple[Mapping[str, object], ...]
    forbidden: tuple[Mapping[str, object], ...]
    accepted: tuple[Mapping[str, object], ...]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _normalized(value: object) -> object:
    if isinstance(value, str):
        decoded = html.unescape(value)
        return " ".join(unicodedata.normalize("NFKC", decoded).split()).casefold()
    if isinstance(value, list):
        return tuple(_normalized(item) for item in value)
    return value


def _contains_exact(source: str, expected: str) -> bool:
    normalized_source = _normalized(source)
    normalized_expected = _normalized(expected)
    if not isinstance(normalized_source, str) or not isinstance(
        normalized_expected, str
    ):
        return False
    pattern = re.compile(
        rf"(?<![\w.]){re.escape(normalized_expected)}(?!\w)(?!\.\d)",
        re.UNICODE,
    )
    return pattern.search(normalized_source) is not None


def _fact_matches(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> bool:
    return all(
        key in actual and _normalized(actual[key]) == _normalized(value)
        for key, value in expected.items()
    )


@dataclass(frozen=True)
class FidelityMetrics:
    """Only aggregate counts/ratios; no OCR, fact, evidence, or summary content."""

    critical_numeric_exact: float
    critical_precision: float
    critical_recall: float
    accepted_output_schema_validity: float
    forbidden_extraction_facts: int
    unsupported_summary_facts: int
    accepted_facts_without_evidence: int
    summary_fact_recall: float = 1.0
    summary_typed_field_recall: float = 1.0

    def __post_init__(self) -> None:
        for value in (
            self.critical_numeric_exact,
            self.critical_precision,
            self.critical_recall,
            self.accepted_output_schema_validity,
            self.summary_fact_recall,
            self.summary_typed_field_recall,
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise FidelityGateError("Fidelity report is invalid.")
        for value in (
            self.forbidden_extraction_facts,
            self.unsupported_summary_facts,
            self.accepted_facts_without_evidence,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise FidelityGateError("Fidelity report is invalid.")

    @classmethod
    def from_report(cls, value: object) -> FidelityMetrics:
        if type(value) is not dict or set(value) != _REPORT_KEYS:
            raise FidelityGateError("Fidelity report is invalid.")
        try:
            return cls(**value)
        except TypeError as exc:
            raise FidelityGateError("Fidelity report is invalid.") from exc

    def as_report(self) -> dict[str, int | float]:
        return asdict(self)

    def assert_release_thresholds(self) -> None:
        thresholds = _RELEASE_THRESHOLDS
        if self.critical_numeric_exact < thresholds["critical_numeric_exact_min"]:
            raise FidelityGateError("Critical numeric OCR exactness is below 99%.")
        if self.critical_precision < thresholds["critical_precision_min"]:
            raise FidelityGateError("Critical extraction precision is below 98%.")
        if self.critical_recall < thresholds["critical_recall_min"]:
            raise FidelityGateError("Critical extraction recall is below 95%.")
        if (
            self.accepted_output_schema_validity
            < thresholds["accepted_output_schema_validity_min"]
        ):
            raise FidelityGateError("Accepted-output schema validity is below 100%.")
        if (
            self.forbidden_extraction_facts
            > thresholds["forbidden_extraction_facts_max"]
        ):
            raise FidelityGateError("Extraction output contains a forbidden fact.")
        if self.unsupported_summary_facts > thresholds["unsupported_summary_facts_max"]:
            raise FidelityGateError("Summary output contains unsupported facts.")
        if (
            self.accepted_facts_without_evidence
            > thresholds["accepted_facts_without_evidence_max"]
        ):
            raise FidelityGateError("Accepted extraction facts are missing evidence.")
        if self.summary_fact_recall < thresholds["summary_fact_recall_min"]:
            raise FidelityGateError("Summary fact selection recall is below 100%.")
        if (
            self.summary_typed_field_recall
            < thresholds["summary_typed_field_recall_min"]
        ):
            raise FidelityGateError(
                "Summary typed-field selection recall is below 100%."
            )


def score_fidelity(
    *,
    numeric_observations: Sequence[tuple[str, str]],
    fact_observations: Sequence[FidelityFactObservation],
    summary_claims: Sequence[Mapping[str, object]],
    schema_results: Sequence[bool],
) -> FidelityMetrics:
    """Compute exact hard-gate metrics without retaining source content."""

    numeric_hits = sum(
        _contains_exact(ocr_source, token) for token, ocr_source in numeric_observations
    )

    matched = 0
    expected_count = 0
    forbidden_count = 0
    accepted_facts: list[Mapping[str, object]] = []
    matched_accepted_facts: list[Mapping[str, object]] = []
    for observation in fact_observations:
        unmatched_expected = set(range(len(observation.expected)))
        expected_count += len(observation.expected)
        accepted_facts.extend(observation.accepted)
        for actual in observation.accepted:
            if any(
                _fact_matches(forbidden, actual) for forbidden in observation.forbidden
            ):
                forbidden_count += 1
            match = next(
                (
                    index
                    for index in sorted(unmatched_expected)
                    if _fact_matches(observation.expected[index], actual)
                ),
                None,
            )
            if match is not None:
                unmatched_expected.remove(match)
                matched += 1
                matched_accepted_facts.append(actual)

    accepted_by_id: dict[str, frozenset[str]] = {}
    facts_without_evidence = 0
    for fact in accepted_facts:
        fact_id = fact.get("fact_id")
        evidence = fact.get("evidence_ids")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(item, str) and item for item in evidence)
        ):
            facts_without_evidence += 1
            evidence_set = frozenset()
        else:
            evidence_set = frozenset(evidence)
        if isinstance(fact_id, str) and fact_id:
            accepted_by_id[fact_id] = evidence_set

    unsupported_claims = 0
    selected_fact_ids: set[str] = set()
    selected_field_paths: dict[str, set[str]] = {}
    for claim in summary_claims:
        fact_id = claim.get("fact_id")
        evidence = claim.get("evidence_ids")
        field_paths = claim.get("field_paths")
        if (
            not isinstance(fact_id, str)
            or fact_id not in accepted_by_id
            or not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(item, str) and item for item in evidence)
            or not frozenset(evidence).issubset(accepted_by_id[fact_id])
            or not isinstance(field_paths, list)
            or not field_paths
            or not all(isinstance(item, str) and item for item in field_paths)
        ):
            unsupported_claims += 1
            continue
        selected_fact_ids.add(fact_id)
        selected_field_paths.setdefault(fact_id, set()).update(field_paths)

    expected_summary_fact_ids = {
        fact_id
        for fact in matched_accepted_facts
        if isinstance((fact_id := fact.get("fact_id")), str) and fact_id
    }
    summary_fact_hits = len(expected_summary_fact_ids.intersection(selected_fact_ids))
    expected_typed_fields = {
        (fact_id, path)
        for fact in matched_accepted_facts
        if isinstance((fact_id := fact.get("fact_id")), str) and fact_id
        for path in fact.get("_summary_required_field_paths", ())
        if isinstance(path, str) and path
    }
    selected_typed_fields = {
        (fact_id, path)
        for fact_id, paths in selected_field_paths.items()
        for path in paths
        if (fact_id, path) in expected_typed_fields
    }

    valid_schema_results = sum(item is True for item in schema_results)
    return FidelityMetrics(
        critical_numeric_exact=_ratio(
            numeric_hits,
            len(numeric_observations),
        ),
        critical_precision=_ratio(matched, len(accepted_facts)),
        critical_recall=_ratio(matched, expected_count),
        accepted_output_schema_validity=_ratio(
            valid_schema_results,
            len(schema_results),
        ),
        forbidden_extraction_facts=forbidden_count,
        unsupported_summary_facts=unsupported_claims,
        accepted_facts_without_evidence=facts_without_evidence,
        summary_fact_recall=(
            _ratio(summary_fact_hits, len(matched_accepted_facts))
            if matched_accepted_facts
            else 1.0
        ),
        summary_typed_field_recall=(
            _ratio(len(selected_typed_fields), len(expected_typed_fields))
            if expected_typed_fields
            else 1.0
        ),
    )
