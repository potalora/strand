"""Tests for strict, evidence-grounded local summary validation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.grounded_summary import (
    SERVER_MEDICAL_DISCLAIMER,
    GroundedSummaryFact,
    GroundedSummaryInput,
    _canonical_json_value,
    _flatten_content,
    _stable_fact_id,
    _stable_uncertainty_id,
    build_maximal_reference_document,
    build_grounded_summary_input,
    validate_and_render_summary,
)


@pytest.fixture
def summary_input() -> GroundedSummaryInput:
    return build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "medication",
                    "name": "Metformin",
                    "dose": "500 mg",
                    "status": "active",
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Metformin 500 mg active",
                "page_number": 2,
                "section": "Medications",
                "field_paths": ["/name", "/dose", "/status"],
            }
        ],
        requested_scope={
            "summary_type": "category",
            "category": "medication",
        },
        uncertainty_labels=[
            {
                "template_id": "medication_end_date_missing",
                "record_ids": ["record-1"],
                "evidence_ids": ["source-evidence-1"],
            }
        ],
    )


def _registries(
    summary_input: GroundedSummaryInput,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        {fact.fact_id: fact for fact in summary_input.facts},
        {item.evidence_id: item for item in summary_input.evidence},
        {item.uncertainty_id: item for item in summary_input.uncertainty_labels},
    )


def _valid_output(summary_input: GroundedSummaryInput) -> dict[str, object]:
    fact = summary_input.facts[0]
    evidence = summary_input.evidence[0]
    uncertainty = summary_input.uncertainty_labels[0]
    return {
        "sections": [
            {
                "heading": "Medications",
                "claims": [
                    {
                        "fact_id": fact.fact_id,
                        "field_paths": ["/name", "/dose", "/status"],
                        "evidence_ids": [evidence.evidence_id],
                    }
                ],
            }
        ],
        "uncertainties": [
            {
                "uncertainty_id": uncertainty.uncertainty_id,
                "fact_ids": [fact.fact_id],
                "evidence_ids": [evidence.evidence_id],
            }
        ],
    }


def test_maximal_reference_document_is_complete_and_deterministic(
    summary_input: GroundedSummaryInput,
) -> None:
    first = build_maximal_reference_document(summary_input)
    second = build_maximal_reference_document(summary_input)
    fact = summary_input.facts[0]
    evidence = summary_input.evidence[0]
    uncertainty = summary_input.uncertainty_labels[0]

    assert first == second
    assert first.model_dump(mode="json") == {
        "sections": [
            {
                "heading": "Medications",
                "claims": [
                    {
                        "fact_id": fact.fact_id,
                        "field_paths": ["/dose", "/name", "/status"],
                        "evidence_ids": [evidence.evidence_id],
                    }
                ],
            }
        ],
        "uncertainties": [
            {
                "uncertainty_id": uncertainty.uncertainty_id,
                "fact_ids": [fact.fact_id],
                "evidence_ids": [evidence.evidence_id],
            }
        ],
    }


def test_model_cannot_supply_free_text_claims_headings_or_uncertainties(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    valid = _valid_output(summary_input)
    fabricated_claim = json.loads(json.dumps(valid))
    fabricated_claim["sections"][0]["claims"][0]["text"] = (  # type: ignore[index]
        "Insulin is active and pancreatic cancer is present."
    )
    fabricated_heading = json.loads(json.dumps(valid))
    fabricated_heading["sections"][0]["heading"] = (  # type: ignore[index]
        "Findings point to pancreatic cancer"
    )
    fabricated_uncertainty = json.loads(json.dumps(valid))
    fabricated_uncertainty["uncertainties"][0]["text"] = (  # type: ignore[index]
        "Consider doubling the medication dose."
    )

    for raw in (fabricated_claim, fabricated_heading, fabricated_uncertainty):
        with pytest.raises(LocalValidationError, match="invalid structure"):
            validate_and_render_summary(
                raw,
                facts=facts,
                evidence=evidence,
                uncertainties=uncertainties,
            )


def test_server_renders_only_selected_validated_fact_fields(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)

    rendered = validate_and_render_summary(
        _valid_output(summary_input),
        facts=facts,
        evidence=evidence,
        uncertainties=uncertainties,
    )

    assert "- Name: Metformin; Dose: 500 mg; Status: active." in rendered.markdown
    assert "Insulin" not in rendered.markdown


def test_summary_rejects_duplicate_section_headings_globally(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    raw = _valid_output(summary_input)
    raw["sections"].append(  # type: ignore[union-attr]
        {
            "heading": "Medications",
            "claims": [],
        }
    )

    with pytest.raises(LocalValidationError, match="invalid structure"):
        validate_and_render_summary(
            raw,
            facts=facts,
            evidence=evidence,
            uncertainties=uncertainties,
        )


def test_summary_rejects_a_fact_claimed_more_than_once_globally(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    raw = _valid_output(summary_input)
    repeated_claim = json.loads(json.dumps(raw["sections"][0]["claims"][0]))  # type: ignore[index]
    raw["sections"].append(  # type: ignore[union-attr]
        {
            "heading": "Overview",
            "claims": [repeated_claim],
        }
    )

    with pytest.raises(LocalValidationError, match="invalid structure"):
        validate_and_render_summary(
            raw,
            facts=facts,
            evidence=evidence,
            uncertainties=uncertainties,
        )


def test_server_rejects_disclaimer_only_output_when_grounded_facts_exist(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)

    with pytest.raises(LocalValidationError, match="no grounded claims"):
        validate_and_render_summary(
            {"sections": [], "uncertainties": []},
            facts=facts,
            evidence=evidence,
            uncertainties=uncertainties,
        )


def test_server_renders_assertion_companion_when_model_selects_only_diagnosis() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "condition-1",
                "content": {
                    "record_type": "condition",
                    "diagnosis": "Type 2 diabetes",
                    "assertion": "negated",
                },
                "evidence_ids": ["diagnosis-evidence", "assertion-evidence"],
            }
        ],
        evidence=[
            {
                "id": "diagnosis-evidence",
                "excerpt": "Type 2 diabetes",
                "field_paths": ["/diagnosis"],
            },
            {
                "id": "assertion-evidence",
                "excerpt": "No evidence of diabetes",
                "field_paths": ["/assertion"],
            },
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    diagnosis_evidence, assertion_evidence = summary_input.evidence

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Conditions",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": ["/diagnosis"],
                            "evidence_ids": [diagnosis_evidence.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={
            diagnosis_evidence.evidence_id: diagnosis_evidence,
            assertion_evidence.evidence_id: assertion_evidence,
        },
    )

    assert "Diagnosis: Type 2 diabetes; Assertion: negated." in rendered.markdown
    assert assertion_evidence.evidence_id in rendered.markdown
    assert rendered.document.sections[0].claims[0].field_paths == (
        "/diagnosis",
        "/assertion",
    )


def test_server_renders_applicable_uncertainty_when_model_omits_it() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "medication-1",
                "content": {
                    "record_type": "medication",
                    "name": "Metformin",
                },
                "evidence_ids": ["medication-evidence"],
            }
        ],
        evidence=[
            {
                "id": "medication-evidence",
                "excerpt": "Metformin",
                "field_paths": ["/name"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
        uncertainty_labels=[
            {
                "template_id": "medication_dose_missing",
                "record_ids": ["medication-1"],
                "evidence_ids": ["medication-evidence"],
            }
        ],
    )
    fact = summary_input.facts[0]
    evidence_item = summary_input.evidence[0]
    uncertainty = summary_input.uncertainty_labels[0]

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Medications",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": ["/name"],
                            "evidence_ids": [evidence_item.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={evidence_item.evidence_id: evidence_item},
        uncertainties={uncertainty.uncertainty_id: uncertainty},
    )

    assert "Medication dose is not available." in rendered.markdown
    assert rendered.document.uncertainties[0].uncertainty_id == (
        uncertainty.uncertainty_id
    )


def test_server_renders_family_history_relationship_companion() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "condition-1",
                "content": {
                    "record_type": "condition",
                    "diagnosis": "Type 2 diabetes",
                    "assertion": "family_history",
                    "relationship": "mother",
                },
                "evidence_ids": [
                    "diagnosis-evidence",
                    "assertion-evidence",
                    "relationship-evidence",
                ],
            }
        ],
        evidence=[
            {
                "id": "diagnosis-evidence",
                "excerpt": "Type 2 diabetes",
                "field_paths": ["/diagnosis"],
            },
            {
                "id": "assertion-evidence",
                "excerpt": "Family history",
                "field_paths": ["/assertion"],
            },
            {
                "id": "relationship-evidence",
                "excerpt": "Mother",
                "field_paths": ["/relationship"],
            },
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    evidence_registry = {item.evidence_id: item for item in summary_input.evidence}
    diagnosis_evidence = summary_input.evidence[0]

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Conditions",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": ["/diagnosis"],
                            "evidence_ids": [diagnosis_evidence.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence=evidence_registry,
    )

    assert (
        "Diagnosis: Type 2 diabetes; Assertion: family\\_history; Relationship: mother."
        in rendered.markdown
    )
    assert rendered.document.sections[0].claims[0].field_paths == (
        "/diagnosis",
        "/assertion",
        "/relationship",
    )


def test_builder_rejects_family_history_without_relationship() -> None:
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "condition-1",
                    "content": {
                        "record_type": "condition",
                        "diagnosis": "Type 2 diabetes",
                        "assertion": "family_history",
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Family history of diabetes",
                    "field_paths": ["/diagnosis", "/assertion"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )


@pytest.mark.parametrize(
    ("record_type", "content", "selected_path", "paired_path"),
    [
        (
            "observation",
            {"name": "HbA1c", "value": "6.5", "unit": "%"},
            "/value",
            "/unit",
        ),
        (
            "medication",
            {"name": "Metformin", "dose_value": "500", "dose_unit": "mg"},
            "/dose_value",
            "/dose_unit",
        ),
        (
            "medication",
            {"name": "Metformin", "dosage": {"amount": 500, "unit": "mg"}},
            "/dosage/amount",
            "/dosage/unit",
        ),
    ],
)
def test_server_renders_numeric_unit_companions(
    record_type: str,
    content: dict[str, object],
    selected_path: str,
    paired_path: str,
) -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {"record_type": record_type, **content},
                "evidence_ids": ["value-evidence", "unit-evidence"],
            }
        ],
        evidence=[
            {
                "id": "value-evidence",
                "excerpt": "Numeric value",
                "field_paths": [selected_path],
            },
            {
                "id": "unit-evidence",
                "excerpt": "Unit",
                "field_paths": [paired_path],
            },
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    value_evidence, unit_evidence = summary_input.evidence

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": (
                        "Observations"
                        if record_type == "observation"
                        else "Medications"
                    ),
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": [selected_path],
                            "evidence_ids": [value_evidence.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={
            value_evidence.evidence_id: value_evidence,
            unit_evidence.evidence_id: unit_evidence,
        },
    )

    claim = rendered.document.sections[0].claims[0]
    assert claim.field_paths == (selected_path, paired_path)
    assert unit_evidence.evidence_id in claim.evidence_ids
    assert ("%" if record_type == "observation" else "mg") in rendered.markdown


def test_server_renders_numeric_value_when_model_selects_only_unit() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "observation-1",
                "content": {
                    "record_type": "observation",
                    "name": "HbA1c",
                    "value": "6.5",
                    "unit": "%",
                },
                "evidence_ids": ["value-evidence", "unit-evidence"],
            }
        ],
        evidence=[
            {
                "id": "value-evidence",
                "excerpt": "6.5",
                "field_paths": ["/value"],
            },
            {
                "id": "unit-evidence",
                "excerpt": "%",
                "field_paths": ["/unit"],
            },
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    value_evidence, unit_evidence = summary_input.evidence

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Observations",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": ["/unit"],
                            "evidence_ids": [unit_evidence.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={
            value_evidence.evidence_id: value_evidence,
            unit_evidence.evidence_id: unit_evidence,
        },
    )

    assert rendered.document.sections[0].claims[0].field_paths == (
        "/unit",
        "/value",
    )
    assert value_evidence.evidence_id in rendered.markdown


@pytest.mark.parametrize("value", ["<5", ">200", "≤5", "≥10", "~5"])
def test_server_renders_comparator_numeric_value_with_unit_companion(
    value: str,
) -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "observation-1",
                "content": {
                    "record_type": "observation",
                    "name": "Bounded result",
                    "value": value,
                    "unit": "mg",
                },
                "evidence_ids": ["value-evidence", "unit-evidence"],
            }
        ],
        evidence=[
            {
                "id": "value-evidence",
                "excerpt": value,
                "field_paths": ["/value"],
            },
            {
                "id": "unit-evidence",
                "excerpt": "mg",
                "field_paths": ["/unit"],
            },
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    value_evidence, unit_evidence = summary_input.evidence

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Observations",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": ["/value"],
                            "evidence_ids": [value_evidence.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={
            value_evidence.evidence_id: value_evidence,
            unit_evidence.evidence_id: unit_evidence,
        },
    )

    claim = rendered.document.sections[0].claims[0]
    assert claim.field_paths == ("/value", "/unit")
    assert unit_evidence.evidence_id in claim.evidence_ids


@pytest.mark.parametrize(
    ("value", "expected_text"),
    [
        (
            {
                "kind": "quantity",
                "comparator": "<",
                "number": 0.05,
                "unit": "mIU/L",
            },
            "<0.05 mIU/L",
        ),
        (
            {
                "kind": "ratio",
                "numerator": 120,
                "denominator": 80,
                "numerator_unit": "mmHg",
                "denominator_unit": "mmHg",
            },
            "120/80 mmHg",
        ),
        (
            {
                "kind": "ratio",
                "numerator": 30,
                "denominator": 1,
                "numerator_unit": "mg",
                "denominator_unit": "g",
            },
            "30 mg/1 g",
        ),
    ],
)
def test_server_renders_typed_observation_value_as_one_evidence_bound_measurement(
    value: dict[str, object],
    expected_text: str,
) -> None:
    value_paths = tuple(f"/value/{key}" for key in sorted(value))
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "observation-1",
                "content": {
                    "record_type": "observation",
                    "name": "Measured result",
                    "value": value,
                },
                "evidence_ids": ["measurement-evidence"],
            }
        ],
        evidence=[
            {
                "id": "measurement-evidence",
                "excerpt": expected_text,
                "field_paths": list(value_paths),
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    evidence_item = summary_input.evidence[0]

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Observations",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": [value_paths[0]],
                            "evidence_ids": [evidence_item.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={evidence_item.evidence_id: evidence_item},
    )

    assert set(rendered.document.sections[0].claims[0].field_paths) == set(value_paths)
    assert rendered.selection_document.sections[0].claims[0].field_paths == (
        value_paths[0],
    )
    expected_rendered = expected_text.replace("<", "\\<")
    assert f"Value: {expected_rendered}" in rendered.markdown


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "quantity", "comparator": "<", "number": 0.05},
        {
            "kind": "quantity",
            "comparator": "!=",
            "number": 0.05,
            "unit": "mIU/L",
        },
        {
            "kind": "ratio",
            "numerator": 120,
            "denominator": 0,
            "numerator_unit": "mmHg",
            "denominator_unit": "mmHg",
        },
        {
            "kind": "ratio",
            "numerator": 120,
            "denominator": 80,
            "unit": "mmHg",
        },
        {
            "kind": "ratio",
            "numerator": 120,
            "denominator": 80,
            "numerator_unit": "mmHg",
            "denominator_unit": "mmHg",
            "diagnosis": "hypertension",
        },
    ],
)
def test_builder_rejects_incomplete_or_non_allowlisted_typed_observation_values(
    value: dict[str, object],
) -> None:
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "observation-1",
                    "content": {
                        "record_type": "observation",
                        "name": "Measured result",
                        "value": value,
                    },
                    "evidence_ids": ["measurement-evidence"],
                }
            ],
            evidence=[
                {
                    "id": "measurement-evidence",
                    "excerpt": "Measured result",
                    "field_paths": ["/value/kind"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )


@pytest.mark.parametrize("kind", ["quantity", "ratio"])
@pytest.mark.parametrize(("unit_length", "accepted"), [(512, True), (513, False)])
def test_typed_observation_unit_bound_is_exactly_512_characters(
    kind: str,
    unit_length: int,
    accepted: bool,
) -> None:
    unit = "u" * unit_length
    value: dict[str, object]
    if kind == "quantity":
        value = {
            "kind": "quantity",
            "comparator": "<",
            "number": 0.05,
            "unit": unit,
        }
        paths = ["/value/comparator", "/value/kind", "/value/number", "/value/unit"]
    else:
        value = {
            "kind": "ratio",
            "numerator": 120,
            "denominator": 80,
            "numerator_unit": unit,
            "denominator_unit": unit,
        }
        paths = [
            "/value/denominator",
            "/value/denominator_unit",
            "/value/kind",
            "/value/numerator",
            "/value/numerator_unit",
        ]

    def build() -> GroundedSummaryInput:
        return build_grounded_summary_input(
            facts=[
                {
                    "record_id": "observation-1",
                    "content": {
                        "record_type": "observation",
                        "name": "Measured result",
                        "value": value,
                    },
                    "evidence_ids": ["measurement-evidence"],
                }
            ],
            evidence=[
                {
                    "id": "measurement-evidence",
                    "excerpt": "Measured result",
                    "field_paths": paths,
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )

    if accepted:
        assert build().facts[0].content_json
    else:
        with pytest.raises(LocalValidationError, match="invalid fact input"):
            build()


@pytest.mark.parametrize("value", ["<5", ">200", "≤5", "≥10", "~5"])
def test_render_rejects_bare_comparator_numeric_value_without_unit(
    value: str,
) -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "observation-1",
                "content": {
                    "record_type": "observation",
                    "name": "Bounded result",
                    "value": value,
                },
                "evidence_ids": ["value-evidence"],
            }
        ],
        evidence=[
            {
                "id": "value-evidence",
                "excerpt": value,
                "field_paths": ["/value"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    value_evidence = summary_input.evidence[0]

    with pytest.raises(LocalValidationError, match="unit"):
        validate_and_render_summary(
            {
                "sections": [
                    {
                        "heading": "Observations",
                        "claims": [
                            {
                                "fact_id": fact.fact_id,
                                "field_paths": ["/value"],
                                "evidence_ids": [value_evidence.evidence_id],
                            }
                        ],
                    }
                ],
                "uncertainties": [],
            },
            facts={fact.fact_id: fact},
            evidence={value_evidence.evidence_id: value_evidence},
        )


def test_render_rejects_bare_numeric_value_without_unit() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "observation-1",
                "content": {
                    "record_type": "observation",
                    "name": "HbA1c",
                    "value": "6.5",
                },
                "evidence_ids": ["value-evidence"],
            }
        ],
        evidence=[
            {
                "id": "value-evidence",
                "excerpt": "6.5",
                "field_paths": ["/value"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    value_evidence = summary_input.evidence[0]

    with pytest.raises(LocalValidationError, match="unit"):
        validate_and_render_summary(
            {
                "sections": [
                    {
                        "heading": "Observations",
                        "claims": [
                            {
                                "fact_id": fact.fact_id,
                                "field_paths": ["/value"],
                                "evidence_ids": [value_evidence.evidence_id],
                            }
                        ],
                    }
                ],
                "uncertainties": [],
            },
            facts={fact.fact_id: fact},
            evidence={value_evidence.evidence_id: value_evidence},
        )


@pytest.mark.parametrize(
    ("value", "rendered_value"),
    [
        ("positive", "positive"),
        ("negative", "negative"),
        ("detected", "detected"),
        (True, "true"),
    ],
)
def test_server_renders_qualitative_observation_value_without_unit(
    value: object,
    rendered_value: str,
) -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "observation-1",
                "content": {
                    "record_type": "observation",
                    "name": "Qualitative result",
                    "value": value,
                },
                "evidence_ids": ["value-evidence"],
            }
        ],
        evidence=[
            {
                "id": "value-evidence",
                "excerpt": rendered_value,
                "field_paths": ["/value"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    value_evidence = summary_input.evidence[0]

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Observations",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": ["/value"],
                            "evidence_ids": [value_evidence.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={value_evidence.evidence_id: value_evidence},
    )

    assert rendered.document.sections[0].claims[0].field_paths == ("/value",)
    assert f"Value: {rendered_value}" in rendered.markdown


def test_model_cannot_place_a_fact_under_a_misleading_heading(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    raw = _valid_output(summary_input)
    raw["sections"][0]["heading"] = "Conditions"  # type: ignore[index]

    with pytest.raises(LocalValidationError, match="section"):
        validate_and_render_summary(
            raw,
            facts=facts,
            evidence=evidence,
            uncertainties=uncertainties,
        )


def test_record_attributed_diagnosis_is_rendered_without_phrase_blacklist() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "condition-1",
                "content": {
                    "record_type": "condition",
                    "diagnosis": "Type 2 diabetes mellitus",
                    "status": "active",
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Active diagnosis: Type 2 diabetes mellitus",
                "page_number": 1,
                "field_paths": ["/diagnosis", "/status"],
            }
        ],
        requested_scope={"summary_type": "category", "category": "condition"},
    )
    fact = summary_input.facts[0]
    evidence_item = summary_input.evidence[0]

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Conditions",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": ["/diagnosis", "/status"],
                            "evidence_ids": [evidence_item.evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={evidence_item.evidence_id: evidence_item},
        uncertainties={},
    )

    assert "Diagnosis: Type 2 diabetes mellitus; Status: active." in rendered.markdown


def test_claim_rejects_unknown_fact_field_and_evidence_ids(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    valid = _valid_output(summary_input)
    unknown_fact = json.loads(json.dumps(valid))
    unknown_fact["sections"][0]["claims"][0]["fact_id"] = "fact-unknown"  # type: ignore[index]
    unknown_field = json.loads(json.dumps(valid))
    unknown_field["sections"][0]["claims"][0]["field_paths"] = ["/insulin"]  # type: ignore[index]
    unknown_evidence = json.loads(json.dumps(valid))
    unknown_evidence["sections"][0]["claims"][0]["evidence_ids"] = [  # type: ignore[index]
        "evidence-unknown"
    ]

    cases = (
        (unknown_fact, "unknown fact"),
        (unknown_field, "unknown fact field"),
        (unknown_evidence, "unknown evidence"),
    )
    for raw, message in cases:
        with pytest.raises(LocalValidationError, match=message):
            validate_and_render_summary(
                raw,
                facts=facts,
                evidence=evidence,
                uncertainties=uncertainties,
            )


def test_every_claim_requires_fact_fields_and_evidence_support(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    fact_id = summary_input.facts[0].fact_id
    raw = {
        "sections": [
            {
                "heading": "Overview",
                "claims": [
                    {
                        "fact_id": fact_id,
                        "field_paths": [],
                        "evidence_ids": [],
                    }
                ],
            }
        ],
        "uncertainties": [],
    }

    with pytest.raises(LocalValidationError, match="support"):
        validate_and_render_summary(
            raw,
            facts=facts,
            evidence=evidence,
            uncertainties=uncertainties,
        )


def test_claim_evidence_must_be_linked_to_its_fact(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    fact_id = summary_input.facts[0].fact_id
    evidence["evidence1_other"] = {
        "evidence_id": "evidence1_other",
        "source_id": "other",
        "excerpt": "Other evidence",
        "page_number": 1,
        "section": None,
        "fact_ids": [fact_id],
        "field_paths": ["/name"],
    }
    raw = _valid_output(summary_input)
    raw["sections"][0]["claims"][0]["evidence_ids"] = ["evidence1_other"]  # type: ignore[index]

    with pytest.raises(LocalValidationError, match="support"):
        validate_and_render_summary(
            raw,
            facts=facts,
            evidence=evidence,
            uncertainties=uncertainties,
        )


def test_uncertainty_must_match_server_owned_support(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    raw = _valid_output(summary_input)
    raw["uncertainties"][0]["fact_ids"] = ["fact-unknown"]  # type: ignore[index]

    with pytest.raises(LocalValidationError, match="uncertainty support"):
        validate_and_render_summary(
            raw,
            facts=facts,
            evidence=evidence,
            uncertainties=uncertainties,
        )


def test_validation_rechecks_uncertainty_applicability_against_canonical_facts() -> (
    None
):
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "medication",
                    "name": "Metformin",
                    "status": "active",
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Metformin active",
                "field_paths": ["/name", "/status"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    evidence_item = summary_input.evidence[0]
    uncertainty_id = _stable_uncertainty_id(
        "record_status_missing",
        (fact.fact_id,),
        (evidence_item.evidence_id,),
    )

    with pytest.raises(LocalValidationError, match="uncertainty"):
        validate_and_render_summary(
            {
                "sections": [],
                "uncertainties": [
                    {
                        "uncertainty_id": uncertainty_id,
                        "fact_ids": [fact.fact_id],
                        "evidence_ids": [evidence_item.evidence_id],
                    }
                ],
            },
            facts={fact.fact_id: fact},
            evidence={evidence_item.evidence_id: evidence_item},
            uncertainties={
                uncertainty_id: {
                    "uncertainty_id": uncertainty_id,
                    "template_id": "record_status_missing",
                    "label": "Record status is not available.",
                    "fact_ids": [fact.fact_id],
                    "evidence_ids": [evidence_item.evidence_id],
                }
            },
        )


def test_rendering_is_deterministic_with_stable_citations(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    raw = _valid_output(summary_input)
    fact_id = summary_input.facts[0].fact_id
    evidence_id = summary_input.evidence[0].evidence_id
    expected = (
        "## Medications\n\n"
        "- Name: Metformin; Dose: 500 mg; Status: active. "
        f"[Fact: {fact_id}; Evidence: {evidence_id}]\n\n"
        "## Uncertainties\n\n"
        "- Medication end date is not available. "
        f"[Facts: {fact_id}; Evidence: {evidence_id}]\n\n"
        f"{SERVER_MEDICAL_DISCLAIMER}"
    )

    first = validate_and_render_summary(
        raw,
        facts=facts,
        evidence=evidence,
        uncertainties=uncertainties,
    )
    second = validate_and_render_summary(
        json.dumps(raw),
        facts=facts,
        evidence=evidence,
        uncertainties=uncertainties,
    )

    assert first.markdown == expected
    assert second.markdown == expected
    assert first.document == second.document


def test_builder_creates_deeply_immutable_canonical_snapshots() -> None:
    nested_content: dict[str, Any] = {
        "record_type": "medication",
        "dosage": {"amount": 500, "unit": "mg"},
        "statuses": ["active"],
    }
    source_evidence: dict[str, Any] = {
        "id": "source-evidence-1",
        "excerpt": "Metformin 500 mg",
        "page_number": 1,
        "section": "Medications",
        "field_paths": ["/dosage/amount", "/dosage/unit"],
    }
    first = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": nested_content,
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[source_evidence],
        requested_scope={"summary_type": "full_health"},
    )
    before = first.model_dump_json()
    nested_content["dosage"]["amount"] = 1000
    nested_content["statuses"].append("stopped")
    source_evidence["excerpt"] = "Tampered"

    assert first.model_dump_json() == before
    assert isinstance(first.facts, tuple)
    assert isinstance(first.facts[0].fields, tuple)
    assert isinstance(first.evidence, tuple)

    changed = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": nested_content,
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Metformin 500 mg",
                "page_number": 1,
                "section": "Medications",
                "field_paths": ["/dosage/amount", "/dosage/unit"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )
    assert changed.facts[0].fact_id != first.facts[0].fact_id


def test_validation_recomputes_content_bound_fact_and_evidence_ids(
    summary_input: GroundedSummaryInput,
) -> None:
    facts, evidence, uncertainties = _registries(summary_input)
    fact = summary_input.facts[0]
    evidence_item = summary_input.evidence[0]
    facts[fact.fact_id] = fact.model_copy(update={"fields": fact.fields[:-1]})
    evidence[evidence_item.evidence_id] = evidence_item.model_copy(
        update={"excerpt": "Tampered evidence"}
    )

    with pytest.raises(LocalValidationError, match="support registry"):
        validate_and_render_summary(
            _valid_output(summary_input),
            facts=facts,
            evidence=evidence,
            uncertainties=uncertainties,
        )


def test_fact_deserialization_rejects_recomputed_non_allowlisted_content(
    summary_input: GroundedSummaryInput,
) -> None:
    fact = summary_input.facts[0]
    content = {
        "record_type": "medication",
        "name": "Metformin",
        "recommended_action": "Double the dose immediately.",
    }
    content_json = _canonical_json_value(content)
    payload = fact.model_dump(mode="json")
    payload.update(
        {
            "content_json": content_json,
            "fields": [
                field.model_dump(mode="json") for field in _flatten_content(content)
            ],
            "fact_id": _stable_fact_id(
                fact.record_id,
                content_json,
                fact.evidence_ids,
            ),
        }
    )

    with pytest.raises(ValueError, match="snapshot"):
        GroundedSummaryFact.model_validate(payload)


def test_validation_rejects_reciprocal_fact_evidence_rebinding() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "medication",
                    "name": "Metformin",
                },
                "evidence_ids": ["source-evidence-1"],
            },
            {
                "record_id": "record-2",
                "content": {
                    "record_type": "medication",
                    "name": "Insulin",
                },
                "evidence_ids": ["source-evidence-2"],
            },
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Metformin",
                "page_number": 1,
                "field_paths": ["/name"],
            },
            {
                "id": "source-evidence-2",
                "excerpt": "Insulin",
                "page_number": 2,
                "field_paths": ["/name"],
            },
        ],
        requested_scope={"summary_type": "category", "category": "medication"},
    )
    metformin, insulin = summary_input.facts
    metformin_evidence, insulin_evidence = summary_input.evidence
    rebound_facts = {
        metformin.fact_id: metformin.model_copy(
            update={"evidence_ids": (insulin_evidence.evidence_id,)}
        ),
        insulin.fact_id: insulin.model_copy(
            update={"evidence_ids": (metformin_evidence.evidence_id,)}
        ),
    }
    rebound_evidence = {
        metformin_evidence.evidence_id: metformin_evidence.model_copy(
            update={"fact_ids": (insulin.fact_id,)}
        ),
        insulin_evidence.evidence_id: insulin_evidence.model_copy(
            update={"fact_ids": (metformin.fact_id,)}
        ),
    }

    with pytest.raises(LocalValidationError, match="support registry"):
        validate_and_render_summary(
            {
                "sections": [
                    {
                        "heading": "Medications",
                        "claims": [
                            {
                                "fact_id": metformin.fact_id,
                                "field_paths": ["/name"],
                                "evidence_ids": [insulin_evidence.evidence_id],
                            }
                        ],
                    }
                ],
                "uncertainties": [],
            },
            facts=rebound_facts,
            evidence=rebound_evidence,
        )


@pytest.mark.parametrize(
    "scope",
    [
        {"summary_type": "invented"},
        {"summary_type": "category", "category": "pancreatic_cancer"},
        {"summary_type": "category"},
        {
            "summary_type": "date_range",
            "date_from": "2025-02-01",
            "date_to": "2025-01-01",
        },
        {"summary_type": "full_health", "raw": "Jane Doe clinical record"},
        {"summary_type": "full_health", "rejected": {"diagnosis": "Cancer"}},
    ],
)
def test_builder_rejects_untyped_or_unsafe_requested_scope(
    scope: dict[str, object],
) -> None:
    with pytest.raises(LocalValidationError, match="scope"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {"name": "Metformin"},
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Metformin",
                    "page_number": 1,
                    "field_paths": ["/name"],
                }
            ],
            requested_scope=scope,
        )


def test_builder_accepts_typed_date_and_single_record_scopes() -> None:
    common = {
        "facts": [
            {
                "record_id": "record-1",
                "content": {"name": "Metformin", "date": "2024-06-01"},
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        "evidence": [
            {
                "id": "source-evidence-1",
                "excerpt": "Metformin",
                "page_number": 1,
                "field_paths": ["/name"],
            }
        ],
    }

    date_scope = build_grounded_summary_input(
        **common,
        requested_scope={
            "summary_type": "date_range",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
        },
    )
    record_scope = build_grounded_summary_input(
        **common,
        requested_scope={
            "summary_type": "single_record",
            "record_ids": ["record-1"],
        },
    )

    assert date_scope.requested_scope.date_from == "2024-01-01"
    assert record_scope.requested_scope.record_ids == ("record-1",)


def test_single_record_scope_requires_the_requested_fact() -> None:
    with pytest.raises(LocalValidationError, match="scope"):
        build_grounded_summary_input(
            facts=[],
            evidence=[],
            requested_scope={
                "summary_type": "single_record",
                "record_ids": ["record-1"],
            },
        )


@pytest.mark.parametrize(
    "fact_date",
    [
        "2024-06-01Tnot-a-date",
        "2024-06-01 12:00:00",
        "2024-06-01T12:00:00+24:00",
        "2024-06-01T12:00:00+04:60",
        "2024-06-01T12:00:00-04:99",
        "2024-13-01",
    ],
)
def test_date_range_scope_rejects_noncanonical_fact_dates(fact_date: str) -> None:
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": "condition",
                        "diagnosis": "Type 2 diabetes",
                        "date": fact_date,
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Type 2 diabetes",
                    "field_paths": ["/diagnosis"],
                }
            ],
            requested_scope={
                "summary_type": "date_range",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31",
            },
        )


@pytest.mark.parametrize(
    "fact_date",
    [
        "2024-06-01",
        "2024-06-01T12:30:00",
        "2024-06-01T12:30:00Z",
        "2024-06-01T12:30:00-04:00",
        "2024-06-01T12:30:00.123456+04:00",
    ],
)
def test_date_range_scope_accepts_canonical_fact_dates(fact_date: str) -> None:
    result = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "condition",
                    "diagnosis": "Type 2 diabetes",
                    "date": fact_date,
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Type 2 diabetes",
                "field_paths": ["/diagnosis"],
            }
        ],
        requested_scope={
            "summary_type": "date_range",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
        },
    )

    assert result.facts[0].record_id == "record-1"


@pytest.mark.parametrize(
    "malformed_end",
    [
        "2024-06-02Tnot-a-date",
        "2024-06-02T12:00:00+24:00",
        "2024-06-02T12:00:00+04:60",
    ],
)
def test_date_range_scope_rejects_a_malformed_secondary_date_field(
    malformed_end: str,
) -> None:
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": "medication",
                        "name": "Metformin",
                        "date": "2024-06-01",
                        "end_date": malformed_end,
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Metformin",
                    "field_paths": ["/name", "/date"],
                }
            ],
            requested_scope={
                "summary_type": "date_range",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31",
            },
        )


def test_date_range_scope_rejects_conflicting_multiple_date_fields() -> None:
    with pytest.raises(LocalValidationError, match="scope"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": "medication",
                        "name": "Metformin",
                        "date": "2024-06-01",
                        "end_date": "2025-01-01",
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Metformin",
                    "field_paths": ["/name", "/date", "/end_date"],
                }
            ],
            requested_scope={
                "summary_type": "date_range",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31",
            },
        )


def test_date_range_scope_accepts_consistent_multiple_date_fields() -> None:
    result = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "medication",
                    "name": "Metformin",
                    "date": "2024-06-01",
                    "end_date": "2024-07-01",
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Metformin",
                "field_paths": ["/name", "/date", "/end_date"],
            }
        ],
        requested_scope={
            "summary_type": "date_range",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
        },
    )

    assert result.facts[0].record_id == "record-1"


def test_builder_rejects_render_controls_and_does_not_echo_phi() -> None:
    for unsafe in ("\u200b", "\u2028", "\u2029"):
        with pytest.raises(LocalValidationError) as error:
            build_grounded_summary_input(
                facts=[
                    {
                        "record_id": "record-1",
                        "content": {"name": f"Jane Doe{unsafe}"},
                        "evidence_ids": ["source-evidence-1"],
                    }
                ],
                evidence=[
                    {
                        "id": "source-evidence-1",
                        "excerpt": "Jane Doe",
                        "page_number": 1,
                        "field_paths": ["/name"],
                    }
                ],
                requested_scope={"summary_type": "full_health"},
            )
        assert "Jane Doe" not in str(error.value)


@pytest.mark.parametrize(
    "raw",
    [
        '{"sections": [], "uncertainties": [], "extra": "Jane Doe"}',
        '{"sections": [], "sections": [], "uncertainties": []}',
        '{"sections": [], "uncertainties": [NaN]}',
        '{"sections": [], "uncertainties": []} trailing',
        b'{"sections": [], "uncertainties": []}',
    ],
)
def test_summary_parsing_is_strict_and_errors_do_not_echo_phi(raw: object) -> None:
    with pytest.raises(LocalValidationError) as error:
        validate_and_render_summary(  # type: ignore[arg-type]
            raw,
            facts={},
            evidence={},
            uncertainties={},
        )

    assert "Jane Doe" not in str(error.value)
    if isinstance(raw, str):
        assert raw not in str(error.value)


def test_deep_recursive_worker_output_fails_closed_without_recursion_error() -> None:
    raw = '{"sections":' + ("[" * 2_000) + ("]" * 2_000) + ',"uncertainties":[]}'

    with pytest.raises(LocalValidationError, match="strict JSON"):
        validate_and_render_summary(
            raw,
            facts={},
            evidence={},
            uncertainties={},
        )


def test_builder_rejects_payload_above_worker_input_cap() -> None:
    long_value = "é" * 4096
    facts = []
    evidence = []
    for index in range(96):
        source_id = f"source-evidence-{index}"
        facts.append(
            {
                "record_id": f"record-{index}",
                "content": {
                    "record_type": "diagnostic_report",
                    "name": long_value,
                    "findings": long_value,
                    "interpretation": long_value,
                },
                "evidence_ids": [source_id],
            }
        )
        evidence.append(
            {
                "id": source_id,
                "excerpt": "é" * 2000,
                "page_number": index + 1,
                "field_paths": ["/name", "/findings", "/interpretation"],
            }
        )

    with pytest.raises(LocalValidationError, match="size limit") as error:
        build_grounded_summary_input(
            facts=facts,
            evidence=evidence,
            requested_scope={"summary_type": "full_health"},
        )

    assert long_value not in str(error.value)


def test_input_model_rejects_payload_above_worker_cap_on_deserialization() -> None:
    long_value = "é" * 4096
    parts = [
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": f"record-{index}",
                    "content": {
                        "record_type": "diagnostic_report",
                        "name": long_value,
                        "findings": long_value,
                        "interpretation": long_value,
                    },
                    "evidence_ids": [f"source-evidence-{index}"],
                }
            ],
            evidence=[
                {
                    "id": f"source-evidence-{index}",
                    "excerpt": "é" * 2000,
                    "page_number": index + 1,
                    "field_paths": ["/name", "/findings", "/interpretation"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )
        for index in range(64)
    ]
    payload = {
        "requested_scope": {"summary_type": "full_health"},
        "facts": [part.facts[0].model_dump(mode="json") for part in parts],
        "evidence": [part.evidence[0].model_dump(mode="json") for part in parts],
        "uncertainty_labels": [],
        "safety_rules": list(parts[0].safety_rules),
    }

    with pytest.raises(ValueError, match="size limit"):
        GroundedSummaryInput.model_validate(payload)


def test_recursive_python_worker_output_fails_closed_without_recursion_error() -> None:
    raw: dict[str, object] = {"sections": [], "uncertainties": []}
    raw["loop"] = raw

    with pytest.raises(LocalValidationError, match="invalid structure"):
        validate_and_render_summary(
            raw,
            facts={},
            evidence={},
            uncertainties={},
        )


def test_builder_rejects_missing_unknown_and_unlinked_evidence() -> None:
    for evidence_ids in ([], ["evidence-unknown"]):
        with pytest.raises(LocalValidationError, match="evidence"):
            build_grounded_summary_input(
                facts=[
                    {
                        "record_id": "record-1",
                        "content": {"name": "Metformin"},
                        "evidence_ids": evidence_ids,
                    }
                ],
                evidence=[
                    {
                        "id": "source-evidence-1",
                        "excerpt": "Metformin",
                        "page_number": 1,
                        "field_paths": ["/name"],
                    }
                ],
                requested_scope={"summary_type": "full_health"},
            )


def test_builder_rejects_raw_rejected_and_unresolved_fact_fields() -> None:
    for field_name in ("raw_upload", "rejected", "unresolved"):
        with pytest.raises(LocalValidationError, match="invalid fact input"):
            build_grounded_summary_input(
                facts=[
                    {
                        "record_id": "record-1",
                        "content": {
                            "name": "Metformin",
                            field_name: "must not leave the validation boundary",
                        },
                        "evidence_ids": ["source-evidence-1"],
                    }
                ],
                evidence=[
                    {
                        "id": "source-evidence-1",
                        "excerpt": "Metformin",
                        "page_number": 1,
                        "field_paths": ["/name"],
                    }
                ],
                requested_scope={"summary_type": "full_health"},
            )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("recommended_action", "Double the dose immediately."),
        ("ocr_text", "Jane Doe DOB 01/02/1980 full raw medical document"),
        ("pending_review", "Pancreatic cancer"),
    ],
)
def test_builder_rejects_non_allowlisted_fact_fields(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": "medication",
                        "name": "Metformin",
                        field_name: value,
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Metformin",
                    "page_number": 1,
                    "field_paths": ["/name"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )


@pytest.mark.parametrize(
    ("record_type", "field_name", "bad_value"),
    [
        ("condition", "diagnosis", False),
        ("condition", "status", 7),
        ("condition", "assertion", "maybe"),
        ("medication", "status", True),
        ("medication", "dose_value", -1),
        ("appointment", "status", "active"),
        ("encounter", "visit_type", "clinic"),
        ("diagnostic_report", "category", "secret"),
        ("service_request", "priority", "whenever"),
        ("observation", "unit", 42),
    ],
)
def test_builder_rejects_wrong_scalar_types_enums_and_numeric_domains(
    record_type: str,
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": record_type,
                        field_name: bad_value,
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Source excerpt",
                    "field_paths": [f"/{field_name}"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )


@pytest.mark.parametrize(
    "scope",
    [
        {"summary_type": "full_health"},
        {"summary_type": "category", "category": "medication"},
        {"summary_type": "single_record", "record_ids": ["record-1"]},
    ],
)
def test_builder_rejects_malformed_fact_dates_under_every_scope(
    scope: dict[str, object],
) -> None:
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": "medication",
                        "name": "Metformin",
                        "date": "not-a-date",
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Metformin",
                    "field_paths": ["/name", "/date"],
                }
            ],
            requested_scope=scope,
        )


@pytest.mark.parametrize(
    "content",
    [
        {
            "record_type": "appointment",
            "title": "Follow-up",
            "start": "2025-02-01T10:00:00Z",
            "end": "2025-02-01T09:00:00Z",
        },
        {
            "record_type": "medication",
            "name": "Metformin",
            "effective_date": "2025-02-01",
            "end_date": "2025-01-01",
        },
        {
            "record_type": "medication",
            "name": "Metformin",
            "date": "2025-02-01",
            "end_date": "2025-01-01",
        },
        {
            "record_type": "condition",
            "diagnosis": "Type 2 diabetes",
            "assertion": "negated",
            "status": "active",
        },
        {
            "record_type": "procedure",
            "name": "Colonoscopy",
            "assertion": "mentioned_not_performed",
            "status": "completed",
        },
        {
            "record_type": "allergy",
            "substance": "Penicillin",
            "assertion": "negated",
            "status": "active",
        },
        {
            "record_type": "diagnostic_report",
            "name": "CT chest",
            "assertion": "mentioned_not_performed",
            "status": "final",
        },
    ],
)
def test_builder_rejects_chronology_and_lifecycle_contradictions(
    content: dict[str, object],
) -> None:
    supported_path = next(
        f"/{field}"
        for field in ("name", "title", "diagnosis", "substance")
        if field in content
    )
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": content,
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Source excerpt",
                    "field_paths": [supported_path],
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )


@pytest.mark.parametrize(
    ("record_type", "field_name", "value"),
    [
        ("medication", "dose_value", "-1"),
        ("medication", "dose_value", "NaN"),
        ("medication", "dose", "-1"),
        ("immunization", "dose", "inf"),
        ("immunization", "dose", "Infinity mL"),
        ("immunization", "dose", "InfinitymL"),
        ("immunization", "dose", "Infinity mcg"),
        ("immunization", "dose", "Infinitymcg"),
        ("immunization", "dose", "Infinity CFU/mL"),
        ("immunization", "dose", "InfinityCFU/mL"),
        ("immunization", "dose", "NaNCFU"),
        ("immunization", "dose", "Infinitycopies"),
        ("observation", "value", "NaN"),
        ("observation", "value", "NaN mg"),
        ("observation", "value", "NaNmg"),
        ("observation", "value", "NaN mmHg"),
        ("observation", "value", "NaNmmHg"),
        ("observation", "value", "NaN bpm"),
        ("observation", "value", "NaNbpm"),
        ("observation", "value", "NaN cells/uL"),
        ("observation", "value", "NaNcells/uL"),
        ("observation", "value", "NaN copies/mL"),
        ("observation", "value", "NaNcopies/mL"),
        ("observation", "value", "NaNcells"),
        ("observation", "value", "NaNmOsm"),
        ("observation", "value", "Inf beats/min"),
        ("observation", "value", "Infbeats/min"),
        ("observation", "value", "Infbeats"),
        ("observation", "value", "Inf%"),
        ("observation", "value", "Infinity"),
    ],
)
def test_builder_rejects_nonfinite_or_nonpositive_numeric_strings(
    record_type: str,
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(LocalValidationError, match="invalid fact input"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": record_type,
                        "name": "Clinical fact",
                        field_name: value,
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Source excerpt",
                    "field_paths": ["/name", f"/{field_name}"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )


@pytest.mark.parametrize(
    ("record_type", "field_name", "value"),
    [
        ("medication", "dose_value", "500 mg"),
        ("immunization", "dose", "0.5 mL"),
        ("observation", "value", "6.5%"),
    ],
)
def test_builder_accepts_valid_compound_numeric_strings(
    record_type: str,
    field_name: str,
    value: str,
) -> None:
    result = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": record_type,
                    "name": "Clinical fact",
                    field_name: value,
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Source excerpt",
                "field_paths": ["/name", f"/{field_name}"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )

    assert result.facts[0].record_id == "record-1"


@pytest.mark.parametrize(
    "value",
    [
        "inflammation",
        "infections",
        "infarcts",
        "infiltrates",
        "infusion",
        "infusions",
        "information",
        "infinite",
        "infant",
        "infants",
        "infrastructure",
        "nanogram",
        "nannies",
        "nanoparticles",
        "nanotechnology",
    ],
)
def test_builder_preserves_ordinary_words_beginning_with_nonfinite_tokens(
    value: str,
) -> None:
    result = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "observation",
                    "name": "Clinical note",
                    "value": value,
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": value,
                "field_paths": ["/name", "/value"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )

    assert result.facts[0].record_id == "record-1"


def test_claim_fields_must_be_supported_by_selected_evidence() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "condition",
                    "diagnosis": "Pancreatic cancer",
                    "status": "active",
                },
                "evidence_ids": ["status-evidence", "diagnosis-evidence"],
            }
        ],
        evidence=[
            {
                "id": "status-evidence",
                "excerpt": "Active",
                "page_number": 1,
                "field_paths": ["/status"],
            },
            {
                "id": "diagnosis-evidence",
                "excerpt": "Pancreatic cancer",
                "page_number": 2,
                "field_paths": ["/diagnosis"],
            },
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    evidence = {item.source_id: item for item in summary_input.evidence}

    with pytest.raises(LocalValidationError, match="field support"):
        validate_and_render_summary(
            {
                "sections": [
                    {
                        "heading": "Conditions",
                        "claims": [
                            {
                                "fact_id": fact.fact_id,
                                "field_paths": ["/diagnosis"],
                                "evidence_ids": [
                                    evidence["status-evidence"].evidence_id
                                ],
                            }
                        ],
                    }
                ],
                "uncertainties": [],
            },
            facts={fact.fact_id: fact},
            evidence={item.evidence_id: item for item in summary_input.evidence},
        )


def test_rendered_claim_discards_evidence_for_unrendered_fields() -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "procedure",
                    "name": "Biopsy",
                    "provider": "Dr Example",
                    "status": "completed",
                },
                "evidence_ids": ["name", "provider", "status"],
            }
        ],
        evidence=[
            {"id": "name", "excerpt": "Biopsy", "field_paths": ["/name"]},
            {
                "id": "provider",
                "excerpt": "Dr Example",
                "field_paths": ["/provider"],
            },
            {"id": "status", "excerpt": "completed", "field_paths": ["/status"]},
        ],
        requested_scope={"summary_type": "full_health"},
    )
    fact = summary_input.facts[0]
    by_source = {item.source_id: item for item in summary_input.evidence}

    rendered = validate_and_render_summary(
        {
            "sections": [
                {
                    "heading": "Procedures",
                    "claims": [
                        {
                            "fact_id": fact.fact_id,
                            "field_paths": ["/name"],
                            "evidence_ids": [
                                by_source["name"].evidence_id,
                                by_source["provider"].evidence_id,
                            ],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        facts={fact.fact_id: fact},
        evidence={item.evidence_id: item for item in summary_input.evidence},
    )

    claim = rendered.document.sections[0].claims[0]
    assert "/status" in claim.field_paths
    assert by_source["name"].evidence_id in claim.evidence_ids
    assert by_source["status"].evidence_id in claim.evidence_ids
    assert by_source["provider"].evidence_id not in claim.evidence_ids


@pytest.mark.parametrize(
    ("scope", "content", "record_id"),
    [
        (
            {"summary_type": "single_record", "record_ids": ["record-1"]},
            {"record_type": "condition", "diagnosis": "Pancreatic cancer"},
            "record-2",
        ),
        (
            {"summary_type": "category", "category": "medication"},
            {"record_type": "condition", "diagnosis": "Pancreatic cancer"},
            "record-1",
        ),
        (
            {
                "summary_type": "date_range",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31",
            },
            {
                "record_type": "condition",
                "diagnosis": "Pancreatic cancer",
                "date": "2025-01-01",
            },
            "record-1",
        ),
    ],
)
def test_builder_rejects_facts_outside_requested_scope(
    scope: dict[str, object],
    content: dict[str, object],
    record_id: str,
) -> None:
    with pytest.raises(LocalValidationError, match="scope"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": record_id,
                    "content": content,
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Pancreatic cancer",
                    "page_number": 1,
                    "field_paths": ["/diagnosis"],
                }
            ],
            requested_scope=scope,
        )


@pytest.mark.parametrize(
    "scope",
    [
        {"summary_type": "single_record", "record_ids": ["record-2"]},
        {"summary_type": "category", "category": "medication"},
        {
            "summary_type": "date_range",
            "date_from": "2025-01-01",
            "date_to": "2025-12-31",
        },
    ],
)
def test_input_model_revalidates_fact_scope_on_deserialization(
    scope: dict[str, object],
) -> None:
    summary_input = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "condition",
                    "diagnosis": "Type 2 diabetes",
                    "date": "2024-06-01",
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Type 2 diabetes",
                "field_paths": ["/diagnosis", "/date"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
    )
    payload = summary_input.model_dump(mode="json")
    payload["requested_scope"] = scope

    with pytest.raises(ValueError, match="scope"):
        GroundedSummaryInput.model_validate(payload)


def test_builder_rejects_arbitrary_uncertainty_prose() -> None:
    with pytest.raises(LocalValidationError, match="uncertainty"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": "medication",
                        "name": "Metformin",
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Metformin",
                    "page_number": 1,
                    "field_paths": ["/name"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
            uncertainty_labels=[
                {
                    "label": "Double the medication dose now.",
                    "record_ids": ["record-1"],
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
        )


@pytest.mark.parametrize(
    ("template_id", "content", "evidence_overrides"),
    [
        (
            "assertion_uncertain",
            {
                "record_type": "condition",
                "diagnosis": "Type 2 diabetes",
                "assertion": "present",
            },
            {},
        ),
        (
            "medication_dose_missing",
            {
                "record_type": "medication",
                "name": "Metformin",
                "dose": "500 mg",
            },
            {},
        ),
        (
            "medication_end_date_missing",
            {
                "record_type": "medication",
                "name": "Metformin",
                "end_date": "2025-01-01",
            },
            {},
        ),
        (
            "record_date_missing",
            {
                "record_type": "condition",
                "diagnosis": "Type 2 diabetes",
                "date": "2025-01-01",
            },
            {},
        ),
        (
            "record_status_missing",
            {
                "record_type": "condition",
                "diagnosis": "Type 2 diabetes",
                "status": "active",
            },
            {},
        ),
        (
            "result_missing",
            {
                "record_type": "observation",
                "name": "A1c",
                "value": "6.5%",
            },
            {},
        ),
        (
            "result_missing",
            {
                "record_type": "condition",
                "diagnosis": "Type 2 diabetes",
            },
            {},
        ),
        (
            "source_detail_missing",
            {
                "record_type": "condition",
                "diagnosis": "Type 2 diabetes",
            },
            {"page_number": 1},
        ),
        (
            "medication_dose_missing",
            {
                "record_type": "condition",
                "diagnosis": "Type 2 diabetes",
            },
            {},
        ),
    ],
)
def test_builder_rejects_inapplicable_uncertainty_templates(
    template_id: str,
    content: dict[str, object],
    evidence_overrides: dict[str, object],
) -> None:
    field_path = "/name" if "name" in content else "/diagnosis"
    if content.get("record_type") == "observation":
        field_path = "/name"

    with pytest.raises(LocalValidationError, match="uncertainty"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": content,
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Source excerpt",
                    "field_paths": [field_path],
                    **evidence_overrides,
                }
            ],
            requested_scope={"summary_type": "full_health"},
            uncertainty_labels=[
                {
                    "template_id": template_id,
                    "record_ids": ["record-1"],
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
        )


def test_builder_rejects_missing_status_when_alternate_statuses_are_present() -> None:
    with pytest.raises(LocalValidationError, match="uncertainty"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": "medication",
                        "name": "Metformin",
                        "statuses": ["active"],
                    },
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Metformin active",
                    "field_paths": ["/name", "/statuses/0"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
            uncertainty_labels=[
                {
                    "template_id": "record_status_missing",
                    "record_ids": ["record-1"],
                    "evidence_ids": ["source-evidence-1"],
                }
            ],
        )


def test_builder_rejects_unrelated_evidence_in_uncertainty_support() -> None:
    with pytest.raises(LocalValidationError, match="uncertainty support"):
        build_grounded_summary_input(
            facts=[
                {
                    "record_id": "record-1",
                    "content": {
                        "record_type": "condition",
                        "diagnosis": "Type 2 diabetes",
                    },
                    "evidence_ids": ["source-evidence-1"],
                },
                {
                    "record_id": "record-2",
                    "content": {
                        "record_type": "condition",
                        "diagnosis": "Hypertension",
                    },
                    "evidence_ids": ["source-evidence-2"],
                },
            ],
            evidence=[
                {
                    "id": "source-evidence-1",
                    "excerpt": "Type 2 diabetes",
                    "field_paths": ["/diagnosis"],
                },
                {
                    "id": "source-evidence-2",
                    "excerpt": "Hypertension",
                    "field_paths": ["/diagnosis"],
                },
            ],
            requested_scope={"summary_type": "full_health"},
            uncertainty_labels=[
                {
                    "template_id": "record_status_missing",
                    "record_ids": ["record-1"],
                    "evidence_ids": [
                        "source-evidence-1",
                        "source-evidence-2",
                    ],
                }
            ],
        )


@pytest.mark.parametrize(
    ("template_id", "content"),
    [
        (
            "assertion_uncertain",
            {
                "record_type": "condition",
                "diagnosis": "Type 2 diabetes",
                "assertion": "uncertain",
            },
        ),
        (
            "medication_dose_missing",
            {"record_type": "medication", "name": "Metformin"},
        ),
        (
            "medication_end_date_missing",
            {"record_type": "medication", "name": "Metformin"},
        ),
        (
            "record_date_missing",
            {"record_type": "condition", "diagnosis": "Type 2 diabetes"},
        ),
        (
            "record_status_missing",
            {"record_type": "condition", "diagnosis": "Type 2 diabetes"},
        ),
        (
            "result_missing",
            {"record_type": "observation", "name": "A1c"},
        ),
    ],
)
def test_builder_accepts_applicable_uncertainty_templates(
    template_id: str,
    content: dict[str, object],
) -> None:
    field_path = "/name" if "name" in content else "/diagnosis"

    result = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": content,
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Source excerpt",
                "field_paths": [field_path],
            }
        ],
        requested_scope={"summary_type": "full_health"},
        uncertainty_labels=[
            {
                "template_id": template_id,
                "record_ids": ["record-1"],
                "evidence_ids": ["source-evidence-1"],
            }
        ],
    )

    assert result.uncertainty_labels[0].template_id == template_id


def test_builder_accepts_missing_source_detail_only_without_location_metadata() -> None:
    result = build_grounded_summary_input(
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "condition",
                    "diagnosis": "Type 2 diabetes",
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Source excerpt",
                "field_paths": ["/diagnosis"],
            }
        ],
        requested_scope={"summary_type": "full_health"},
        uncertainty_labels=[
            {
                "template_id": "source_detail_missing",
                "record_ids": ["record-1"],
                "evidence_ids": ["source-evidence-1"],
            }
        ],
    )

    assert result.uncertainty_labels[0].template_id == "source_detail_missing"
