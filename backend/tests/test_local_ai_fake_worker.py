from __future__ import annotations

from app.services.local_ai.extraction_schema import FACT_CATEGORY_NAMES
from app.services.local_ai.extraction_validator import validate_clinical_extraction
from app.services.local_ai.fake_worker import _fixed_result
from app.services.local_ai.grounded_summary import (
    build_grounded_summary_input,
    validate_and_render_summary,
)
from app.services.local_ai.types import ModelRole


def test_pipeline_valid_ocr_echoes_requested_page_number() -> None:
    result = _fixed_result(
        ModelRole.OCR,
        {"page_number": 7},
        pipeline_valid=True,
    )

    assert result == {
        "markdown": "# Synthetic OCR\n\nFixturemed 10 mg daily is active.",
        "page_number": 7,
    }


def test_pipeline_valid_extraction_returns_validator_compatible_empty_document() -> (
    None
):
    pages = {3: "# Synthetic OCR"}

    result = _fixed_result(
        ModelRole.EXTRACTION,
        {
            "page_markdown": [
                {"page_number": page_number, "markdown": markdown}
                for page_number, markdown in pages.items()
            ]
        },
        pipeline_valid=True,
    )
    validated = validate_clinical_extraction(
        result,
        pages,
        upload_id="upload-1",
        strict_local=True,
    )

    assert validated.schema_version == "clinical-document-extraction.v1"
    assert validated.patient is None
    assert all(getattr(validated, category) == [] for category in FACT_CATEGORY_NAMES)
    assert validated.unresolved_fields == []
    assert validated.rejected_fields == []


def test_pipeline_valid_extraction_returns_grounded_synthetic_fact() -> None:
    page_number = 3
    ocr_result = _fixed_result(
        ModelRole.OCR,
        {"page_number": page_number},
        pipeline_valid=True,
    )
    markdown = ocr_result["markdown"]
    assert isinstance(markdown, str)

    result = _fixed_result(
        ModelRole.EXTRACTION,
        {
            "page_markdown": [
                {"page_number": page_number, "markdown": markdown},
            ]
        },
        pipeline_valid=True,
    )
    validated = validate_clinical_extraction(
        result,
        {page_number: markdown},
        upload_id="upload-1",
        strict_local=True,
    )

    assert len(validated.medications) == 1
    medication = validated.medications[0]
    assert medication.fact_id == "synthetic-medication-1"
    assert medication.name == "Fixturemed"
    assert medication.evidence_id is not None
    assert len(validated.evidence) == 1
    assert validated.evidence[0].id == medication.evidence_id
    assert "medications[0].name" in validated.evidence[0].field_paths


def test_pipeline_valid_summary_returns_validator_compatible_references() -> None:
    summary_input = build_grounded_summary_input(
        requested_scope={"summary_type": "full_health"},
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
                "excerpt": "Metformin remains active.",
                "page_number": 1,
                "section": "Medications",
                "field_paths": ["/name", "/status"],
            }
        ],
        uncertainty_labels=[],
    )
    payload = summary_input.model_dump(mode="json")

    result = _fixed_result(ModelRole.SUMMARY, payload, pipeline_valid=True)
    rendered = validate_and_render_summary(
        result,
        facts={item.fact_id: item for item in summary_input.facts},
        evidence={item.evidence_id: item for item in summary_input.evidence},
        uncertainties={},
    )

    claim = rendered.selection_document.sections[0].claims[0]
    assert rendered.selection_document.sections[0].heading == "Overview"
    assert claim.fact_id == summary_input.facts[0].fact_id
    assert claim.field_paths
    assert claim.evidence_ids == summary_input.facts[0].evidence_ids


def test_default_fake_worker_results_remain_unchanged() -> None:
    assert _fixed_result(ModelRole.OCR, {"page_number": 7}) == {
        "markdown": "# Synthetic OCR",
        "page_number": 1,
    }
    assert _fixed_result(ModelRole.EXTRACTION, {}) == {
        "entities": [],
        "evidence": [],
    }
    assert _fixed_result(ModelRole.SUMMARY, {}) == {
        "sections": [{"facts": [], "title": "Synthetic summary"}]
    }
