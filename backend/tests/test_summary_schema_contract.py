"""Contract tests for generated summary request and response schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.summary import (
    BuildPromptRequest,
    GenerateSummaryRequest,
    GenerateSummaryResponse,
    PasteResponseRequest,
)
from app.services.local_ai.grounded_summary import (
    MAX_FACTS,
    MAX_RAW_OUTPUT_CHARACTERS,
)

PATIENT_ID = UUID("00000000-0000-0000-0000-000000000001")
SUMMARY_ID = UUID("00000000-0000-0000-0000-000000000002")


def _generated_response(processing_mode: str) -> dict[str, object]:
    return {
        "id": SUMMARY_ID,
        "processing_mode": processing_mode,
        "model_provenance": None,
        "typed_response": {"sections": [], "uncertainties": []},
        "natural_language": None,
        "json_data": None,
        "record_count": 0,
        "duplicate_warning": None,
        "de_identification_report": None,
        "model_used": "test-model",
        "generated_at": datetime.now(timezone.utc),
    }


@pytest.mark.parametrize(
    "processing_mode",
    ["validated_strict_local", "custom_local", "cloud_assisted"],
)
def test_generated_response_requires_typed_response(
    processing_mode: str,
) -> None:
    adapter = TypeAdapter(GenerateSummaryResponse)
    payload = _generated_response(processing_mode)

    assert adapter.validate_python(payload).typed_response is not None

    payload["typed_response"] = None
    with pytest.raises(ValidationError):
        adapter.validate_python(payload)


def test_generated_response_schema_is_discriminated_by_processing_mode() -> None:
    schema = TypeAdapter(GenerateSummaryResponse).json_schema()

    assert schema["discriminator"]["propertyName"] == "processing_mode"
    assert len(schema["oneOf"]) == 3


@pytest.mark.parametrize(
    ("response_mode", "provenance_mode"),
    [
        ("custom_local", "cloud_assisted"),
        ("cloud_assisted", "custom_local"),
    ],
)
def test_generated_response_rejects_mismatched_routed_provenance(
    response_mode: str,
    provenance_mode: str,
) -> None:
    payload = _generated_response(response_mode)
    payload["model_provenance"] = {
        "processing_mode": provenance_mode,
        "provider": "test-provider",
        "model": "test-model",
    }

    with pytest.raises(ValidationError):
        TypeAdapter(GenerateSummaryResponse).validate_python(payload)


@pytest.mark.parametrize(
    "field_name",
    ["custom_system_prompt", "custom_user_prompt"],
)
def test_generate_request_bounds_custom_prompts(field_name: str) -> None:
    GenerateSummaryRequest(patient_id=PATIENT_ID, **{field_name: "x" * 4096})

    with pytest.raises(ValidationError):
        GenerateSummaryRequest(patient_id=PATIENT_ID, **{field_name: "x" * 4097})


def test_paste_response_request_bounds_provider_output() -> None:
    PasteResponseRequest(
        prompt_id=SUMMARY_ID,
        response_text="x" * MAX_RAW_OUTPUT_CHARACTERS,
    )

    with pytest.raises(ValidationError):
        PasteResponseRequest(
            prompt_id=SUMMARY_ID,
            response_text="x" * (MAX_RAW_OUTPUT_CHARACTERS + 1),
        )


@pytest.mark.parametrize("request_type", [BuildPromptRequest, GenerateSummaryRequest])
def test_summary_requests_bound_record_identifiers(request_type: type) -> None:
    record_ids = [UUID(int=index + 1) for index in range(MAX_FACTS)]
    request_type(patient_id=PATIENT_ID, record_ids=record_ids)

    with pytest.raises(ValidationError):
        request_type(
            patient_id=PATIENT_ID,
            record_ids=[*record_ids, UUID(int=MAX_FACTS + 1)],
        )


def test_build_prompt_bounds_and_allowlists_record_types() -> None:
    BuildPromptRequest(
        patient_id=PATIENT_ID,
        record_types=["family_history", "imaging"],
    )
    BuildPromptRequest(
        patient_id=PATIENT_ID,
        record_types=["medication"] * MAX_FACTS,
    )

    with pytest.raises(ValidationError):
        BuildPromptRequest(
            patient_id=PATIENT_ID,
            record_types=["medication"] * (MAX_FACTS + 1),
        )
    with pytest.raises(ValidationError):
        BuildPromptRequest(
            patient_id=PATIENT_ID,
            record_types=["unsupported_record_type"],
        )
