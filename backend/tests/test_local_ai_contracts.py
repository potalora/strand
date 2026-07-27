from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from typing import get_type_hints

import pytest

from app.services.local_ai.errors import (
    LocalAIError,
    LocalPolicyError,
    LocalValidationError,
    LocalWorkerError,
    LocalWorkerTimeout,
)
from app.services.local_ai.types import (
    ClinicalExtraction,
    ExtractionBackend,
    GroundedSummary,
    ModelIdentity,
    ModelRole,
    OCRBackend,
    OCRPageRequest,
    OCRPageResult,
    ProcessingMode,
    SummaryBackend,
)


def _ocr_model() -> ModelIdentity:
    return ModelIdentity(
        role=ModelRole.OCR,
        repository="sahilchachra/ovisocr2-int4-mlx",
        revision="0123456789abcdef0123456789abcdef01234567",
        quantization="int4",
        runtime="mlx-vlm-0.5.0",
    )


def test_ocr_request_is_serializable_and_carries_immutable_model_identity() -> None:
    model = _ocr_model()
    request = OCRPageRequest(
        job_id="job-1",
        page_number=1,
        image_path="/private/job-1/page-0001.png",
        image_sha256="a" * 64,
        model=model,
        max_output_tokens=8192,
    )

    assert asdict(request)["model"]["repository"] == model.repository
    with pytest.raises(FrozenInstanceError):
        model.repository = "other/repository"  # type: ignore[misc]


def test_processing_modes_are_explicit_and_stable() -> None:
    assert {mode.value for mode in ProcessingMode} == {
        "validated_strict_local",
        "custom_local",
        "cloud_assisted",
        "prompt_only",
    }


def test_worker_contracts_are_async_and_keep_results_serializable() -> None:
    assert get_type_hints(OCRBackend.parse_page)["return"] is OCRPageResult
    assert get_type_hints(ExtractionBackend.extract)["return"] is ClinicalExtraction
    assert get_type_hints(SummaryBackend.summarize)["return"] is GroundedSummary
    assert asdict(
        OCRPageResult(
            page_number=1,
            markdown="# Page one",
            width=100,
            height=200,
            warnings=[],
            content_sha256="b" * 64,
            model=_ocr_model(),
        )
    )["page_number"] == 1


def test_local_ai_errors_have_stable_safe_codes_and_retryability() -> None:
    assert LocalAIError.code == "local_ai_error"
    assert LocalPolicyError.code == "local_policy_error"
    assert LocalWorkerError.code == "local_worker_error"
    assert LocalWorkerTimeout.code == "local_worker_timeout"
    assert LocalValidationError.code == "local_validation_error"
    assert LocalWorkerTimeout.retryable is True
    assert LocalPolicyError.retryable is False
