from __future__ import annotations

import json
import logging
import random
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.services.local_ai import pipeline as pipeline_module
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.extraction_schema import (
    CLINICAL_EXTRACTION_SCHEMA_VERSION,
    FACT_CATEGORY_NAMES,
    ClinicalDocumentExtraction,
)
from app.services.local_ai.pipeline import StrictLocalPipeline

_PAGE = "Glucose 95 mg/dL. Patient Alpha born 1970-01-01."


def _empty_extraction() -> dict[str, object]:
    return {
        "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
        "patient": None,
        **{category: [] for category in FACT_CATEGORY_NAMES},
        "unresolved_fields": [],
        "rejected_fields": [],
    }


def _lab() -> dict[str, object]:
    return {
        "name": "Glucose",
        "value": "95",
        "unit": "mg/dL",
        "verbatim": "Glucose 95 mg/dL",
        "page_number": 1,
        "evidence_excerpt": "Glucose 95 mg/dL",
    }


def _lab_extraction() -> dict[str, object]:
    extraction = _empty_extraction()
    extraction["labs"] = [_lab()]
    return extraction


def _assert_content_free_rejection(
    result: ClinicalDocumentExtraction,
    expected: str,
    *,
    canary: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert result.labs == []
    assert result.rejected_fields == [expected]
    assert canary not in json.dumps(result.model_dump(mode="json"))
    assert canary not in caplog.text


def test_candidate_validator_ordinary_exception_is_content_free_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Validator Canary MRN 10101"
    real_validate = pipeline_module.validate_clinical_extraction

    def fail_candidate(raw, *args, **kwargs):
        if isinstance(raw, dict) and raw.get("labs"):
            raise RuntimeError(canary)
        return real_validate(raw, *args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "validate_clinical_extraction",
        fail_candidate,
    )
    caplog.set_level(logging.DEBUG)

    result = StrictLocalPipeline._validate_extraction_result(
        _lab_extraction(),
        {1: _PAGE},
        upload_id="candidate-validator-exception",
    )

    _assert_content_free_rejection(
        result,
        "chunks[0].labs[0]:fact_validation_failed",
        canary=canary,
        caplog=caplog,
    )


def test_patient_validator_ordinary_exception_is_content_free_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Identity Canary MRN 20202"
    extraction = _empty_extraction()
    extraction["patient"] = {"name": "Patient Alpha"}
    real_validate = pipeline_module.validate_clinical_extraction

    def fail_patient(raw, *args, **kwargs):
        if isinstance(raw, dict) and raw.get("patient"):
            raise RuntimeError(canary)
        return real_validate(raw, *args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "validate_clinical_extraction",
        fail_patient,
    )
    caplog.set_level(logging.DEBUG)

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: _PAGE},
        upload_id="patient-validator-exception",
    )

    assert result.patient is None
    assert result.rejected_fields == ["chunks[0].patient.name:fact_validation_failed"]
    assert canary not in json.dumps(result.model_dump(mode="json"))
    assert canary not in caplog.text


def test_candidate_adapter_ordinary_exception_is_content_free_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Adapter Canary MRN 30303"

    def fail_adapter(_validated):
        raise RuntimeError(canary)

    monkeypatch.setattr(pipeline_module, "to_extracted_entities", fail_adapter)
    caplog.set_level(logging.DEBUG)

    result = StrictLocalPipeline._validate_extraction_result(
        _lab_extraction(),
        {1: _PAGE},
        upload_id="candidate-adapter-exception",
    )

    _assert_content_free_rejection(
        result,
        "chunks[0].labs[0]:adapter_rejected",
        canary=canary,
        caplog=caplog,
    )


def test_duplicate_signature_ordinary_exception_is_content_free_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Signature Canary MRN 40404"

    def fail_signature(_fact):
        raise RuntimeError(canary)

    monkeypatch.setattr(
        pipeline_module,
        "clinical_fact_duplicate_signature",
        fail_signature,
    )
    caplog.set_level(logging.DEBUG)

    result = StrictLocalPipeline._validate_extraction_result(
        _lab_extraction(),
        {1: _PAGE},
        upload_id="candidate-signature-exception",
    )

    _assert_content_free_rejection(
        result,
        "chunks[0].labs[0]:fact_validation_failed",
        canary=canary,
        caplog=caplog,
    )


def test_merge_ordinary_exception_becomes_stable_local_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Merge Canary MRN 50505"

    def fail_merge(_chunks):
        raise RuntimeError(canary)

    monkeypatch.setattr(
        StrictLocalPipeline,
        "_merge_extraction_chunks",
        fail_merge,
    )
    caplog.set_level(logging.DEBUG)

    with pytest.raises(
        LocalValidationError,
        match="Local extraction merge is invalid",
    ) as raised:
        StrictLocalPipeline._validate_extraction_result(
            {
                "result_type": "chunked_clinical_extraction.v1",
                "chunks": [
                    {
                        "page_numbers": [1],
                        "extraction": _empty_extraction(),
                    },
                    {
                        "page_numbers": [2],
                        "extraction": _empty_extraction(),
                    },
                ],
            },
            {1: "Page one.", 2: "Page two."},
            upload_id="merge-exception",
        )

    assert canary not in str(raised.value)
    assert canary not in caplog.text


def test_final_validation_ordinary_exception_becomes_stable_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Final Validator Canary MRN 60606"
    real_validate = pipeline_module.validate_clinical_extraction

    def fail_final(raw, *args, **kwargs):
        if isinstance(raw, dict) and all(
            category in raw for category in FACT_CATEGORY_NAMES
        ):
            raise RuntimeError(canary)
        return real_validate(raw, *args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "validate_clinical_extraction",
        fail_final,
    )
    caplog.set_level(logging.DEBUG)

    with pytest.raises(
        LocalValidationError,
        match="Local extraction invariant is invalid",
    ) as raised:
        StrictLocalPipeline._validate_extraction_result(
            _empty_extraction(),
            {1: "No clinical facts."},
            upload_id="final-validator-exception",
        )

    assert canary not in str(raised.value)
    assert canary not in caplog.text


def test_evidence_conversion_ordinary_exception_becomes_stable_fatal_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Evidence Canary MRN 70707"
    validated = StrictLocalPipeline._validate_extraction_result(
        _lab_extraction(),
        {1: _PAGE},
        upload_id="evidence-conversion-exception",
    )
    validated._evidence = (SimpleNamespace(id=canary),)
    caplog.set_level(logging.DEBUG)

    with pytest.raises(
        LocalValidationError,
        match="Local extraction evidence is invalid",
    ) as raised:
        StrictLocalPipeline._persistable_evidence(validated)

    assert canary not in str(raised.value)
    assert canary not in caplog.text


def test_bounded_model_json_fuzz_never_leaks_an_ordinary_exception() -> None:
    rng = random.Random(20260728)
    mutations: tuple[object, ...] = (
        None,
        True,
        False,
        0,
        -1,
        1.5,
        "",
        "x",
        "\ud800",
        "\x00",
        "a" * 513,
        [],
        {},
        [1],
        {"unexpected": "value"},
    )

    for case_index in range(256):
        raw: object = _lab_extraction()
        mode = rng.randrange(5)
        if mode == 0:
            raw = deepcopy(rng.choice(mutations))
        elif mode == 1:
            assert isinstance(raw, dict)
            raw[rng.choice(tuple(raw))] = deepcopy(rng.choice(mutations))
        elif mode == 2:
            assert isinstance(raw, dict)
            fact = raw["labs"][0]
            assert isinstance(fact, dict)
            field = rng.choice((*tuple(fact), "unexpected"))
            fact[field] = deepcopy(rng.choice(mutations))
        elif mode == 3:
            assert isinstance(raw, dict)
            raw["patient"] = deepcopy(rng.choice(mutations))
        else:
            raw = {
                "result_type": "chunked_clinical_extraction.v1",
                "chunks": [
                    {
                        "page_numbers": deepcopy(
                            rng.choice(([1], [], [True], [1, 1], ["1"], [{}]))
                        ),
                        "extraction": raw,
                    }
                ],
            }

        try:
            result = StrictLocalPipeline._validate_extraction_result(
                raw,
                {1: _PAGE},
                upload_id=f"bounded-fuzz-{case_index}",
            )
        except LocalValidationError:
            continue
        except Exception as exc:  # pragma: no cover - assertion target
            pytest.fail(
                f"ordinary exception escaped for case {case_index}: "
                f"{type(exc).__name__}"
            )
        assert isinstance(result, ClinicalDocumentExtraction)
