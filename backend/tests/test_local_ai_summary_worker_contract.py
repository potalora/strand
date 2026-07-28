from __future__ import annotations

import json
import sys
from pathlib import Path

from app.services.local_ai.grounded_summary import build_grounded_summary_input

WORKER_SRC = (
    Path(__file__).resolve().parents[2] / "workers" / "local_ai" / "apple_mlx" / "src"
)
sys.path.insert(0, str(WORKER_SRC))


class _Tokenizer:
    model_max_length = 65_536

    def encode(self, value: str, **_kwargs: object) -> list[str]:
        return list(value)


class _Processor:
    tokenizer = _Tokenizer()


class _Config:
    max_position_embeddings = 65_536


class _Model:
    config = _Config()


def test_reviewed_backend_grounding_contract_is_accepted_by_real_qwen_worker() -> None:
    from local_ai_mlx_worker.common import LoadedRole
    from local_ai_mlx_worker.qwen_summary import run_summary

    summary_input = build_grounded_summary_input(
        requested_scope={"summary_type": "full_health"},
        facts=[
            {
                "record_id": "record-1",
                "content": {
                    "record_type": "medication",
                    "name": "Metformin",
                    "dose_value": 500,
                    "dose_unit": "mg",
                    "end_date": "2024-07-01",
                    "status": "active",
                },
                "evidence_ids": ["source-evidence-1"],
            }
        ],
        evidence=[
            {
                "id": "source-evidence-1",
                "excerpt": "Metformin 500 mg active through July 1, 2024.",
                "page_number": 1,
                "section": "Medications",
                "field_paths": [
                    "/name",
                    "/dose_value",
                    "/dose_unit",
                    "/end_date",
                    "/status",
                ],
            }
        ],
        uncertainty_labels=[],
    )
    fact = summary_input.facts[0]
    payload = summary_input.model_dump(mode="json")
    payload.update(
        {
            "job_id": "job-1",
            "manifest_path": "/verified-pack/locked-manifest.json",
            "model_dir": "/verified-pack",
            "manifest_identity": {
                "schema_version": 1,
                "pack_revision": "apple-m4-16gb-v1",
                "platform": "apple_silicon",
                "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
                "validation_suite_version": "fixtures-v1",
                "roles": ["extraction", "ocr", "summary"],
                "role": "summary",
                "repository": "owner/summary",
                "revision": "0" * 40,
                "quantization": "4bit",
                "license": "apache-2.0",
                "attribution": "https://huggingface.co/owner/summary",
                "manifest_sha256": "a" * 64,
            },
            "max_output_tokens": 4096,
        }
    )
    assert set(payload) == {
        "job_id",
        "requested_scope",
        "facts",
        "evidence",
        "uncertainty_labels",
        "safety_rules",
        "manifest_path",
        "model_dir",
        "manifest_identity",
        "max_output_tokens",
    }

    loaded = LoadedRole(
        role="summary",
        model=_Model(),
        processor=_Processor(),
        model_path="/verified-pack/summary",
        decode_limits={"max_input_tokens": 32_768, "max_output_tokens": 4096},
        repository_files_used=frozenset({"config.json", "model.safetensors"}),
    )
    result = run_summary(
        payload,
        loaded=loaded,
        generate_fn=lambda **_kwargs: json.dumps(
            {
                "sections": [
                    {
                        "heading": "Medications",
                        "claims": [
                            {
                                "fact_id": fact.fact_id,
                                "field_paths": [
                                    "/name",
                                    "/dose_value",
                                    "/dose_unit",
                                    "/end_date",
                                    "/status",
                                ],
                                "evidence_ids": list(fact.evidence_ids),
                            }
                        ],
                    }
                ],
                "uncertainties": [],
            }
        ),
    )

    assert result["sections"][0]["claims"][0]["fact_id"] == fact.fact_id


def test_qwen_worker_accepts_typed_comparator_and_ratio_observation_values() -> None:
    from local_ai_mlx_worker.common import LoadedRole
    from local_ai_mlx_worker.qwen_summary import run_summary

    summary_input = build_grounded_summary_input(
        requested_scope={"summary_type": "full_health"},
        facts=[
            {
                "record_id": "observation-comparator",
                "content": {
                    "record_type": "observation",
                    "name": "TSH",
                    "value": {
                        "kind": "quantity",
                        "comparator": "<",
                        "number": 0.05,
                        "unit": "mIU/L",
                    },
                },
                "evidence_ids": ["comparator-evidence"],
            },
            {
                "record_id": "observation-ratio",
                "content": {
                    "record_type": "observation",
                    "name": "Blood pressure",
                    "value": {
                        "kind": "ratio",
                        "numerator": 120,
                        "denominator": 80,
                        "numerator_unit": "mmHg",
                        "denominator_unit": "mmHg",
                    },
                },
                "evidence_ids": ["ratio-evidence"],
            },
        ],
        evidence=[
            {
                "id": "comparator-evidence",
                "excerpt": "TSH <0.05 mIU/L",
                "field_paths": [
                    "/name",
                    "/value/comparator",
                    "/value/kind",
                    "/value/number",
                    "/value/unit",
                ],
            },
            {
                "id": "ratio-evidence",
                "excerpt": "Blood pressure 120/80 mmHg",
                "field_paths": [
                    "/name",
                    "/value/denominator",
                    "/value/denominator_unit",
                    "/value/kind",
                    "/value/numerator",
                    "/value/numerator_unit",
                ],
            },
        ],
    )
    payload = summary_input.model_dump(mode="json")
    payload.update(
        {
            "job_id": "job-observations",
            "manifest_path": "/verified-pack/locked-manifest.json",
            "model_dir": "/verified-pack",
            "manifest_identity": {
                "schema_version": 1,
                "pack_revision": "apple-m4-16gb-v1",
                "platform": "apple_silicon",
                "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
                "validation_suite_version": "fixtures-v1",
                "roles": ["extraction", "ocr", "summary"],
                "role": "summary",
                "repository": "owner/summary",
                "revision": "0" * 40,
                "quantization": "4bit",
                "license": "apache-2.0",
                "attribution": "https://huggingface.co/owner/summary",
                "manifest_sha256": "a" * 64,
            },
            "max_output_tokens": 4096,
        }
    )
    loaded = LoadedRole(
        role="summary",
        model=_Model(),
        processor=_Processor(),
        model_path="/verified-pack/summary",
        decode_limits={"max_input_tokens": 32_768, "max_output_tokens": 4096},
        repository_files_used=frozenset({"config.json", "model.safetensors"}),
    )

    result = run_summary(
        payload,
        loaded=loaded,
        generate_fn=lambda **_kwargs: json.dumps(
            {
                "sections": [
                    {
                        "heading": "Observations",
                        "claims": [
                            {
                                "fact_id": fact.fact_id,
                                "field_paths": [
                                    next(
                                        field.path
                                        for field in fact.fields
                                        if field.path.startswith("/value/")
                                    )
                                ],
                                "evidence_ids": list(fact.evidence_ids),
                            }
                            for fact in summary_input.facts
                        ],
                    }
                ],
                "uncertainties": [],
            }
        ),
    )

    assert len(result["sections"][0]["claims"]) == 2
