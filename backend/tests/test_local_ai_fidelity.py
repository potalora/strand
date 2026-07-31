"""Exact scoring gates for the strict-local synthetic and private fixture suites."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

FIXTURE_PATH = (
    Path(__file__).parent
    / "fidelity"
    / "local_ai"
    / "fixtures"
    / "synthetic"
    / "corpus-v1.json"
)
FIXTURE_SHA256 = "5654f214d4fb0a404d3abf3572812f209e05b20be0deb379dddd7db663d1a826"


def _model_extractions() -> list[dict[str, object]]:
    return [
        {
            "schema_version": "clinical-document-extraction.v1",
            "medications": [
                {
                    "name": "Metformin",
                    "dose_value": "500",
                    "dose_unit": "mg",
                    "status": "active",
                    "verbatim": (
                        "Medication: Metformin 500 mg by mouth twice daily; active."
                    ),
                    "page_number": 1,
                    "evidence_excerpt": (
                        "Medication: Metformin 500 mg by mouth twice daily; active."
                    ),
                }
            ],
            "labs": [
                {
                    "name": "Hemoglobin A1c",
                    "value": "6.8",
                    "unit": "%",
                    "date": "2026-07-01",
                    "verbatim": ("Laboratory: Hemoglobin A1c 6.8 % on 2026-07-01."),
                    "page_number": 1,
                    "evidence_excerpt": (
                        "Laboratory: Hemoglobin A1c 6.8 % on 2026-07-01."
                    ),
                }
            ],
        },
        {
            "schema_version": "clinical-document-extraction.v1",
            "labs": [
                {
                    "name": "Potassium",
                    "value": "4.1",
                    "unit": "mmol/L",
                    "verbatim": "Potassium | 4.1 | mmol/L",
                    "page_number": 1,
                    "evidence_excerpt": "Potassium | 4.1 | mmol/L",
                },
                {
                    "name": "TSH",
                    "value": "< 0.05",
                    "unit": "mIU/L",
                    "verbatim": "TSH | < 0.05 | mIU/L",
                    "page_number": 1,
                    "evidence_excerpt": "TSH | < 0.05 | mIU/L",
                },
            ],
            "vital_signs": [
                {
                    "name": "Blood pressure",
                    "value": "120/80",
                    "unit": "mmHg",
                    "verbatim": "Blood pressure | 120/80 | mmHg",
                    "page_number": 1,
                    "evidence_excerpt": "Blood pressure | 120/80 | mmHg",
                }
            ],
        },
        {
            "schema_version": "clinical-document-extraction.v1",
            "medications": [
                {
                    "name": "Warfarin",
                    "dose_value": "2.5",
                    "dose_unit": "mg",
                    "status": "stopped",
                    "date": "2026-06-30",
                    "verbatim": "Warfarin 2.5 mg was stopped on 2026-06-30.",
                    "page_number": 1,
                    "evidence_excerpt": ("Warfarin 2.5 mg was stopped on 2026-06-30."),
                }
            ],
            "conditions": [
                {
                    "name": "colon cancer",
                    "assertion": "family_history",
                    "verbatim": "FHx: colon cancer.",
                    "page_number": 1,
                    "evidence_excerpt": "FHx: colon cancer.",
                }
            ],
        },
        {
            "schema_version": "clinical-document-extraction.v1",
            "medications": [
                {
                    "name": "Insulin glargine",
                    "dose_value": "12",
                    "dose_unit": "units",
                    "verbatim": (
                        "Insulin glargine 12 units nightly started 2026-07-14."
                    ),
                    "page_number": 1,
                    "evidence_excerpt": (
                        "Insulin glargine 12 units nightly started 2026-07-14."
                    ),
                }
            ],
        },
        {
            "schema_version": "clinical-document-extraction.v1",
            "labs": [
                {
                    "name": "Glucose",
                    "value": "104",
                    "unit": "mg/dL",
                    "date": "2026-07-16",
                    "verbatim": "Glucose 104 mg/dL collected 2026-07-16.",
                    "page_number": 1,
                    "evidence_excerpt": "Glucose 104 mg/dL collected 2026-07-16.",
                }
            ],
        },
        {
            "schema_version": "clinical-document-extraction.v1",
            "medications": [
                {
                    "name": "Levothyroxine",
                    "dose_value": "75",
                    "dose_unit": "mcg",
                    "date": "2026-07-18",
                    "verbatim": "Levothyroxine 75 mcg daily on 2026-07-18.",
                    "page_number": 1,
                    "evidence_excerpt": "Levothyroxine 75 mcg daily on 2026-07-18.",
                }
            ],
        },
    ]


class _PerfectFidelityManager:
    def __init__(self, *, partial_typed_kind: str | None = None) -> None:
        corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self._ocr = ["\n".join(item["render"]["lines"]) for item in corpus["documents"]]
        self._extractions = _model_extractions()
        self._partial_typed_kind = partial_typed_kind
        self.calls: list[str] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def run(self, role: object, payload: dict[str, Any]) -> object:
        role_name = getattr(role, "value", str(role))
        self.calls.append(role_name)
        if role_name == "summary":
            assert "scratch_dir" not in payload
        else:
            assert Path(payload["scratch_dir"]).is_absolute()
        if role_name == "ocr":
            index = self.calls.count("ocr") - 1
            return {"markdown": self._ocr[index], "page_number": 1}
        if role_name == "extraction":
            index = self.calls.count("extraction") - 1
            return self._extractions[index]

        evidence = {
            item["evidence_id"]: item
            for item in payload["evidence"]
            if isinstance(item, dict)
        }
        claims: list[dict[str, object]] = []
        for fact in payload["facts"]:
            assert isinstance(fact, dict)
            evidence_ids = fact["evidence_ids"]
            assert isinstance(evidence_ids, list)
            content = json.loads(fact["content_json"])
            paths = sorted(
                {
                    path
                    for evidence_id in evidence_ids
                    for path in evidence[evidence_id]["field_paths"]
                }
            )
            typed_value = content.get("value")
            if (
                isinstance(typed_value, dict)
                and typed_value.get("kind") == self._partial_typed_kind
            ):
                paths = ["/value/kind"]
            claims.append(
                {
                    "fact_id": fact["fact_id"],
                    "field_paths": paths,
                    "evidence_ids": evidence_ids,
                }
            )
        return {
            "sections": [{"heading": "Overview", "claims": claims}],
            "uncertainties": [],
        }


def _fake_manifest() -> object:
    from app.services.local_ai.manifest import (
        LocalAIManifest,
        ManifestArtifact,
        ManifestFile,
    )
    from app.services.local_ai.types import ModelRole

    artifacts = tuple(
        ManifestArtifact(
            role=role,
            repository=f"example/{role.value}",
            revision="a" * 40,
            quantization="4bit",
            license="apache-2.0",
            attribution=f"Example {role.value}",
            decode_limits={
                "max_input_tokens": 32_768,
                "max_output_tokens": 8_192,
            },
            files=(
                ManifestFile(
                    path="config.json",
                    sha256="b" * 64,
                    size=1,
                ),
            ),
        )
        for role in ModelRole
    )
    return LocalAIManifest(
        schema_version=1,
        pack_revision="fidelity-test-v1",
        platform="apple_silicon",
        runtime={"name": "mlx-vlm", "version": "0.5.0"},
        validation_suite_version="local-ai-fixtures-v1",
        artifacts=artifacts,
    )


def _expected_from_corpus() -> tuple[list[str], list[dict[str, object]]]:
    corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    numeric_tokens: list[str] = []
    facts: list[dict[str, object]] = []
    for document in corpus["documents"]:
        numeric_tokens.extend(document["critical_numeric_tokens"])
        facts.extend(document["expected_facts"])
    return numeric_tokens, facts


def _perfect_observations() -> tuple[
    list[str],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    numeric_tokens, expected = _expected_from_corpus()
    ocr_pages = [f"source value {token}" for token in numeric_tokens]
    facts = [
        {
            **fact,
            "fact_id": f"fact-{index}",
            "evidence_ids": [f"evidence-{index}"],
        }
        for index, fact in enumerate(expected, start=1)
    ]
    claims = [
        {
            "fact_id": fact["fact_id"],
            "evidence_ids": list(fact["evidence_ids"]),
            "field_paths": ["/name"],
        }
        for fact in facts
    ]
    return ocr_pages, facts, claims


def test_synthetic_corpus_covers_required_numeric_and_clinical_edge_cases() -> None:
    from app.services.local_ai.fidelity_runner import (
        RELEASE_SYNTHETIC_DOCUMENT_COUNT,
    )

    corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert corpus["schema_version"] == 1
    assert RELEASE_SYNTHETIC_DOCUMENT_COUNT == 6
    assert len(corpus["documents"]) == RELEASE_SYNTHETIC_DOCUMENT_COUNT
    assert {item["render"]["format"] for item in corpus["documents"]} == {
        "pdf",
        "tiff",
    }
    assert {item["render"]["style"] for item in corpus["documents"]} >= {
        "clean",
        "table",
        "skew-low-contrast",
        "handwriting",
        "poor-illumination",
        "poor-scan",
    }
    documents_by_id = {item["id"]: item for item in corpus["documents"]}
    assert documents_by_id["handwriting-medication-negation"][
        "critical_numeric_tokens"
    ] == ["12", "units", "2026-07-14"]
    assert documents_by_id["poor-illumination-laboratory"][
        "critical_numeric_tokens"
    ] == ["104", "mg/dL", "2026-07-16"]
    assert documents_by_id["poor-scan-follow-up"]["critical_numeric_tokens"] == [
        "75",
        "mcg",
        "2026-07-18",
    ]
    assert all(
        document["forbidden_facts"]
        for document in (
            documents_by_id["handwriting-medication-negation"],
            documents_by_id["poor-illumination-laboratory"],
            documents_by_id["poor-scan-follow-up"],
        )
    )
    source = json.dumps(corpus, ensure_ascii=False)
    for required in (
        "6.8",
        "120/80",
        "< 0.05",
        "mg",
        "2026-07-01",
        "No evidence of pneumonia",
        "FHx: colon cancer",
        "cancelled",
        "skew-low-contrast",
        "handwriting",
        "poor-illumination",
        "poor-scan",
        "PATIENT SUMMARY",
    ):
        assert required in source


def test_fidelity_summary_projection_keeps_typed_comparator_and_ratio_values() -> None:
    from app.services.local_ai.extraction_schema import LabFact, VitalSignFact
    from app.services.local_ai.fidelity_runner import _summary_content

    comparator = LabFact(
        name="TSH",
        value="< 0.05",
        unit="mIU/L",
        verbatim="TSH | < 0.05 | mIU/L",
        page_number=1,
        evidence_excerpt="TSH | < 0.05 | mIU/L",
    )
    ratio = VitalSignFact(
        name="Blood pressure",
        value="120/80",
        unit="mmHg",
        verbatim="Blood pressure | 120/80 | mmHg",
        page_number=1,
        evidence_excerpt="Blood pressure | 120/80 | mmHg",
    )

    assert _summary_content("labs", comparator)["value"] == {
        "kind": "quantity",
        "comparator": "<",
        "number": 0.05,
        "unit": "mIU/L",
    }
    assert _summary_content("vital_signs", ratio)["value"] == {
        "kind": "ratio",
        "numerator": 120,
        "denominator": 80,
        "numerator_unit": "mmHg",
        "denominator_unit": "mmHg",
    }
    assert "unit" not in _summary_content("labs", comparator)
    assert "unit" not in _summary_content("vital_signs", ratio)


def test_synthetic_documents_render_as_deterministic_raster_pdf_and_tiff(
    tmp_path: Path,
) -> None:
    from app.services.local_ai.fidelity_runner import (
        load_fidelity_corpus,
        materialize_fidelity_documents,
    )

    corpus = load_fidelity_corpus(FIXTURE_PATH)
    first = materialize_fidelity_documents(corpus, tmp_path / "first")
    second = materialize_fidelity_documents(corpus, tmp_path / "second")

    assert [item.document_id for item in first] == [
        "medication-lab-negation",
        "table-and-operators",
        "skew-low-contrast-lifecycle",
        "handwriting-medication-negation",
        "poor-illumination-laboratory",
        "poor-scan-follow-up",
    ]
    assert [item.path.suffix for item in first] == [
        ".pdf",
        ".tiff",
        ".pdf",
        ".tiff",
        ".pdf",
        ".tiff",
    ]
    assert [item.path.read_bytes() for item in first] == [
        item.path.read_bytes() for item in second
    ]
    assert first[0].path.read_bytes().startswith(b"%PDF-")
    assert first[1].path.read_bytes().startswith((b"II*\x00", b"MM\x00*"))


def test_private_fidelity_sources_require_explicit_ground_truth(
    tmp_path: Path,
) -> None:
    from app.services.local_ai.errors import LocalValidationError
    from app.services.local_ai.fidelity_runner import load_private_fidelity_corpus

    source = tmp_path / "private-note.pdf"
    source.write_bytes(b"%PDF-private-fixture")
    corpus_path = tmp_path / "local-ai-corpus-v1.json"
    payload = {
        "schema_version": 1,
        "suite_version": "local-ai-fidelity-v1",
        "documents": [
            {
                "id": "private-note",
                "source_file": source.name,
                "critical_numeric_tokens": ["7.1"],
                "expected_facts": [],
                "forbidden_facts": [],
            }
        ],
    }
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LocalValidationError, match="corpus is invalid"):
        load_private_fidelity_corpus(tmp_path)

    payload["documents"][0]["expected_facts"] = [
        {
            "category": "labs",
            "name": "Hemoglobin A1c",
            "value": "7.1",
            "unit": "%",
        }
    ]
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")
    corpus = load_private_fidelity_corpus(tmp_path)
    assert corpus.documents[0].source_path == source


@pytest.mark.parametrize("source_file", ["../outside.pdf", "linked.pdf"])
def test_private_fidelity_sources_reject_path_escape_and_symlinks(
    tmp_path: Path,
    source_file: str,
) -> None:
    from app.services.local_ai.errors import LocalValidationError
    from app.services.local_ai.fidelity_runner import load_private_fidelity_corpus

    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"%PDF-outside")
    if source_file == "linked.pdf":
        (tmp_path / source_file).symlink_to(outside)
    payload = {
        "schema_version": 1,
        "suite_version": "local-ai-fidelity-v1",
        "documents": [
            {
                "id": "private-note",
                "source_file": source_file,
                "critical_numeric_tokens": ["7.1"],
                "expected_facts": [
                    {
                        "category": "labs",
                        "name": "Hemoglobin A1c",
                        "value": "7.1",
                        "unit": "%",
                    }
                ],
                "forbidden_facts": [],
            }
        ],
    }
    (tmp_path / "local-ai-corpus-v1.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(LocalValidationError, match="corpus is invalid"):
        load_private_fidelity_corpus(tmp_path)


def test_exact_fidelity_metrics_accept_a_fully_grounded_run() -> None:
    from app.services.local_ai.fidelity_metrics import (
        FidelityFactObservation,
        score_fidelity,
    )

    numeric_tokens, expected = _expected_from_corpus()
    pages, facts, claims = _perfect_observations()

    metrics = score_fidelity(
        numeric_observations=list(zip(numeric_tokens, pages, strict=True)),
        fact_observations=[
            FidelityFactObservation(
                expected=tuple(expected),
                forbidden=(),
                accepted=tuple(facts),
            )
        ],
        summary_claims=claims,
        schema_results=[True, True, True],
    )

    assert metrics.critical_numeric_exact == 1.0
    assert metrics.critical_precision == 1.0
    assert metrics.critical_recall == 1.0
    assert metrics.accepted_output_schema_validity == 1.0
    assert metrics.forbidden_extraction_facts == 0
    assert metrics.unsupported_summary_facts == 0
    assert metrics.accepted_facts_without_evidence == 0
    metrics.assert_release_thresholds()


def test_fidelity_gate_rejects_an_empty_summary_with_zero_fact_recall() -> None:
    from app.services.local_ai.fidelity_metrics import (
        FidelityFactObservation,
        FidelityGateError,
        score_fidelity,
    )

    pages, facts, _claims = _perfect_observations()
    numeric_tokens, expected = _expected_from_corpus()

    metrics = score_fidelity(
        numeric_observations=list(zip(numeric_tokens, pages, strict=True)),
        fact_observations=[
            FidelityFactObservation(
                expected=tuple(expected),
                forbidden=(),
                accepted=tuple(facts),
            )
        ],
        summary_claims=[],
        schema_results=[True],
    )

    assert metrics.summary_fact_recall == 0.0
    with pytest.raises(FidelityGateError, match="fact selection recall"):
        metrics.assert_release_thresholds()


def test_fidelity_gate_requires_every_typed_measurement_leaf() -> None:
    from app.services.local_ai.fidelity_metrics import (
        FidelityFactObservation,
        FidelityGateError,
        score_fidelity,
    )

    required_paths = [
        "/value/comparator",
        "/value/kind",
        "/value/number",
        "/value/unit",
    ]
    accepted = {
        "category": "labs",
        "name": "TSH",
        "value": "< 0.05",
        "unit": "mIU/L",
        "fact_id": "fact-tsh",
        "evidence_ids": ["evidence-tsh"],
        "_summary_required_field_paths": required_paths,
    }
    observation = FidelityFactObservation(
        expected=(
            {
                "category": "labs",
                "name": "TSH",
                "value": "< 0.05",
                "unit": "mIU/L",
            },
        ),
        forbidden=(),
        accepted=(accepted,),
    )

    complete = score_fidelity(
        numeric_observations=[("< 0.05", "TSH < 0.05 mIU/L")],
        fact_observations=[observation],
        summary_claims=[
            {
                "fact_id": "fact-tsh",
                "evidence_ids": ["evidence-tsh"],
                "field_paths": required_paths,
            }
        ],
        schema_results=[True],
    )
    omitted_leaf = score_fidelity(
        numeric_observations=[("< 0.05", "TSH < 0.05 mIU/L")],
        fact_observations=[observation],
        summary_claims=[
            {
                "fact_id": "fact-tsh",
                "evidence_ids": ["evidence-tsh"],
                "field_paths": required_paths[:-1],
            }
        ],
        schema_results=[True],
    )

    assert complete.summary_fact_recall == 1.0
    assert complete.summary_typed_field_recall == 1.0
    complete.assert_release_thresholds()
    assert omitted_leaf.summary_fact_recall == 1.0
    assert omitted_leaf.summary_typed_field_recall == 0.75
    with pytest.raises(FidelityGateError, match="typed-field selection recall"):
        omitted_leaf.assert_release_thresholds()


def test_fidelity_summary_claims_preserve_selected_field_paths() -> None:
    from types import SimpleNamespace

    from app.services.local_ai.fidelity_runner import _summary_claims

    document = SimpleNamespace(
        sections=(
            SimpleNamespace(
                claims=(
                    SimpleNamespace(
                        fact_id="fact-tsh",
                        evidence_ids=("evidence-tsh",),
                        field_paths=(
                            "/value/comparator",
                            "/value/kind",
                            "/value/number",
                            "/value/unit",
                        ),
                    ),
                ),
            ),
        )
    )

    assert _summary_claims(document) == [
        {
            "fact_id": "fact-tsh",
            "evidence_ids": ["evidence-tsh"],
            "field_paths": [
                "/value/comparator",
                "/value/kind",
                "/value/number",
                "/value/unit",
            ],
        }
    ]


def test_numeric_exactness_accepts_sentence_punctuation_not_longer_numbers() -> None:
    from app.services.local_ai.fidelity_metrics import score_fidelity

    punctuated = score_fidelity(
        numeric_observations=[("6.8", "Hemoglobin A1c was 6.8.")],
        fact_observations=[],
        summary_claims=[],
        schema_results=[True],
    )
    longer = score_fidelity(
        numeric_observations=[("6.8", "Hemoglobin A1c was 6.80.")],
        fact_observations=[],
        summary_claims=[],
        schema_results=[True],
    )

    assert punctuated.critical_numeric_exact == 1.0
    assert longer.critical_numeric_exact == 0.0


def test_numeric_exactness_decodes_html_entities_in_ocr_markup() -> None:
    from app.services.local_ai.fidelity_metrics import score_fidelity

    metrics = score_fidelity(
        numeric_observations=[
            ("< 0.05", "<tr><td>TSH</td><td>&lt; 0.05</td><td>mIU/L</td></tr>")
        ],
        fact_observations=[],
        summary_claims=[],
        schema_results=[True],
    )

    assert metrics.critical_numeric_exact == 1.0


def test_numeric_exactness_scores_repeated_tokens_per_document() -> None:
    from app.services.local_ai.fidelity_metrics import score_fidelity

    metrics = score_fidelity(
        numeric_observations=[
            ("95", "Glucose 95 mg/dL"),
            ("95", "Glucose result missing"),
        ],
        fact_observations=[],
        summary_claims=[],
        schema_results=[True],
    )

    assert metrics.critical_numeric_exact == 0.5


def test_extraction_scoring_cannot_swap_facts_between_documents() -> None:
    from app.services.local_ai.fidelity_metrics import (
        FidelityFactObservation,
        score_fidelity,
    )

    first = {
        "category": "conditions",
        "name": "First condition",
        "fact_id": "first",
        "evidence_ids": ["first-evidence"],
    }
    second = {
        "category": "conditions",
        "name": "Second condition",
        "fact_id": "second",
        "evidence_ids": ["second-evidence"],
    }
    metrics = score_fidelity(
        numeric_observations=[("1", "1")],
        fact_observations=[
            FidelityFactObservation(
                expected=({"category": "conditions", "name": "First condition"},),
                forbidden=(),
                accepted=(second,),
            ),
            FidelityFactObservation(
                expected=({"category": "conditions", "name": "Second condition"},),
                forbidden=(),
                accepted=(first,),
            ),
        ],
        summary_claims=[],
        schema_results=[True],
    )

    assert metrics.critical_precision == 0.0
    assert metrics.critical_recall == 0.0


def test_forbidden_extraction_facts_fail_even_when_precision_exceeds_threshold() -> (
    None
):
    from app.services.local_ai.fidelity_metrics import (
        FidelityFactObservation,
        FidelityGateError,
        score_fidelity,
    )

    expected = tuple(
        {"category": "conditions", "name": f"Expected condition {index}"}
        for index in range(99)
    )
    accepted = tuple(
        {
            **fact,
            "fact_id": f"fact-{index}",
            "evidence_ids": [f"evidence-{index}"],
        }
        for index, fact in enumerate(expected)
    ) + (
        {
            "category": "conditions",
            "name": "Forbidden condition",
            "fact_id": "forbidden",
            "evidence_ids": ["forbidden-evidence"],
        },
    )
    metrics = score_fidelity(
        numeric_observations=[("1", "1")],
        fact_observations=[
            FidelityFactObservation(
                expected=expected,
                forbidden=(
                    {
                        "category": "conditions",
                        "name": "Forbidden condition",
                    },
                ),
                accepted=accepted,
            )
        ],
        summary_claims=[],
        schema_results=[True],
    )

    assert metrics.critical_precision == 0.99
    assert metrics.critical_recall == 1.0
    assert metrics.forbidden_extraction_facts == 1
    with pytest.raises(FidelityGateError, match="forbidden fact"):
        metrics.assert_release_thresholds()


def test_fidelity_gate_rejects_numeric_drift_false_fact_and_missing_evidence() -> None:
    from app.services.local_ai.fidelity_metrics import (
        FidelityFactObservation,
        FidelityGateError,
        score_fidelity,
    )

    numeric_tokens, expected = _expected_from_corpus()
    pages, facts, claims = _perfect_observations()
    pages = [page.replace("6.8", "six point eight") for page in pages]
    facts[0] = {
        **facts[0],
        "name": "Invented diagnosis",
        "evidence_ids": [],
    }
    claims.append({"fact_id": "unsupported-fact", "evidence_ids": []})

    metrics = score_fidelity(
        numeric_observations=list(zip(numeric_tokens, pages, strict=True)),
        fact_observations=[
            FidelityFactObservation(
                expected=tuple(expected),
                forbidden=(),
                accepted=tuple(facts),
            )
        ],
        summary_claims=claims,
        schema_results=[True, False, True],
    )

    assert metrics.critical_numeric_exact < 0.99
    assert metrics.critical_recall < 0.95
    assert metrics.accepted_facts_without_evidence == 1
    assert metrics.unsupported_summary_facts == 2
    with pytest.raises(FidelityGateError):
        metrics.assert_release_thresholds()


def test_fidelity_report_is_atomic_strict_and_contains_no_fixture_content(
    tmp_path: Path,
) -> None:
    from app.services.local_ai.fidelity_metrics import FidelityMetrics
    from app.services.local_ai.fidelity_runner import (
        build_fidelity_report,
        load_fidelity_report,
        write_fidelity_report,
    )

    report = build_fidelity_report(
        metrics=FidelityMetrics(
            critical_numeric_exact=1.0,
            critical_precision=1.0,
            critical_recall=1.0,
            accepted_output_schema_validity=1.0,
            forbidden_extraction_facts=0,
            unsupported_summary_facts=0,
            accepted_facts_without_evidence=0,
        ),
        fixture_suite_version="local-ai-fidelity-v1",
        fixture_suite_sha256=FIXTURE_SHA256,
        manifest_sha256="a" * 64,
        synthetic_documents=3,
        private_documents=0,
    )
    output = tmp_path / "artifacts" / "local-ai-fidelity.json"
    write_fidelity_report(output, report)

    loaded = load_fidelity_report(output)
    loaded.assert_release_thresholds()
    assert loaded == report
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "content_free",
        "fixture_suite_version",
        "fixture_suite_sha256",
        "manifest_sha256",
        "synthetic_documents",
        "private_documents",
        "metrics",
        "private_metrics",
    }
    serialized = json.dumps(payload)
    for forbidden in ("Hemoglobin", "Metformin", "pneumonia", "colon cancer"):
        assert forbidden not in serialized


def test_fidelity_report_rejects_untrusted_suite_labels() -> None:
    from app.services.local_ai.errors import LocalValidationError
    from app.services.local_ai.fidelity_metrics import FidelityMetrics
    from app.services.local_ai.fidelity_runner import build_fidelity_report

    with pytest.raises(LocalValidationError, match="report is invalid"):
        build_fidelity_report(
            metrics=FidelityMetrics(
                critical_numeric_exact=1.0,
                critical_precision=1.0,
                critical_recall=1.0,
                accepted_output_schema_validity=1.0,
                forbidden_extraction_facts=0,
                unsupported_summary_facts=0,
                accepted_facts_without_evidence=0,
            ),
            fixture_suite_version="patient-jane-doe",
            fixture_suite_sha256=FIXTURE_SHA256,
            manifest_sha256="a" * 64,
            synthetic_documents=3,
            private_documents=0,
        )


def test_release_gate_rejects_any_replacement_synthetic_corpus(
    tmp_path: Path,
) -> None:
    from app.services.local_ai.errors import LocalValidationError
    from app.services.local_ai.fidelity_runner import (
        load_fidelity_corpus,
        validate_release_fidelity_corpus,
    )

    replacement = tmp_path / "corpus-v1.json"
    replacement.write_bytes(FIXTURE_PATH.read_bytes() + b"\n")
    corpus = load_fidelity_corpus(replacement)

    with pytest.raises(LocalValidationError, match="release fidelity corpus"):
        validate_release_fidelity_corpus(corpus)


def test_private_metrics_cannot_rescue_a_failed_synthetic_gate() -> None:
    from app.services.local_ai.fidelity_metrics import (
        FidelityGateError,
        FidelityMetrics,
    )
    from app.services.local_ai.fidelity_runner import build_fidelity_report

    passing = FidelityMetrics(
        critical_numeric_exact=1.0,
        critical_precision=1.0,
        critical_recall=1.0,
        accepted_output_schema_validity=1.0,
        forbidden_extraction_facts=0,
        unsupported_summary_facts=0,
        accepted_facts_without_evidence=0,
    )
    failed_synthetic = FidelityMetrics(
        **{
            **passing.as_report(),
            "critical_numeric_exact": 0.98,
        }
    )
    report = build_fidelity_report(
        metrics=failed_synthetic,
        private_metrics=passing,
        fixture_suite_version="local-ai-fidelity-v1",
        fixture_suite_sha256=FIXTURE_SHA256,
        manifest_sha256="a" * 64,
        synthetic_documents=3,
        private_documents=1,
    )

    with pytest.raises(FidelityGateError, match="below 99%"):
        report.assert_release_thresholds()


@pytest.mark.asyncio
async def test_real_fidelity_runner_drives_exact_role_chain_for_every_fixture(
    tmp_path: Path,
) -> None:
    from app.services.local_ai.fidelity_runner import (
        load_fidelity_corpus,
        run_fidelity_suite,
    )

    manifest = _fake_manifest()
    manifest_path = tmp_path / "manifest.lock.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )
    model_dir = tmp_path / "model-pack"
    model_dir.mkdir()
    manager = _PerfectFidelityManager()

    report = await run_fidelity_suite(
        manifest=manifest,
        manifest_path=manifest_path,
        model_dir=model_dir,
        corpus=load_fidelity_corpus(FIXTURE_PATH),
        scratch_root=tmp_path / "scratch",
        manager=manager,
    )

    assert manager.started is True
    assert manager.stopped is True
    assert manager.calls == [
        "ocr",
        "extraction",
        "ocr",
        "extraction",
        "ocr",
        "extraction",
        "ocr",
        "extraction",
        "ocr",
        "extraction",
        "ocr",
        "extraction",
        "summary",
    ]
    assert report.metrics.summary_fact_recall == 1.0
    assert report.metrics.summary_typed_field_recall == 1.0
    report.assert_release_thresholds()
    assert report.synthetic_documents == 6
    assert report.private_documents == 0
    assert not list((tmp_path / "scratch").glob("fixtures-*"))
    assert not list((tmp_path / "scratch" / "jobs").iterdir())


@pytest.mark.parametrize("partial_typed_kind", ["quantity", "ratio"])
@pytest.mark.asyncio
async def test_real_fidelity_runner_scores_worker_typed_leaf_omissions_before_expansion(
    partial_typed_kind: str,
    tmp_path: Path,
) -> None:
    from app.services.local_ai.fidelity_metrics import FidelityGateError
    from app.services.local_ai.fidelity_runner import (
        load_fidelity_corpus,
        run_fidelity_suite,
    )

    manifest = _fake_manifest()
    manifest_path = tmp_path / "manifest.lock.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )
    model_dir = tmp_path / "model-pack"
    model_dir.mkdir()

    report = await run_fidelity_suite(
        manifest=manifest,
        manifest_path=manifest_path,
        model_dir=model_dir,
        corpus=load_fidelity_corpus(FIXTURE_PATH),
        scratch_root=tmp_path / "scratch",
        manager=_PerfectFidelityManager(partial_typed_kind=partial_typed_kind),
    )

    assert report.metrics.summary_fact_recall == 1.0
    assert report.metrics.summary_typed_field_recall < 1.0
    with pytest.raises(FidelityGateError, match="typed-field selection recall"):
        report.assert_release_thresholds()


@pytest.mark.asyncio
async def test_fidelity_cli_writes_report_and_enforces_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.services.local_ai.fidelity_metrics import FidelityMetrics
    from app.services.local_ai.fidelity_runner import build_fidelity_report
    from scripts import run_local_ai_fidelity as cli

    report = build_fidelity_report(
        metrics=FidelityMetrics(
            critical_numeric_exact=1.0,
            critical_precision=1.0,
            critical_recall=1.0,
            accepted_output_schema_validity=1.0,
            forbidden_extraction_facts=0,
            unsupported_summary_facts=0,
            accepted_facts_without_evidence=0,
        ),
        fixture_suite_version="local-ai-fidelity-v1",
        fixture_suite_sha256=FIXTURE_SHA256,
        manifest_sha256="a" * 64,
        synthetic_documents=3,
        private_documents=0,
    )

    async def fake_run(**_kwargs: object) -> object:
        return report

    monkeypatch.setattr(cli, "run_installed_fidelity_suite", fake_run)
    output = tmp_path / "fidelity.json"
    exit_code = await cli._execute(cli._parser().parse_args(["--output", str(output)]))

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["content_free"] is True


@pytest.mark.local_model
@pytest.mark.fidelity
@pytest.mark.timeout(900)
@pytest.mark.asyncio
async def test_real_local_model_fidelity_report_passes_all_hard_gates(
    tmp_path: Path,
) -> None:
    """Generate and validate the real installed-pack synthetic/private report."""
    if os.environ.get("LOCAL_AI_ENABLED", "").casefold() != "true":
        pytest.skip(
            "set LOCAL_AI_ENABLED=true to run the real local-model fidelity gate"
        )

    from app.config import settings
    from app.services.local_ai.fidelity_runner import (
        load_fidelity_corpus,
        load_fidelity_report,
        run_installed_fidelity_suite,
        write_fidelity_report,
    )

    private_value = os.environ.get("REAL_MEDICAL_FIXTURES_DIR")
    report = await run_installed_fidelity_suite(
        corpus_path=FIXTURE_PATH,
        manifest_path=Path(settings.local_ai_manifest_path),
        model_root=Path(settings.local_ai_model_dir),
        scratch_root=Path(settings.local_ai_scratch_dir) / "fidelity",
        private_fixtures_dir=Path(private_value) if private_value else None,
    )
    output_value = os.environ.get("LOCAL_AI_FIDELITY_REPORT")
    output = (
        Path(output_value).expanduser() if output_value else tmp_path / "report.json"
    )
    write_fidelity_report(output, report)

    persisted = load_fidelity_report(output)
    persisted.assert_release_thresholds()
    assert persisted.synthetic_documents == len(
        load_fidelity_corpus(FIXTURE_PATH).documents
    )
