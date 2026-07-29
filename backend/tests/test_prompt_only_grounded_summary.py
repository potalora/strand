from __future__ import annotations

import json
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.encryption import encrypt_field
from app.models.ai_summary import AISummaryPrompt
from app.models.local_ai import ExtractionEvidence
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.services.local_ai.grounded_summary import SERVER_MEDICAL_DISCLAIMER
from tests.conftest import auth_headers, create_test_patient, seed_test_records

_INPUT_MARKER = "INPUT_JSON="
_JSON_MARKER = "\n\n---JSON---\n"


def _transport(prompt: dict[str, object]) -> dict[str, object]:
    user_prompt = prompt["user_prompt"]
    assert isinstance(user_prompt, str)
    assert _INPUT_MARKER in user_prompt
    value = json.loads(user_prompt.rsplit(_INPUT_MARKER, 1)[1])
    assert isinstance(value, dict)
    return value


def _first_selection(prompt: dict[str, object]) -> dict[str, object]:
    transport = _transport(prompt)
    facts = transport["facts"]
    evidence = transport["evidence"]
    assert isinstance(facts, list) and facts
    assert isinstance(evidence, list) and evidence
    fact = facts[0]
    assert isinstance(fact, dict)
    evidence_by_id = {
        item["evidence_id"]: item for item in evidence if isinstance(item, dict)
    }
    evidence_id = fact["evidence_ids"][0]
    linked = evidence_by_id[evidence_id]
    return {
        "sections": [
            {
                "heading": "Overview",
                "claims": [
                    {
                        "fact_id": fact["fact_id"],
                        "field_paths": linked["field_paths"],
                        "evidence_ids": [evidence_id],
                    }
                ],
            }
        ],
        "uncertainties": [],
    }


@pytest.mark.asyncio
async def test_prompt_survivor_inherits_owner_scoped_archived_strict_evidence_until_undo(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-grounding-merged-evidence@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    survivor = HealthRecord(
        user_id=UUID(user_id),
        patient_id=patient.id,
        record_type="medication",
        fhir_resource_type="MedicationRequest",
        fhir_resource={
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {"text": "Metformin"},
            "dosageInstruction": [
                {
                    "doseAndRate": [
                        {"doseQuantity": {"value": 500, "unit": "mg"}},
                    ],
                    "route": {"text": "oral"},
                    "timing": {"code": {"text": "twice daily"}},
                }
            ],
        },
        source_format="fhir",
        source_file_id=None,
        display_text="Metformin",
        status="active",
        ai_extracted=False,
    )
    db_session.add(survivor)
    await db_session.flush()
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="archived-note.pdf",
        mime_type="application/pdf",
        file_hash="a" * 64,
        storage_path="/private/archived-note.pdf",
        processing_mode="validated_strict_local",
    )
    db_session.add(upload)
    await db_session.flush()
    archived = HealthRecord(
        user_id=UUID(user_id),
        patient_id=patient.id,
        record_type="medication",
        fhir_resource_type="MedicationRequest",
        fhir_resource={"resourceType": "MedicationRequest"},
        source_format="local_ai",
        source_file_id=upload.id,
        display_text="Metformin",
        status="active",
        ai_extracted=True,
        is_duplicate=True,
        merged_into_id=survivor.id,
    )
    db_session.add(archived)
    await db_session.flush()
    marker = "Metformin evidence from archived strict child."
    db_session.add(
        ExtractionEvidence(
            user_id=UUID(user_id),
            upload_id=upload.id,
            health_record_id=archived.id,
            page_number=1,
            section="Medications",
            excerpt=marker,
            start_offset=0,
            end_offset=len(marker),
            field_paths=["medications[0].name", "medications[0].dose"],
            source_metadata={"evidence_id": "archived-evidence"},
        )
    )
    await db_session.commit()

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "single_record",
            "record_ids": [str(survivor.id)],
        },
    )

    assert response.status_code == 200, response.text
    transport = _transport(response.json())
    assert len(transport["facts"]) == 1
    assert [item["excerpt"] for item in transport["evidence"]] == [marker]
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(
                AISummaryPrompt.id == UUID(response.json()["id"]),
                AISummaryPrompt.user_id == UUID(user_id),
            )
        )
    ).scalar_one()
    assert prompt.scope_filter["selected_record_ids"] == [str(survivor.id)]
    assert str(archived.id) not in prompt.scope_filter["selected_record_ids"]

    archived.is_duplicate = False
    archived.merged_into_id = None
    await db_session.commit()
    after_undo = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "single_record",
            "record_ids": [str(survivor.id)],
        },
    )
    assert after_undo.status_code == 200
    assert marker not in {
        item["excerpt"] for item in _transport(after_undo.json())["evidence"]
    }


@pytest.mark.asyncio
async def test_build_prompt_emits_deidentified_reference_registry_and_exact_scope(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-grounding-build@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    patient_name = "Bartholomew Quibblesworth"
    patient.name_encrypted = encrypt_field(patient_name)
    records = await seed_test_records(db_session, user_id, patient.id, count=2)
    records[0].display_text = f"Visit for {patient_name} on 2024-01-01"
    records[0].code_display = f"Visit for {patient_name} on 2024-01-01"
    records[0].fhir_resource["code"]["coding"][0]["display"] = (
        f"Visit for {patient_name} on 2024-01-01"
    )
    await db_session.commit()

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "full",
            "record_ids": [str(records[0].id)],
            "output_format": "both",
        },
    )

    assert response.status_code == 200
    body = response.json()
    serialized = f"{body['system_prompt']}\n{body['user_prompt']}"
    assert patient_name not in serialized
    assert "2024-01-01" not in serialized
    assert "Return exactly one JSON object" in body["system_prompt"]
    assert "Never emit clinical free text" in body["system_prompt"]
    assert body["processing_mode"] == "prompt_only"
    assert body["record_count"] == 1
    assert set(_transport(body)) == {
        "requested_scope",
        "facts",
        "evidence",
        "uncertainty_labels",
        "safety_rules",
    }
    transport = _transport(body)
    outbound = json.dumps(transport, sort_keys=True)
    assert str(records[0].id) not in outbound
    assert "record_ids" not in transport["requested_scope"]
    assert "record_id" not in transport["facts"][0]
    assert "source_id" not in transport["evidence"][0]
    assert transport["facts"][0]["fact_id"] == "fact_ref_0001"
    assert transport["evidence"][0]["evidence_id"] == "evidence_ref_0001"
    assert body["de_identification_report"]["names_scrubbed"] >= 1
    assert body["de_identification_report"]["dates_generalized"] >= 1

    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(
                AISummaryPrompt.id == UUID(body["id"]),
                AISummaryPrompt.user_id == UUID(str(user_id)),
            )
        )
    ).scalar_one()
    assert prompt.processing_mode == "prompt_only"
    assert prompt.scope_filter["record_ids"] == [str(records[0].id)]
    assert prompt.scope_filter["selected_record_ids"] == [str(records[0].id)]
    assert prompt.scope_filter["record_types"] is None
    assert prompt.scope_filter["output_format"] == "both"
    assert len(prompt.scope_filter["grounding_input_sha256"]) == 64


@pytest.mark.asyncio
async def test_build_prompt_redacts_split_identifiers_numeric_values_and_local_paths(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-grounding-labelled-identifiers@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    records = await seed_test_records(db_session, user_id, patient.id, count=1)
    record = records[0]
    record.record_type = "questionnaire_response"
    record.code_display = "/home/pedro/private/intake.json"
    record.display_text = "/home/pedro/private/intake.json"
    record.fhir_resource = {
        "resourceType": "QuestionnaireResponse",
        "status": "completed",
        "questionnaire": "/home/pedro/private/intake.json",
        "item": [
            {
                "text": "Member ID",
                "answer": [{"valueInteger": 8675309}],
            },
            {
                "text": "Accession number",
                "answer": [{"valueString": "ZXCV1234"}],
            },
        ],
    }
    await db_session.commit()

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "single_record",
            "record_ids": [str(record.id)],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    outbound = f"{body['system_prompt']}\n{body['user_prompt']}"
    assert "8675309" not in outbound
    assert "ZXCV1234" not in outbound
    assert "/home/pedro/private/intake.json" not in outbound
    assert "[HEALTH_PLAN]" in outbound
    assert "[ACCOUNT]" in outbound
    assert "[LOCAL_PATH]" in outbound
    transport = _transport(body)
    evidence = transport["evidence"]
    assert isinstance(evidence, list) and evidence
    assert "8675309" not in evidence[0]["excerpt"]
    assert "ZXCV1234" not in evidence[0]["excerpt"]


@pytest.mark.parametrize(
    "date_fields",
    [
        {},
        {"date_from": "2024-01-01T00:00:00Z"},
    ],
)
@pytest.mark.asyncio
async def test_build_prompt_rejects_one_sided_date_scope(
    client: AsyncClient,
    db_session: AsyncSession,
    date_fields: dict[str, str],
) -> None:
    headers, user_id = await auth_headers(
        client,
        email=f"prompt-grounding-date-scope-{len(date_fields)}@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=1)

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "date_range",
            **date_fields,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Both dates are required for a date range summary"
    }


@pytest.mark.asyncio
async def test_build_prompt_rejects_category_scope_without_category(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-grounding-missing-category@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=1)

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "category",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Category is required for a category summary"}


@pytest.mark.asyncio
async def test_build_prompt_rejects_conflicting_record_selectors(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-grounding-conflicting-scope@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    records = await seed_test_records(db_session, user_id, patient.id, count=1)

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "full",
            "category": "medication",
            "record_ids": [str(records[0].id)],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Choose only one record selector for a grounded prompt"
    }


@pytest.mark.parametrize(
    "extra_fields",
    [
        {"record_types": ["condition"]},
        {"category": "condition", "record_types": ["condition"]},
    ],
)
@pytest.mark.asyncio
async def test_build_prompt_rejects_each_conflicting_selector_pair(
    client: AsyncClient,
    db_session: AsyncSession,
    extra_fields: dict[str, object],
) -> None:
    headers, user_id = await auth_headers(
        client,
        email=f"prompt-grounding-selector-pair-{len(str(extra_fields))}@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    records = await seed_test_records(db_session, user_id, patient.id, count=1)
    request_body: dict[str, object] = {
        "patient_id": str(patient.id),
        "summary_type": "full",
        **extra_fields,
    }
    if "category" not in extra_fields:
        request_body["record_ids"] = [str(records[0].id)]

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json=request_body,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Choose only one record selector for a grounded prompt"
    }


@pytest.mark.asyncio
async def test_build_prompt_rejects_date_range_with_record_selector(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-grounding-date-selector@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    records = await seed_test_records(db_session, user_id, patient.id, count=1)

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "date_range",
            "date_from": "2024-01-01T00:00:00Z",
            "date_to": "2024-01-31T23:59:59Z",
            "record_ids": [str(records[0].id)],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Date ranges cannot be combined with another grounded prompt selector"
    }


@pytest.mark.asyncio
async def test_build_prompt_rejects_duplicate_record_ids(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-grounding-duplicate-records@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    records = await seed_test_records(db_session, user_id, patient.id, count=1)

    response = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "full",
            "record_ids": [str(records[0].id), str(records[0].id)],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Grounded prompt record identifiers must be unique"
    }


@pytest.mark.parametrize("output_format", ["natural_language", "json", "both"])
@pytest.mark.asyncio
async def test_paste_response_accepts_only_references_and_server_renders_format(
    client: AsyncClient,
    db_session: AsyncSession,
    output_format: str,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email=f"prompt-grounding-paste-{output_format}@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "full",
            "output_format": output_format,
        },
    )
    selection = _first_selection(built.json())

    response = await client.post(
        "/api/v1/summary/paste-response",
        headers=headers,
        json={
            "prompt_id": built.json()["id"],
            "response_text": json.dumps(selection),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["typed_response"] != selection
    assert "fact_ref_0001" not in json.dumps(body["typed_response"])
    assert "evidence_ref_0001" not in json.dumps(body["typed_response"])
    if output_format == "natural_language":
        assert body["natural_language"].startswith("## Overview")
        assert body["natural_language"].endswith(SERVER_MEDICAL_DISCLAIMER)
        assert body["json_data"] is None
    elif output_format == "json":
        assert body["natural_language"] is None
        assert body["json_data"] == body["typed_response"]
    else:
        assert body["natural_language"].startswith("## Overview")
        assert body["natural_language"].endswith(SERVER_MEDICAL_DISCLAIMER)
        assert body["json_data"] == body["typed_response"]

    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(
                AISummaryPrompt.id == UUID(built.json()["id"])
            )
        )
    ).scalar_one()
    assert prompt.typed_response == body["typed_response"]
    assert prompt.response_source == "pasted_grounded"
    assert prompt.response_format == output_format
    if output_format == "natural_language":
        assert prompt.response_text == body["natural_language"]
    elif output_format == "json":
        assert json.loads(prompt.response_text) == body["typed_response"]
    else:
        markdown, raw_json = prompt.response_text.split(_JSON_MARKER, 1)
        assert markdown == body["natural_language"]
        assert json.loads(raw_json) == body["typed_response"]


@pytest.mark.parametrize(
    "response_text",
    [
        "Insulin might be appropriate.",
        '{"summary":"Insulin could be considered.","sections":[],"uncertainties":[]}',
        (
            '{"sections":[{"heading":"Overview","claims":[{"fact_id":"fact1_unknown",'
            '"field_paths":["/name"],"evidence_ids":["evidence1_unknown"]}]}],'
            '"uncertainties":[]}'
        ),
    ],
)
@pytest.mark.asyncio
async def test_paste_response_rejects_non_reference_output_without_persistence(
    client: AsyncClient,
    db_session: AsyncSession,
    response_text: str,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email=f"prompt-grounding-reject-{abs(hash(response_text))}@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )

    response = await client.post(
        "/api/v1/summary/paste-response",
        headers=headers,
        json={
            "prompt_id": built.json()["id"],
            "response_text": response_text,
        },
    )

    assert response.status_code == 400
    assert response_text not in response.text
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(
                AISummaryPrompt.id == UUID(built.json()["id"])
            )
        )
    ).scalar_one()
    await db_session.refresh(prompt)
    assert prompt.response_text is None
    assert prompt.typed_response is None
    assert prompt.response_pasted_at is None


@pytest.mark.asyncio
async def test_paste_response_fails_closed_when_prompt_registry_changed(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-grounding-stale@example.com",
    )
    patient = await create_test_patient(db_session, user_id)
    records = await seed_test_records(db_session, user_id, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    selection = _first_selection(built.json())
    records[0].code_display = "Changed after prompt creation"
    records[0].display_text = "Changed after prompt creation"
    records[0].fhir_resource["code"]["coding"][0]["display"] = (
        "Changed after prompt creation"
    )
    await db_session.commit()

    response = await client.post(
        "/api/v1/summary/paste-response",
        headers=headers,
        json={
            "prompt_id": built.json()["id"],
            "response_text": json.dumps(selection),
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Prompt records changed; build a new prompt."}
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(
                AISummaryPrompt.id == UUID(built.json()["id"])
            )
        )
    ).scalar_one()
    await db_session.refresh(prompt)
    assert prompt.response_text is None
    assert prompt.typed_response is None


@pytest.mark.asyncio
async def test_paste_response_keeps_prompt_owner_scoped(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    owner_headers, owner_id = await auth_headers(
        client,
        email="prompt-grounding-owner@example.com",
    )
    patient = await create_test_patient(db_session, owner_id)
    await seed_test_records(db_session, owner_id, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=owner_headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    selection = _first_selection(built.json())
    other_headers, _other_id = await auth_headers(
        client,
        email="prompt-grounding-other@example.com",
    )

    response = await client.post(
        "/api/v1/summary/paste-response",
        headers=other_headers,
        json={
            "prompt_id": built.json()["id"],
            "response_text": json.dumps(selection),
        },
    )

    assert response.status_code == 404
