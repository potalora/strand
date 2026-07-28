from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.encryption import encrypt_field
from app.models.ai_summary import AISummaryPrompt
from app.models.record import HealthRecord
from app.services.local_ai.grounded_summary import SERVER_MEDICAL_DISCLAIMER
from tests.conftest import auth_headers, create_test_patient, seed_test_records

_INPUT_MARKER = "INPUT_JSON="


def _valid_pasted_reference_response(prompt: dict[str, object]) -> str:
    user_prompt = prompt["user_prompt"]
    assert isinstance(user_prompt, str)
    transport = json.loads(user_prompt.rsplit(_INPUT_MARKER, 1)[1])
    fact = transport["facts"][0]
    evidence_id = fact["evidence_ids"][0]
    evidence = next(
        item for item in transport["evidence"] if item["evidence_id"] == evidence_id
    )
    return json.dumps(
        {
            "sections": [
                {
                    "heading": "Overview",
                    "claims": [
                        {
                            "fact_id": fact["fact_id"],
                            "field_paths": evidence["field_paths"],
                            "evidence_ids": [evidence_id],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        }
    )


@pytest.mark.asyncio
async def test_build_prompt_scrubs_known_patient_name(
    client: AsyncClient, db_session: AsyncSession
):
    """The patient's own name must be removed from the prompt before it leaves the app.

    Regression for the PHI gap where build_prompt called scrub_phi WITHOUT the
    patient's known identifiers, so free-text names (which the regex patterns
    can't catch) reached the model.
    """
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    patient.name_encrypted = encrypt_field("Pedro Otalora")
    db_session.add(patient)
    await db_session.commit()

    # A record whose free text embeds the patient's name (display_text + note).
    rec = HealthRecord(
        id=uuid4(),
        patient_id=patient.id,
        user_id=UUID(uid) if isinstance(uid, str) else uid,
        record_type="document",
        fhir_resource_type="DocumentReference",
        fhir_resource={
            "resourceType": "DocumentReference",
            "note": [{"text": "Pedro Otalora seen in clinic; prescribed Rifaximin."}],
        },
        source_format="fhir_r4",
        status="current",
        display_text="Rifaximin visit note for Pedro Otalora",
        effective_date=datetime(2024, 3, 1, tzinfo=timezone.utc),
    )
    db_session.add(rec)
    await db_session.commit()

    resp = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # The patient's name must NOT appear anywhere in the outgoing prompt payload.
    assert "Pedro" not in data["user_prompt"]
    assert "Otalora" not in data["user_prompt"]
    assert "Pedro" not in data["copyable_payload"]
    assert "[PATIENT]" in data["user_prompt"]
    # Clinical content survives de-identification.
    assert "Rifaximin" in data["user_prompt"]
    # The de-identification report records the scrub.
    assert data["de_identification_report"].get("names_scrubbed", 0) >= 1


@pytest.mark.asyncio
async def test_build_prompt_unauthenticated(client: AsyncClient):
    """POST /summary/build-prompt without token returns 401."""
    resp = await client.post(
        "/api/v1/summary/build-prompt",
        json={"patient_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_build_prompt_success(client: AsyncClient, db_session: AsyncSession):
    """Build prompt returns all expected fields per handoff."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Verify all handoff fields
    for field in [
        "id",
        "summary_type",
        "system_prompt",
        "user_prompt",
        "target_model",
        "suggested_config",
        "record_count",
        "de_identification_report",
        "copyable_payload",
        "generated_at",
    ]:
        assert field in data, f"Missing field: {field}"

    assert data["summary_type"] == "full"
    assert data["record_count"] == 5
    assert data["target_model"] == "gemini-3.5-flash"
    assert isinstance(data["suggested_config"], dict)
    assert "temperature" in data["suggested_config"]


@pytest.mark.asyncio
async def test_build_prompt_no_records(client: AsyncClient, db_session: AsyncSession):
    """Build prompt with no records returns 400."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)

    resp = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_build_prompt_patient_not_found(
    client: AsyncClient, db_session: AsyncSession
):
    """Build prompt with non-existent patient returns 404."""
    headers, _ = await auth_headers(client)
    resp = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": "00000000-0000-0000-0000-000000000000",
            "summary_type": "full",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_build_prompt_no_diagnosis_disclaimer(
    client: AsyncClient, db_session: AsyncSession
):
    """System prompt contains no-diagnosis disclaimer."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=3)

    resp = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    data = resp.json()
    assert "Do not provide diagnoses" in data["system_prompt"]
    assert "treatment recommendations" in data["system_prompt"]


@pytest.mark.asyncio
async def test_build_prompt_target_model(client: AsyncClient, db_session: AsyncSession):
    """Target model is the configured prompt target (gemini-3.5-flash)."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=1)

    resp = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id)},
    )
    assert resp.json()["target_model"] == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_list_prompts(client: AsyncClient, db_session: AsyncSession):
    """Fix 6: List prompts returns full items with all fields."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=3)

    # Build a prompt first
    await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )

    resp = await client.get("/api/v1/summary/prompts", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) >= 1

    item = data["items"][0]
    for field in [
        "id",
        "summary_type",
        "system_prompt",
        "user_prompt",
        "target_model",
        "suggested_config",
        "record_count",
        "de_identification_report",
        "copyable_payload",
        "generated_at",
    ]:
        assert field in item, f"Missing field in list item: {field}"


@pytest.mark.asyncio
async def test_prompt_history_and_detail_return_saved_mode_and_provenance(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Saved summary history carries its immutable, content-free model identity."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    prompt_id = UUID(built.json()["id"])
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(AISummaryPrompt.id == prompt_id)
        )
    ).scalar_one()
    provenance = {
        "processing_mode": "validated_strict_local",
        "manifest_sha256": "d" * 64,
        "pack_revision": "apple-m4-16gb-v1",
        "model": {
            "role": "summary",
            "repository": "mlx-community/Qwen3.5-9B-MLX-4bit",
            "revision": "9" * 40,
            "quantization": "4bit",
            "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        },
    }
    prompt.processing_mode = "validated_strict_local"
    prompt.model_provenance = provenance
    prompt.response_text = "Saved strict-local summary."
    prompt.response_format = "natural_language"
    await db_session.commit()

    history = await client.get("/api/v1/summary/prompts", headers=headers)
    detail = await client.get(
        f"/api/v1/summary/prompts/{prompt_id}",
        headers=headers,
    )

    assert history.status_code == 200
    history_item = next(
        item for item in history.json()["items"] if item["id"] == str(prompt_id)
    )
    assert history_item["processing_mode"] == "validated_strict_local"
    assert history_item["model_provenance"] == provenance
    assert detail.status_code == 200
    assert detail.json()["processing_mode"] == "validated_strict_local"
    assert detail.json()["model_provenance"] == provenance


@pytest.mark.asyncio
async def test_prompt_history_omits_secret_shaped_saved_provenance(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Persisted provenance is revalidated before crossing the API boundary."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    prompt_id = UUID(built.json()["id"])
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(AISummaryPrompt.id == prompt_id)
        )
    ).scalar_one()
    secret_canary = "sk-proj-history-canary"
    prompt.processing_mode = "validated_strict_local"
    prompt.model_provenance = {
        "processing_mode": "validated_strict_local",
        "manifest_sha256": "d" * 64,
        "pack_revision": "apple-m4-16gb-v1",
        "model": {
            "role": "summary",
            "repository": "mlx-community/Qwen3.5-9B-MLX-4bit",
            "revision": "9" * 40,
            "quantization": "4bit",
            "runtime": {"name": "mlx-vlm", "version": secret_canary},
        },
    }
    await db_session.commit()

    history = await client.get("/api/v1/summary/prompts", headers=headers)
    detail = await client.get(
        f"/api/v1/summary/prompts/{prompt_id}",
        headers=headers,
    )

    assert history.status_code == 200
    history_item = next(
        item for item in history.json()["items"] if item["id"] == str(prompt_id)
    )
    assert history_item["processing_mode"] == "validated_strict_local"
    assert history_item["model_provenance"] is None
    assert secret_canary not in history.text
    assert detail.status_code == 200
    assert detail.json()["model_provenance"] is None
    assert secret_canary not in detail.text


@pytest.mark.asyncio
async def test_prompt_history_omits_patient_name_shaped_saved_provenance(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """A routed model identity cannot echo known patient PHI through history."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    patient_name = "Bartholomew Quibblesworth"
    patient.name_encrypted = encrypt_field(patient_name)
    await seed_test_records(db_session, uid, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id), "summary_type": "full"},
    )
    prompt_id = UUID(built.json()["id"])
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(AISummaryPrompt.id == prompt_id)
        )
    ).scalar_one()
    patient_name_model = patient_name.replace(" ", "-")
    prompt.processing_mode = "cloud_assisted"
    prompt.model_provenance = {
        "processing_mode": "cloud_assisted",
        "provider": "gemini",
        "model": patient_name_model,
    }
    await db_session.commit()

    history = await client.get("/api/v1/summary/prompts", headers=headers)
    detail = await client.get(
        f"/api/v1/summary/prompts/{prompt_id}",
        headers=headers,
    )

    assert history.status_code == 200
    history_item = next(
        item for item in history.json()["items"] if item["id"] == str(prompt_id)
    )
    assert history_item["processing_mode"] == "cloud_assisted"
    assert history_item["model_provenance"] is None
    assert patient_name_model not in history.text
    assert detail.status_code == 200
    assert detail.json()["model_provenance"] is None
    assert patient_name_model not in detail.text


@pytest.mark.asyncio
async def test_paste_response(client: AsyncClient, db_session: AsyncSession):
    """Fix 5: Paste response returns {id, prompt_id, response_pasted_at}."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=3)

    # Build prompt
    build_resp = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id)},
    )
    prompt_id = build_resp.json()["id"]

    # Paste a reference-only response
    resp = await client.post(
        "/api/v1/summary/paste-response",
        headers=headers,
        json={
            "prompt_id": prompt_id,
            "response_text": _valid_pasted_reference_response(build_resp.json()),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == prompt_id
    assert data["prompt_id"] == prompt_id
    assert "response_pasted_at" in data
    assert data["response_pasted_at"] is not None


@pytest.mark.asyncio
async def test_paste_response_rejects_medical_guidance(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id)},
    )

    unsafe_responses = [
        "The patient has pneumonia.",
        "Pneumonia is the likely diagnosis.",
        "Double the metformin dose.",
        "Metformin ought to be discontinued.",
        "It would be best to start insulin.",
        "My recommendation is to increase lisinopril.",
        "Recommended: start insulin.",
        "Plan: start insulin.",
        "Start insulin now.",
        "Initiate insulin now.",
        "The patient is diagnosed with pneumonia.",
        "You may want to start insulin.",
        "You might consider beginning insulin.",
        "I suspect pneumonia.",
        "- Start insulin.",
        "Stаrt insulin.",
        "```text\nSt\\x61rt insulin.\n```",
        '```json\n{"D\\u006fuble the metformin dose.": true}\n```',
        'prefix {"D\\u006fuble the metformin dose.": true}',
        "**Start** insulin.",
        "<b>Start</b> insulin.",
        "Sтart insulin.",
        "You may want to **start** insulin.",
        "I **suspect** pneumonia.",
        "Starting insulin may help.",
        "Insulin may help.",
    ]
    for unsafe_response in unsafe_responses:
        response = await client.post(
            "/api/v1/summary/paste-response",
            headers=headers,
            json={
                "prompt_id": built.json()["id"],
                "response_text": unsafe_response,
            },
        )

        assert response.status_code == 400
        assert response.json() == {
            "detail": "Pasted response must be valid reference-only JSON."
        }
        assert unsafe_response not in response.text

    prompt = await client.get(
        f"/api/v1/summary/prompts/{built.json()['id']}",
        headers=headers,
    )
    assert prompt.json()["response_text"] is None


@pytest.mark.parametrize(
    ("case_name", "locked_fields"),
    [
        ("strict_mode", {"processing_mode": "validated_strict_local"}),
        ("local_source", {"response_source": "local_ai"}),
        (
            "typed_response",
            {"typed_response": {"sections": [], "disclaimer": "locked"}},
        ),
        (
            "strict_provenance",
            {
                "model_provenance": {
                    "processing_mode": "validated_strict_local",
                    "manifest_sha256": "d" * 64,
                    "pack_revision": "apple-m4-16gb-v1",
                }
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_paste_response_cannot_overwrite_grounded_or_strict_summary(
    client: AsyncClient,
    db_session: AsyncSession,
    case_name: str,
    locked_fields: dict[str, object],
) -> None:
    headers, uid = await auth_headers(
        client,
        email=f"paste-overwrite-{case_name}@example.com",
    )
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id)},
    )
    prompt = (
        await db_session.execute(
            select(AISummaryPrompt).where(
                AISummaryPrompt.id == UUID(built.json()["id"])
            )
        )
    ).scalar_one()
    prompt.response_text = "Existing grounded response."
    prompt.typed_response = {"existing": "typed response"}
    prompt.model_provenance = {
        "processing_mode": "cloud_assisted",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
    }
    prompt.response_source = "provider"
    prompt.response_pasted_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    for field, value in locked_fields.items():
        setattr(prompt, field, value)
    await db_session.commit()
    before = {
        "response_text": prompt.response_text,
        "typed_response": prompt.typed_response,
        "model_provenance": prompt.model_provenance,
        "response_source": prompt.response_source,
        "response_pasted_at": prompt.response_pasted_at,
    }
    canary = f"overwrite-canary-{case_name}"

    response = await client.post(
        "/api/v1/summary/paste-response",
        headers=headers,
        json={"prompt_id": built.json()["id"], "response_text": canary},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Grounded summaries cannot be overwritten."}
    assert canary not in response.text
    await db_session.refresh(prompt)
    assert {
        "response_text": prompt.response_text,
        "typed_response": prompt.typed_response,
        "model_provenance": prompt.model_provenance,
        "response_source": prompt.response_source,
        "response_pasted_at": prompt.response_pasted_at,
    } == before


@pytest.mark.asyncio
async def test_paste_response_server_renders_disclaimer(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=1)
    built = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id)},
    )

    response = await client.post(
        "/api/v1/summary/paste-response",
        headers=headers,
        json={
            "prompt_id": built.json()["id"],
            "response_text": _valid_pasted_reference_response(built.json()),
        },
    )

    assert response.status_code == 200
    prompt = await client.get(
        f"/api/v1/summary/prompts/{built.json()['id']}",
        headers=headers,
    )
    assert prompt.json()["response_text"].endswith(SERVER_MEDICAL_DISCLAIMER)
    assert prompt.json()["typed_response"] is not None


@pytest.mark.asyncio
async def test_build_prompt_with_record_types(
    client: AsyncClient, db_session: AsyncSession
):
    """Test that record_types filters correctly."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "full",
            "record_types": ["medication", "observation"],
        },
    )
    # Should succeed (200) or return 400 if no records match
    assert resp.status_code in (200, 400)
    if resp.status_code == 200:
        data = resp.json()
        assert data["record_count"] > 0


@pytest.mark.asyncio
async def test_list_responses(client: AsyncClient, db_session: AsyncSession):
    """List responses only returns prompts with pasted responses."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=3)

    # Build 2 prompts
    build1 = await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id)},
    )
    await client.post(
        "/api/v1/summary/build-prompt",
        headers=headers,
        json={"patient_id": str(patient.id)},
    )

    # Paste response on only the first one
    await client.post(
        "/api/v1/summary/paste-response",
        headers=headers,
        json={
            "prompt_id": build1.json()["id"],
            "response_text": _valid_pasted_reference_response(build1.json()),
        },
    )

    resp = await client.get("/api/v1/summary/responses", headers=headers)
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["id"] == build1.json()["id"]
