"""Tests for patient-demographics extraction + backfill during ingestion.

These guard the deterministic de-identification defense: the ``patients`` row
must carry the encrypted name/MRN/DOB so ``scrub_phi`` can strip the patient's
own identifiers before text reaches Gemini. Regression target — historically
``name_encrypted`` stayed NULL and the patient name leaked to the LLM.
"""

from __future__ import annotations

import uuid

import pytest

from app.middleware.encryption import decrypt_field
from app.models.patient import Patient
from app.models.user import User
from app.services.ai.patient_phi import patient_scrub_args
from app.services.ai.phi_scrubber import scrub_phi
from app.services.ingestion.patient_demographics import (
    backfill_patient_demographics,
    extract_epic_demographics,
    extract_fhir_demographics,
    normalize_name,
)


# --- normalize_name -------------------------------------------------------
def test_normalize_hl7_pid5_carets():
    assert normalize_name("Otalora^Pedro^^^^") == "Otalora Pedro"


def test_normalize_epic_comma():
    assert normalize_name("OTALORA,PEDRO") == "OTALORA PEDRO"


def test_normalize_empty():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""


# --- extract_fhir_demographics --------------------------------------------
def test_extract_fhir_given_family():
    res = {
        "name": [{"given": ["Jane", "Q"], "family": "Doe"}],
        "birthDate": "1990-05-15",
        "gender": "female",
        "identifier": [{"type": {"coding": [{"code": "MR"}]}, "value": "MRN999"}],
    }
    d = extract_fhir_demographics(res)
    assert d["name"] == "Jane Q Doe"
    assert d["dob"] == "1990-05-15"
    assert d["mrn"] == "MRN999"
    assert d["gender"] == "female"


def test_extract_fhir_text_name():
    d = extract_fhir_demographics({"name": [{"text": "John Smith"}]})
    assert d["name"] == "John Smith"


# --- extract_epic_demographics --------------------------------------------
def test_extract_epic_patient_tsv(tmp_path):
    p = tmp_path / "PATIENT.tsv"
    p.write_text(
        "PAT_ID\tPAT_NAME\tBIRTH_DATE\tPAT_MRN_ID\n"
        "Z1\tOTALORA,PEDRO\t7/31/1996 12:00:00 AM\tMRN123\n"
    )
    d = extract_epic_demographics(tmp_path)
    assert d["name"] == "OTALORA PEDRO"
    assert d["dob"] == "7/31/1996"
    assert d["mrn"] == "MRN123"


def test_extract_epic_missing_file(tmp_path):
    assert extract_epic_demographics(tmp_path) == {}


# --- backfill_patient_demographics ----------------------------------------
async def _make_user(db_session) -> uuid.UUID:
    """Create a persisted User so patient FK constraints are satisfied."""
    user = User(
        id=uuid.uuid4(),
        login_identifier=f"demo-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="x",
    )
    db_session.add(user)
    await db_session.commit()
    return user.id


@pytest.mark.asyncio
async def test_backfill_fills_blank_patient(db_session):
    uid = await _make_user(db_session)
    patient = Patient(id=uuid.uuid4(), user_id=uid)
    db_session.add(patient)
    await db_session.commit()

    changed = await backfill_patient_demographics(
        db_session, patient, name="Otalora^Pedro", dob="19960731", mrn="MRN42"
    )
    assert changed is True
    assert decrypt_field(patient.name_encrypted) == "Otalora Pedro"
    assert decrypt_field(patient.birth_date_encrypted) == "19960731"
    assert decrypt_field(patient.mrn_encrypted) == "MRN42"


@pytest.mark.asyncio
async def test_backfill_does_not_overwrite_existing(db_session):
    from app.middleware.encryption import encrypt_field

    uid = await _make_user(db_session)
    patient = Patient(id=uuid.uuid4(), user_id=uid)
    patient.name_encrypted = encrypt_field("Existing Name")
    db_session.add(patient)
    await db_session.commit()

    changed = await backfill_patient_demographics(db_session, patient, name="New Name")
    assert changed is False
    assert decrypt_field(patient.name_encrypted) == "Existing Name"


@pytest.mark.asyncio
async def test_backfilled_name_is_scrubbed_from_text(db_session):
    """End-to-end: a backfilled name must redact from free text via scrub_phi."""
    uid = await _make_user(db_session)
    patient = Patient(id=uuid.uuid4(), user_id=uid)
    db_session.add(patient)
    await db_session.commit()
    await backfill_patient_demographics(db_session, patient, name="Otalora^Pedro")

    text = "Thank you for getting back to me Pedro! Signed, PEDRO OTALORA."
    scrubbed, report = scrub_phi(text, **patient_scrub_args(patient))
    assert "Pedro" not in scrubbed
    assert "PEDRO" not in scrubbed
    assert "OTALORA" not in scrubbed
    assert report.get("names_scrubbed", 0) >= 3
