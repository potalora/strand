"""Tests for targeted patient-context PHI scrubbing.

These cover `patient_scrub_args`, which decrypts a patient's known identifiers
(name / MRN / birth date) into keyword arguments for `scrub_phi` so the
patient's own PHI is removed from free text before it is sent to Gemini.

This is *targeted* de-identification — it removes identifiers we already know.
Detecting arbitrary or provider names requires NER and is tracked separately.
"""

from __future__ import annotations

import uuid


from app.middleware.encryption import encrypt_field
from app.models.patient import Patient
from app.services.ai.patient_phi import patient_scrub_args
from app.services.ai.phi_scrubber import scrub_phi


def _patient(
    name: str | None = None,
    mrn: str | None = None,
    dob: str | None = None,
) -> Patient:
    """Build an unpersisted Patient with encrypted demographics for testing."""
    p = Patient(user_id=uuid.uuid4())
    p.name_encrypted = encrypt_field(name) if name else None
    p.mrn_encrypted = encrypt_field(mrn) if mrn else None
    p.birth_date_encrypted = encrypt_field(dob) if dob else None
    return p


def test_decrypts_patient_name_into_names_list():
    args = patient_scrub_args(_patient(name="Pedro Otalora"))
    assert args["patient_names"] == ["Pedro Otalora"]


def test_decrypts_mrn_and_dob():
    args = patient_scrub_args(
        _patient(name="Jane Doe", mrn="MRN12345678", dob="1990-05-15")
    )
    assert args["patient_mrn"] == "MRN12345678"
    assert args["patient_dob"] == "1990-05-15"


def test_none_returns_empty_dict():
    assert patient_scrub_args(None) == {}


def test_patient_with_no_demographics_returns_empty_dict():
    # All *_encrypted fields are None -> nothing to scrub.
    assert patient_scrub_args(Patient(user_id=uuid.uuid4())) == {}


def test_multiple_patients_aggregate_names():
    args = patient_scrub_args(
        [_patient(name="Pedro Otalora"), _patient(name="Maria Otalora")]
    )
    assert set(args["patient_names"]) == {"Pedro Otalora", "Maria Otalora"}


def test_duplicate_names_are_deduplicated():
    args = patient_scrub_args(
        [_patient(name="Pedro Otalora"), _patient(name="Pedro Otalora")]
    )
    assert args["patient_names"] == ["Pedro Otalora"]


def test_decryption_failure_is_non_fatal():
    # A corrupt/invalid blob must NOT raise — the pipeline falls through to the
    # regex scrubber rather than crashing (or leaking by aborting the caller).
    p = Patient(user_id=uuid.uuid4())
    p.name_encrypted = b"not-valid-ciphertext"
    assert patient_scrub_args(p) == {}


def test_end_to_end_known_patient_name_is_scrubbed_clinical_content_preserved():
    p = _patient(name="Pedro Otalora")
    text = "Pedro, it was great to meet you in clinic. Please try Rifaximin."
    scrubbed, report = scrub_phi(text, **patient_scrub_args(p))

    assert "Pedro" not in scrubbed
    assert "[PATIENT]" in scrubbed
    assert report.get("names_scrubbed", 0) >= 1
    # Clinical content must survive de-identification.
    assert "Rifaximin" in scrubbed
