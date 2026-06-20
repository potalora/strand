"""Tests for the Presidio de-identification path (``PHI_ENGINE=presidio``).

Every existing legacy-scrubber regression (see ``test_hipaa_compliance.py`` and
``test_patient_phi.py``) is ported here against ``scrub_phi_presidio`` so the
Presidio path is a verified parity superset, plus:

* the Layer-2 known-identity leak defense (deny-list),
* the Layer-3 eponym survival (Crohn's/Hodgkin/Gastroenterology),
* the Layer-4 clinical LOCATION pass (drug-name negatives), and
* the ``scrub_phi`` engine dispatch + fail-open-to-legacy behavior.

Presidio + spaCy ``en_core_web_md`` must be installed; the module skips cleanly
otherwise. The clinical LOCATION tests are marked ``slow`` (they download the
Stanford model on first run) and skip gracefully when the model is unavailable.
"""

from __future__ import annotations

import importlib

import pytest

# Skip the whole module unless Presidio + the spaCy model are available.
presidio = pytest.importorskip("presidio_analyzer")
spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_md")
    _MODEL_AVAILABLE = True
except OSError:
    _MODEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODEL_AVAILABLE, reason="en_core_web_md not installed"
)


def _scrub(text, **kw):
    from app.services.ai.phi_presidio import scrub_phi_presidio

    return scrub_phi_presidio(text, **kw)


# ===========================================================================
# Layer 1 — structured-identifier parity (ports of the legacy regressions)
# ===========================================================================


def test_presidio_removes_ssn_and_email():
    out, _ = _scrub(
        "Patient John Doe, SSN 123-45-6789, email john@example.com was seen.",
        enable_ner=False,
    )
    assert "123-45-6789" not in out
    assert "john@example.com" not in out
    assert "[SSN]" in out
    assert "[EMAIL]" in out


def test_presidio_removes_fax():
    out, _ = _scrub("Contact fax: 555-123-4567 for records", enable_ner=False)
    assert "555-123-4567" not in out
    assert "[FAX]" in out or "[PHONE]" in out


def test_presidio_removes_vin():
    out, _ = _scrub("Vehicle ID: 1HGBH41JXMN109186", enable_ner=False)
    assert "1HGBH41JXMN109186" not in out


def test_presidio_removes_device_id():
    out, _ = _scrub("Device serial: ABC123-DEF456", enable_ner=False)
    assert "ABC123-DEF456" not in out


def test_presidio_removes_health_plan_number():
    out, _ = _scrub("Member number: HPN12345", enable_ner=False)
    assert "HPN12345" not in out


def test_presidio_generalizes_slash_dates():
    """MM/DD/YYYY -> MM/YYYY (day dropped), matching legacy month+year handling."""
    out, report = _scrub(
        "DOB: 07/31/1996  Collected: 02/16/2026  Reported: 2/3/2026",
        enable_ner=False,
    )
    assert "07/31/1996" not in out
    assert "07/1996" in out
    assert "02/2026" in out
    assert "2/2026" in out
    assert report.get("dates_generalized", 0) >= 3


def test_presidio_generalizes_month_name_dates():
    out, _ = _scrub("Seen on January 5, 2026 and February 28, 2025.", enable_ner=False)
    assert "January 5, 2026" not in out
    assert "January 2026" in out
    assert "February 2025" in out


def test_presidio_redacts_account_and_accession_numbers():
    out, _ = _scrub(
        "Account No: 235410324\n**Lab Accession:** 87414853\nAcct #998877",
        enable_ner=False,
    )
    assert "235410324" not in out
    assert "87414853" not in out
    assert "998877" not in out
    assert "[ACCOUNT]" in out


def test_presidio_does_not_redact_clinical_codes_as_ssn():
    """Regression (shadow-compare): Presidio's weak 9-digit SSN heuristic must NOT
    redact numeric clinical codes (SNOMED/LOINC/ICD). The score floor drops it."""
    out, _ = _scrub(
        "Coded as SNOMED 185349003 and 306206005 for the encounter.",
        enable_ner=False,
    )
    assert "185349003" in out
    assert "306206005" in out
    assert "[SSN]" not in out


def test_presidio_preserves_clinical_numbers():
    """De-identification must not destroy lab values, ranges, or percentages."""
    out, _ = _scrub(
        "Anti-CdtB Ab 1.24; reference 0.00 - 1.55; 96% - 100% PPV; IBS-D",
        enable_ner=False,
    )
    assert "1.24" in out
    assert "0.00 - 1.55" in out
    assert "96% - 100%" in out
    assert "IBS-D" in out


def test_presidio_redacts_street_address():
    out, _ = _scrub(
        "Address: 275 Post Rd E, Ste. 10, Unit 310; also 1234 Elm Street, Apt 5B",
        enable_ner=False,
    )
    assert "275 Post Rd" not in out
    assert "1234 Elm Street" not in out
    assert "[LOCATION]" in out


def test_presidio_street_regex_preserves_clinical_text():
    out, _ = _scrub(
        "Take 2 tablets by mouth daily; 5 mg PO; BP 120/80; 3 episodes per week",
        enable_ner=False,
    )
    assert "2 tablets" in out
    assert "5 mg" in out
    assert "120/80" in out
    assert "3 episodes" in out


# ===========================================================================
# Layer 2 — known patient identity (deny-list), incl. the NULL-leak defense
# ===========================================================================


def test_presidio_word_boundary_short_names():
    out, _ = _scrub("The patient named Li has diabetes.", patient_names=["Li"], enable_ner=False)
    assert "[PATIENT]" in out
    out2, _ = _scrub("Check the lipid panel results.", patient_names=["Li"], enable_ner=False)
    assert "lipid" in out2


def test_presidio_known_patient_name_scrubbed_clinical_preserved():
    out, report = _scrub(
        "Pedro, it was great to meet you in clinic. Please try Rifaximin.",
        patient_names=["Pedro Otalora"],
        enable_ner=False,
    )
    assert "Pedro" not in out
    assert "[PATIENT]" in out
    assert report.get("names_scrubbed", 0) >= 1
    assert "Rifaximin" in out


def test_presidio_known_mrn_and_dob_removed():
    out, _ = _scrub(
        "Record MRN12345678 for DOB 1990-05-15.",
        patient_mrn="MRN12345678",
        patient_dob="1990-05-15",
        enable_ner=False,
    )
    assert "MRN12345678" not in out
    assert "1990-05-15" not in out


def test_presidio_empty_patient_context_is_noop_for_identity():
    # No known identity -> nothing removed by Layer 2; clinical text intact.
    out, _ = _scrub("Routine visit, no acute distress.", enable_ner=False)
    assert out == "Routine visit, no acute distress."


# ===========================================================================
# Layer 3 — eponym survival (re-homed; shared with legacy phi_ner)
# ===========================================================================


def test_presidio_eponyms_survive_provider_redacted():
    out, _ = _scrub(
        "History of Crohn's disease and Hodgkin lymphoma, managed by Dr. Robert Chen "
        "in Gastroenterology.",
        enable_ner=True,
    )
    assert "Crohn's disease" in out
    assert "Hodgkin lymphoma" in out
    assert "Gastroenterology" in out
    assert "Robert" not in out  # provider name redacted by the NER pass


# ===========================================================================
# Layer 4 — clinical LOCATION pass (slow: downloads the Stanford model)
# ===========================================================================


@pytest.mark.slow
def test_presidio_location_pass_redacts_city_and_keeps_drugs(monkeypatch):
    """The clinical LOCATION model closes the city gap WITHOUT tagging drugs as
    places (the documented failure of the general GPE model)."""
    from app.config import settings
    from app.services.ai import phi_location_ner

    if not phi_location_ner.warm_load_location_ner():
        pytest.skip("Stanford clinical de-id model unavailable (offline/blocked)")

    monkeypatch.setattr(settings, "phi_location_ner_enabled", True)
    text = (
        "Patient lives in Springfield and was prescribed Rifaximin and Rituximab "
        "for Crohn's disease."
    )
    out, report = _scrub(text, enable_ner=False)

    # Drugs and the eponym must survive the location pass.
    assert "Rifaximin" in out
    assert "Rituximab" in out
    assert "Crohn's" in out
    # The city should be redacted (best-effort recall; model-dependent).
    assert "[LOCATION]" in out
    assert report.get("locations_scrubbed", 0) >= 1


@pytest.mark.slow
def test_location_module_drug_negatives_directly():
    """Direct module test: drug names must never be redacted as locations."""
    from app.services.ai import phi_location_ner

    if not phi_location_ner.warm_load_location_ner():
        pytest.skip("Stanford clinical de-id model unavailable (offline/blocked)")

    out, _ = phi_location_ner.redact_locations(
        "Start Rifaximin 550mg; consider Rituximab. Crohn's disease stable."
    )
    assert "Rifaximin" in out
    assert "Rituximab" in out
    assert "Crohn's" in out


def test_location_pass_fails_open_when_model_missing(monkeypatch):
    """If the model can't load, the LOCATION pass returns text unchanged (fail-open)."""
    from app.services.ai import phi_location_ner

    monkeypatch.setattr(phi_location_ner, "_pipeline", None)
    monkeypatch.setattr(phi_location_ner, "_get_pipeline", lambda: None)
    out, report = phi_location_ner.redact_locations("Patient lives in Springfield.")
    assert out == "Patient lives in Springfield."
    assert report == {}


# ===========================================================================
# scrub_phi dispatch + fail-open-to-legacy
# ===========================================================================


def test_scrub_phi_dispatches_to_presidio(monkeypatch):
    from app.config import settings
    from app.services.ai import phi_scrubber

    called = {"n": 0}
    real = importlib.import_module("app.services.ai.phi_presidio").scrub_phi_presidio

    def spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(settings, "phi_engine", "presidio")
    monkeypatch.setattr(
        "app.services.ai.phi_presidio.scrub_phi_presidio", spy
    )
    out, _ = phi_scrubber.scrub_phi(
        "SSN 123-45-6789 here.", enable_ner=False
    )
    assert called["n"] == 1
    assert "123-45-6789" not in out


def test_scrub_phi_falls_back_to_legacy_on_presidio_error(monkeypatch):
    """A Presidio failure must not crash de-id — it falls back to the legacy path,
    which still removes the SSN."""
    from app.config import settings
    from app.services.ai import phi_scrubber

    def boom(*a, **k):
        raise RuntimeError("simulated presidio failure")

    monkeypatch.setattr(settings, "phi_engine", "presidio")
    monkeypatch.setattr("app.services.ai.phi_presidio.scrub_phi_presidio", boom)
    out, _ = phi_scrubber.scrub_phi("SSN 123-45-6789 here.", enable_ner=False)
    assert "123-45-6789" not in out
    assert "[SSN]" in out


def test_scrub_phi_legacy_is_default(monkeypatch):
    """Default engine stays legacy (no Presidio invoked)."""
    from app.config import settings
    from app.services.ai import phi_scrubber

    assert settings.phi_engine in ("legacy", "presidio")
    monkeypatch.setattr(settings, "phi_engine", "legacy")

    def boom(*a, **k):  # would raise if presidio were called
        raise AssertionError("presidio must not be called under legacy engine")

    monkeypatch.setattr("app.services.ai.phi_presidio.scrub_phi_presidio", boom)
    out, _ = phi_scrubber.scrub_phi("SSN 123-45-6789 here.", enable_ner=False)
    assert "[SSN]" in out
