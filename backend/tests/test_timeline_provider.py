"""Unit tests for timeline provider-display extraction (C1, Timeline half).

The Timeline is a compact chronological view; it carries a flat human-readable
provider string per event rather than the full FHIR participant/performer
structure that RecordDetail renders. These tests pin the extraction logic that
mirrors the frontend `performerName` helper: prefer a `display`, drop opaque
UUID-only references, and read the R4B paths per resource type.
"""

from __future__ import annotations

from app.services.timeline_service import extract_provider_display


class TestExtractProviderDisplay:
    def test_encounter_participant_individual_display(self) -> None:
        fhir = {
            "resourceType": "Encounter",
            "participant": [
                {"individual": {"reference": "Practitioner/abc", "display": "Dr. Jane Smith"}}
            ],
        }
        assert extract_provider_display(fhir, "encounter") == "Dr. Jane Smith"

    def test_observation_performer_flat_display(self) -> None:
        fhir = {
            "resourceType": "Observation",
            "performer": [{"reference": "Practitioner/xyz", "display": "Quest Diagnostics"}],
        }
        assert extract_provider_display(fhir, "observation") == "Quest Diagnostics"

    def test_procedure_performer_actor_display(self) -> None:
        fhir = {
            "resourceType": "Procedure",
            "performer": [{"actor": {"reference": "Practitioner/p1", "display": "Dr. Lee"}}],
        }
        assert extract_provider_display(fhir, "procedure") == "Dr. Lee"

    def test_uuid_only_reference_is_dropped_as_opaque(self) -> None:
        # No display + a "Type/<uuid>" reference renders as nothing (opaque).
        fhir = {
            "resourceType": "Observation",
            "performer": [
                {"reference": "Practitioner/3f2504e0-4f89-41d3-9a0c-0305e82c3301"}
            ],
        }
        assert extract_provider_display(fhir, "observation") is None

    def test_non_uuid_reference_without_display_is_kept(self) -> None:
        fhir = {
            "resourceType": "Observation",
            "performer": [{"reference": "Practitioner/Dr-Smith"}],
        }
        assert extract_provider_display(fhir, "observation") == "Practitioner/Dr-Smith"

    def test_missing_provider_returns_none(self) -> None:
        assert extract_provider_display({"resourceType": "Observation"}, "observation") is None
        assert extract_provider_display(None, "observation") is None
        assert extract_provider_display({}, "encounter") is None

    def test_first_named_provider_wins_over_opaque(self) -> None:
        fhir = {
            "resourceType": "Encounter",
            "participant": [
                {"individual": {"reference": "Practitioner/3f2504e0-4f89-41d3-9a0c-0305e82c3301"}},
                {"individual": {"reference": "Practitioner/x", "display": "Dr. Real Name"}},
            ],
        }
        assert extract_provider_display(fhir, "encounter") == "Dr. Real Name"
