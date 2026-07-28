from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from app.services.dedup.field_merger import apply_field_update, revert_field_update
from app.services.ingestion.content_hash import content_hash


def _make_record(
    fhir_resource: dict, record_type: str = "medication", display_text: str = "Test"
):
    rec = MagicMock()
    rec.id = uuid4()
    rec.fhir_resource = fhir_resource.copy()
    rec.record_type = record_type
    rec.display_text = display_text
    rec.fhir_resource_type = fhir_resource.get("resourceType", "Unknown")
    rec.merge_metadata = None
    return rec


class TestApplyFieldUpdate:
    """Tests for field-level FHIR merge."""

    def test_update_all_changed_fields(self):
        primary = _make_record(
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "dosageInstruction": [{"text": "500mg daily"}],
                "medicationCodeableConcept": {"text": "Metformin 500mg"},
            }
        )
        secondary = _make_record(
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "dosageInstruction": [{"text": "1000mg daily"}],
                "medicationCodeableConcept": {"text": "Metformin 1000mg"},
            }
        )

        result = apply_field_update(primary, secondary, field_overrides=None)

        assert result["updated_resource"]["dosageInstruction"] == [
            {"text": "1000mg daily"}
        ]
        assert (
            result["updated_resource"]["medicationCodeableConcept"]["text"]
            == "Metformin 1000mg"
        )
        assert "dosageInstruction" in result["merge_metadata"]["fields_updated"]
        assert result["merge_metadata"]["previous_values"]["dosageInstruction"] == [
            {"text": "500mg daily"}
        ]

    def test_cherry_pick_specific_fields(self):
        primary = _make_record(
            {
                "resourceType": "Condition",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "code": {"text": "Hypertension"},
            }
        )
        secondary = _make_record(
            {
                "resourceType": "Condition",
                "clinicalStatus": {"coding": [{"code": "resolved"}]},
                "code": {"text": "Essential Hypertension"},
            }
        )

        result = apply_field_update(
            primary, secondary, field_overrides=["clinicalStatus"]
        )

        # Only clinicalStatus should be updated
        assert (
            result["updated_resource"]["clinicalStatus"]["coding"][0]["code"]
            == "resolved"
        )
        # code should remain unchanged
        assert result["updated_resource"]["code"]["text"] == "Hypertension"
        assert "clinicalStatus" in result["merge_metadata"]["fields_updated"]
        assert "code" in result["merge_metadata"]["fields_kept"]

    def test_preserves_resource_type_and_metadata(self):
        primary = _make_record(
            {
                "resourceType": "Observation",
                "status": "final",
                "valueQuantity": {"value": 120},
                "_extraction_metadata": {"entity_class": "vital"},
            }
        )
        secondary = _make_record(
            {
                "resourceType": "Observation",
                "status": "final",
                "valueQuantity": {"value": 130},
                "_extraction_metadata": {"entity_class": "vital"},
            }
        )

        result = apply_field_update(primary, secondary, field_overrides=None)
        assert result["updated_resource"]["resourceType"] == "Observation"
        assert "_extraction_metadata" in result["updated_resource"]

    def test_display_text_regenerated(self):
        primary = _make_record(
            {
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"text": "Old"},
            },
            display_text="Old",
        )
        secondary = _make_record(
            {
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"text": "New"},
            },
            display_text="New",
        )

        result = apply_field_update(primary, secondary, field_overrides=None)
        assert result["display_text"]  # Should have some display text


class TestRevertFieldUpdate:
    """Tests for undoing a field-level merge."""

    def test_revert_restores_previous_values(self):
        rec = _make_record(
            {
                "resourceType": "MedicationRequest",
                "dosageInstruction": [{"text": "1000mg daily"}],
            }
        )
        rec.merge_metadata = {
            "previous_values": {
                "dosageInstruction": [{"text": "500mg daily"}],
            },
            "fields_updated": ["dosageInstruction"],
        }

        revert_field_update(rec)

        assert rec.fhir_resource["dosageInstruction"] == [{"text": "500mg daily"}]
        assert rec.merge_metadata is None

    def test_revert_noop_when_no_previous_values(self):
        rec = _make_record({"resourceType": "Condition", "code": {"text": "Test"}})
        rec.merge_metadata = {"merge_type": "duplicate"}

        revert_field_update(rec)
        assert rec.fhir_resource["code"]["text"] == "Test"

    def test_revert_recomputes_content_hash(self):
        """After reverting, content_hash must match the NEW (reverted) resource,
        not the stale pre-revert resource."""
        rec = _make_record(
            {
                "resourceType": "MedicationRequest",
                "dosageInstruction": [{"text": "1000mg daily"}],
            }
        )
        # Stale hash from before revert (the post-merge resource).
        rec.content_hash = content_hash(rec.fhir_resource)
        rec.merge_metadata = {
            "previous_values": {
                "dosageInstruction": [{"text": "500mg daily"}],
            },
            "fields_updated": ["dosageInstruction"],
        }

        revert_field_update(rec)

        # Hash now reflects the reverted resource, not the stale one.
        assert rec.content_hash == content_hash(rec.fhir_resource)
        assert rec.content_hash == content_hash(
            {
                "resourceType": "MedicationRequest",
                "dosageInstruction": [{"text": "500mg daily"}],
            }
        )
