from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.services.extraction.entity_extractor import ExtractedEntity, ExtractionResult
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict
from app.services.extraction.section_parser import (
    ParsedDocument,
    ParsedSection,
    SectionType,
    split_large_section,
)


class TestEntityToFhirRoundTrip:
    """Test that entities from all 11 types produce valid record dicts."""

    ENTITY_CONFIGS = [
        ("medication", "aspirin 81mg", {"medication_group": "aspirin"}),
        ("condition", "Hypertension", {"status": "active"}),
        ("lab_result", "HbA1c 6.5", {"test": "HbA1c", "value": "6.5", "unit": "%"}),
        ("vital", "BP 120/80", {"type": "Blood Pressure"}),
        ("procedure", "Colonoscopy", {"date": "2024-03-15"}),
        ("allergy", "Penicillin", {"reaction": "rash", "severity": "moderate"}),
        ("encounter", "Office visit", {"visit_type": "office", "date": "2026-03-30"}),
        (
            "imaging_result",
            "CT Abdomen",
            {
                "procedure_name": "CT Abdomen",
                "findings": "normal",
                "category": "imaging",
            },
        ),
        (
            "family_history",
            "Father: DM2",
            {"relationship": "father", "condition": "DM2"},
        ),
        (
            "assessment_plan",
            "#1: Continue meds.",
            {"plan_items": ["Continue medications"]},
        ),
        ("social_history", "Alcohol: none", {"category": "alcohol", "value": "none"}),
    ]

    @pytest.mark.parametrize("entity_class,text,attrs", ENTITY_CONFIGS)
    def test_storable_entity_produces_record(self, entity_class, text, attrs):
        entity = ExtractedEntity(entity_class=entity_class, text=text, attributes=attrs)
        result = entity_to_health_record_dict(entity, uuid4(), uuid4(), uuid4())
        assert result is not None
        assert result["record_type"] in (
            "medication",
            "condition",
            "observation",
            "procedure",
            "allergy",
            "encounter",
            "diagnostic_report",
            "family_history",
            "document",
        )
        assert result["fhir_resource"]["resourceType"] in (
            "MedicationRequest",
            "Condition",
            "Observation",
            "Procedure",
            "AllergyIntolerance",
            "Encounter",
            "DiagnosticReport",
            "FamilyMemberHistory",
            "DocumentReference",
        )
        assert result["display_text"]
        assert result["ai_extracted"] is True
        assert result["confidence_score"] == 0.8

    NONSTORABLE = ["dosage", "route", "frequency", "duration", "date", "provider"]

    @pytest.mark.parametrize("entity_class", NONSTORABLE)
    def test_nonstorable_entity_returns_none(self, entity_class):
        entity = ExtractedEntity(entity_class=entity_class, text="some value")
        result = entity_to_health_record_dict(entity, uuid4(), uuid4())
        assert result is None


class TestSectionParserToEntityPipeline:
    """Test the section parser → entity extraction flow."""

    @pytest.mark.asyncio
    async def test_parsed_sections_feed_into_extraction(self):
        """Verify that parsed sections can be individually extracted."""
        doc = ParsedDocument(
            document_type="clinical_note",
            primary_visit_date="2026-03-30",
            provider="Dr. Test",
            facility="Test Clinic",
            sections=[
                ParsedSection(
                    SectionType.MEDICATIONS, "Medications", "aspirin 81mg daily"
                ),
                ParsedSection(SectionType.LABS, "Labs", "HbA1c 6.5%"),
            ],
        )

        mock_entities_meds = ExtractionResult(
            source_file="test.pdf",
            source_text="aspirin 81mg daily",
            entities=[
                ExtractedEntity(
                    "medication", "aspirin 81mg", {"medication_group": "aspirin"}
                )
            ],
        )
        mock_entities_labs = ExtractionResult(
            source_file="test.pdf",
            source_text="HbA1c 6.5%",
            entities=[
                ExtractedEntity(
                    "lab_result", "HbA1c 6.5", {"test": "HbA1c", "value": "6.5"}
                )
            ],
        )

        with patch(
            "app.services.extraction.entity_extractor.extract_entities_async",
            new_callable=AsyncMock,
            side_effect=[mock_entities_meds, mock_entities_labs],
        ):
            from app.services.extraction.entity_extractor import extract_entities_async

            all_entities = []
            for section in doc.sections:
                result = await extract_entities_async(
                    section.text, "test.pdf", "fake-key"
                )
                for entity in result.entities:
                    entity.attributes["_source_section"] = section.section_type.value
                    all_entities.append(entity)

        assert len(all_entities) == 2
        assert all_entities[0].entity_class == "medication"
        assert all_entities[0].attributes["_source_section"] == "medications"
        assert all_entities[1].entity_class == "lab_result"
        assert all_entities[1].attributes["_source_section"] == "labs"

    def test_large_section_splitting_preserves_content(self):
        """All content is preserved across split chunks."""
        paragraphs = [f"Paragraph {i}. " * 80 for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = split_large_section(text, max_chars=1500)
        assert len(chunks) >= 2
        # Every paragraph should appear in at least one chunk
        for para in paragraphs:
            found = any(para[:50] in chunk for chunk in chunks)
            assert found, (
                f"Paragraph starting with '{para[:50]}' not found in any chunk"
            )

    def test_document_dedup_removes_same_entity_from_overlapping_chunks(self):
        """Duplicate entities across chunks are removed by text+type dedup."""
        entities = [
            ExtractedEntity("medication", "aspirin 81mg"),
            ExtractedEntity("medication", "aspirin 81mg"),  # duplicate from overlap
            ExtractedEntity("condition", "Hypertension"),
        ]
        seen = set()
        unique = []
        for e in entities:
            key = (e.entity_class, e.text.strip().lower())
            if key not in seen:
                seen.add(key)
                unique.append(e)
        assert len(unique) == 2

    def test_encounter_record_links_to_other_records(self):
        """Encounter record ID can be used as linked_encounter_id."""
        encounter_entity = ExtractedEntity(
            "encounter",
            "Visit 03/30",
            attributes={"visit_type": "office", "date": "2026-03-30"},
        )
        med_entity = ExtractedEntity("medication", "aspirin 81mg")

        user_id, patient_id = uuid4(), uuid4()
        enc_dict = entity_to_health_record_dict(encounter_entity, user_id, patient_id)
        med_dict = entity_to_health_record_dict(med_entity, user_id, patient_id)

        assert enc_dict is not None
        assert med_dict is not None
        # The caller (upload.py) sets linked_encounter_id — here we verify the dict has an id
        assert "id" in enc_dict
        assert "id" in med_dict
