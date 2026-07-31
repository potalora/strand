from __future__ import annotations

from datetime import datetime


from app.services.ingestion.cda_dedup import deduplicate_across_documents


def _make_record(
    record_type: str = "condition",
    fhir_resource_type: str = "Condition",
    code_system: str = "http://snomed.info/sct",
    code_value: str = "73211009",
    code_display: str = "Diabetes mellitus",
    effective_date: datetime | None = datetime(2024, 3, 15),
    display_text: str = "Diabetes mellitus",
    source_document: str = "doc_a.xml",
    source_format: str = "cda",
) -> dict:
    """Build a realistic parsed record dict."""
    return {
        "record_type": record_type,
        "fhir_resource_type": fhir_resource_type,
        "code_system": code_system,
        "code_value": code_value,
        "code_display": code_display,
        "effective_date": effective_date,
        "display_text": display_text,
        "source_format": source_format,
        "fhir_resource": {
            "resourceType": fhir_resource_type,
            "code": {
                "coding": [
                    {
                        "system": code_system,
                        "code": code_value,
                        "display": code_display,
                    }
                ]
            },
            "_extraction_metadata": {
                "source_document": source_document,
                "parser": "cda",
            },
        },
    }


class TestDeduplicateAcrossDocuments:
    """Tests for intra-upload cross-document deduplication."""

    def test_identical_records_collapsed(self):
        """Identical records from 2 docs collapse to 1, provenance lists both."""
        records = [
            _make_record(source_document="doc_a.xml"),
            _make_record(source_document="doc_b.xml"),
        ]

        unique, stats = deduplicate_across_documents(records)

        assert len(unique) == 1
        assert stats.total_parsed == 2
        assert stats.unique_records == 1
        assert stats.duplicates_collapsed == 1
        source_docs = unique[0]["fhir_resource"]["_extraction_metadata"][
            "source_documents"
        ]
        assert source_docs == ["doc_a.xml", "doc_b.xml"]

    def test_different_codes_kept(self):
        """Records with different codes are both kept."""
        records = [
            _make_record(code_value="73211009", code_display="Diabetes mellitus"),
            _make_record(
                code_value="38341003",
                code_display="Hypertension",
                display_text="Hypertension",
            ),
        ]

        unique, stats = deduplicate_across_documents(records)

        assert len(unique) == 2
        assert stats.duplicates_collapsed == 0

    def test_same_code_different_dates_kept(self):
        """Same code but different dates are different clinical events."""
        records = [
            _make_record(effective_date=datetime(2024, 1, 10)),
            _make_record(effective_date=datetime(2024, 6, 20)),
        ]

        unique, stats = deduplicate_across_documents(records)

        assert len(unique) == 2
        assert stats.duplicates_collapsed == 0

    def test_empty_input(self):
        """Empty input returns empty output with zero stats."""
        unique, stats = deduplicate_across_documents([])

        assert unique == []
        assert stats.total_parsed == 0
        assert stats.unique_records == 0
        assert stats.duplicates_collapsed == 0
        assert stats.records_per_document == {}

    def test_single_document_passthrough(self):
        """Records from a single document pass through with no dedup."""
        records = [
            _make_record(code_value="73211009", source_document="doc_a.xml"),
            _make_record(
                code_value="38341003",
                code_display="Hypertension",
                display_text="Hypertension",
                source_document="doc_a.xml",
            ),
        ]

        unique, stats = deduplicate_across_documents(records)

        assert len(unique) == 2
        assert stats.duplicates_collapsed == 0
        assert stats.records_per_document == {"doc_a.xml": 2}

    def test_stats_records_per_document(self):
        """Stats correctly track per-document record counts."""
        records = [
            _make_record(code_value="73211009", source_document="doc_a.xml"),
            _make_record(code_value="73211009", source_document="doc_b.xml"),
            _make_record(
                code_value="38341003",
                code_display="Hypertension",
                display_text="Hypertension",
                source_document="doc_b.xml",
            ),
            _make_record(
                record_type="medication",
                fhir_resource_type="MedicationRequest",
                code_value="197361",
                code_display="Metformin",
                display_text="Metformin 500mg",
                source_document="doc_c.xml",
            ),
        ]

        unique, stats = deduplicate_across_documents(records)

        assert stats.total_parsed == 4
        assert stats.unique_records == 3
        assert stats.duplicates_collapsed == 1
        assert stats.records_per_document == {
            "doc_a.xml": 1,
            "doc_b.xml": 2,
            "doc_c.xml": 1,
        }

    def test_none_effective_date_dedup(self):
        """Records with None effective_date can still be deduped."""
        records = [
            _make_record(effective_date=None, source_document="doc_a.xml"),
            _make_record(effective_date=None, source_document="doc_b.xml"),
        ]

        unique, stats = deduplicate_across_documents(records)

        assert len(unique) == 1
        assert stats.duplicates_collapsed == 1
