"""Tests for CDA-to-FHIR parser."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


from app.services.ingestion.cda_parser import parse_cda_document

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "synthetic_cda"
DOC0001 = FIXTURES_DIR / "DOC0001.XML"
DOC0002 = FIXTURES_DIR / "DOC0002.XML"


@dataclass
class FakeXDMDocument:
    """Stand-in for XDMDocument used in tests."""

    uri: str
    hash: str
    size: int
    creation_time: str
    mime_type: str
    author_institution: str


def _sha1_of(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


class TestCDAConversion:
    """Test CDA-to-FHIR conversion produces valid records."""

    def test_converts_cda_to_fhir_records(self):
        """DOC0001 produces records with all required fields."""
        records = parse_cda_document(DOC0001)
        assert len(records) > 0
        for rec in records:
            assert "record_type" in rec
            assert "fhir_resource_type" in rec
            assert "fhir_resource" in rec
            assert rec["source_format"] == "cda_r2"
            assert "display_text" in rec
            assert isinstance(rec["display_text"], str)
            assert len(rec["display_text"]) > 0

    def test_tags_source_document_in_metadata(self):
        """Records include _extraction_metadata with source info."""
        manifest_doc = FakeXDMDocument(
            uri="DOC0001.XML",
            hash=_sha1_of(DOC0001),
            size=DOC0001.stat().st_size,
            creation_time="20240115120000",
            mime_type="text/xml",
            author_institution="Synthetic Health Clinic",
        )
        records = parse_cda_document(DOC0001, manifest_doc=manifest_doc)
        assert len(records) > 0
        for rec in records:
            meta = rec["fhir_resource"].get("_extraction_metadata")
            assert meta is not None
            assert meta["source_format"] == "cda_r2"
            assert meta["source_document"] == "DOC0001.XML"
            assert meta["source_institution"] == "Synthetic Health Clinic"

    def test_produces_expected_resource_types(self):
        """DOC0001 should produce at least some known resource types."""
        records = parse_cda_document(DOC0001)
        types = {r["fhir_resource_type"] for r in records}
        # The renderer produces AllergyIntolerance, MedicationStatement, Condition,
        # DocumentReference from DOC0001
        expected_some = {"AllergyIntolerance", "MedicationStatement", "Condition"}
        assert types & expected_some, f"Expected some of {expected_some}, got {types}"

    def test_doc0002_produces_records(self):
        """DOC0002 (with immunization section) produces records."""
        records = parse_cda_document(DOC0002)
        assert len(records) > 0
        types = {r["fhir_resource_type"] for r in records}
        assert "Immunization" in types, f"Expected Immunization in {types}"

    def test_skips_patient_resources(self):
        """Patient resources should not appear in output."""
        records = parse_cda_document(DOC0001)
        patient_records = [r for r in records if r["fhir_resource_type"] == "Patient"]
        assert len(patient_records) == 0

    def test_handles_nonexistent_file(self):
        """Nonexistent file returns empty list."""
        records = parse_cda_document(Path("/nonexistent/file.xml"))
        assert records == []

    def test_handles_malformed_xml(self, tmp_path):
        """Malformed XML returns empty list."""
        bad_file = tmp_path / "bad.xml"
        bad_file.write_text("<not valid CDA><unclosed")
        records = parse_cda_document(bad_file)
        assert records == []


class TestHashValidation:
    """Test SHA-1 hash validation against XDM manifest."""

    def test_hash_passes_when_matching(self):
        """When hash matches, records are returned."""
        correct_hash = _sha1_of(DOC0001)
        manifest_doc = FakeXDMDocument(
            uri="DOC0001.XML",
            hash=correct_hash,
            size=DOC0001.stat().st_size,
            creation_time="20240115120000",
            mime_type="text/xml",
            author_institution="Synthetic Health Clinic",
        )
        records = parse_cda_document(DOC0001, manifest_doc=manifest_doc)
        assert len(records) > 0

    def test_hash_fails_when_mismatched(self):
        """When hash does not match, returns empty list."""
        manifest_doc = FakeXDMDocument(
            uri="DOC0001.XML",
            hash="0000000000000000000000000000000000000000",
            size=DOC0001.stat().st_size,
            creation_time="20240115120000",
            mime_type="text/xml",
            author_institution="Synthetic Health Clinic",
        )
        records = parse_cda_document(DOC0001, manifest_doc=manifest_doc)
        assert records == []
