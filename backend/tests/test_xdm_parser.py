from __future__ import annotations

from pathlib import Path


from app.services.ingestion.xdm_parser import parse_xdm_metadata

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "synthetic_cda"


class TestXDMParser:
    """Unit tests for IHE XDM METADATA.XML parser."""

    def test_parse_valid_metadata_document_count(self) -> None:
        """Parse valid METADATA.XML and get correct document count (2 XML + 1 PDF)."""
        result = parse_xdm_metadata(FIXTURES_DIR / "METADATA.XML")
        assert result is not None
        assert len(result.documents) == 3

    def test_extract_patient_demographics(self) -> None:
        """Extract patient name and DOB from PID fields."""
        result = parse_xdm_metadata(FIXTURES_DIR / "METADATA.XML")
        assert result is not None
        assert result.patient_name == "Doe^Jane"
        assert result.patient_dob == "19900115"

    def test_extract_patient_id(self) -> None:
        """Extract patient ID from sourcePatientId."""
        result = parse_xdm_metadata(FIXTURES_DIR / "METADATA.XML")
        assert result is not None
        assert result.patient_id is not None
        assert "MRN-12345" in result.patient_id

    def test_extract_document_hashes(self) -> None:
        """All documents should have non-empty SHA-1 hashes."""
        result = parse_xdm_metadata(FIXTURES_DIR / "METADATA.XML")
        assert result is not None
        for doc in result.documents:
            assert doc.hash, f"Document {doc.uri} has empty hash"
            assert len(doc.hash) == 40  # SHA-1 hex length

    def test_extract_author_institution(self) -> None:
        """Extract author institution from Classification element."""
        result = parse_xdm_metadata(FIXTURES_DIR / "METADATA.XML")
        assert result is not None
        xml_docs = [d for d in result.documents if d.mime_type == "text/xml"]
        assert len(xml_docs) >= 1
        assert xml_docs[0].author_institution == "Synthetic Health Clinic"

    def test_pdf_document_mime_type(self) -> None:
        """PDF documents included with correct mime type."""
        result = parse_xdm_metadata(FIXTURES_DIR / "METADATA.XML")
        assert result is not None
        pdf_docs = [d for d in result.documents if d.mime_type == "application/pdf"]
        assert len(pdf_docs) == 1
        assert pdf_docs[0].uri == "SCAN0001.PDF"

    def test_missing_metadata_returns_none(self, tmp_path: Path) -> None:
        """Handle missing METADATA.XML by returning None."""
        result = parse_xdm_metadata(tmp_path / "nonexistent" / "METADATA.XML")
        assert result is None

    def test_malformed_xml_returns_none(self, tmp_path: Path) -> None:
        """Handle malformed XML by returning None."""
        bad_file = tmp_path / "METADATA.XML"
        bad_file.write_text("<not-valid-xml><unclosed>")
        result = parse_xdm_metadata(bad_file)
        assert result is None
