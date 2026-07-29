from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.services.extraction.text_extractor import (
    FileType,
    detect_file_type,
    extract_text_from_rtf,
)

# Path to real test data (may not exist in CI)
REQUESTED_RECORD_DIR = (
    Path(__file__).resolve().parent.parent.parent / "Requested Record"
)
RTF_DIR = REQUESTED_RECORD_DIR / "Rich Text"
PDF_DIR = REQUESTED_RECORD_DIR / "Media"
TIFF_DIR = REQUESTED_RECORD_DIR / "Media"

HAS_REAL_DATA = REQUESTED_RECORD_DIR.exists()
HAS_API_KEY = bool(os.environ.get("GEMINI_API_KEY"))


# ---------- File type detection ----------


def test_detect_pdf():
    assert detect_file_type(Path("report.pdf")) == FileType.PDF


def test_detect_rtf():
    assert detect_file_type(Path("note.rtf")) == FileType.RTF


def test_detect_tiff():
    assert detect_file_type(Path("scan.tif")) == FileType.TIFF
    assert detect_file_type(Path("scan.tiff")) == FileType.TIFF


def test_detect_unknown():
    assert detect_file_type(Path("file.doc")) == FileType.UNKNOWN
    assert detect_file_type(Path("file.txt")) == FileType.UNKNOWN
    assert detect_file_type(Path("file.xlsx")) == FileType.UNKNOWN


def test_detect_case_insensitive():
    assert detect_file_type(Path("report.PDF")) == FileType.PDF
    assert detect_file_type(Path("note.RTF")) == FileType.RTF


# ---------- RTF extraction (local, no API) ----------


@pytest.mark.skipif(not HAS_REAL_DATA, reason="No real test data at Requested Record/")
def test_rtf_text_extraction():
    """Extract text from a real RTF file and verify it contains clinical content."""
    rtf_files = list(RTF_DIR.glob("*.RTF")) + list(RTF_DIR.glob("*.rtf"))
    if not rtf_files:
        pytest.skip("No RTF files found")

    rtf_path = rtf_files[0]
    text = extract_text_from_rtf(rtf_path)
    assert len(text) > 50, f"Extracted text too short: {len(text)} chars"
    # RTF clinical notes should contain some recognizable content
    text_lower = text.lower()
    assert any(
        term in text_lower
        for term in [
            "patient",
            "date",
            "history",
            "note",
            "assessment",
            "plan",
            "diagnosis",
            "provider",
            "md",
            "dr",
        ]
    ), "RTF text doesn't appear to contain clinical content"


def test_rtf_extraction_with_synthetic():
    """Test RTF extraction with a synthetic RTF string."""
    import tempfile

    rtf_content = r"""{\rtf1\ansi\deff0
{\fonttbl{\f0 Times New Roman;}}
\f0\fs24 Patient seen for follow-up. Assessment: Hypertension controlled. Plan: Continue Lisinopril 10mg daily.
}"""
    with tempfile.NamedTemporaryFile(suffix=".rtf", mode="w", delete=False) as f:
        f.write(rtf_content)
        f.flush()
        text = extract_text_from_rtf(Path(f.name))

    assert "Hypertension" in text
    assert "Lisinopril" in text
    os.unlink(f.name)


# ---------- PDF extraction via Gemini (slow, requires API) ----------


@pytest.mark.slow
@pytest.mark.private_cloud_fidelity
@pytest.mark.skipif(
    not HAS_REAL_DATA or not HAS_API_KEY,
    reason="Requires real data and GEMINI_API_KEY",
)
@pytest.mark.asyncio
async def test_pdf_text_extraction_via_gemini():
    """Send a real PDF to Gemini 3 Flash and verify extracted text."""
    from app.services.extraction.text_extractor import extract_text_from_pdf

    pdf_files = list(PDF_DIR.glob("**/*.pdf")) + list(PDF_DIR.glob("**/*.PDF"))
    if not pdf_files:
        pytest.skip("No PDF files found")

    pdf_path = pdf_files[0]
    api_key = os.environ["GEMINI_API_KEY"]
    text = await extract_text_from_pdf(pdf_path, api_key)

    assert len(text) > 20, f"Extracted text too short: {len(text)} chars"


@pytest.mark.slow
@pytest.mark.private_cloud_fidelity
@pytest.mark.skipif(
    not HAS_REAL_DATA or not HAS_API_KEY,
    reason="Requires real data and GEMINI_API_KEY",
)
@pytest.mark.asyncio
async def test_pdf_extraction_preserves_structure():
    """Verify that PDF extraction preserves some structure (headers, sections)."""
    from app.services.extraction.text_extractor import extract_text_from_pdf

    pdf_files = list(PDF_DIR.glob("**/*.pdf")) + list(PDF_DIR.glob("**/*.PDF"))
    if not pdf_files:
        pytest.skip("No PDF files found")

    pdf_path = pdf_files[0]
    api_key = os.environ["GEMINI_API_KEY"]
    text = await extract_text_from_pdf(pdf_path, api_key)

    # Check it has some line breaks indicating preserved structure
    assert "\n" in text, "Extracted text has no line breaks — structure lost"


# ---------- TIFF OCR via Gemini (slow, requires API) ----------


@pytest.mark.slow
@pytest.mark.private_cloud_fidelity
@pytest.mark.skipif(
    not HAS_REAL_DATA or not HAS_API_KEY,
    reason="Requires real data and GEMINI_API_KEY",
)
@pytest.mark.asyncio
async def test_tiff_ocr_via_gemini():
    """Send a TIFF to Gemini 3 Flash for OCR and verify text is non-empty."""
    from app.services.extraction.text_extractor import extract_text_from_tiff

    tiff_files = (
        list(TIFF_DIR.glob("**/*.tif"))
        + list(TIFF_DIR.glob("**/*.tiff"))
        + list(TIFF_DIR.glob("**/*.TIFF"))
        + list(TIFF_DIR.glob("**/*.TIF"))
    )
    if not tiff_files:
        pytest.skip("No TIFF files found")

    tiff_path = tiff_files[0]
    api_key = os.environ["GEMINI_API_KEY"]
    text = await extract_text_from_tiff(tiff_path, api_key)

    assert len(text) > 10, f"OCR text too short: {len(text)} chars"


# ---------- Dispatcher ----------


@pytest.mark.asyncio
async def test_extract_text_unsupported_type():
    """Verify unsupported file types raise ValueError."""
    from app.services.extraction.text_extractor import extract_text

    with pytest.raises(ValueError, match="Unsupported file type"):
        await extract_text(Path("document.doc"), "fake-key")


@pytest.mark.asyncio
async def test_extract_text_rtf_dispatch():
    """Verify dispatcher routes RTF correctly (no API needed)."""
    import tempfile

    from app.services.extraction.text_extractor import extract_text

    rtf_content = r"""{\rtf1\ansi Test content for extraction.}"""
    with tempfile.NamedTemporaryFile(suffix=".rtf", mode="w", delete=False) as f:
        f.write(rtf_content)
        f.flush()
        text, file_type = await extract_text(Path(f.name), "unused-key")

    assert file_type == FileType.RTF
    assert "Test content" in text
    os.unlink(f.name)
