from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from google import genai
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)


class SectionType(str, Enum):
    MEDICATIONS = "medications"
    ASSESSMENT = "assessment"
    CLINICAL_NOTE = "clinical_note"
    LABS = "labs"
    REVIEW_OF_SYSTEMS = "review_of_systems"
    HISTORY = "history"
    PHYSICAL_EXAM = "physical_exam"
    ASSESSMENT_PLAN = "assessment_plan"
    IMAGING = "imaging"
    FAMILY_HISTORY = "family_history"
    SOCIAL_HISTORY = "social_history"
    ALLERGIES = "allergies"
    PROCEDURES = "procedures"
    VITALS = "vitals"
    OTHER = "other"


@dataclass
class ParsedSection:
    section_type: SectionType
    title: str
    text: str
    char_range: tuple[int, int] | None = None


@dataclass
class ParsedDocument:
    document_type: str
    primary_visit_date: str | None
    provider: str | None
    facility: str | None
    sections: list[ParsedSection] = field(default_factory=list)


_SECTION_PARSER_PROMPT = """\
You are a clinical document parser. Given the full text of a medical document, identify its
logical sections in the order they appear and return structured JSON.

For EACH section return:
- "type": one of medications, assessment, clinical_note, labs, review_of_systems, history,
  physical_exam, assessment_plan, imaging, family_history, social_history, allergies,
  procedures, vitals, other
- "anchor": the section's heading or first line, copied VERBATIM from the document (the exact
  characters as they appear, ~10-60 chars). Do NOT paraphrase. Do NOT return the section body.

Also return top-level "document_type", "primary_visit_date", "provider", "facility" when
identifiable (else null). Return sections in document order. Do NOT echo the document text;
return only types and verbatim anchors.
"""


class _SectionAnchor(BaseModel):
    type: str
    anchor: str


class _SectionParseSchema(BaseModel):
    document_type: str = "unknown"
    primary_visit_date: str | None = None
    provider: str | None = None
    facility: str | None = None
    sections: list[_SectionAnchor]


def resolve_sections(text: str, raw_sections: list[dict]) -> list[ParsedSection]:
    """Locate each section's anchor in `text` and slice locally (boundaries-only).

    Each raw section is {"type": <str>, "anchor": <verbatim snippet from the doc>}.
    Anchors are located in document order via forward search; an anchor not found is
    dropped. Full-coverage invariant: any text before the first anchor becomes a
    leading OTHER section, and each section runs to the next section's start, so the
    concatenation of section texts reconstructs `text` exactly. If no anchor resolves,
    the whole document is returned as a single OTHER section.
    """
    located: list[tuple[int, SectionType, str]] = []  # (position, type, anchor)
    search_from = 0
    for s in raw_sections:
        if not isinstance(s, dict):
            continue
        anchor = s.get("anchor") or ""
        if not anchor:
            continue
        pos = text.find(anchor, search_from)
        if pos == -1:
            pos = text.find(anchor)  # model may have returned slightly out of order
        if pos == -1:
            logger.debug("section anchor not found (type=%s); dropping", s.get("type"))
            continue
        try:
            stype = SectionType(s.get("type"))
        except (ValueError, KeyError):
            stype = SectionType.OTHER
        located.append((pos, stype, anchor))
        search_from = pos + len(anchor)

    if not located:
        return [ParsedSection(SectionType.OTHER, "Full Document", text, (0, len(text)))]

    located.sort(key=lambda t: t[0])
    sections: list[ParsedSection] = []

    first_pos = located[0][0]
    if first_pos > 0:
        sections.append(
            ParsedSection(SectionType.OTHER, "Preamble", text[0:first_pos], (0, first_pos))
        )

    for i, (pos, stype, anchor) in enumerate(located):
        end = located[i + 1][0] if i + 1 < len(located) else len(text)
        sections.append(ParsedSection(stype, anchor, text[pos:end], (pos, end)))

    return sections


async def parse_sections(text: str, api_key: str) -> ParsedDocument:
    """Parse a clinical document into logical sections using Gemini.

    Falls back to a single OTHER section if the LLM call fails.
    """
    if not text or len(text.strip()) < 10:
        return ParsedDocument(
            document_type="unknown",
            primary_visit_date=None,
            provider=None,
            facility=None,
            sections=[ParsedSection(SectionType.OTHER, "Full Document", text or "")],
        )

    try:
        llm_response = await _call_gemini_for_sections(text, api_key)
    except Exception:
        logger.exception("Section parsing failed, falling back to single section")
        return ParsedDocument(
            document_type="unknown",
            primary_visit_date=None,
            provider=None,
            facility=None,
            sections=[ParsedSection(SectionType.OTHER, "Full Document", text)],
        )

    # Handle case where Gemini returns a JSON array instead of an object.
    # If the response is a list, treat it as the sections array directly.
    if isinstance(llm_response, list):
        raw_sections = llm_response
        doc_type, visit_date, provider, facility = "unknown", None, None, None
    else:
        raw_sections = llm_response.get("sections", [])
        doc_type = llm_response.get("document_type", "unknown")
        visit_date = llm_response.get("primary_visit_date")
        provider = llm_response.get("provider")
        facility = llm_response.get("facility")

    sections = resolve_sections(text, raw_sections)

    return ParsedDocument(
        document_type=doc_type,
        primary_visit_date=visit_date,
        provider=provider,
        facility=facility,
        sections=sections,
    )


async def _call_gemini_for_sections(text: str, api_key: str) -> dict:
    """Call Gemini Flash to parse document sections (type + verbatim anchor only)."""
    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=f"{_SECTION_PARSER_PROMPT}\n\n---\n\nDOCUMENT TEXT:\n{text}",
        config=genai.types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=_SectionParseSchema,
        ),
    )
    return json.loads(response.text)


def split_large_section(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split a large section into chunks at paragraph or sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        return _merge_chunks(paragraphs, max_chars, overlap, separator="\n\n")

    sentences = text.replace(". ", ".\n").split("\n")
    return _merge_chunks(sentences, max_chars, overlap, separator=" ")


def _merge_chunks(
    parts: list[str], max_chars: int, overlap: int, separator: str
) -> list[str]:
    """Merge small parts into chunks respecting max_chars, adding overlap."""
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part
        if len(candidate) > max_chars and current:
            chunks.append(current)
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + separator + part
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
