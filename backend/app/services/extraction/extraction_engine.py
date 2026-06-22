"""Local / hybrid clinical-extraction orchestrator (WS-A).

Ties together the local NER fast-path (:mod:`local_ner`) and the medspaCy
clinical-context stage (:mod:`clinical_context`) into a section-aware extraction
pipeline that runs **on-device**, escalating only the hard parts to
Gemini/LangExtract.

Engines (``EXTRACTION_ENGINE``):

* ``local``  — local NER + ConText only. Fast, fully on-device (no Gemini, no
  PHI round-trip). Recognizes the two highest-volume entity types
  (medications → chemicals, conditions → diseases); other content is not
  extracted. Best for med/problem-list-heavy structured documents.
* ``hybrid`` — local fast-path for sections the local model covers well; any
  section the local model can't represent (labs, vitals, procedures, allergies,
  imaging, social/family history, A&P, ROS, exam) or whose local spans are
  low-confidence (< threshold) **escalates** to the Gemini path. Parity with the
  pure-Gemini engine, at a fraction of the latency on the common case.

All external dependencies (the NER engine, the context stage, and the Gemini
per-section extractor) are injected, so the escalation policy is fully unit-
testable with fakes — no model or network required.

Privacy: the local path runs on the **original** (unscrubbed) text and never
leaves the device, so it needs no PHI de-identification. Only escalated section
text is sent to Gemini, and the injected ``gemini_section_extract`` callable is
responsible for scrubbing it first (honoring the rule that all health data is
de-identified before any Gemini call).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from statistics import fmean

from app.services.extraction.clinical_context import ClinicalContext, postprocess_entities
from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_validator import normalize_entity_text
from app.services.extraction.local_ner import (
    LocalNerEngine,
    LocalSpan,
    label_to_entity_class,
    span_to_coding,
)
from app.services.extraction.section_parser import ParsedSection, SectionType

logger = logging.getLogger(__name__)

GeminiSectionExtractor = Callable[[str], Awaitable[list[ExtractedEntity]]]

# Section types whose content the local NER (CHEMICAL/DISEASE only) cannot
# adequately represent — they carry labs/vitals/procedures/allergies/imaging/
# social/family/plan content that needs the Gemini path in hybrid mode. OTHER and
# CLINICAL_NOTE (free-text / unsectioned narrative) are included: without a clean
# clinical-list structure we can't trust the local fast-path to be complete, so
# hybrid escalates them for recall. The local fast-path is kept for exactly the
# sections it is good at — medications, problem/history lists, plain assessments.
_ESCALATE_SECTION_TYPES: frozenset[SectionType] = frozenset(
    {
        SectionType.LABS,
        SectionType.VITALS,
        SectionType.PROCEDURES,
        SectionType.ALLERGIES,
        SectionType.IMAGING,
        SectionType.SOCIAL_HISTORY,
        SectionType.FAMILY_HISTORY,
        SectionType.ASSESSMENT_PLAN,
        SectionType.REVIEW_OF_SYSTEMS,
        SectionType.PHYSICAL_EXAM,
        SectionType.OTHER,
        SectionType.CLINICAL_NOTE,
    }
)

# Per-span confidence proxy. scispaCy NER exposes no probability, so a span that
# resolves to a real terminology code is treated as high-confidence and one that
# doesn't as low — driving the < threshold escalation decision.
_CODED_CONFIDENCE = 0.9
_UNCODED_CONFIDENCE = 0.5

# A section with no local entities but at least this much text is "substantive":
# the local model missed everything, so in hybrid mode let Gemini try.
_MIN_SUBSTANTIVE_CHARS = 40

# Disposition of a ConText-negated condition (the negation GUARD). Default
# "drop": an absent/refuted finding ("no chest pain", "denies diabetes") is not
# added to the active problem list — matching the Gemini baseline and the eval
# fixtures, which treat negated findings as must-not-extract. The complementary
# behavior in ``entity_to_fhir`` (a condition arriving with status="negated" maps
# to FHIR ``inactive`` clinicalStatus) is preserved and untouched, so set this to
# "inactive" to instead RECORD negated findings as inactive conditions.
NEGATED_CONDITION_DISPOSITION = "drop"  # "drop" | "inactive"


@dataclass
class EngineResult:
    """Output of a local/hybrid extraction run."""

    entities: list[ExtractedEntity]
    sections: list[ParsedSection]
    document_metadata: dict
    stats: dict = field(default_factory=dict)


def _span_to_entity(
    span: LocalSpan, is_negated: bool, is_family: bool, is_historical: bool,
    is_hypothetical: bool, section_type: SectionType,
) -> ExtractedEntity | None:
    """Convert one asserted local span into a storable entity (or drop it)."""
    cls = label_to_entity_class(span.label)
    if cls is None:
        return None
    if is_hypothetical:
        # "if you develop X", "rule out X", educational/conditional — not a finding.
        return None

    coding = span_to_coding(span)
    confidence = _CODED_CONFIDENCE if coding is not None else _UNCODED_CONFIDENCE
    attrs: dict = {"_source_section": section_type.value, "_local_ner": True}

    if cls == "condition":
        if is_family:
            # Attributed to a relative → family history, not the patient's condition.
            return ExtractedEntity(
                entity_class="family_history",
                text=span.text,
                attributes={**attrs, "condition": span.text, "relationship": "unknown"},
                start_pos=span.start_char,
                end_pos=span.end_char,
                confidence=confidence,
            )
        if is_negated:
            if NEGATED_CONDITION_DISPOSITION == "drop":
                return None
            attrs["status"] = "negated"  # → FHIR inactive in entity_to_fhir
        elif is_historical:
            attrs["status"] = "historical"
        else:
            attrs["status"] = "active"
        return ExtractedEntity(
            entity_class="condition",
            text=span.text,
            attributes=attrs,
            start_pos=span.start_char,
            end_pos=span.end_char,
            confidence=confidence,
        )

    # medication (CHEMICAL)
    if is_negated:
        # "not taking lisinopril", "denies aspirin use" — not an active med record.
        return None
    attrs["medication_group"] = span.text
    return ExtractedEntity(
        entity_class="medication",
        text=span.text,
        attributes=attrs,
        start_pos=span.start_char,
        end_pos=span.end_char,
        confidence=confidence,
    )


def _section_confidence(spans: list[LocalSpan]) -> float | None:
    """Mean coded/uncoded confidence over a section's spans (None if no spans)."""
    if not spans:
        return None
    return fmean(
        _CODED_CONFIDENCE if span_to_coding(s) is not None else _UNCODED_CONFIDENCE
        for s in spans
    )


def _should_escalate(
    section: ParsedSection, spans: list[LocalSpan], confidence: float | None,
    threshold: float,
) -> bool:
    """Decide whether a section should escalate to Gemini (hybrid only)."""
    if section.section_type in _ESCALATE_SECTION_TYPES:
        return True
    if confidence is not None and confidence < threshold:
        return True
    if confidence is None and len(section.text.strip()) >= _MIN_SUBSTANTIVE_CHARS:
        # Substantive section, local model found nothing — let Gemini try.
        return True
    return False


async def run_clinical_extraction(
    text: str,
    *,
    engine: str,
    ner: LocalNerEngine,
    context: ClinicalContext,
    gemini_section_extract: GeminiSectionExtractor | None,
    confidence_threshold: float,
    primary_visit_date: str | None = None,
) -> EngineResult:
    """Run the local/hybrid extraction pipeline over ``text``.

    ``engine`` is ``"local"`` or ``"hybrid"``. ``gemini_section_extract`` is
    awaited for each escalated section (hybrid only) and must return validated
    entities for already-scrubbed section text; it may be ``None`` for ``local``.
    """
    if engine not in ("local", "hybrid"):
        raise ValueError(f"run_clinical_extraction: unsupported engine {engine!r}")

    # Section detection + NER + ConText are synchronous, CPU-bound spaCy/medspaCy
    # inference. Offload them to a worker thread so this coroutine (run by the
    # background extraction worker) doesn't pin the event loop and stall
    # concurrent API requests (D3). The model objects are read-only during
    # prediction; only WHERE the compute runs changes, not its result.
    sections = await asyncio.to_thread(context.detect_sections, text)
    all_entities: list[ExtractedEntity] = []
    escalated_sections = 0
    local_entity_count = 0
    escalated_entity_count = 0

    for section in sections:
        spans = await asyncio.to_thread(ner.extract, section.text)
        confidence = _section_confidence(spans)
        escalate = engine == "hybrid" and _should_escalate(
            section, spans, confidence, confidence_threshold
        )

        if escalate and gemini_section_extract is not None:
            escalated_sections += 1
            try:
                gem_entities = await gemini_section_extract(section.text)
            except Exception:  # noqa: BLE001 - one bad section must not abort the doc
                logger.warning(
                    "Gemini escalation failed for section %s; skipping",
                    section.section_type.value, exc_info=True,
                )
                gem_entities = []
            for e in gem_entities:
                e.attributes.setdefault("_source_section", section.section_type.value)
            all_entities.extend(gem_entities)
            escalated_entity_count += len(gem_entities)
            continue

        # Local path for this section: assert spans with ConText, convert, drop.
        # ConText assertion is CPU-bound spaCy work — offload it too (D3).
        assertions = await asyncio.to_thread(context.assert_spans, section.text, spans)
        local_entities: list[ExtractedEntity] = []
        for a in assertions:
            entity = _span_to_entity(
                a.span, a.is_negated, a.is_family, a.is_historical,
                a.is_hypothetical, section.section_type,
            )
            if entity is not None:
                local_entities.append(entity)
        # Postprocess drop-rules (parity with entity_validator) on local output.
        local_entities = postprocess_entities(local_entities)
        all_entities.extend(local_entities)
        local_entity_count += len(local_entities)

    # Within-document dedup (same normalization the Gemini path uses).
    seen: set[tuple[str, str]] = set()
    unique: list[ExtractedEntity] = []
    for e in all_entities:
        key = (e.entity_class, normalize_entity_text(e.text))
        if key not in seen:
            seen.add(key)
            unique.append(e)

    document_metadata = {
        "document_type": "clinical_note",
        "primary_visit_date": primary_visit_date,
        "provider": None,
        "facility": None,
        "section_count": len(sections),
        "extraction_engine": engine,
    }
    stats = {
        "engine": engine,
        "section_count": len(sections),
        "escalated_sections": escalated_sections,
        "local_entities": local_entity_count,
        "escalated_entities": escalated_entity_count,
        "total_entities": len(unique),
    }
    logger.info(
        "Local extraction (%s): %d sections, %d escalated, %d entities",
        engine, len(sections), escalated_sections, len(unique),
    )
    return EngineResult(
        entities=unique,
        sections=sections,
        document_metadata=document_metadata,
        stats=stats,
    )
