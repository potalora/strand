from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.patient import Patient
from app.models.record import HealthRecord
from app.services.ai.patient_phi import patient_scrub_args
from app.services.ai.phi_scrubber import scrub_phi

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical records summarizer. Your task is to organize and summarize the following de-identified health records into a clear, structured overview.

IMPORTANT RULES:
- Do NOT provide any diagnoses, treatment recommendations, medical advice, or clinical decision support.
- Summarize the factual medical information ONLY.
- If information is unclear or potentially conflicting, note this without interpretation.
- Organize information chronologically within each category.
- Use clear section headers.

OUTPUT FORMAT:
Use structured markdown with sections organized by category and chronological order."""

CATEGORY_PROMPTS = {
    "full": "Provide a comprehensive chronological overview of ALL health records below.",
    "condition": "Summarize all conditions and diagnoses from the records below.",
    "observation": "Summarize all lab results, vitals, and observations from the records below.",
    "medication": "Summarize all medications, prescriptions, and medication history from the records below.",
    "encounter": "Summarize all clinical encounters and visits from the records below.",
    "immunization": "Summarize the immunization history from the records below.",
    "procedure": "Summarize all procedures from the records below.",
    "document": "Summarize the clinical documents and notes from the records below.",
}


async def build_prompt(
    db: AsyncSession,
    user_id: UUID,
    patient_id: UUID,
    summary_type: str = "full",
    category: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    record_ids: list[UUID] | None = None,
    record_types: list[str] | None = None,
) -> dict:
    """Build a complete de-identified prompt for AI summarization.

    Returns the prompt package (NO API calls made).
    """
    # Fetch records
    query = select(HealthRecord).where(
        HealthRecord.user_id == user_id,
        HealthRecord.patient_id == patient_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    )

    if record_ids:
        query = query.where(HealthRecord.id.in_(record_ids))
    elif record_types:
        query = query.where(HealthRecord.record_type.in_(record_types))
    elif category and category != "full":
        query = query.where(HealthRecord.record_type == category)

    if date_from:
        query = query.where(HealthRecord.effective_date >= date_from)
    if date_to:
        query = query.where(HealthRecord.effective_date <= date_to)

    query = query.order_by(HealthRecord.effective_date.asc().nullslast())
    result = await db.execute(query)
    records = result.scalars().all()

    if not records:
        raise ValueError("No records found matching the criteria")

    # Build the record text
    record_texts = []
    for r in records:
        text = _format_record(r)
        record_texts.append(text)

    combined_text = "\n\n---\n\n".join(record_texts)

    # De-identify. Pass the patient's known identifiers so their own name / MRN
    # / DOB are stripped (regex patterns alone don't catch free-text names).
    patient = (
        await db.execute(
            select(Patient).where(Patient.id == patient_id, Patient.user_id == user_id)
        )
    ).scalar_one_or_none()
    scrubbed_text, deidentification_report = scrub_phi(
        combined_text, **patient_scrub_args(patient)
    )

    # Build user prompt
    prompt_instruction = CATEGORY_PROMPTS.get(
        category or summary_type, CATEGORY_PROMPTS["full"]
    )
    user_prompt = f"""{prompt_instruction}

The following de-identified health records are provided for summarization:

{scrubbed_text}

Please provide a structured summary following the rules in the system prompt."""

    # Build copyable payload
    copyable = f"""System: {SYSTEM_PROMPT}

User: {user_prompt}"""

    return {
        "summary_type": summary_type,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "target_model": settings.prompt_target_model,
        "suggested_config": {
            "temperature": settings.prompt_suggested_temperature,
            "max_output_tokens": settings.prompt_suggested_max_tokens,
            "thinking_level": settings.prompt_suggested_thinking_level,
        },
        "record_count": len(records),
        "de_identification_report": deidentification_report,
        "copyable_payload": copyable,
    }


def _format_record(record: HealthRecord) -> str:
    """Format a single health record as text for prompt inclusion."""
    parts = [f"[{record.record_type.upper()}] {record.display_text}"]

    if record.effective_date:
        # HIPAA Safe Harbor: emit the year only (month/day are date elements).
        parts.append(f"Date: {record.effective_date.strftime('%Y')}")

    if record.status:
        parts.append(f"Status: {record.status}")

    fhir = record.fhir_resource or {}

    # Extract value for observations
    if record.record_type == "observation":
        vq = fhir.get("valueQuantity", {})
        if vq:
            parts.append(f"Value: {vq.get('value', '')} {vq.get('unit', '')}")
        vs = fhir.get("valueString")
        if vs:
            parts.append(f"Value: {vs}")
        ref = fhir.get("referenceRange", [])
        if ref:
            r = ref[0]
            low = r.get("low", {}).get("value", "")
            high = r.get("high", {}).get("value", "")
            if low or high:
                parts.append(f"Reference: {low} - {high}")

    # Dosage for medications
    if record.record_type == "medication":
        dosage = fhir.get("dosageInstruction", [])
        if dosage:
            parts.append(f"Dosage: {dosage[0].get('text', '')}")

    # Notes
    notes = fhir.get("note", [])
    if notes:
        parts.append(f"Note: {notes[0].get('text', '')}")

    return "\n".join(parts)
