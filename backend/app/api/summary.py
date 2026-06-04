from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.models.ai_summary import AISummaryPrompt
from app.models.patient import Patient
from app.models.record import HealthRecord
from app.models.summary_item import SummaryItem
from app.schemas.summary import (
    BuildPromptRequest,
    GenerateSummaryRequest,
    GenerateSummaryResponse,
    DuplicateWarning,
    PasteResponseRequest,
    PromptResponse,
    SummaryItemCreate,
    SummaryItemResponse,
)
from app.services.ai.prompt_builder import build_prompt

router = APIRouter(prefix="/summary", tags=["summary"])


@router.post("/build-prompt", response_model=PromptResponse)
async def build_prompt_endpoint(
    body: BuildPromptRequest,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> PromptResponse:
    """Build a de-identified prompt. Returns the prompt, NOT an AI response."""
    # Verify patient belongs to user
    result = await db.execute(
        select(Patient).where(Patient.id == body.patient_id, Patient.user_id == user_id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        prompt_data = await build_prompt(
            db=db,
            user_id=user_id,
            patient_id=body.patient_id,
            summary_type=body.summary_type,
            category=body.category,
            date_from=body.date_from,
            date_to=body.date_to,
            record_ids=body.record_ids,
            record_types=body.record_types,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store the prompt
    prompt_record = AISummaryPrompt(
        id=uuid4(),
        user_id=user_id,
        patient_id=body.patient_id,
        summary_type=body.summary_type,
        scope_filter={
            "category": body.category,
            "date_from": body.date_from.isoformat() if body.date_from else None,
            "date_to": body.date_to.isoformat() if body.date_to else None,
        },
        system_prompt=prompt_data["system_prompt"],
        user_prompt=prompt_data["user_prompt"],
        target_model=prompt_data["target_model"],
        suggested_config=prompt_data["suggested_config"],
        record_count=prompt_data["record_count"],
        de_identification_log=prompt_data["de_identification_report"],
        generated_at=datetime.now(timezone.utc),
    )
    db.add(prompt_record)
    await db.commit()
    await db.refresh(prompt_record)

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.build_prompt",
        resource_type="ai_summary",
        resource_id=prompt_record.id,
        ip_address=request.client.host if request.client else None,
        details={"summary_type": body.summary_type, "record_count": prompt_data["record_count"]},
    )

    return PromptResponse(
        id=prompt_record.id,
        summary_type=prompt_data["summary_type"],
        system_prompt=prompt_data["system_prompt"],
        user_prompt=prompt_data["user_prompt"],
        target_model=prompt_data["target_model"],
        suggested_config=prompt_data["suggested_config"],
        record_count=prompt_data["record_count"],
        de_identification_report=prompt_data["de_identification_report"],
        copyable_payload=prompt_data["copyable_payload"],
        generated_at=prompt_record.generated_at,
    )


@router.get("/prompts")
async def list_prompts(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """List previously built prompts."""
    result = await db.execute(
        select(AISummaryPrompt)
        .where(AISummaryPrompt.user_id == user_id)
        .order_by(AISummaryPrompt.generated_at.desc())
    )
    prompts = result.scalars().all()
    items = []
    for p in prompts:
        copyable = f"System: {p.system_prompt}\n\nUser: {p.user_prompt}"
        items.append({
            "id": str(p.id),
            "summary_type": p.summary_type,
            "system_prompt": p.system_prompt,
            "user_prompt": p.user_prompt,
            "target_model": p.target_model,
            "suggested_config": p.suggested_config,
            "record_count": p.record_count,
            "de_identification_report": p.de_identification_log,
            "copyable_payload": copyable,
            "generated_at": p.generated_at.isoformat() if p.generated_at else None,
        })

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.list_prompts",
        resource_type="ai_summary",
        ip_address=request.client.host if request.client else None,
    )

    return {"items": items}


@router.get("/prompts/{prompt_id}")
async def get_prompt(
    prompt_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get prompt detail for re-copying."""
    result = await db.execute(
        select(AISummaryPrompt).where(
            AISummaryPrompt.id == prompt_id,
            AISummaryPrompt.user_id == user_id,
        )
    )
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    copyable = f"System: {prompt.system_prompt}\n\nUser: {prompt.user_prompt}"

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.view_prompt",
        resource_type="ai_summary",
        resource_id=prompt_id,
        ip_address=request.client.host if request.client else None,
    )

    return {
        "id": str(prompt.id),
        "summary_type": prompt.summary_type,
        "system_prompt": prompt.system_prompt,
        "user_prompt": prompt.user_prompt,
        "target_model": prompt.target_model,
        "suggested_config": prompt.suggested_config,
        "record_count": prompt.record_count,
        "de_identification_report": prompt.de_identification_log,
        "copyable_payload": copyable,
        "response_text": prompt.response_text,
        "response_format": prompt.response_format,
        "generated_at": prompt.generated_at.isoformat() if prompt.generated_at else None,
    }


@router.post("/paste-response")
async def paste_response(
    body: PasteResponseRequest,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """User pastes AI response back for storage."""
    result = await db.execute(
        select(AISummaryPrompt).where(
            AISummaryPrompt.id == body.prompt_id,
            AISummaryPrompt.user_id == user_id,
        )
    )
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    prompt.response_text = body.response_text
    prompt.response_pasted_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(prompt)

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.paste_response",
        resource_type="ai_summary",
        resource_id=body.prompt_id,
        ip_address=request.client.host if request.client else None,
    )

    return {
        "id": str(prompt.id),
        "prompt_id": str(prompt.id),
        "response_pasted_at": prompt.response_pasted_at.isoformat(),
    }


@router.get("/responses")
async def list_responses(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """List stored responses."""
    result = await db.execute(
        select(AISummaryPrompt)
        .where(
            AISummaryPrompt.user_id == user_id,
            AISummaryPrompt.response_text.isnot(None),
        )
        .order_by(AISummaryPrompt.response_pasted_at.desc())
    )
    prompts = result.scalars().all()

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.list_responses",
        resource_type="ai_summary",
        ip_address=request.client.host if request.client else None,
    )

    return {
        "items": [
            {
                "id": str(p.id),
                "summary_type": p.summary_type,
                "record_count": p.record_count,
                "response_text": p.response_text[:200] if p.response_text else None,
                "response_pasted_at": p.response_pasted_at.isoformat()
                if p.response_pasted_at
                else None,
            }
            for p in prompts
        ],
        "total": len(prompts),
    }


@router.post("/generate", response_model=GenerateSummaryResponse)
async def generate_summary_endpoint(
    body: GenerateSummaryRequest,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> GenerateSummaryResponse:
    """Generate an AI summary by calling Gemini 3 Flash."""
    # Verify patient belongs to user
    result = await db.execute(
        select(Patient).where(Patient.id == body.patient_id, Patient.user_id == user_id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    from app.services.ai.summarizer import generate_summary

    try:
        summary_data = await generate_summary(
            db=db,
            user_id=user_id,
            patient_id=body.patient_id,
            summary_type=body.summary_type,
            category=body.category,
            date_from=body.date_from,
            date_to=body.date_to,
            output_format=body.output_format,
            custom_system_prompt=body.custom_system_prompt,
            custom_user_prompt=body.custom_user_prompt,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import json

    # Store in DB
    response_text = summary_data.get("natural_language") or ""
    if summary_data.get("json_data"):
        if response_text:
            response_text += "\n\n---JSON---\n" + json.dumps(summary_data["json_data"], indent=2)
        else:
            response_text = json.dumps(summary_data["json_data"], indent=2)

    prompt_record = AISummaryPrompt(
        id=uuid4(),
        user_id=user_id,
        patient_id=body.patient_id,
        summary_type=body.summary_type,
        scope_filter={
            "category": body.category,
            "date_from": body.date_from.isoformat() if body.date_from else None,
            "date_to": body.date_to.isoformat() if body.date_to else None,
            "output_format": body.output_format,
        },
        system_prompt=summary_data["system_prompt"],
        user_prompt=summary_data["user_prompt"],
        target_model=summary_data["model_used"],
        suggested_config={
            "temperature": 0.3,
            "max_output_tokens": 8192,
        },
        record_count=summary_data["record_count"],
        de_identification_log=summary_data["de_identification_report"],
        response_text=response_text,
        response_pasted_at=datetime.now(timezone.utc),
        response_source="api",
        response_format=body.output_format,
        api_model_used=summary_data["model_used"],
        api_tokens_used=summary_data.get("tokens_used"),
        generated_at=datetime.now(timezone.utc),
    )
    db.add(prompt_record)
    await db.commit()
    await db.refresh(prompt_record)

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.generate",
        resource_type="ai_summary",
        resource_id=prompt_record.id,
        ip_address=request.client.host if request.client else None,
        details={"summary_type": body.summary_type, "record_count": summary_data["record_count"]},
    )

    dup_warning = None
    if summary_data.get("duplicate_warning"):
        dw = summary_data["duplicate_warning"]
        dup_warning = DuplicateWarning(
            total_records=dw["total_records"],
            deduped_records=dw["deduped_records"],
            duplicates_excluded=dw["duplicates_excluded"],
            message=dw.get("message"),
        )

    return GenerateSummaryResponse(
        id=prompt_record.id,
        natural_language=summary_data.get("natural_language"),
        json_data=summary_data.get("json_data"),
        record_count=summary_data["record_count"],
        duplicate_warning=dup_warning,
        de_identification_report=summary_data["de_identification_report"],
        model_used=summary_data["model_used"],
        generated_at=prompt_record.generated_at,
    )


@router.post("/items", response_model=SummaryItemResponse)
async def add_summary_item(
    body: SummaryItemCreate,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> SummaryItemResponse:
    """Stage a record into the user's "Add to summary" basket.

    Idempotent: re-adding a record returns the existing item rather than erroring
    on the unique (user_id, record_id) constraint. 404 if the record does not
    exist or is not owned by the user.
    """
    record_result = await db.execute(
        select(HealthRecord).where(
            HealthRecord.id == body.record_id,
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
        )
    )
    if record_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Record not found")

    existing = await db.execute(
        select(SummaryItem).where(
            SummaryItem.user_id == user_id,
            SummaryItem.record_id == body.record_id,
        )
    )
    item = existing.scalar_one_or_none()

    if item is None:
        item = SummaryItem(id=uuid4(), user_id=user_id, record_id=body.record_id)
        db.add(item)
        try:
            await db.commit()
            await db.refresh(item)
        except IntegrityError:
            # Lost a race against a concurrent add — fetch the winner.
            await db.rollback()
            existing = await db.execute(
                select(SummaryItem).where(
                    SummaryItem.user_id == user_id,
                    SummaryItem.record_id == body.record_id,
                )
            )
            item = existing.scalar_one()

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.item_add",
        resource_type="summary_item",
        resource_id=item.id,
        ip_address=request.client.host if request.client else None,
        details={"record_id": str(body.record_id)},
    )

    return SummaryItemResponse(
        id=item.id, record_id=item.record_id, created_at=item.created_at
    )


@router.get("/items")
async def list_summary_items(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """List the user's summary basket, joined to record display_text/record_type."""
    result = await db.execute(
        select(SummaryItem, HealthRecord.display_text, HealthRecord.record_type)
        .join(HealthRecord, SummaryItem.record_id == HealthRecord.id)
        .where(SummaryItem.user_id == user_id)
        .order_by(SummaryItem.created_at.desc())
    )
    rows = result.all()

    items = [
        {
            "id": str(item.id),
            "record_id": str(item.record_id),
            "display_text": display_text,
            "record_type": record_type,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item, display_text, record_type in rows
    ]

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.items_list",
        resource_type="summary_item",
        ip_address=request.client.host if request.client else None,
    )

    return {"items": items, "total": len(items)}


@router.delete("/items/{item_id}", status_code=204)
async def delete_summary_item(
    item_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Remove a record from the user's summary basket. 404 if not owned."""
    result = await db.execute(
        select(SummaryItem).where(
            SummaryItem.id == item_id,
            SummaryItem.user_id == user_id,
        )
    )
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Summary item not found")

    await db.delete(item)
    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.item_remove",
        resource_type="summary_item",
        resource_id=item_id,
        ip_address=request.client.host if request.client else None,
    )

    return Response(status_code=204)
