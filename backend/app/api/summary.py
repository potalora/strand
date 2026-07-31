from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.models.ai_summary import AISummaryPrompt
from app.models.local_ai import LocalAIJob
from app.models.patient import Patient
from app.models.record import HealthRecord
from app.models.summary_item import SummaryItem
from app.schemas.summary import (
    BuildPromptRequest,
    CloudAssistedGenerateSummaryResponse,
    CustomLocalGenerateSummaryResponse,
    DuplicateWarning,
    GenerateSummaryRequest,
    GenerateSummaryResponse,
    PasteResponseRequest,
    PasteResponseResponse,
    PromptDetailResponse,
    PromptListResponse,
    PromptResponse,
    StrictLocalSummaryAccepted,
    SummaryItemCreate,
    SummaryItemResponse,
)
from app.services.ai.prompt_grounding import (
    build_grounded_prompt_package,
    validate_pasted_grounded_response,
)
from app.services.local_ai.errors import (
    LocalAIError,
    LocalPolicyError,
    LocalValidationError,
)
from app.services.local_ai.grounded_summary import SERVER_SAFETY_RULES
from app.services.local_ai.processing_snapshot import (
    revalidate_strict_snapshot_admission,
    resolve_new_job_snapshot,
)
from app.services.local_ai.summary_runner import local_summary_runner
from app.services.local_ai.types import ProcessingMode

router = APIRouter(prefix="/summary", tags=["summary"])


def _processing_mode_for_api(value: object) -> ProcessingMode | None:
    """Return only a recognized persisted execution mode."""
    try:
        return ProcessingMode(value)
    except (TypeError, ValueError):
        return None


def _saved_model_provenance_for_api(
    prompt: AISummaryPrompt,
    patient: Patient | None,
) -> dict[str, object] | None:
    """Rebuild bounded model provenance before returning persisted history."""
    mode = _processing_mode_for_api(prompt.processing_mode)
    raw = prompt.model_provenance
    if (
        mode is None
        or not isinstance(raw, dict)
        or raw.get("processing_mode") != mode.value
    ):
        return None

    try:
        if mode is ProcessingMode.VALIDATED_STRICT_LOCAL:
            model = raw["model"]
            runtime = model["runtime"]
            if not isinstance(model, dict) or not isinstance(runtime, dict):
                return None
            from app.services.ai.summarizer import (
                _safe_strict_local_model_provenance,
            )

            return _safe_strict_local_model_provenance(
                manifest_sha256=raw["manifest_sha256"],
                pack_revision=raw["pack_revision"],
                repository=model["repository"],
                revision=model["revision"],
                quantization=model["quantization"],
                runtime=runtime,
            )

        if mode in {
            ProcessingMode.CUSTOM_LOCAL,
            ProcessingMode.CLOUD_ASSISTED,
        }:
            provider = raw.get("provider")
            if not isinstance(provider, str):
                return None
            from app.services.ai.patient_phi import patient_scrub_args
            from app.services.ai.summarizer import _safe_routed_model_provenance

            return _safe_routed_model_provenance(
                processing_mode=mode,
                provider=provider,
                model=raw.get("model"),
                known_patient_identifiers=patient_scrub_args(patient),
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return None


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
        prompt_data = await build_grounded_prompt_package(
            db=db,
            user_id=user_id,
            patient=patient,
            summary_type=body.summary_type,
            category=body.category,
            date_from=body.date_from,
            date_to=body.date_to,
            record_ids=body.record_ids,
            record_types=body.record_types,
            output_format=body.output_format,
        )
    except (ValueError, LocalValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store the prompt
    prompt_record = AISummaryPrompt(
        id=uuid4(),
        user_id=user_id,
        patient_id=body.patient_id,
        summary_type=body.summary_type,
        processing_mode=ProcessingMode.PROMPT_ONLY.value,
        scope_filter={
            "category": body.category,
            "date_from": body.date_from.isoformat() if body.date_from else None,
            "date_to": body.date_to.isoformat() if body.date_to else None,
            "record_ids": (
                [str(record_id) for record_id in body.record_ids]
                if body.record_ids
                else None
            ),
            "record_types": body.record_types,
            "selected_record_ids": prompt_data["selected_record_ids"],
            "grounded_scope": prompt_data["grounded_scope"],
            "grounding_input_sha256": prompt_data["grounding_input_sha256"],
            "output_format": body.output_format,
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
        details={
            "summary_type": body.summary_type,
            "record_count": prompt_data["record_count"],
        },
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
        processing_mode=ProcessingMode.PROMPT_ONLY,
    )


@router.get("/prompts", response_model=PromptListResponse)
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
    patient_ids = {prompt.patient_id for prompt in prompts}
    patients = (
        (
            await db.execute(
                select(Patient).where(
                    Patient.id.in_(patient_ids),
                    Patient.user_id == user_id,
                )
            )
        )
        .scalars()
        .all()
        if patient_ids
        else []
    )
    patients_by_id = {patient.id: patient for patient in patients}
    items = []
    for p in prompts:
        copyable = f"System: {p.system_prompt}\n\nUser: {p.user_prompt}"
        items.append(
            {
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
                "processing_mode": (
                    mode.value
                    if (mode := _processing_mode_for_api(p.processing_mode))
                    else None
                ),
                "model_provenance": _saved_model_provenance_for_api(
                    p,
                    patients_by_id.get(p.patient_id),
                ),
            }
        )

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.list_prompts",
        resource_type="ai_summary",
        ip_address=request.client.host if request.client else None,
    )

    return {"items": items}


@router.get("/prompts/{prompt_id}", response_model=PromptDetailResponse)
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
    patient = (
        await db.execute(
            select(Patient).where(
                Patient.id == prompt.patient_id,
                Patient.user_id == user_id,
            )
        )
    ).scalar_one_or_none()

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
        "typed_response": prompt.typed_response,
        "processing_mode": (
            mode.value
            if (mode := _processing_mode_for_api(prompt.processing_mode))
            else None
        ),
        "model_provenance": _saved_model_provenance_for_api(prompt, patient),
        "generated_at": prompt.generated_at.isoformat()
        if prompt.generated_at
        else None,
    }


@router.post("/paste-response", response_model=PasteResponseResponse)
async def paste_response(
    body: PasteResponseRequest,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> PasteResponseResponse:
    """Validate pasted references and store only server-rendered output."""
    result = await db.execute(
        select(AISummaryPrompt)
        .where(
            AISummaryPrompt.id == body.prompt_id,
            AISummaryPrompt.user_id == user_id,
        )
        .with_for_update()
    )
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    provenance_mode = (
        prompt.model_provenance.get("processing_mode")
        if isinstance(prompt.model_provenance, dict)
        else None
    )
    if (
        prompt.processing_mode == "validated_strict_local"
        or prompt.response_source == "local_ai"
        or prompt.typed_response is not None
        or provenance_mode == "validated_strict_local"
    ):
        raise HTTPException(
            status_code=409,
            detail="Grounded summaries cannot be overwritten.",
        )

    try:
        rendered = await validate_pasted_grounded_response(
            db,
            prompt=prompt,
            user_id=user_id,
            raw_response=body.response_text,
        )
    except LocalValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail="Pasted response must be valid reference-only JSON.",
        ) from exc
    except ValueError as exc:
        detail = str(exc)
        if detail == "Prompt records changed; build a new prompt.":
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(
            status_code=409,
            detail="Prompt grounding snapshot is unavailable; build a new prompt.",
        ) from exc

    scope = prompt.scope_filter if isinstance(prompt.scope_filter, dict) else {}
    output_format = scope.get("output_format", "natural_language")
    if output_format not in {"natural_language", "json", "both"}:
        raise HTTPException(
            status_code=409,
            detail="Prompt grounding snapshot is unavailable; build a new prompt.",
        )
    typed_response = rendered.document.model_dump(mode="json")
    encoded_response = json.dumps(typed_response, indent=2)
    if output_format == "natural_language":
        response_text = rendered.markdown
    elif output_format == "json":
        response_text = encoded_response
    else:
        response_text = rendered.markdown + "\n\n---JSON---\n" + encoded_response
    completed_at = datetime.now(timezone.utc)
    prompt.typed_response = typed_response
    prompt.response_text = response_text
    prompt.response_pasted_at = completed_at
    prompt.response_source = "pasted_grounded"
    prompt.response_format = output_format
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

    return PasteResponseResponse(
        id=prompt.id,
        prompt_id=prompt.id,
        response_pasted_at=prompt.response_pasted_at,
        typed_response=rendered.document,
        natural_language=(
            rendered.markdown if output_format in {"natural_language", "both"} else None
        ),
        json_data=(typed_response if output_format in {"json", "both"} else None),
    )


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


@router.get("/providers")
async def list_providers(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """List selectable LLM providers and the default routed summary provider.

    Returns display metadata only — never API keys.
    """
    from app.services.ai.llm.registry import available_providers, provider_name_for

    await log_audit_event(
        db,
        user_id=user_id,
        action="summary.providers.list",
        resource_type="ai_summary",
        ip_address=request.client.host if request.client else None,
    )

    return {"providers": available_providers(), "default": provider_name_for("summary")}


@router.post(
    "/generate",
    response_model=GenerateSummaryResponse,
    responses={status.HTTP_202_ACCEPTED: {"model": StrictLocalSummaryAccepted}},
)
async def generate_summary_endpoint(
    body: GenerateSummaryRequest,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> GenerateSummaryResponse | StrictLocalSummaryAccepted:
    """Generate a summary using the selected explicit processing mode."""
    if body.processing_mode is ProcessingMode.PROMPT_ONLY:
        raise HTTPException(
            status_code=400,
            detail="Prompt-only mode cannot generate a summary; use /summary/build-prompt.",
        )
    if body.processing_mode is ProcessingMode.VALIDATED_STRICT_LOCAL and any(
        value is not None
        for value in (
            body.provider,
            body.model,
            body.custom_system_prompt,
            body.custom_user_prompt,
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Validated strict-local summaries do not allow provider, model, or custom prompts.",
        )
    if body.processing_mode is ProcessingMode.CUSTOM_LOCAL and any(
        value is not None for value in (body.provider, body.model)
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Custom-local summaries use the validated stored loopback route "
                "and do not allow provider or model overrides."
            ),
        )
    # Verify patient belongs to user
    result = await db.execute(
        select(Patient).where(Patient.id == body.patient_id, Patient.user_id == user_id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if body.processing_mode is ProcessingMode.VALIDATED_STRICT_LOCAL:
        try:
            snapshot = await resolve_new_job_snapshot(
                db,
                user_id,
                ProcessingMode.VALIDATED_STRICT_LOCAL,
            )
            if snapshot.mode is not ProcessingMode.VALIDATED_STRICT_LOCAL:
                raise LocalPolicyError("Validated strict-local summary is unavailable.")
            prompt_record = AISummaryPrompt(
                id=uuid4(),
                user_id=user_id,
                patient_id=body.patient_id,
                summary_type=body.summary_type,
                processing_mode=ProcessingMode.VALIDATED_STRICT_LOCAL.value,
                scope_filter={
                    "category": body.category,
                    "date_from": body.date_from.isoformat() if body.date_from else None,
                    "date_to": body.date_to.isoformat() if body.date_to else None,
                    "record_ids": [str(item) for item in body.record_ids or []],
                    "output_format": body.output_format,
                },
                system_prompt="\n".join(SERVER_SAFETY_RULES),
                user_prompt="Server-selected evidence-grounded record facts only.",
                target_model="locked-local-summary",
                suggested_config={"temperature": 0, "worker_role": "summary"},
                record_count=0,
                de_identification_log=None,
                generated_at=datetime.now(timezone.utc),
            )
            job = LocalAIJob(
                user_id=user_id,
                summary_prompt_id=prompt_record.id,
                kind="summary",
                processing_mode=ProcessingMode.VALIDATED_STRICT_LOCAL.value,
                manifest_snapshot=snapshot.manifest_snapshot,
                manifest_sha256=snapshot.manifest_sha256,
                status="queued",
                stage="queued",
                progress={"stage": "queued"},
                audit_metadata={"summary_type": body.summary_type},
            )
            db.add_all([prompt_record, job])
            await revalidate_strict_snapshot_admission(db, snapshot)
            # This durability boundary is deliberate: cancellation and recovery
            # must see the summary job before a worker can be invoked.
            await db.commit()
            await db.refresh(prompt_record)
            await db.refresh(job)

        except (ValueError, LocalAIError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        local_summary_runner.enqueue(job.id)

        await log_audit_event(
            db,
            user_id=user_id,
            action="summary.generate",
            resource_type="ai_summary",
            resource_id=prompt_record.id,
            ip_address=request.client.host if request.client else None,
            details={
                "summary_type": body.summary_type,
                "record_count": 0,
                "processing_mode": ProcessingMode.VALIDATED_STRICT_LOCAL.value,
            },
        )
        return Response(
            content=StrictLocalSummaryAccepted(
                id=prompt_record.id,
                job_id=job.id,
                processing_mode=ProcessingMode.VALIDATED_STRICT_LOCAL.value,
                created_at=job.created_at,
            ).model_dump_json(),
            media_type="application/json",
            status_code=status.HTTP_202_ACCEPTED,
        )

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
            record_ids=body.record_ids,
            output_format=body.output_format,
            custom_system_prompt=body.custom_system_prompt,
            custom_user_prompt=body.custom_user_prompt,
            provider=body.provider,
            model=body.model,
            processing_mode=body.processing_mode,
        )
    except (ValueError, LocalAIError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    import json

    # Store in DB
    response_text = summary_data.get("natural_language") or ""
    if summary_data.get("json_data"):
        if response_text:
            response_text += "\n\n---JSON---\n" + json.dumps(
                summary_data["json_data"], indent=2
            )
        else:
            response_text = json.dumps(summary_data["json_data"], indent=2)

    prompt_record = AISummaryPrompt(
        id=uuid4(),
        user_id=user_id,
        patient_id=body.patient_id,
        summary_type=body.summary_type,
        processing_mode=body.processing_mode.value,
        scope_filter={
            "category": body.category,
            "date_from": body.date_from.isoformat() if body.date_from else None,
            "date_to": body.date_to.isoformat() if body.date_to else None,
            "record_ids": [str(item) for item in body.record_ids or []],
            "output_format": body.output_format,
        },
        system_prompt=summary_data["system_prompt"],
        user_prompt=summary_data["user_prompt"],
        target_model=summary_data["model_used"],
        suggested_config={
            "temperature": 0,
            "max_output_tokens": settings.gemini_summary_max_tokens,
        },
        record_count=summary_data["record_count"],
        de_identification_log=summary_data["de_identification_report"],
        response_text=response_text,
        typed_response=summary_data["typed_response"],
        response_pasted_at=datetime.now(timezone.utc),
        response_source="api",
        response_format=body.output_format,
        api_model_used=summary_data["model_used"],
        api_tokens_used=summary_data.get("tokens_used"),
        model_provenance=summary_data.get("model_provenance"),
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
        details={
            "summary_type": body.summary_type,
            "record_count": summary_data["record_count"],
        },
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

    response_type = (
        CustomLocalGenerateSummaryResponse
        if body.processing_mode is ProcessingMode.CUSTOM_LOCAL
        else CloudAssistedGenerateSummaryResponse
    )
    return response_type(
        id=prompt_record.id,
        processing_mode=body.processing_mode,
        model_provenance=prompt_record.model_provenance,
        typed_response=prompt_record.typed_response,
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
