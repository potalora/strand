from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory

# Global semaphore to limit concurrent Gemini API calls across all uploads
_gemini_semaphore: asyncio.Semaphore | None = None

# Extraction semaphore to limit concurrent file extractions (layer above Gemini semaphore)
_extraction_semaphore: asyncio.Semaphore | None = None

# Worker task reference
_worker_task: asyncio.Task | None = None


def _get_gemini_semaphore() -> asyncio.Semaphore:
    global _gemini_semaphore
    if _gemini_semaphore is None:
        _gemini_semaphore = asyncio.Semaphore(settings.gemini_concurrency_limit)
    return _gemini_semaphore


def _get_extraction_semaphore() -> asyncio.Semaphore:
    global _extraction_semaphore
    if _extraction_semaphore is None:
        _extraction_semaphore = asyncio.Semaphore(settings.extraction_concurrency)
    return _extraction_semaphore


async def _extraction_worker() -> None:
    """DB-polling background worker: claim pending files and process them.

    Uses a rolling window approach: claims one file at a time, acquires a
    semaphore slot, then fires off processing without waiting. This keeps all
    extraction slots busy instead of blocking on the slowest file in a batch.
    """
    sem = _get_extraction_semaphore()
    poll_interval = 2
    stuck_check_interval = 60
    last_stuck_check = datetime.now(timezone.utc)

    logger.info("Extraction worker started (concurrency=%d, poll=%ds)", settings.extraction_concurrency, poll_interval)

    while True:
        try:
            now = datetime.now(timezone.utc)
            if (now - last_stuck_check).total_seconds() >= stuck_check_interval:
                await _recover_stuck_files()
                last_stuck_check = now

            # Claim one file at a time for rolling window
            claimed = await _claim_pending_files(1)
            if not claimed:
                await asyncio.sleep(poll_interval)
                continue

            upload_id, file_path, user_id = claimed[0]
            logger.info("Claimed file %s for extraction", upload_id)

            # Block until a slot opens, then fire-and-forget
            await sem.acquire()
            asyncio.create_task(
                _process_and_release(sem, upload_id, Path(file_path), user_id)
            )

        except Exception:
            logger.exception("Extraction worker encountered an error, recovering")
            await asyncio.sleep(5)


async def _claim_pending_files(batch_size: int) -> list[tuple[str, str, str]]:
    """Claim pending extraction files using SELECT FOR UPDATE SKIP LOCKED."""
    async with async_session_factory() as db:
        try:
            result = await db.execute(
                text(
                    "SELECT id, storage_path, user_id FROM uploaded_files "
                    "WHERE ingestion_status = 'pending_extraction' "
                    "AND file_category = 'unstructured' "
                    "ORDER BY created_at ASC "
                    "LIMIT :batch_size "
                    "FOR UPDATE SKIP LOCKED"
                ),
                {"batch_size": batch_size},
            )
            rows = result.fetchall()

            if not rows:
                return []

            # Mark claimed files as processing
            ids = [row[0] for row in rows]
            now = datetime.now(timezone.utc)
            await db.execute(
                text(
                    "UPDATE uploaded_files "
                    "SET ingestion_status = 'processing', "
                    "processing_started_at = :now "
                    "WHERE id = ANY(:ids)"
                ),
                {"now": now, "ids": ids},
            )
            await db.commit()

            return [(str(row[0]), row[1], str(row[2])) for row in rows]

        except Exception as e:
            logger.exception("Failed to claim pending files: %s", e)
            await db.rollback()
            return []


async def _recover_stuck_files() -> None:
    """Reset files stuck in 'processing' beyond the timeout.

    Increments retry_count. After max retries, marks as 'failed'.
    """
    timeout = timedelta(minutes=settings.extraction_timeout_minutes)
    cutoff = datetime.now(timezone.utc) - timeout
    max_retries = settings.extraction_max_retries

    async with async_session_factory() as db:
        try:
            # Reset retriable files back to pending
            await db.execute(
                text(
                    "UPDATE uploaded_files "
                    "SET ingestion_status = 'pending_extraction', "
                    "processing_started_at = NULL, "
                    "retry_count = COALESCE(retry_count, 0) + 1 "
                    "WHERE ingestion_status = 'processing' "
                    "AND file_category = 'unstructured' "
                    "AND processing_started_at < :cutoff "
                    "AND COALESCE(retry_count, 0) < :max_retries"
                ),
                {"cutoff": cutoff, "max_retries": max_retries},
            )

            # Mark files that exceeded max retries as failed
            await db.execute(
                text(
                    "UPDATE uploaded_files "
                    "SET ingestion_status = 'failed', "
                    "ingestion_errors = '[{\"error\": \"Processing timed out after maximum retries.\", \"error_type\": \"TimeoutError\"}]'::jsonb, "
                    "processing_completed_at = :now "
                    "WHERE ingestion_status = 'processing' "
                    "AND file_category = 'unstructured' "
                    "AND processing_started_at < :cutoff "
                    "AND COALESCE(retry_count, 0) >= :max_retries"
                ),
                {"cutoff": cutoff, "max_retries": max_retries, "now": datetime.now(timezone.utc)},
            )

            await db.commit()
        except Exception:
            logger.exception("Failed to recover stuck files")
            await db.rollback()


async def _process_and_release(
    sem: asyncio.Semaphore, upload_id: UUID | str, file_path: Path, user_id: UUID | str
) -> None:
    """Process one file then release the semaphore."""
    try:
        uid = UUID(str(upload_id)) if not isinstance(upload_id, UUID) else upload_id
        uid_user = UUID(str(user_id)) if not isinstance(user_id, UUID) else user_id
        await _process_unstructured(uid, file_path, uid_user)
    finally:
        sem.release()


def start_extraction_worker() -> None:
    """Start the DB-polling extraction worker. Called from main.py lifespan."""
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_extraction_worker())

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.services.extraction.section_parser import parse_sections, split_large_section
from app.middleware.audit import log_audit_event
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.schemas.upload import (
    BatchUploadResponse,
    ConfirmExtractionRequest,
    ExtractedEntitySchema,
    ExtractionResultResponse,
    PendingExtractionFile,
    UnstructuredUploadResponse,
    UploadHistoryResponse,
    UploadResponse,
    UploadStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


# --- Security helpers ---

MAGIC_BYTES = {
    ".pdf": b"%PDF",
    ".rtf": b"{\\rtf",
    ".tif": [b"\x49\x49\x2a\x00", b"\x4d\x4d\x00\x2a"],  # LE and BE TIFF
    ".tiff": [b"\x49\x49\x2a\x00", b"\x4d\x4d\x00\x2a"],
}


def _validate_magic_bytes(content: bytes, ext: str) -> bool:
    """Validate file content matches expected magic bytes for the extension."""
    expected = MAGIC_BYTES.get(ext)
    if expected is None:
        return True  # No magic bytes check for unknown types
    if isinstance(expected, list):
        return any(content[:len(sig)] == sig for sig in expected)
    return content[:len(expected)] == expected


def _safe_file_path(upload_dir: Path, user_id: UUID, original_filename: str) -> Path:
    """Generate a safe file path preventing path traversal attacks."""
    # Preserve original extension only
    ext = Path(original_filename).suffix.lower()
    safe_name = f"{user_id}_{uuid4().hex}{ext}"
    file_path = (upload_dir / safe_name).resolve()

    # Validate the resolved path is within the upload directory
    upload_dir_resolved = upload_dir.resolve()
    if not str(file_path).startswith(str(upload_dir_resolved)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    return file_path


# --- Endpoints ---


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Upload a FHIR JSON or ZIP file for ingestion."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = _safe_file_path(upload_dir, user_id, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        f.write(content)

    # Run ingestion synchronously for now (small files)
    from app.services.ingestion.coordinator import ingest_file

    result = await ingest_file(
        db=db,
        user_id=user_id,
        file_path=file_path,
        original_filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
    )

    await log_audit_event(
        db,
        user_id=user_id,
        action="file.upload",
        resource_type="uploaded_file",
        resource_id=UUID(result["upload_id"]),
        details={"filename": file.filename, "records": result["records_inserted"]},
    )

    return UploadResponse(
        upload_id=result["upload_id"],
        status=result["status"],
        records_inserted=result["records_inserted"],
        errors=result.get("errors", []),
        unstructured_uploads=result.get("unstructured_uploads", []),
    )


@router.post("/epic-export", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_epic_export(
    file: UploadFile,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Upload an Epic EHI Tables export (ZIP)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = _safe_file_path(upload_dir, user_id, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        # C6: Size check for epic exports
        max_bytes = settings.max_epic_export_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Epic export too large. Maximum size: {settings.max_epic_export_size_mb}MB",
            )
        f.write(content)

    from app.services.ingestion.coordinator import ingest_file

    result = await ingest_file(
        db=db,
        user_id=user_id,
        file_path=file_path,
        original_filename=file.filename,
        mime_type=file.content_type or "application/zip",
    )

    return UploadResponse(
        upload_id=result["upload_id"],
        status=result["status"],
        records_inserted=result["records_inserted"],
        errors=result.get("errors", []),
    )


@router.get("/pending-extraction")
async def get_pending_extractions(
    statuses: str | None = None,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List files by extraction status for this user.

    Args:
        statuses: Comma-separated list of statuses to filter by.
                  Defaults to 'pending_extraction'.
    """
    status_list = [s.strip() for s in statuses.split(",")] if statuses else ["pending_extraction"]

    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.user_id == user_id)
        .where(UploadedFile.ingestion_status.in_(status_list))
        .order_by(UploadedFile.created_at.desc())
    )
    files = result.scalars().all()

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.pending_extraction.list",
        resource_type="uploaded_file",
        resource_id=None,
        details={"count": len(files), "statuses": status_list},
    )

    return {
        "files": [
            PendingExtractionFile(
                id=str(f.id),
                filename=f.filename,
                mime_type=f.mime_type,
                file_category=f.file_category,
                file_size_bytes=f.file_size_bytes,
                created_at=f.created_at.isoformat() if f.created_at else None,
                ingestion_status=f.ingestion_status,
            ).model_dump()
            for f in files
        ],
        "total": len(files),
    }


@router.get("/extraction-progress")
async def extraction_progress(
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get extraction progress counts for the user's unstructured files."""
    from sqlalchemy import func, case

    result = await db.execute(
        select(
            func.count().label("total"),
            func.count().filter(UploadedFile.ingestion_status.in_(["completed", "awaiting_confirmation", "awaiting_review", "completed_with_merges"])).label("completed"),
            func.count().filter(UploadedFile.ingestion_status == "processing").label("processing"),
            func.count().filter(UploadedFile.ingestion_status == "failed").label("failed"),
            func.count().filter(UploadedFile.ingestion_status == "pending_extraction").label("pending"),
            func.coalesce(
                func.sum(
                    case(
                        (UploadedFile.ingestion_status.in_(["completed", "awaiting_confirmation", "awaiting_review", "completed_with_merges"]), UploadedFile.record_count),
                        else_=0,
                    )
                ),
                0,
            ).label("records_created"),
        )
        .where(UploadedFile.user_id == user_id)
        .where(UploadedFile.file_category == "unstructured")
    )
    row = result.one()

    return {
        "total": row.total,
        "completed": row.completed,
        "processing": row.processing,
        "failed": row.failed,
        "pending": row.pending,
        "records_created": row.records_created,
    }


@router.post("/trigger-extraction")
async def trigger_extraction(
    body: dict,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Trigger text+entity extraction for pending unstructured files."""
    from app.schemas.upload import TriggerExtractionRequest

    req = TriggerExtractionRequest(**body)
    upload_ids = [UUID(uid) for uid in req.upload_ids]

    # Bulk fetch only uploads owned by this user (HIPAA: row-level security)
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id.in_(upload_ids),
            UploadedFile.user_id == user_id,
        )
    )
    uploads = {u.id: u for u in result.scalars().all()}

    triggered = []
    failed = []

    for uid in upload_ids:
        upload = uploads.get(uid)
        if not upload:
            failed.append({"upload_id": str(uid), "status": "not_found"})
            continue
        if upload.ingestion_status not in ("pending_extraction", "processing", "failed", "awaiting_confirmation"):
            failed.append({"upload_id": str(uid), "status": upload.ingestion_status})
            continue

        # Set to pending_extraction — the DB-polling worker picks them up automatically
        if upload.ingestion_status in ("failed", "awaiting_confirmation", "processing"):
            upload.ingestion_status = "pending_extraction"
        triggered.append(upload)

    if triggered:
        await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.trigger_extraction",
        resource_type="uploaded_file",
        resource_id=None,
        details={"triggered": len(triggered), "failed": len(failed)},
    )

    return {
        "triggered": len(triggered),
        "failed": len(failed),
        "results": [
            {"upload_id": str(u.id), "status": "pending_extraction"} for u in triggered
        ] + failed,
    }


@router.get("/{upload_id}/status", response_model=UploadStatusResponse)
async def get_upload_status(
    upload_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UploadStatusResponse:
    """Get ingestion job status."""
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    return UploadStatusResponse(
        upload_id=str(upload.id),
        filename=upload.filename,
        ingestion_status=upload.ingestion_status,
        record_count=upload.record_count,
        total_file_count=upload.total_file_count or 1,
        ingestion_progress=upload.ingestion_progress or {},
        ingestion_errors=upload.ingestion_errors or [],
        processing_started_at=upload.processing_started_at,
        processing_completed_at=upload.processing_completed_at,
    )


@router.get("/{upload_id}/errors")
async def get_upload_errors(
    upload_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get ingestion errors for a specific upload."""
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    return {"errors": upload.ingestion_errors or []}


@router.get("/history", response_model=UploadHistoryResponse)
async def get_upload_history(
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UploadHistoryResponse:
    """Upload history with record counts."""
    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.user_id == user_id)
        .order_by(UploadedFile.created_at.desc())
    )
    uploads = result.scalars().all()

    items = []
    for u in uploads:
        items.append({
            "id": str(u.id),
            "filename": u.filename,
            "ingestion_status": u.ingestion_status,
            "record_count": u.record_count,
            "file_size_bytes": u.file_size_bytes,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        })

    return UploadHistoryResponse(items=items, total=len(items))


ALLOWED_UNSTRUCTURED = {".pdf", ".rtf", ".tif", ".tiff"}


async def _process_unstructured(upload_id: UUID, file_path: Path, user_id: UUID) -> None:
    """Background task: extract text then entities from an unstructured file.

    Uses section-aware extraction: Gemini parses document structure first,
    then LangExtract runs per-section (respecting the 2000-char buffer).
    """
    from app.services.extraction.text_extractor import extract_text, detect_file_type as _detect_file_type
    from app.services.extraction.text_extractor import FileType as _FileType
    from app.services.extraction.entity_extractor import extract_entities_async
    from app.services.ai.phi_scrubber import scrub_phi

    async with async_session_factory() as db:
        result = await db.execute(
            select(UploadedFile).where(UploadedFile.id == upload_id)
        )
        upload = result.scalar_one_or_none()
        if not upload:
            return

        try:
            upload.processing_started_at = datetime.now(timezone.utc)
            upload.ingestion_status = "processing"
            await db.commit()

            sem = _get_gemini_semaphore()

            # Step 1: Extract text (Gemini for PDF/TIFF, local for RTF)
            file_type_enum = _detect_file_type(file_path)
            if file_type_enum == _FileType.RTF:
                extracted_text, file_type = await extract_text(file_path, settings.gemini_api_key)
            else:
                async with sem:
                    extracted_text, file_type = await extract_text(file_path, settings.gemini_api_key)
            text = extracted_text
            upload.extracted_text = text
            await db.commit()

            # Step 2: Scrub PHI before entity extraction — local, no semaphore
            scrubbed_text, deident_report = scrub_phi(text)

            # Step 3: Section parsing (skip Gemini call for small docs)
            if len(scrubbed_text) < settings.small_doc_threshold:
                from app.services.extraction.section_parser import ParsedDocument, ParsedSection, SectionType
                parsed_doc = ParsedDocument(
                    sections=[ParsedSection(
                        section_type=SectionType.OTHER,
                        title="Full Document",
                        text=scrubbed_text,
                        char_range=(0, len(scrubbed_text)),
                    )],
                    document_type="clinical_note",
                    primary_visit_date=None,
                    provider=None,
                    facility=None,
                )
            else:
                async with sem:
                    parsed_doc = await parse_sections(scrubbed_text, settings.gemini_api_key)

            upload.extraction_sections = {
                "sections": [
                    {"type": s.section_type.value, "title": s.title, "char_range": s.char_range}
                    for s in parsed_doc.sections
                ]
            }
            upload.document_metadata = {
                "document_type": parsed_doc.document_type,
                "primary_visit_date": parsed_doc.primary_visit_date,
                "provider": parsed_doc.provider,
                "facility": parsed_doc.facility,
                "section_count": len(parsed_doc.sections),
            }
            await db.commit()

            # Step 4: Per-section entity extraction (with small-chunk batching)
            all_entities = []
            extraction_tasks = []
            current_batch = ""
            current_section = None

            for section in parsed_doc.sections:
                chunks = split_large_section(section.text)
                for chunk in chunks:
                    if current_batch and len(current_batch) + len(chunk) + 1 <= 2000:
                        current_batch += "\n" + chunk
                    else:
                        if current_batch:
                            extraction_tasks.append((current_batch, current_section))
                        current_batch = chunk
                        current_section = section.section_type.value
            if current_batch:
                extraction_tasks.append((current_batch, current_section))

            # Process sections concurrently (configurable)
            section_sem = asyncio.Semaphore(settings.section_extraction_concurrency)

            async def extract_chunk(text_chunk: str, section_type: str):
                async with section_sem:
                    async with sem:
                        chunk_result = await extract_entities_async(
                            text_chunk, upload.filename, settings.gemini_api_key
                        )
                return chunk_result, section_type

            tasks = [extract_chunk(chunk, stype) for chunk, stype in extraction_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    logger.error("Section extraction failed: %s", r)
                    continue
                extraction_result, section_type = r
                if extraction_result.error:
                    logger.warning("Extraction error in section %s: %s", section_type, extraction_result.error)
                    continue
                for entity in extraction_result.entities:
                    entity.attributes["_source_section"] = section_type
                    all_entities.append(entity)

            # Deduplicate entities within the same document (same text + same type)
            seen = set()
            unique_entities = []
            for entity in all_entities:
                key = (entity.entity_class, entity.text.strip().lower())
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)

            # Store entities on upload record
            upload.extraction_entities = [
                {
                    "entity_class": e.entity_class,
                    "text": e.text,
                    "attributes": e.attributes,
                    "start_pos": e.start_pos,
                    "end_pos": e.end_pos,
                    "confidence": e.confidence,
                }
                for e in unique_entities
            ]
            await db.commit()

            # Step 5: Auto-confirm if patient exists (with encounter linking)
            from app.models.patient import Patient
            from app.services.extraction.entity_to_fhir import entity_to_health_record_dict

            patient_result = await db.execute(
                select(Patient).where(Patient.user_id == user_id).limit(1)
            )
            patient = patient_result.scalar_one_or_none()

            if patient:
                from app.services.ingestion.reextraction import soft_delete_prior_extracted
                replaced = await soft_delete_prior_extracted(db, upload_id)
                if replaced:
                    logger.info("Re-extraction replaced %d prior records for %s", replaced, upload_id)

                encounter_id = None
                created_records = []

                for entity in unique_entities:
                    record_dict = entity_to_health_record_dict(
                        entity, user_id, patient.id, upload_id
                    )
                    if record_dict is None:
                        continue
                    record_dict["source_section"] = entity.attributes.get("_source_section")
                    record = HealthRecord(**record_dict)
                    db.add(record)
                    created_records.append((record, entity))

                    # Track encounter ID for linking
                    if entity.entity_class == "encounter":
                        await db.flush()
                        encounter_id = record.id

                # Link all records to the encounter
                if encounter_id:
                    for record, _ in created_records:
                        if record.id != encounter_id:
                            record.linked_encounter_id = encounter_id

                # Create cross-references from A&P DocumentReference to other records
                ap_records = [(r, e) for r, e in created_records if e.entity_class == "assessment_plan"]
                non_ap_records = [(r, e) for r, e in created_records if e.entity_class != "assessment_plan"]
                if ap_records and non_ap_records:
                    from app.models.cross_reference import RecordCrossReference
                    await db.flush()  # Ensure all record IDs are assigned
                    for ap_record, _ in ap_records:
                        for other_record, other_entity in non_ap_records:
                            if other_entity.entity_class in ("encounter",):
                                continue  # Don't cross-ref the encounter itself
                            ref_type = {
                                "medication": "prescribes",
                                "condition": "addresses",
                                "lab_result": "supports",
                                "vital": "supports",
                                "procedure": "addresses",
                                "allergy": "addresses",
                                "imaging_result": "supports",
                                "family_history": "supports",
                                "social_history": "supports",
                            }.get(other_entity.entity_class, "addresses")
                            xref = RecordCrossReference(
                                document_record_id=ap_record.id,
                                referenced_record_id=other_record.id,
                                reference_type=ref_type,
                            )
                            db.add(xref)

                await db.commit()

                # Run dedup scan in background
                upload.ingestion_status = "dedup_scanning"
                upload.record_count = len(created_records)
                await db.commit()

                from app.services.ingestion.coordinator import _run_dedup_background
                asyncio.create_task(
                    _run_dedup_background(upload_id, patient.id, user_id)
                )
            else:
                # No patient found — fall back to manual confirmation
                upload.ingestion_status = "awaiting_confirmation"

            upload.processing_completed_at = datetime.now(timezone.utc)
            await db.commit()

        except Exception as e:
            # H4: Log full error internally, expose only error type to client
            logger.error("Unstructured processing failed for %s: %s", upload_id, e, exc_info=True)
            error_type = type(e).__name__
            upload.ingestion_status = "failed"
            upload.ingestion_errors = [{"error": "Processing failed. Please retry or contact support.", "error_type": error_type}]
            upload.processing_completed_at = datetime.now(timezone.utc)
            await db.commit()


@router.post(
    "/unstructured",
    response_model=UnstructuredUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_unstructured(
    file: UploadFile,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UnstructuredUploadResponse:
    """Upload a PDF, RTF, or TIFF for AI-powered text and entity extraction."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_UNSTRUCTURED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_UNSTRUCTURED)}",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = _safe_file_path(upload_dir, user_id, file.filename)
    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    # M1: Validate magic bytes
    if not _validate_magic_bytes(content, ext):
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match expected format for {ext}",
        )

    with open(file_path, "wb") as f:
        f.write(content)

    file_hash = hashlib.sha256(content).hexdigest()

    from app.services.ingestion.reextraction import find_prior_extracted_upload
    prior = await find_prior_extracted_upload(db, user_id, file_hash)

    upload_record = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        file_size_bytes=len(content),
        file_hash=file_hash,
        storage_path=str(file_path),
        ingestion_status="duplicate_file" if prior else "pending_extraction",
        file_category="unstructured",
    )
    if prior:
        upload_record.ingestion_progress = {
            "duplicate_of": str(prior.id),
            "record_count": prior.record_count or 0,
        }
    db.add(upload_record)
    await db.commit()
    await db.refresh(upload_record)

    await log_audit_event(
        db,
        user_id=user_id,
        action="file.upload.unstructured",
        resource_type="uploaded_file",
        resource_id=upload_record.id,
        details={"filename": file.filename, "file_type": ext},
    )

    # Worker will pick up the file automatically via DB polling (duplicate_file rows are skipped)

    from app.services.extraction.text_extractor import detect_file_type
    file_type = detect_file_type(file_path)

    return UnstructuredUploadResponse(
        upload_id=str(upload_record.id),
        status=upload_record.ingestion_status,
        file_type=file_type.value,
    )


@router.post(
    "/unstructured-batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_unstructured_batch(
    files: list[UploadFile],
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> BatchUploadResponse:
    """Upload multiple unstructured files for concurrent processing."""
    from app.services.extraction.text_extractor import detect_file_type
    from app.services.ingestion.reextraction import find_prior_extracted_upload

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_UNSTRUCTURED:
            continue

        file_path = _safe_file_path(upload_dir, user_id, file.filename)
        content = await file.read()
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            continue

        if not _validate_magic_bytes(content, ext):
            continue

        with open(file_path, "wb") as f:
            f.write(content)

        file_hash = hashlib.sha256(content).hexdigest()
        prior = await find_prior_extracted_upload(db, user_id, file_hash)

        upload_record = UploadedFile(
            id=uuid4(),
            user_id=user_id,
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            file_size_bytes=len(content),
            file_hash=file_hash,
            storage_path=str(file_path),
            ingestion_status="duplicate_file" if prior else "pending_extraction",
            file_category="unstructured",
        )
        if prior:
            upload_record.ingestion_progress = {
                "duplicate_of": str(prior.id),
                "record_count": prior.record_count or 0,
            }
        db.add(upload_record)
        await db.flush()

        await log_audit_event(
            db,
            user_id=user_id,
            action="file.upload.unstructured",
            resource_type="uploaded_file",
            resource_id=upload_record.id,
            details={"filename": file.filename, "file_type": ext},
        )

        file_type = detect_file_type(file_path)
        results.append(UnstructuredUploadResponse(
            upload_id=str(upload_record.id),
            status=upload_record.ingestion_status,
            file_type=file_type.value,
        ))

    await db.commit()

    # Worker will pick up files automatically via DB polling (duplicate_file rows are skipped)

    return BatchUploadResponse(uploads=results, total=len(results))


@router.get("/{upload_id}/extraction", response_model=ExtractionResultResponse)
async def get_extraction_results(
    upload_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> ExtractionResultResponse:
    """Get extraction results for an unstructured upload."""
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    entities = []
    if upload.extraction_entities:
        entities = [
            ExtractedEntitySchema(**e) for e in upload.extraction_entities
        ]

    preview = None
    if upload.extracted_text:
        preview = upload.extracted_text[:500]

    error = None
    if upload.ingestion_errors:
        errors = upload.ingestion_errors
        if errors and isinstance(errors, list) and len(errors) > 0:
            error = errors[0].get("error", str(errors[0])) if isinstance(errors[0], dict) else str(errors[0])

    return ExtractionResultResponse(
        upload_id=str(upload.id),
        status=upload.ingestion_status,
        extracted_text_preview=preview,
        entities=entities,
        error=error,
    )


@router.post("/{upload_id}/confirm-extraction")
async def confirm_extraction(
    upload_id: UUID,
    body: ConfirmExtractionRequest,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Confirm extracted entities and save them as HealthRecords."""
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    if not body.patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")

    from app.services.extraction.entity_extractor import ExtractedEntity
    from app.services.extraction.entity_to_fhir import entity_to_health_record_dict

    patient_uuid = UUID(body.patient_id)
    created_count = 0

    from app.services.ingestion.reextraction import soft_delete_prior_extracted
    replaced = await soft_delete_prior_extracted(db, upload_id)
    if replaced:
        logger.info("Manual re-confirm replaced %d prior records for %s", replaced, upload_id)

    for entity_data in body.confirmed_entities:
        entity = ExtractedEntity(
            entity_class=entity_data.entity_class,
            text=entity_data.text,
            attributes=entity_data.attributes,
            start_pos=entity_data.start_pos,
            end_pos=entity_data.end_pos,
            confidence=entity_data.confidence,
        )

        record_dict = entity_to_health_record_dict(
            entity=entity,
            user_id=user_id,
            patient_id=patient_uuid,
            source_file_id=upload_id,
        )
        if record_dict is None:
            continue

        health_record = HealthRecord(**record_dict)
        db.add(health_record)
        created_count += 1

    await db.commit()

    # Run dedup in background
    upload.ingestion_status = "dedup_scanning"
    upload.record_count = created_count
    await db.commit()

    from app.services.ingestion.coordinator import _run_dedup_background
    asyncio.create_task(
        _run_dedup_background(upload_id, patient_uuid, user_id)
    )

    await log_audit_event(
        db,
        user_id=user_id,
        action="extraction.confirm",
        resource_type="uploaded_file",
        resource_id=upload_id,
        details={"records_created": created_count, "patient_id": body.patient_id},
    )

    return {
        "upload_id": str(upload_id),
        "records_created": created_count,
        "status": "completed",
    }


@router.get("/{upload_id}/review")
async def get_upload_review(
    upload_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get dedup review data for an upload."""
    from app.models.deduplication import DedupCandidate

    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Fetch all candidates for this upload
    candidates_result = await db.execute(
        select(DedupCandidate).where(
            DedupCandidate.source_upload_id == upload_id,
        )
    )
    candidates = candidates_result.scalars().all()

    auto_merged = []
    needs_review: dict[str, list] = {}

    for c in candidates:
        rec_a = await db.get(HealthRecord, c.record_a_id)
        rec_b = await db.get(HealthRecord, c.record_b_id)
        if not rec_a or not rec_b:
            continue

        entry = {
            "candidate_id": str(c.id),
            "primary": {
                "id": str(rec_a.id),
                "display_text": rec_a.display_text or "",
                "record_type": rec_a.record_type,
                "fhir_resource": rec_a.fhir_resource,
            },
            "secondary": {
                "id": str(rec_b.id),
                "display_text": rec_b.display_text or "",
                "record_type": rec_b.record_type,
                "fhir_resource": rec_b.fhir_resource,
            },
            "similarity_score": c.similarity_score,
            "llm_classification": c.llm_classification,
            "llm_confidence": c.llm_confidence,
            "llm_explanation": c.llm_explanation,
            "field_diff": c.field_diff,
            "merged_at": c.resolved_at.isoformat() if c.resolved_at else None,
        }

        if c.status == "merged" and c.auto_resolved:
            auto_merged.append(entry)
        elif c.status == "pending":
            rtype = rec_a.record_type
            needs_review.setdefault(rtype, []).append(entry)

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.review.view",
        resource_type="uploaded_file",
        resource_id=upload_id,
    )

    return {
        "upload": {
            "id": str(upload.id),
            "filename": upload.filename,
            "uploaded_at": upload.created_at.isoformat() if upload.created_at else None,
            "record_count": upload.record_count,
            "status": upload.ingestion_status,
            "dedup_summary": upload.dedup_summary,
        },
        "auto_merged": auto_merged,
        "needs_review": needs_review,
    }


@router.post("/{upload_id}/review/resolve")
async def resolve_review(
    upload_id: UUID,
    body: dict,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Bulk resolve dedup candidates for an upload."""
    from app.models.deduplication import DedupCandidate
    from app.models.provenance import Provenance
    from app.services.dedup.field_merger import apply_field_update
    from app.services.ingestion.content_hash import content_hash

    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    resolutions = body.get("resolutions", [])
    resolved_count = 0

    for resolution in resolutions:
        candidate_id = UUID(resolution["candidate_id"])
        action = resolution["action"]
        field_overrides = resolution.get("field_overrides")

        candidate = await db.get(DedupCandidate, candidate_id)
        if not candidate or candidate.source_upload_id != upload_id:
            continue

        rec_a = await db.get(HealthRecord, candidate.record_a_id)
        rec_b = await db.get(HealthRecord, candidate.record_b_id)
        if not rec_a or not rec_b:
            continue

        now = datetime.now(timezone.utc)

        if action == "merge":
            rec_b.is_duplicate = True
            rec_b.merged_into_id = rec_a.id
            rec_b.merge_metadata = {
                "merged_from": str(rec_b.id),
                "merged_at": now.isoformat(),
                "merge_type": "duplicate",
                "source_upload_id": str(upload_id),
            }
            candidate.status = "merged"
            candidate.resolved_by = user_id
            candidate.resolved_at = now
            db.add(Provenance(
                record_id=rec_a.id,
                action="merge",
                agent=f"user/{user_id}",
                source_file_id=upload_id,
                details={"merged_record_id": str(rec_b.id), "action": "merge"},
            ))

        elif action == "update":
            merge_result = apply_field_update(rec_a, rec_b, field_overrides)
            rec_a.fhir_resource = merge_result["updated_resource"]
            rec_a.content_hash = content_hash(rec_a.fhir_resource)
            rec_a.display_text = merge_result["display_text"]
            rec_a.merge_metadata = merge_result["merge_metadata"]
            rec_b.is_duplicate = True
            rec_b.merged_into_id = rec_a.id
            candidate.status = "merged"
            candidate.resolved_by = user_id
            candidate.resolved_at = now
            db.add(Provenance(
                record_id=rec_a.id,
                action="field_update",
                agent=f"user/{user_id}",
                source_file_id=upload_id,
                details={
                    "merged_record_id": str(rec_b.id),
                    "fields_updated": merge_result["merge_metadata"].get("fields_updated", []),
                },
            ))

        elif action in ("dismiss", "keep_both"):
            candidate.status = "dismissed"
            candidate.resolved_by = user_id
            candidate.resolved_at = now

        resolved_count += 1

    # Check if all candidates are resolved
    pending_result = await db.execute(
        select(DedupCandidate).where(
            DedupCandidate.source_upload_id == upload_id,
            DedupCandidate.status == "pending",
        )
    )
    remaining = pending_result.scalars().all()
    if not remaining:
        upload.ingestion_status = "completed"

    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.review.resolve",
        resource_type="uploaded_file",
        resource_id=upload_id,
        details={"resolutions_count": resolved_count},
    )

    return {"resolved": resolved_count, "remaining": len(remaining)}


@router.post("/{upload_id}/review/undo-merge")
async def undo_merge(
    upload_id: UUID,
    body: dict,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Undo an auto-merged dedup candidate."""
    from app.models.deduplication import DedupCandidate
    from app.services.dedup.field_merger import revert_field_update

    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    candidate_id = UUID(body["candidate_id"])
    candidate = await db.get(DedupCandidate, candidate_id)
    if not candidate or candidate.source_upload_id != upload_id:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if candidate.status != "merged":
        raise HTTPException(status_code=400, detail="Candidate is not merged")

    # Restore secondary record
    rec_b = await db.get(HealthRecord, candidate.record_b_id)
    if rec_b:
        rec_b.is_duplicate = False
        rec_b.merged_into_id = None

    # Revert field changes on primary if this was a field update
    rec_a = await db.get(HealthRecord, candidate.record_a_id)
    if rec_a and rec_a.merge_metadata and rec_a.merge_metadata.get("previous_values"):
        revert_field_update(rec_a)

    # Reset candidate
    candidate.status = "pending"
    candidate.resolved_by = None
    candidate.resolved_at = None

    # Update upload status if needed
    if upload.ingestion_status == "completed":
        upload.ingestion_status = "awaiting_review"

    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.review.undo",
        resource_type="uploaded_file",
        resource_id=upload_id,
        details={"candidate_id": str(candidate_id)},
    )

    return {"status": "undone", "candidate_id": str(candidate_id)}
