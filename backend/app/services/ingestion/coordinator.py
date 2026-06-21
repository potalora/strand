from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.patient import Patient
from app.models.uploaded_file import UploadedFile
from app.services.ingestion.cda_dedup import deduplicate_across_documents
from app.services.ingestion.cda_parser import parse_cda_document
from app.services.ingestion.epic_parser import parse_epic_export
from app.services.ingestion.fhir_parser import parse_fhir_bundle
from app.services.ingestion.idempotent_inserter import idempotent_insert_records
from app.services.ingestion.patient_demographics import (
    backfill_patient_demographics,
    extract_epic_demographics,
    extract_fhir_demographics,
)
from app.services.ingestion.xdm_parser import parse_xdm_metadata
from app.utils.file_utils import decrypt_file_to, is_encrypted_file

logger = logging.getLogger(__name__)

# --- SEC-DOS-02: zip-bomb defenses ---------------------------------------
#
# The upload size limits only bound the COMPRESSED archive; a tiny zip can
# declare/inflate to many gigabytes (the classic "zip bomb"). These caps bound
# the DECOMPRESSED output instead, and are enforced both up front (declared
# header totals) and while streaming each member (the header can lie). They are
# module-level so tests can tighten them without a multi-GB fixture.
_ZIP_MAX_ENTRIES = 100_000
_ZIP_MAX_MEMBER_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB / member
_ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES = 50 * 1024 * 1024 * 1024  # 50 GiB total
_ZIP_MAX_COMPRESSION_RATIO = 100.0  # reject members inflating > 100x
_ZIP_EXTRACT_CHUNK = 1024 * 1024  # 1 MiB streaming chunk


def _zip_safe_target(temp_dir: Path, member_name: str) -> Path | None:
    """Resolve a zip member to a path inside ``temp_dir``, or None if it escapes.

    Defends against zip-slip (``../`` / absolute paths) by rejecting any member
    whose resolved path is not contained within the extraction directory.
    """
    target = (temp_dir / member_name).resolve()
    base = temp_dir.resolve()
    try:
        target.relative_to(base)
    except ValueError:
        logger.warning("Skipping zip member outside extract dir: %s", member_name)
        return None
    return target


def _safe_extract_zip(zf: zipfile.ZipFile, temp_dir: Path) -> None:
    """Extract a zip member-by-member with zip-bomb defenses (SEC-DOS-02).

    Replaces ``zf.extractall`` (which has no decompression cap). First rejects up
    front on the declared totals — member count, per-member size, per-member
    compression ratio, and the summed uncompressed size — then extracts each
    member streaming its bytes and re-checking the byte budget against the actual
    decompressed output (header sizes can be forged). Raises ``HTTPException``
    (413/400) on any breach so the upload endpoint returns a clean rejection.
    """
    infos = [i for i in zf.infolist() if not i.is_dir()]

    if len(infos) > _ZIP_MAX_ENTRIES:
        raise HTTPException(
            status_code=400,
            detail=f"ZIP has too many entries (>{_ZIP_MAX_ENTRIES})",
        )

    declared_total = 0
    for info in infos:
        if info.file_size > _ZIP_MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise HTTPException(status_code=413, detail="ZIP member too large")
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > _ZIP_MAX_COMPRESSION_RATIO:
                raise HTTPException(
                    status_code=413, detail="ZIP compression ratio too high (suspected zip bomb)"
                )
        declared_total += info.file_size
    if declared_total > _ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise HTTPException(
            status_code=413, detail="ZIP uncompressed size exceeds the allowed budget"
        )

    # Stream each member, enforcing the REAL decompressed byte budget — the
    # declared sizes above can lie, so count actual bytes as we write.
    extracted_total = 0
    for info in infos:
        target = _zip_safe_target(temp_dir, info.filename)
        if target is None:
            continue  # zip-slip — skip defensively
        target.parent.mkdir(parents=True, exist_ok=True)
        member_bytes = 0
        with zf.open(info, "r") as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(_ZIP_EXTRACT_CHUNK)
                if not chunk:
                    break
                member_bytes += len(chunk)
                extracted_total += len(chunk)
                if member_bytes > _ZIP_MAX_MEMBER_UNCOMPRESSED_BYTES:
                    raise HTTPException(status_code=413, detail="ZIP member too large")
                if extracted_total > _ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="ZIP uncompressed size exceeds the allowed budget",
                    )
                dst.write(chunk)


def _find_patient_resource_streaming(file_path: Path) -> dict | None:
    """Locate the first Patient resource in a FHIR bundle by streaming (ijson).

    SEC-INJ-03: avoids ``json.load``-ing the entire (up to 500MB) bundle into RAM
    just to read the patient demographics — that negated the streaming the bundle
    parser already uses. Fail-open: a malformed / non-bundle file yields ``None``
    and ingestion proceeds with the default patient.
    """
    import ijson

    try:
        with open(file_path, "rb") as f:
            for entry in ijson.items(f, "entry.item"):
                if not isinstance(entry, dict):
                    continue
                resource = entry.get("resource")
                if isinstance(resource, dict) and resource.get("resourceType") == "Patient":
                    return resource
    except Exception as e:  # noqa: BLE001 - fail-open; never block ingestion
        logger.warning("Streaming Patient lookup failed for %s: %s", file_path, e)
    return None


async def get_or_create_patient(
    db: AsyncSession, user_id: UUID, fhir_data: dict | None = None
) -> Patient:
    """Get existing patient for user or create a default one."""
    result = await db.execute(select(Patient).where(Patient.user_id == user_id))
    patient = result.scalar_one_or_none()
    demo = extract_fhir_demographics(fhir_data) if fhir_data else {}

    if patient:
        # Backfill any identifiers a prior (blank) creation missed — critical so
        # the deterministic PHI scrubber can strip the patient's own name.
        if demo:
            await backfill_patient_demographics(
                db, patient,
                name=demo.get("name"), mrn=demo.get("mrn"),
                dob=demo.get("dob"), gender=demo.get("gender"),
            )
        return patient

    patient = Patient(
        id=uuid4(),
        user_id=user_id,
        fhir_id=fhir_data.get("id") if fhir_data else None,
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    if demo:
        await backfill_patient_demographics(
            db, patient,
            name=demo.get("name"), mrn=demo.get("mrn"),
            dob=demo.get("dob"), gender=demo.get("gender"),
        )
    return patient


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _is_cda_xml(file_path: Path) -> bool:
    """Check if an XML file is a CDA ClinicalDocument."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            header = f.read(500)
            return "ClinicalDocument" in header
    except (OSError, UnicodeDecodeError):
        return False


def detect_file_type(file_path: Path) -> str:
    """Detect whether a file is FHIR JSON, Epic TSV directory, CDA XML, or ZIP."""
    if file_path.is_dir():
        tsv_files = list(file_path.glob("*.tsv"))
        if tsv_files:
            return "epic_ehi"
        return "unknown"

    suffix = file_path.suffix.lower()
    if suffix == ".zip":
        return "zip"
    if suffix == ".json":
        return "fhir_r4"
    if suffix == ".tsv":
        return "epic_ehi_single"
    if suffix == ".xml" and _is_cda_xml(file_path):
        return "cda_xml"
    return "unknown"


def _find_xdm_metadata(root_dir: Path) -> Path | None:
    """Recursively find METADATA.XML with XDM SubmitObjectsRequest root."""
    for metadata_path in root_dir.rglob("METADATA.XML"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                header = f.read(500)
                if "SubmitObjectsRequest" in header:
                    return metadata_path
        except (OSError, UnicodeDecodeError):
            continue
    return None


async def ingest_file(
    db: AsyncSession,
    user_id: UUID,
    file_path: Path,
    original_filename: str,
    mime_type: str = "application/octet-stream",
) -> dict:
    """Main ingestion entry point. Detects file type and routes to appropriate parser."""
    # CRYPTO-02 (issue #54): structured uploads are encrypted at rest in the
    # framed AES-256-GCM format. The streaming parsers (ijson, member-by-member
    # zip extraction with the W9 caps) need PLAINTEXT input, so an encrypted
    # source is stream-decrypted ONCE to a temp file here and ALL downstream
    # logic (detect_file_type, compute_file_hash, the _ingest_* dispatch) runs on
    # that temp ``work_path``. ``decrypt_file_to`` streams frame-by-frame, so a
    # multi-GB file decrypts with bounded memory (the W9 OOM guarantee is kept).
    # ``storage_path`` stays the ORIGINAL encrypted path so future reads decrypt
    # again; the temp is unlinked in the ``finally`` even on error. Legacy
    # plaintext files and directory uploads skip the temp entirely.
    work_path = file_path
    temp_decrypted: Path | None = None
    if file_path.is_file() and is_encrypted_file(file_path):
        temp_dir = Path(settings.temp_extract_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        # Preserve the original suffix so detect_file_type routes correctly.
        temp_decrypted = temp_dir / f"decrypt_{uuid4().hex}{file_path.suffix}"
        decrypt_file_to(file_path, temp_decrypted)
        work_path = temp_decrypted

    try:
        file_type = detect_file_type(work_path)
        # Hash + size are computed on the PLAINTEXT work_path so the dedup hash is
        # stable across re-uploads despite the random per-frame encryption nonces.
        file_hash = compute_file_hash(work_path) if work_path.is_file() else "directory"
        file_size = work_path.stat().st_size if work_path.is_file() else 0

        # Create upload record — storage_path is the ORIGINAL (encrypted) path.
        upload = UploadedFile(
            id=uuid4(),
            user_id=user_id,
            filename=original_filename,
            mime_type=mime_type,
            file_size_bytes=file_size,
            file_hash=file_hash,
            storage_path=str(file_path),
            ingestion_status="processing",
            processing_started_at=datetime.now(timezone.utc),
        )
        db.add(upload)
        await db.commit()
        await db.refresh(upload)

        patient = await get_or_create_patient(db, user_id)

        try:
            if file_type == "fhir_r4":
                stats = await _ingest_fhir(db, user_id, patient.id, upload.id, work_path)
            elif file_type == "epic_ehi":
                stats = await _ingest_epic_dir(db, user_id, patient.id, upload.id, work_path)
            elif file_type == "zip":
                stats = await _ingest_zip(db, user_id, patient.id, upload.id, work_path)
            elif file_type == "cda_xml":
                stats = await _ingest_cda_standalone(db, user_id, patient.id, upload.id, work_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Set initial completion stats before dedup
            upload.record_count = stats.get("records_inserted", 0)
            upload.ingestion_errors = stats.get("errors", [])
            upload.ingestion_progress = {
                "total_entries": stats.get("total_entries", 0),
                "records_inserted": stats.get("records_inserted", 0),
                "records_skipped": stats.get("records_skipped", 0),
                "records_updated": stats.get("records_updated", 0),
                "records_unchanged": stats.get("records_unchanged", 0),
            }

            # Run dedup scan in background so the upload endpoint returns immediately
            upload.ingestion_status = "dedup_scanning"
            await db.commit()

            asyncio.create_task(
                _run_dedup_background(upload.id, patient.id, user_id)
            )

            return {
                "upload_id": str(upload.id),
                "status": "dedup_scanning",
                "records_inserted": stats.get("records_inserted", 0),
                "errors": stats.get("errors", []),
                "unstructured_uploads": stats.get("unstructured_files", []),
            }

        except Exception as e:
            logger.error("Ingestion failed for %s: %s", original_filename, e)
            upload.ingestion_status = "failed"
            upload.ingestion_errors = [{"error": str(e)}]
            upload.processing_completed_at = datetime.now(timezone.utc)
            await db.commit()
            raise
    finally:
        # Always drop the decrypted temp (it carries plaintext PHI), even on error.
        if temp_decrypted is not None:
            temp_decrypted.unlink(missing_ok=True)


async def _run_dedup_background(
    upload_id: UUID,
    patient_id: UUID,
    user_id: UUID,
) -> None:
    """Run dedup scanning in the background with its own DB session."""
    from app.database import async_session_factory
    from app.services.dedup.orchestrator import run_upload_dedup

    try:
        async with async_session_factory() as db:
            upload = await db.get(UploadedFile, upload_id)
            if not upload:
                logger.error("Background dedup: upload %s not found", upload_id)
                return

            dedup_summary = await run_upload_dedup(
                upload_id, patient_id, user_id, db
            )
            upload.dedup_summary = dedup_summary.to_dict()

            if dedup_summary.needs_review > 0:
                upload.ingestion_status = "awaiting_review"
            elif dedup_summary.auto_merged > 0:
                upload.ingestion_status = "completed_with_merges"
            else:
                upload.ingestion_status = "completed"

            upload.processing_completed_at = datetime.now(timezone.utc)
            await db.commit()
            logger.info(
                "Background dedup completed for %s: %d candidates, %d auto-merged, %d need review",
                upload_id, dedup_summary.total_candidates,
                dedup_summary.auto_merged, dedup_summary.needs_review,
            )
    except Exception:
        logger.exception("Background dedup failed for %s", upload_id)
        try:
            async with async_session_factory() as db:
                upload = await db.get(UploadedFile, upload_id)
                if upload:
                    upload.ingestion_status = "completed"
                    upload.processing_completed_at = datetime.now(timezone.utc)
                    await db.commit()
        except Exception:
            logger.exception("Failed to update upload status after dedup error")


async def _ingest_fhir(
    db: AsyncSession,
    user_id: UUID,
    patient_id: UUID,
    upload_id: UUID,
    file_path: Path,
) -> dict:
    """Ingest a FHIR R4 JSON file."""
    # SEC-INJ-03: locate the bundle's Patient by streaming (ijson) instead of
    # json.load-ing the whole (up to 500MB) bundle into RAM, which defeated the
    # streaming the parser already uses.
    patient_resource = _find_patient_resource_streaming(file_path)
    if patient_resource is not None:
        patient = await get_or_create_patient(db, user_id, patient_resource)
        patient_id = patient.id

    return await parse_fhir_bundle(
        file_path=file_path,
        user_id=user_id,
        patient_id=patient_id,
        source_file_id=upload_id,
        db=db,
    )


async def _backfill_patient_by_id(
    db: AsyncSession, patient_id: UUID, demo: dict
) -> None:
    """Load a patient by id and backfill demographics (best-effort)."""
    if not demo or not any(demo.values()):
        return
    patient = (
        await db.execute(select(Patient).where(Patient.id == patient_id))
    ).scalar_one_or_none()
    if patient is None:
        return
    await backfill_patient_demographics(
        db, patient,
        name=demo.get("name"), mrn=demo.get("mrn"),
        dob=demo.get("dob"), gender=demo.get("gender"),
    )


async def _ingest_epic_dir(
    db: AsyncSession,
    user_id: UUID,
    patient_id: UUID,
    upload_id: UUID,
    dir_path: Path,
) -> dict:
    """Ingest an Epic EHI Tables export directory."""
    # Backfill patient identifiers from PATIENT.tsv before records are inserted.
    await _backfill_patient_by_id(db, patient_id, extract_epic_demographics(dir_path))
    return await parse_epic_export(
        export_dir=dir_path,
        user_id=user_id,
        patient_id=patient_id,
        source_file_id=upload_id,
        db=db,
    )


async def _ingest_cda_standalone(
    db: AsyncSession,
    user_id: UUID,
    patient_id: UUID,
    upload_id: UUID,
    file_path: Path,
) -> dict:
    """Ingest a standalone CDA XML file (not inside an XDM package)."""
    stats: dict = {
        "total_entries": 0,
        "records_inserted": 0,
        "records_skipped": 0,
        "errors": [],
    }

    try:
        records = parse_cda_document(file_path, manifest_doc=None)
        stats["total_entries"] = len(records)
    except Exception as e:
        stats["errors"].append({"file": file_path.name, "error": str(e)})
        return stats

    if not records:
        stats["errors"].append({"error": "No records extracted from CDA document"})
        return stats

    for rec in records:
        rec["user_id"] = user_id
        rec["patient_id"] = patient_id
        rec["source_file_id"] = upload_id

    batch_size = settings.ingestion_batch_size
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        result = await idempotent_insert_records(db, batch)
        stats["records_inserted"] += result["inserted"]
        stats["records_updated"] = stats.get("records_updated", 0) + result["updated"]
        stats["records_unchanged"] = stats.get("records_unchanged", 0) + result["unchanged"]
        await db.commit()

    logger.info(
        "Standalone CDA ingestion: %d entries, %d inserted",
        stats["total_entries"],
        stats["records_inserted"],
    )
    return stats


async def _ingest_xdm(
    db: AsyncSession,
    user_id: UUID,
    patient_id: UUID,
    upload_id: UUID,
    xdm_dir: Path,
    metadata_path: Path,
) -> dict:
    """Ingest an IHE XDM package containing CDA XML documents."""
    stats: dict = {
        "total_entries": 0,
        "records_inserted": 0,
        "records_skipped": 0,
        "errors": [],
        "unstructured_files": [],
    }

    # Parse manifest
    manifest = parse_xdm_metadata(metadata_path)
    if not manifest:
        stats["errors"].append({"error": "Failed to parse METADATA.XML"})
        return stats

    # Backfill patient identifiers from the XDM manifest (HL7 PID-5/PID-7) so the
    # deterministic PHI scrubber can strip the patient's own name from CDA text.
    await _backfill_patient_by_id(
        db, patient_id,
        {"name": manifest.patient_name, "dob": manifest.patient_dob,
         "mrn": None, "gender": None},
    )

    # Filter to XML documents only
    xml_docs = [d for d in manifest.documents if d.mime_type == "text/xml"]
    skipped_docs = [d for d in manifest.documents if d.mime_type != "text/xml"]

    # Log skipped files
    for doc in skipped_docs:
        stats["errors"].append({
            "file": doc.uri,
            "reason": "structured_preferred",
            "message": "Skipped: CDA XML documents provide higher-fidelity structured data",
        })

    if not xml_docs:
        stats["errors"].append({"error": "No CDA XML documents found in manifest"})
        return stats

    # Parse each CDA document
    all_records: list[dict] = []
    for doc in xml_docs:
        doc_path = xdm_dir / doc.uri
        if not doc_path.exists():
            stats["errors"].append({"file": doc.uri, "error": "File not found"})
            continue

        try:
            records = parse_cda_document(doc_path, doc)
            stats["total_entries"] += len(records)
            all_records.extend(records)
        except Exception as e:
            stats["errors"].append({"file": doc.uri, "error": str(e)})

    if not all_records:
        stats["errors"].append({"error": "No records extracted from CDA documents"})
        return stats

    # Intra-upload cross-document dedup
    unique_records, dedup_stats = deduplicate_across_documents(all_records)
    stats["records_skipped"] += dedup_stats.duplicates_collapsed

    # Add user/patient/source_file IDs to each record
    for rec in unique_records:
        rec["user_id"] = user_id
        rec["patient_id"] = patient_id
        rec["source_file_id"] = upload_id

    # Bulk insert in batches
    batch_size = settings.ingestion_batch_size
    for i in range(0, len(unique_records), batch_size):
        batch = unique_records[i : i + batch_size]
        result = await idempotent_insert_records(db, batch)
        stats["records_inserted"] += result["inserted"]
        stats["records_updated"] = stats.get("records_updated", 0) + result["updated"]
        stats["records_unchanged"] = stats.get("records_unchanged", 0) + result["unchanged"]
        await db.commit()

    logger.info(
        "XDM ingestion: %d docs, %d total entries, %d unique, %d inserted",
        len(xml_docs),
        dedup_stats.total_parsed,
        dedup_stats.unique_records,
        stats["records_inserted"],
    )
    return stats


async def _ingest_zip(
    db: AsyncSession,
    user_id: UUID,
    patient_id: UUID,
    upload_id: UUID,
    zip_path: Path,
) -> dict:
    """Extract and ingest a ZIP file with mixed content support."""
    temp_dir = Path(settings.temp_extract_dir) / str(upload_id)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # SEC-DOS-02: bounded, member-by-member extraction (zip-bomb guard).
            _safe_extract_zip(zf, temp_dir)

        # Check for IHE XDM package first
        metadata_path = _find_xdm_metadata(temp_dir)
        if metadata_path:
            logger.info("Detected IHE XDM package: %s", metadata_path)
            xdm_dir = metadata_path.parent
            return await _ingest_xdm(db, user_id, patient_id, upload_id, xdm_dir, metadata_path)

        # Collect all files, excluding schema dirs and readme
        all_files = list(temp_dir.rglob("*"))

        tsv_files = []
        json_files = []
        unstructured_files = []

        for f in all_files:
            if not f.is_file():
                continue
            # Skip schema directories and readme files
            parts_lower = [p.lower() for p in f.parts]
            if any("schema" in p for p in parts_lower):
                continue
            if f.stem.lower() == "readme":
                continue

            suffix = f.suffix.lower()
            if suffix == ".tsv":
                tsv_files.append(f)
            elif suffix == ".json":
                json_files.append(f)
            elif suffix in (".pdf", ".rtf", ".tif", ".tiff"):
                unstructured_files.append(f)

        stats = {
            "total_entries": 0,
            "records_inserted": 0,
            "records_skipped": 0,
            "errors": [],
            "unstructured_files": [],
        }

        # Process structured content
        if tsv_files:
            tsv_dir = tsv_files[0].parent
            epic_stats = await _ingest_epic_dir(db, user_id, patient_id, upload_id, tsv_dir)
            stats["total_entries"] += epic_stats.get("total_files", 0)
            stats["records_inserted"] += epic_stats.get("records_inserted", 0)
            stats["records_skipped"] += epic_stats.get("records_skipped", 0)
            stats["errors"].extend(epic_stats.get("errors", []))

        if json_files:
            for jf in json_files:
                try:
                    result = await _ingest_fhir(db, user_id, patient_id, upload_id, jf)
                    stats["total_entries"] += result.get("total_entries", 0)
                    stats["records_inserted"] += result.get("records_inserted", 0)
                    stats["records_skipped"] += result.get("records_skipped", 0)
                    stats["errors"].extend(result.get("errors", []))
                except Exception as e:
                    stats["errors"].append({"file": jf.name, "error": str(e)})

        # Queue unstructured files for extraction
        if unstructured_files:
            for uf in unstructured_files:
                try:
                    # Copy to upload dir with UUID filename
                    dest_name = f"{uuid4()}{uf.suffix}"
                    dest_path = Path(settings.upload_dir) / dest_name
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(uf, dest_path)

                    # Determine mime type
                    suffix = uf.suffix.lower()
                    mime_map = {
                        ".pdf": "application/pdf",
                        ".rtf": "application/rtf",
                        ".tif": "image/tiff",
                        ".tiff": "image/tiff",
                    }

                    unstr_upload = UploadedFile(
                        id=uuid4(),
                        user_id=user_id,
                        filename=uf.name,
                        mime_type=mime_map.get(suffix, "application/octet-stream"),
                        file_size_bytes=uf.stat().st_size,
                        file_hash=compute_file_hash(uf),
                        storage_path=str(dest_path),
                        ingestion_status="pending_extraction",
                        file_category="unstructured",
                    )
                    db.add(unstr_upload)
                    stats["unstructured_files"].append({
                        "upload_id": str(unstr_upload.id),
                        "filename": uf.name,
                        "status": "pending_extraction",
                    })
                except Exception as e:
                    stats["errors"].append({"file": uf.name, "error": str(e)})

            await db.commit()

        if not tsv_files and not json_files and not unstructured_files:
            raise ValueError("ZIP contains no processable files")

        return stats
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
