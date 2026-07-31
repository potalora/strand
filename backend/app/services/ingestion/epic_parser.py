from __future__ import annotations

import csv
import logging
import os
import sqlite3
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ingestion.epic_mappers.base import EpicMapper
from app.services.ingestion.epic_mappers.allergies import AllergyMapper
from app.services.ingestion.epic_mappers.documents import DocInformationMapper
from app.services.ingestion.epic_mappers.encounter_dx import EncounterDxMapper
from app.services.ingestion.epic_mappers.encounters import PatEncMapper
from app.services.ingestion.epic_mappers.family_hx import FamilyHxMapper
from app.services.ingestion.epic_mappers.immunizations import ImmuneMapper
from app.services.ingestion.epic_mappers.medications import OrderMedMapper
from app.services.ingestion.epic_mappers.problems import (
    MedicalHxMapper,
    ProblemListMapper,
)
from app.services.ingestion.epic_mappers.procedures import OrderProcMapper
from app.services.ingestion.epic_mappers.referrals import ReferralMapper
from app.services.ingestion.epic_mappers.results import OrderResultsMapper
from app.services.ingestion.epic_mappers.social_hx import SocialHxMapper
from app.services.ingestion.epic_mappers.vitals import VitalsMapper
from app.services.ingestion.fhir_parser import build_display_text
from app.services.ingestion.idempotent_inserter import idempotent_insert_records
from app.services.ingestion.identity import epic_identity

logger = logging.getLogger(__name__)

EPIC_TABLE_MAPPERS: dict[str, EpicMapper] = {
    "PROBLEM_LIST": ProblemListMapper(),
    "PROBLEM_LIST_ALL": ProblemListMapper(),
    "MEDICAL_HX": MedicalHxMapper(),
    "ORDER_MED": OrderMedMapper(),
    "ORDER_RESULTS": OrderResultsMapper(),
    "PAT_ENC": PatEncMapper(),
    "DOC_INFORMATION": DocInformationMapper(),
    "ALLERGY": AllergyMapper(),
    "IMMUNE": ImmuneMapper(),
    "ORDER_PROC": OrderProcMapper(),
    "IP_FLWSHT_MEAS": VitalsMapper(),
    "REFERRAL": ReferralMapper(),
    "PAT_ENC_DX": EncounterDxMapper(),
    "SOCIAL_HX": SocialHxMapper(),
    "FAMILY_HX": FamilyHxMapper(),
}

RECORD_TYPE_MAP = {
    "Condition": "condition",
    "MedicationRequest": "medication",
    "Observation": "observation",
    "Encounter": "encounter",
    "DocumentReference": "document",
    "Immunization": "immunization",
    "Procedure": "procedure",
    "AllergyIntolerance": "allergy",
    "ServiceRequest": "service_request",
    "FamilyMemberHistory": "condition",
    "CareTeam": "care_team",
    "ImmunizationRecommendation": "immunization",
    "QuestionnaireResponse": "questionnaire_response",
}


class _VitalsValueIndex:
    """Disk-backed lookup for Epic's normalized flowsheet value companion."""

    _COMPANION_NAME = "V_EHI_FLO_MEAS_VALUE.tsv"
    _INSERT_BATCH_SIZE = 100

    def __init__(self, export_dir: Path) -> None:
        self._companion_path = export_dir / self._COMPANION_NAME
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> _VitalsValueIndex:
        if not self._companion_path.is_file():
            return self

        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="medtimeline-epic-vitals-"
        )
        database_path = Path(self._temporary_directory.name) / "values.sqlite3"
        try:
            connection = sqlite3.connect(database_path)
            self._connection = connection
            os.chmod(database_path, 0o600)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute(
                """
                CREATE TABLE values_by_key (
                    fsd_id TEXT NOT NULL,
                    line TEXT NOT NULL,
                    measurement TEXT NOT NULL,
                    units TEXT NOT NULL,
                    PRIMARY KEY (fsd_id, line)
                ) WITHOUT ROWID
                """
            )
            self._populate(connection)
        except Exception:
            self.close()
            raise
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        self.close()

    def _populate(self, connection: sqlite3.Connection) -> None:
        required = {"FSD_ID", "LINE", "MEAS_VALUE_EXTERNAL"}
        batch: list[tuple[str, str, str, str]] = []
        with self._companion_path.open("r", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Epic vitals companion table is invalid.")
            for row in reader:
                fsd_id = (row.get("FSD_ID") or "").strip()
                line = (row.get("LINE") or "").strip()
                measurement = (row.get("MEAS_VALUE_EXTERNAL") or "").strip()
                if not fsd_id or not line or not measurement:
                    continue
                batch.append(
                    (
                        fsd_id,
                        line,
                        measurement,
                        (row.get("UNITS") or "").strip(),
                    )
                )
                if len(batch) >= self._INSERT_BATCH_SIZE:
                    self._insert_batch(connection, batch)
                    batch.clear()
            if batch:
                self._insert_batch(connection, batch)
        connection.commit()

    @staticmethod
    def _insert_batch(
        connection: sqlite3.Connection,
        batch: list[tuple[str, str, str, str]],
    ) -> None:
        connection.executemany(
            """
            INSERT OR REPLACE INTO values_by_key
                (fsd_id, line, measurement, units)
            VALUES (?, ?, ?, ?)
            """,
            batch,
        )

    def get(self, row: dict[str, str]) -> dict[str, str]:
        if self._connection is None:
            return {}
        fsd_id = (row.get("FSD_ID") or "").strip()
        line = (row.get("LINE") or "").strip()
        if not fsd_id or not line:
            return {}
        result = self._connection.execute(
            """
            SELECT measurement, units
            FROM values_by_key
            WHERE fsd_id = ? AND line = ?
            """,
            (fsd_id, line),
        ).fetchone()
        if result is None:
            return {}
        return {"MEAS_VALUE": result[0], "UNITS": result[1]}

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None


def _enrich_epic_row(
    table_name: str,
    row: dict[str, str],
    vitals_values: _VitalsValueIndex | None,
) -> dict[str, str]:
    if (
        table_name != "IP_FLWSHT_MEAS"
        or vitals_values is None
        or (row.get("MEAS_VALUE") or "").strip()
    ):
        return row
    companion = vitals_values.get(row)
    return {**row, **companion} if companion else row


async def parse_epic_export(
    export_dir: Path,
    user_id: UUID,
    patient_id: UUID,
    source_file_id: UUID | None,
    db: AsyncSession,
    batch_size: int = 100,
    progress_callback: Any = None,
) -> dict:
    """Process an Epic EHI Tables export directory.

    Files are processed one at a time, rows streamed row-by-row.
    Returns detailed stats including per-file breakdown.
    """
    tsv_files = sorted(export_dir.glob("*.tsv"))
    total_files = len(tsv_files)
    stats: dict[str, Any] = {
        "total_files": total_files,
        "files_processed": 0,
        "records_inserted": 0,
        "records_skipped": 0,
        "errors": [],
        "files_detail": [],
        "files_skipped": [],
    }

    for file_idx, tsv_path in enumerate(tsv_files):
        table_name = tsv_path.stem.upper()
        mapper = EPIC_TABLE_MAPPERS.get(table_name)
        if not mapper:
            stats["files_skipped"].append(table_name)
            stats["records_skipped"] += 1
            continue

        logger.info(
            "Processing Epic table: %s (%d/%d)", table_name, file_idx + 1, total_files
        )
        batch = []
        row_count = 0
        rows_inserted = 0
        rows_skipped = 0

        values_context = (
            _VitalsValueIndex(export_dir)
            if table_name == "IP_FLWSHT_MEAS"
            else nullcontext(None)
        )
        try:
            with values_context as vitals_values:
                with open(tsv_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row_idx, row in enumerate(reader):
                        row_count += 1
                        try:
                            enriched_row = _enrich_epic_row(
                                table_name,
                                row,
                                vitals_values,
                            )
                            fhir_resource = mapper.to_fhir(enriched_row)
                            if not fhir_resource:
                                rows_skipped += 1
                                continue

                            resource_type = fhir_resource.get("resourceType", "Unknown")
                            record_type = RECORD_TYPE_MAP.get(
                                resource_type,
                                resource_type.lower(),
                            )

                            from app.services.ingestion.fhir_parser import (
                                extract_categories,
                                extract_coding,
                                extract_effective_date,
                                extract_effective_date_end,
                                extract_status,
                            )

                            code_system, code_value, code_display = extract_coding(
                                fhir_resource
                            )

                            mapped = {
                                "user_id": user_id,
                                "patient_id": patient_id,
                                "source_file_id": source_file_id,
                                "record_type": record_type,
                                "fhir_resource_type": resource_type,
                                "fhir_resource": fhir_resource,
                                "source_format": "epic_ehi",
                                "effective_date": extract_effective_date(fhir_resource),
                                "effective_date_end": extract_effective_date_end(
                                    fhir_resource
                                ),
                                "status": extract_status(fhir_resource),
                                "category": extract_categories(fhir_resource),
                                "code_system": code_system,
                                "code_value": code_value,
                                "code_display": code_display,
                                "display_text": build_display_text(
                                    fhir_resource,
                                    resource_type,
                                ),
                            }

                            ident = epic_identity(
                                table_name,
                                mapper.primary_key_columns,
                                row,
                            )
                            if ident is not None:
                                mapped["external_id"] = ident.external_id
                                mapped["source_system"] = ident.source_system

                            batch.append(mapped)

                            if len(batch) >= batch_size:
                                result = await idempotent_insert_records(db, batch)
                                rows_inserted += result["inserted"]
                                stats["records_inserted"] += result["inserted"]
                                stats["records_updated"] = (
                                    stats.get("records_updated", 0) + result["updated"]
                                )
                                stats["records_unchanged"] = (
                                    stats.get("records_unchanged", 0)
                                    + result["unchanged"]
                                )
                                batch.clear()
                                await db.commit()

                        except Exception as e:
                            stats["errors"].append(
                                {"file": table_name, "row": row_idx, "error": str(e)}
                            )
                            continue

            if batch:
                result = await idempotent_insert_records(db, batch)
                rows_inserted += result["inserted"]
                stats["records_inserted"] += result["inserted"]
                stats["records_updated"] = (
                    stats.get("records_updated", 0) + result["updated"]
                )
                stats["records_unchanged"] = (
                    stats.get("records_unchanged", 0) + result["unchanged"]
                )
                batch.clear()
                await db.commit()

        except Exception as e:
            stats["errors"].append({"file": table_name, "error": str(e)})
            logger.error("Error processing %s: %s", table_name, e)

        stats["files_processed"] += 1
        stats["files_detail"].append(
            {
                "table_name": table_name,
                "rows_found": row_count,
                "rows_inserted": rows_inserted,
                "rows_skipped": rows_skipped,
            }
        )
        logger.info(
            "Processed %s: %d rows, %d inserted", table_name, row_count, rows_inserted
        )

        if progress_callback:
            await progress_callback(
                file_idx + 1, total_files, stats["records_inserted"]
            )

    logger.info(
        "Epic export processing complete: %d files, %d records, %d errors",
        stats["files_processed"],
        stats["records_inserted"],
        len(stats["errors"]),
    )
    return stats
