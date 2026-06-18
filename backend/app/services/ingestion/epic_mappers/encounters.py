from __future__ import annotations

from app.services.ingestion.epic_mappers.base import EpicMapper


class PatEncMapper(EpicMapper):
    """Map PAT_ENC rows to FHIR Encounter resources."""

    source_table = "PAT_ENC"
    primary_key_columns = ["PAT_ENC_CSN_ID"]

    def to_fhir(self, row: dict[str, str]) -> dict | None:
        contact_date = self.parse_epic_date(self.safe_get(row, "CONTACT_DATE"))
        if not contact_date:
            return None

        status_raw = self.safe_get(row, "APPT_STATUS_C_NAME").lower()
        status = "finished"
        if "completed" in status_raw or "complete" in status_raw:
            status = "finished"
        elif "cancelled" in status_raw or "canceled" in status_raw:
            status = "cancelled"
        elif "no show" in status_raw:
            status = "cancelled"
        elif "scheduled" in status_raw:
            status = "planned"

        enc_class = "AMB"
        fin_class = self.safe_get(row, "FIN_CLASS_C_NAME").lower()
        if "inpatient" in fin_class:
            enc_class = "IMP"
        elif "emergency" in fin_class:
            enc_class = "EMER"

        resource = {
            "resourceType": "Encounter",
            "status": status,
            "class": {"code": enc_class},
            "period": {"start": contact_date.isoformat()},
        }

        # Visit type / title — gives the renderer a meaningful header beyond the
        # bare class code (e.g. "Office Visit", "Telehealth", "Hospital Encounter").
        enc_type = (
            self.safe_get(row, "ENC_TYPE_C_NAME")
            or self.safe_get(row, "APPT_PRC_ID_PRC_NAME")
        )
        if enc_type:
            resource["type"] = [{"text": enc_type}]

        dept = self.safe_get(row, "DEPARTMENT_ID_EXTERNAL_NAME")
        if dept:
            resource["location"] = [{"location": {"display": dept}}]

        # B6 — facility / managing organization (serviceProvider was 0%).
        # Prefer an explicit facility/location-name column; fall back to the
        # department name so the field is populated whenever any is available.
        facility = (
            self.safe_get(row, "LOC_ID_LOC_NAME")
            or self.safe_get(row, "DEPARTMENT_ID_DEPARTMENT_NAME")
            or dept
        )
        if facility:
            resource["serviceProvider"] = {"display": facility}

        # Provider — the visit provider, falling back to the PCP so an encounter
        # without a named visit provider still surfaces a clinician.
        provider = (
            self.safe_get(row, "VISIT_PROV_ID_PROV_NAME")
            or self.safe_get(row, "PCP_PROV_ID_PROV_NAME")
        )
        title = self.safe_get(row, "VISIT_PROV_TITLE_NAME")
        if provider:
            display = f"{provider}, {title}" if title else provider
            resource["participant"] = [{"individual": {"display": display}}]

        discharge_date = self.parse_epic_date(self.safe_get(row, "HOSP_DISCHRG_TIME"))
        if discharge_date:
            resource["period"]["end"] = discharge_date.isoformat()

        reason = self.safe_get(row, "CONTACT_COMMENT")
        if reason:
            resource["reasonCode"] = [{"text": reason}]

        return resource
