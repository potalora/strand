from __future__ import annotations

from app.services.extraction import terminology
from app.services.ingestion.epic_mappers.base import EpicMapper


class OrderMedMapper(EpicMapper):
    """Map ORDER_MED rows to FHIR MedicationRequest resources."""

    source_table = "ORDER_MED"
    primary_key_columns = ["ORDER_MED_ID"]

    def to_fhir(self, row: dict[str, str]) -> dict | None:
        med_name = self.safe_get(row, "DISPLAY_NAME") or self.safe_get(
            row, "MEDICATION_ID_MEDICATION_NAME"
        )
        if not med_name:
            return None

        start_date = self.parse_epic_date(self.safe_get(row, "START_DATE"))
        end_date = self.parse_epic_date(self.safe_get(row, "END_DATE"))
        authored = self.parse_epic_date(self.safe_get(row, "ORDERING_DATE"))
        status_raw = self.safe_get(row, "ORDER_STATUS_C_NAME").lower()

        status = "active"
        if "completed" in status_raw or "sent" in status_raw:
            status = "completed"
        elif "cancel" in status_raw or "discontinue" in status_raw:
            status = "cancelled"

        med_cc: dict = {"text": med_name}
        # B1 — RxNorm coding for structured meds (0% before this).
        coding = terminology.lookup_medication(med_name)
        if coding:
            med_cc["coding"] = [coding.as_coding()]

        resource = {
            "resourceType": "MedicationRequest",
            "status": status,
            "intent": "order",
            "medicationCodeableConcept": med_cc,
            "category": [{"text": "community"}],
        }

        if authored:
            resource["authoredOn"] = authored.isoformat()

        dosage = self.safe_get(row, "DOSAGE")
        description = self.safe_get(row, "DESCRIPTION")
        if dosage or description:
            sig = dosage or description
            instruction: dict = {"text": sig}
            # B4 — parse the sig into structured dose + timing.
            parsed = terminology.parse_dosage(sig)
            if parsed["dose_value"] is not None:
                dose_quantity: dict = {"value": parsed["dose_value"]}
                if parsed["dose_unit"]:
                    dose_quantity["unit"] = parsed["dose_unit"]
                instruction["doseAndRate"] = [{"doseQuantity": dose_quantity}]
            if parsed["frequency"]:
                instruction["timing"] = {
                    "repeat": {
                        "frequency": parsed["frequency"],
                        "period": parsed["period"],
                        "periodUnit": parsed["period_unit"],
                    }
                }
            elif parsed["as_needed"]:
                instruction["asNeeded"] = True
            resource["dosageInstruction"] = [instruction]

        quantity = self.safe_get(row, "QUANTITY")
        refills = self.safe_get(row, "REFILLS")
        if quantity or refills:
            disp = {}
            if quantity:
                disp["quantity"] = {"value": quantity}
            if refills:
                disp["numberOfRepeatsAllowed"] = refills
            resource["dispenseRequest"] = disp

        if start_date or end_date:
            period = {}
            if start_date:
                period["start"] = start_date.isoformat()
            if end_date:
                period["end"] = end_date.isoformat()
            resource["effectivePeriod"] = period

        prescriber = self.safe_get(row, "MED_PRESC_PROV_ID_PROV_NAME")
        if prescriber:
            resource["requester"] = {"display": prescriber}

        route = self.safe_get(row, "MED_ROUTE_C_NAME")
        if route and resource.get("dosageInstruction"):
            resource["dosageInstruction"][0]["route"] = {"text": route}

        return resource
