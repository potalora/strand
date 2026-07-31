"""Column mapping specifications for all 14 Epic EHI table mappers.

Each TableSpec defines:
- tsv_columns: mapping of TSV column name → FHIR resource path
- gate_columns: columns that cause row skip when empty
- resource_type: expected FHIR resourceType
- record_type: expected health_records.record_type value
- example_row: a sample TSV row dict for unit testing
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnMapping:
    """Maps a single TSV column to its expected FHIR location."""

    tsv_column: str
    fhir_path: str  # dot-separated path within FHIR resource
    transform: str = "direct"  # direct, date, status_map, numeric, conditional
    description: str = ""


@dataclass
class TableSpec:
    """Full specification for an Epic EHI table mapper."""

    table_name: str
    mapper_class: str
    resource_type: str
    record_type: str
    gate_columns: list[str]
    columns: list[ColumnMapping]
    gate_groups: list[list[str]] | None = None
    example_row: dict[str, str] = field(default_factory=dict)

    def has_required_data(self, row: dict[str, str]) -> bool:
        """Return whether every required group has a populated alternative."""

        groups = self.gate_groups or [self.gate_columns]
        return all(
            any((row.get(column) or "").strip() for column in group) for group in groups
        )


# ---------------------------------------------------------------------------
# Table specifications for all 14 mappers
# ---------------------------------------------------------------------------

PROBLEM_LIST_SPEC = TableSpec(
    table_name="PROBLEM_LIST",
    mapper_class="ProblemListMapper",
    resource_type="Condition",
    record_type="condition",
    gate_columns=["DX_ID_DX_NAME", "DESCRIPTION"],  # both must be empty to skip
    columns=[
        ColumnMapping("DX_ID_DX_NAME", "code.text"),
        ColumnMapping(
            "DESCRIPTION", "code.text", description="Fallback for DX_ID_DX_NAME"
        ),
        ColumnMapping("NOTED_DATE", "onsetDateTime", transform="date"),
        ColumnMapping("RESOLVED_DATE", "abatementDateTime", transform="date"),
        ColumnMapping(
            "PROBLEM_STATUS_C_NAME",
            "clinicalStatus.coding.0.code",
            transform="status_map",
        ),
        ColumnMapping(
            "CHRONIC_YN",
            "category.1.text",
            transform="conditional",
            description="Only added when Y",
        ),
        ColumnMapping("PROBLEM_CMT", "note.0.text"),
    ],
    example_row={
        "DX_ID_DX_NAME": "Type 2 Diabetes",
        "DESCRIPTION": "Type 2 Diabetes Mellitus",
        "NOTED_DATE": "3/15/2020 12:00:00 AM",
        "RESOLVED_DATE": "",
        "PROBLEM_STATUS_C_NAME": "Active",
        "CHRONIC_YN": "Y",
        "PROBLEM_CMT": "Monitor A1c quarterly",
    },
)

MEDICAL_HX_SPEC = TableSpec(
    table_name="MEDICAL_HX",
    mapper_class="MedicalHxMapper",
    resource_type="Condition",
    record_type="condition",
    gate_columns=["DX_ID_DX_NAME"],
    columns=[
        ColumnMapping("DX_ID_DX_NAME", "code.text"),
        ColumnMapping("MEDICAL_HX_DATE", "onsetDateTime", transform="date"),
        ColumnMapping("MED_HX_ANNOTATION", "note.0.text"),
    ],
    example_row={
        "DX_ID_DX_NAME": "Appendectomy",
        "MEDICAL_HX_DATE": "5/10/2015 12:00:00 AM",
        "MED_HX_ANNOTATION": "Laparoscopic, no complications",
    },
)

ORDER_MED_SPEC = TableSpec(
    table_name="ORDER_MED",
    mapper_class="OrderMedMapper",
    resource_type="MedicationRequest",
    record_type="medication",
    gate_columns=[
        "DISPLAY_NAME",
        "MEDICATION_ID_MEDICATION_NAME",
    ],  # both must be empty to skip
    columns=[
        ColumnMapping("DISPLAY_NAME", "medicationCodeableConcept.text"),
        ColumnMapping(
            "MEDICATION_ID_MEDICATION_NAME",
            "medicationCodeableConcept.text",
            description="Fallback for DISPLAY_NAME",
        ),
        ColumnMapping("ORDERING_DATE", "authoredOn", transform="date"),
        ColumnMapping("START_DATE", "effectivePeriod.start", transform="date"),
        ColumnMapping("END_DATE", "effectivePeriod.end", transform="date"),
        ColumnMapping("ORDER_STATUS_C_NAME", "status", transform="status_map"),
        ColumnMapping("DOSAGE", "dosageInstruction.0.text"),
        ColumnMapping(
            "DESCRIPTION", "dosageInstruction.0.text", description="Fallback for DOSAGE"
        ),
        ColumnMapping("QUANTITY", "dispenseRequest.quantity.value"),
        ColumnMapping("REFILLS", "dispenseRequest.numberOfRepeatsAllowed"),
        ColumnMapping("MED_PRESC_PROV_ID_PROV_NAME", "requester.display"),
        ColumnMapping("MED_ROUTE_C_NAME", "dosageInstruction.0.route.text"),
    ],
    example_row={
        "DISPLAY_NAME": "Metformin 500mg",
        "MEDICATION_ID_MEDICATION_NAME": "Metformin",
        "ORDERING_DATE": "3/18/2020 12:00:00 AM",
        "START_DATE": "3/20/2020 12:00:00 AM",
        "END_DATE": "",
        "ORDER_STATUS_C_NAME": "Active",
        "DOSAGE": "500mg twice daily",
        "DESCRIPTION": "Metformin 500mg oral tablet",
        "QUANTITY": "60",
        "REFILLS": "3",
        "MED_PRESC_PROV_ID_PROV_NAME": "Dr. Smith",
        "MED_ROUTE_C_NAME": "Oral",
    },
)

ORDER_RESULTS_SPEC = TableSpec(
    table_name="ORDER_RESULTS",
    mapper_class="OrderResultsMapper",
    resource_type="Observation",
    record_type="observation",
    gate_columns=["COMPONENT_ID_NAME"],
    columns=[
        ColumnMapping("COMPONENT_ID_NAME", "code.text"),
        ColumnMapping("ORD_NUM_VALUE", "valueQuantity.value", transform="numeric"),
        ColumnMapping(
            "ORD_VALUE", "valueString", description="Fallback when non-numeric"
        ),
        ColumnMapping("REFERENCE_UNIT", "valueQuantity.unit"),
        ColumnMapping(
            "REFERENCE_LOW", "referenceRange.0.low.value", transform="numeric"
        ),
        ColumnMapping(
            "REFERENCE_HIGH", "referenceRange.0.high.value", transform="numeric"
        ),
        ColumnMapping("RESULT_DATE", "effectiveDateTime", transform="date"),
        ColumnMapping("RESULT_STATUS_C_NAME", "status", transform="status_map"),
        ColumnMapping(
            "RESULT_FLAG_C_NAME",
            "interpretation.0.coding.0.code",
            transform="status_map",
        ),
        ColumnMapping("COMPON_LNC_ID_LNC_LONG_NAME", "code.coding.0.display"),
    ],
    example_row={
        "COMPONENT_ID_NAME": "Hemoglobin A1c",
        "ORD_VALUE": "6.8",
        "ORD_NUM_VALUE": "6.8",
        "REFERENCE_UNIT": "%",
        "REFERENCE_LOW": "4.0",
        "REFERENCE_HIGH": "5.6",
        "RESULT_DATE": "1/10/2024 12:00:00 AM",
        "RESULT_STATUS_C_NAME": "Final",
        "RESULT_FLAG_C_NAME": "High",
        "COMPON_LNC_ID_LNC_LONG_NAME": "Hemoglobin A1c/Hemoglobin.total in Blood",
    },
)

PAT_ENC_SPEC = TableSpec(
    table_name="PAT_ENC",
    mapper_class="PatEncMapper",
    resource_type="Encounter",
    record_type="encounter",
    gate_columns=["CONTACT_DATE"],
    columns=[
        ColumnMapping("CONTACT_DATE", "period.start", transform="date"),
        ColumnMapping("APPT_STATUS_C_NAME", "status", transform="status_map"),
        ColumnMapping("FIN_CLASS_C_NAME", "class.code", transform="status_map"),
        ColumnMapping("DEPARTMENT_ID_EXTERNAL_NAME", "location.0.location.display"),
        ColumnMapping("VISIT_PROV_ID_PROV_NAME", "participant.0.individual.display"),
        ColumnMapping(
            "VISIT_PROV_TITLE_NAME",
            "participant.0.individual.display",
            description="Appended to provider name",
        ),
        ColumnMapping("HOSP_DISCHRG_TIME", "period.end", transform="date"),
        ColumnMapping("CONTACT_COMMENT", "reasonCode.0.text"),
    ],
    example_row={
        "CONTACT_DATE": "1/10/2024 12:00:00 AM",
        "APPT_STATUS_C_NAME": "Completed",
        "FIN_CLASS_C_NAME": "Outpatient",
        "DEPARTMENT_ID_EXTERNAL_NAME": "Internal Medicine",
        "VISIT_PROV_ID_PROV_NAME": "Dr. Smith",
        "VISIT_PROV_TITLE_NAME": "MD",
        "HOSP_DISCHRG_TIME": "",
        "CONTACT_COMMENT": "Annual checkup",
    },
)

DOC_INFORMATION_SPEC = TableSpec(
    table_name="DOC_INFORMATION",
    mapper_class="DocInformationMapper",
    resource_type="DocumentReference",
    record_type="document",
    gate_columns=["DOC_INFO_TYPE_C_NAME"],
    columns=[
        ColumnMapping("DOC_INFO_TYPE_C_NAME", "type.text"),
        ColumnMapping("DOC_RECV_TIME", "date", transform="date"),
        ColumnMapping("DOC_STAT_C_NAME", "status", transform="status_map"),
        ColumnMapping("DOC_DESCR", "description"),
        ColumnMapping("RECV_BY_USER_ID_NAME", "author.0.display"),
        ColumnMapping(
            "IS_SCANNED_YN",
            "category.0.text",
            transform="conditional",
            description="Only added when Y",
        ),
    ],
    example_row={
        "DOC_INFO_TYPE_C_NAME": "Progress Note",
        "DOC_RECV_TIME": "1/10/2024 12:00:00 AM",
        "DOC_STAT_C_NAME": "Active",
        "DOC_DESCR": "Office Visit Progress Note",
        "RECV_BY_USER_ID_NAME": "Dr. Smith",
        "IS_SCANNED_YN": "N",
    },
)

ALLERGY_SPEC = TableSpec(
    table_name="ALLERGY",
    mapper_class="AllergyMapper",
    resource_type="AllergyIntolerance",
    record_type="allergy",
    gate_columns=["ALLERGEN_ID_ALLERGEN_NAME"],
    columns=[
        ColumnMapping("ALLERGEN_ID_ALLERGEN_NAME", "code.text"),
        ColumnMapping("DATE_NOTED", "recordedDate", transform="date"),
        ColumnMapping("SEVERITY_C_NAME", "reaction.0.severity", transform="status_map"),
        ColumnMapping(
            "ALRGY_STATUS_C_NAME",
            "clinicalStatus.coding.0.code",
            transform="status_map",
        ),
        ColumnMapping("REACTION", "reaction.0.manifestation.0.text"),
    ],
    example_row={
        "ALLERGEN_ID_ALLERGEN_NAME": "Penicillin",
        "DATE_NOTED": "3/15/2021 12:00:00 AM",
        "SEVERITY_C_NAME": "Severe",
        "ALRGY_STATUS_C_NAME": "Active",
        "REACTION": "Hives",
    },
)

IMMUNE_SPEC = TableSpec(
    table_name="IMMUNE",
    mapper_class="ImmuneMapper",
    resource_type="Immunization",
    record_type="immunization",
    gate_columns=["IMMUNZATN_ID_NAME"],
    columns=[
        ColumnMapping("IMMUNZATN_ID_NAME", "vaccineCode.text"),
        ColumnMapping("IMMUNE_DATE", "occurrenceDateTime", transform="date"),
        ColumnMapping("DOSE", "doseQuantity.value"),
        ColumnMapping("ROUTE_C_NAME", "route.text"),
        ColumnMapping("SITE_C_NAME", "site.text"),
        ColumnMapping("MFG_C_NAME", "manufacturer.display"),
        ColumnMapping("LOT", "lotNumber"),
        ColumnMapping("IMMNZTN_STATUS_C_NAME", "status", transform="status_map"),
    ],
    example_row={
        "IMMUNZATN_ID_NAME": "COVID-19 Vaccine",
        "IMMUNE_DATE": "1/15/2021 12:00:00 AM",
        "DOSE": "0.3 mL",
        "ROUTE_C_NAME": "Intramuscular",
        "SITE_C_NAME": "Left Deltoid",
        "MFG_C_NAME": "Pfizer",
        "LOT": "EL9261",
        "IMMNZTN_STATUS_C_NAME": "Administered",
    },
)

ORDER_PROC_SPEC = TableSpec(
    table_name="ORDER_PROC",
    mapper_class="OrderProcMapper",
    resource_type="Procedure",
    record_type="procedure",
    gate_columns=[
        "DESCRIPTION",
        "PROC_NAME",
        "ORDER_TYPE_C_NAME",
        "DISPLAY_NAME",
    ],  # all must be empty to skip
    columns=[
        ColumnMapping("DESCRIPTION", "code.text"),
        ColumnMapping("ORDER_INST", "performedDateTime", transform="date"),
        ColumnMapping("ORDER_STATUS_C_NAME", "status", transform="status_map"),
        ColumnMapping("AUTHRZING_PROV_ID_PROV_NAME", "performer.0.actor.display"),
    ],
    example_row={
        "DESCRIPTION": "CT Abdomen with Contrast",
        "ORDER_INST": "6/10/2023 12:00:00 AM",
        "ORDER_STATUS_C_NAME": "Completed",
        "AUTHRZING_PROV_ID_PROV_NAME": "Dr. Smith",
    },
)

VITALS_SPEC = TableSpec(
    table_name="IP_FLWSHT_MEAS",
    mapper_class="VitalsMapper",
    resource_type="Observation",
    record_type="observation",
    gate_columns=[
        "FLO_MEAS_NAME",
        "DISP_NAME",
        "FLO_MEAS_ID_FLO_MEAS_NAME",
        "MEAS_VALUE",
    ],
    columns=[
        ColumnMapping("FLO_MEAS_NAME", "code.text"),
        ColumnMapping(
            "MEAS_VALUE",
            "valueQuantity.value",
            transform="numeric",
            description="Numeric values; falls back to valueString",
        ),
        ColumnMapping("UNITS", "valueQuantity.unit"),
        ColumnMapping("RECORDED_TIME", "effectiveDateTime", transform="date"),
    ],
    gate_groups=[
        ["FLO_MEAS_NAME", "DISP_NAME", "FLO_MEAS_ID_FLO_MEAS_NAME"],
        ["MEAS_VALUE"],
    ],
    example_row={
        "FLO_MEAS_NAME": "Blood Pressure Systolic",
        "MEAS_VALUE": "120",
        "UNITS": "mmHg",
        "RECORDED_TIME": "2/15/2024 10:30:00 AM",
    },
)

REFERRAL_SPEC = TableSpec(
    table_name="REFERRAL",
    mapper_class="ReferralMapper",
    resource_type="ServiceRequest",
    record_type="service_request",
    gate_columns=[
        "RSN_FOR_RFL_C_NAME",
        "REFERRAL_PROV_ID_PROV_NAME",
    ],  # both must be empty to skip
    columns=[
        ColumnMapping("RSN_FOR_RFL_C_NAME", "code.text"),
        ColumnMapping("REFERRING_PROV_ID_REFERRING_PROV_NAM", "requester.display"),
        ColumnMapping("REFERRAL_PROV_ID_PROV_NAME", "performer.0.display"),
        ColumnMapping("RFL_STATUS_C_NAME", "status", transform="status_map"),
        ColumnMapping("START_DATE", "authoredOn", transform="date"),
        ColumnMapping("EXP_DATE", "occurrencePeriod.end", transform="date"),
    ],
    example_row={
        "RSN_FOR_RFL_C_NAME": "Cardiology Consultation",
        "REFERRING_PROV_ID_REFERRING_PROV_NAM": "Dr. Jones",
        "REFERRAL_PROV_ID_PROV_NAME": "Dr. Specialist",
        "RFL_STATUS_C_NAME": "Completed",
        "START_DATE": "3/1/2024 12:00:00 AM",
        "EXP_DATE": "6/1/2024 12:00:00 AM",
    },
)

ENCOUNTER_DX_SPEC = TableSpec(
    table_name="PAT_ENC_DX",
    mapper_class="EncounterDxMapper",
    resource_type="Condition",
    record_type="condition",
    gate_columns=["DX_ID_DX_NAME"],
    columns=[
        ColumnMapping("DX_ID_DX_NAME", "code.text"),
        ColumnMapping("CONTACT_DATE", "recordedDate", transform="date"),
        ColumnMapping("PRIMARY_DX_YN", "_primaryDiagnosis", transform="conditional"),
        ColumnMapping("ANNOTATION", "note.0.text"),
    ],
    example_row={
        "DX_ID_DX_NAME": "Acute Bronchitis",
        "CONTACT_DATE": "2/10/2024 12:00:00 AM",
        "PRIMARY_DX_YN": "Y",
        "ANNOTATION": "Follow-up in 2 weeks",
    },
)

SOCIAL_HX_SPEC = TableSpec(
    table_name="SOCIAL_HX",
    mapper_class="SocialHxMapper",
    resource_type="Observation",
    record_type="observation",
    gate_columns=[
        "SOCIAL_HX_TYPE_C_NAME",
        "HX_TYPE",
        "TOBACCO_USER_C_NAME",
        "SOCIAL_HX_COMMENT",
        "COMMENT",
        "SMOKING_TOBA_USE_C_NAME",
    ],  # all must be empty
    columns=[
        ColumnMapping("SOCIAL_HX_TYPE_C_NAME", "code.text"),
        ColumnMapping("SOCIAL_HX_COMMENT", "valueString"),
        ColumnMapping("CONTACT_DATE", "effectiveDateTime", transform="date"),
    ],
    example_row={
        "SOCIAL_HX_TYPE_C_NAME": "Tobacco Use",
        "SOCIAL_HX_COMMENT": "Former smoker, quit 2015",
        "CONTACT_DATE": "1/5/2024 12:00:00 AM",
    },
)

FAMILY_HX_SPEC = TableSpec(
    table_name="FAMILY_HX",
    mapper_class="FamilyHxMapper",
    resource_type="FamilyMemberHistory",
    record_type="condition",
    gate_columns=["FAM_MEDICAL_DX_ID_DX_NAME"],
    columns=[
        ColumnMapping("FAM_MEDICAL_DX_ID_DX_NAME", "condition.0.code.text"),
        ColumnMapping("RELATION_C_NAME", "relationship.text"),
        ColumnMapping(
            "AGE_OF_ONSET",
            "condition.0.onsetAge.value",
            transform="numeric",
            description="Numeric → onsetAge, non-numeric → onsetString",
        ),
    ],
    example_row={
        "FAM_MEDICAL_DX_ID_DX_NAME": "Type 2 Diabetes",
        "RELATION_C_NAME": "Mother",
        "AGE_OF_ONSET": "55",
    },
)


# All specs indexed by table name
ALL_TABLE_SPECS: dict[str, TableSpec] = {
    spec.table_name: spec
    for spec in [
        PROBLEM_LIST_SPEC,
        MEDICAL_HX_SPEC,
        ORDER_MED_SPEC,
        ORDER_RESULTS_SPEC,
        PAT_ENC_SPEC,
        DOC_INFORMATION_SPEC,
        ALLERGY_SPEC,
        IMMUNE_SPEC,
        ORDER_PROC_SPEC,
        VITALS_SPEC,
        REFERRAL_SPEC,
        ENCOUNTER_DX_SPEC,
        SOCIAL_HX_SPEC,
        FAMILY_HX_SPEC,
    ]
}
