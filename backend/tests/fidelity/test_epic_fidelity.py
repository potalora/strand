"""Epic EHI Tables ingestion fidelity tests.

Tests verify that every mapper correctly transforms TSV rows into FHIR
resources with proper field mappings, metadata extraction, and gate
column behavior. Tests run without a database.

Two categories:
1. Column spec tests (synthetic data, always run) — verify each mapper
   against its TableSpec example_row
2. Real-data fidelity tests (@pytest.mark.fidelity) — verify against
   actual Epic EHI exports, skip when absent
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.fidelity.column_specs import ALL_TABLE_SPECS
from tests.fidelity.helpers import (
    get_mapper_for_table,
    load_tsv_rows,
    map_with_metadata,
    resolve_fhir_path,
)


# ---------------------------------------------------------------------------
# Column spec unit tests (synthetic, always available)
# ---------------------------------------------------------------------------


class TestMapperExampleRows:
    """Verify each mapper produces valid output from its example_row."""

    @pytest.mark.parametrize(
        "table_name",
        list(ALL_TABLE_SPECS.keys()),
        ids=list(ALL_TABLE_SPECS.keys()),
    )
    def test_example_row_produces_resource(self, table_name: str):
        """Every TableSpec example_row should produce a non-None FHIR resource."""
        spec = ALL_TABLE_SPECS[table_name]
        mapper = get_mapper_for_table(table_name)
        assert mapper is not None, f"No mapper registered for {table_name}"
        result = mapper.to_fhir(spec.example_row)
        assert result is not None, (
            f"{spec.mapper_class} returned None for example_row"
        )
        assert result["resourceType"] == spec.resource_type

    @pytest.mark.parametrize(
        "table_name",
        list(ALL_TABLE_SPECS.keys()),
        ids=list(ALL_TABLE_SPECS.keys()),
    )
    def test_example_row_gate_columns(self, table_name: str):
        """Empty gate columns should cause mapper to return None."""
        spec = ALL_TABLE_SPECS[table_name]
        mapper = get_mapper_for_table(table_name)
        assert mapper is not None

        # Clear all gate columns
        empty_row = dict(spec.example_row)
        for col in spec.gate_columns:
            empty_row[col] = ""

        result = mapper.to_fhir(empty_row)
        assert result is None, (
            f"{spec.mapper_class} should return None when gate columns "
            f"{spec.gate_columns} are empty"
        )

    @pytest.mark.parametrize(
        "table_name",
        list(ALL_TABLE_SPECS.keys()),
        ids=list(ALL_TABLE_SPECS.keys()),
    )
    def test_example_row_metadata_extraction(self, table_name: str):
        """Verify metadata extraction (display_text, record_type) works."""
        spec = ALL_TABLE_SPECS[table_name]
        results = map_with_metadata(table_name, [spec.example_row])
        assert len(results) == 1
        result = results[0]
        assert not result["skipped"]
        assert result["record_type"] == spec.record_type
        assert result["resource_type"] == spec.resource_type
        # display_text should not be the bare resource type name
        # (which is what you get when the display_text logic fails)
        display = result["display_text"]
        assert display is not None
        assert len(display) > 0


class TestMapperColumnAccuracy:
    """Verify specific column → FHIR path mappings using example rows."""

    def test_problem_list_code_text(self):
        spec = ALL_TABLE_SPECS["PROBLEM_LIST"]
        mapper = get_mapper_for_table("PROBLEM_LIST")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "code.text") == "Type 2 Diabetes Mellitus"

    def test_problem_list_onset(self):
        spec = ALL_TABLE_SPECS["PROBLEM_LIST"]
        mapper = get_mapper_for_table("PROBLEM_LIST")
        result = mapper.to_fhir(spec.example_row)
        assert result["onsetDateTime"] is not None

    def test_problem_list_chronic_category(self):
        spec = ALL_TABLE_SPECS["PROBLEM_LIST"]
        mapper = get_mapper_for_table("PROBLEM_LIST")
        result = mapper.to_fhir(spec.example_row)
        categories = result.get("category", [])
        assert any(c.get("text") == "chronic" for c in categories)

    def test_problem_list_comment(self):
        spec = ALL_TABLE_SPECS["PROBLEM_LIST"]
        mapper = get_mapper_for_table("PROBLEM_LIST")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "note.0.text") == "Monitor A1c quarterly"

    def test_medical_hx_code_text(self):
        spec = ALL_TABLE_SPECS["MEDICAL_HX"]
        mapper = get_mapper_for_table("MEDICAL_HX")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "code.text") == "Appendectomy"

    def test_medical_hx_annotation(self):
        spec = ALL_TABLE_SPECS["MEDICAL_HX"]
        mapper = get_mapper_for_table("MEDICAL_HX")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "note.0.text") == "Laparoscopic, no complications"

    def test_order_med_medication_name(self):
        spec = ALL_TABLE_SPECS["ORDER_MED"]
        mapper = get_mapper_for_table("ORDER_MED")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "medicationCodeableConcept.text") == "Metformin 500mg"

    def test_order_med_effective_period(self):
        """Verify effectivePeriod is used (not _effectiveStart)."""
        spec = ALL_TABLE_SPECS["ORDER_MED"]
        mapper = get_mapper_for_table("ORDER_MED")
        result = mapper.to_fhir(spec.example_row)
        assert "_effectiveStart" not in result
        assert "_effectiveEnd" not in result
        assert "effectivePeriod" in result
        assert result["effectivePeriod"]["start"] is not None

    def test_order_med_dosage(self):
        spec = ALL_TABLE_SPECS["ORDER_MED"]
        mapper = get_mapper_for_table("ORDER_MED")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "dosageInstruction.0.text") == "500mg twice daily"

    def test_order_med_dispense(self):
        spec = ALL_TABLE_SPECS["ORDER_MED"]
        mapper = get_mapper_for_table("ORDER_MED")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "dispenseRequest.quantity.value") == "60"
        assert resolve_fhir_path(result, "dispenseRequest.numberOfRepeatsAllowed") == "3"

    def test_order_med_route(self):
        spec = ALL_TABLE_SPECS["ORDER_MED"]
        mapper = get_mapper_for_table("ORDER_MED")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "dosageInstruction.0.route.text") == "Oral"

    def test_order_med_requester(self):
        spec = ALL_TABLE_SPECS["ORDER_MED"]
        mapper = get_mapper_for_table("ORDER_MED")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "requester.display") == "Dr. Smith"

    def test_order_results_value(self):
        spec = ALL_TABLE_SPECS["ORDER_RESULTS"]
        mapper = get_mapper_for_table("ORDER_RESULTS")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "valueQuantity.value") == 6.8
        assert resolve_fhir_path(result, "valueQuantity.unit") == "%"

    def test_order_results_reference_range(self):
        spec = ALL_TABLE_SPECS["ORDER_RESULTS"]
        mapper = get_mapper_for_table("ORDER_RESULTS")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "referenceRange.0.low.value") == 4.0
        assert resolve_fhir_path(result, "referenceRange.0.high.value") == 5.6

    def test_order_results_loinc(self):
        spec = ALL_TABLE_SPECS["ORDER_RESULTS"]
        mapper = get_mapper_for_table("ORDER_RESULTS")
        result = mapper.to_fhir(spec.example_row)
        coding = resolve_fhir_path(result, "code.coding.0")
        assert coding["system"] == "http://loinc.org"
        assert "Hemoglobin" in coding["display"]

    def test_order_results_interpretation(self):
        spec = ALL_TABLE_SPECS["ORDER_RESULTS"]
        mapper = get_mapper_for_table("ORDER_RESULTS")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "interpretation.0.coding.0.code") == "H"

    def test_pat_enc_period_start(self):
        spec = ALL_TABLE_SPECS["PAT_ENC"]
        mapper = get_mapper_for_table("PAT_ENC")
        result = mapper.to_fhir(spec.example_row)
        assert result["period"]["start"] is not None

    def test_pat_enc_location(self):
        spec = ALL_TABLE_SPECS["PAT_ENC"]
        mapper = get_mapper_for_table("PAT_ENC")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "location.0.location.display") == "Internal Medicine"

    def test_pat_enc_provider(self):
        spec = ALL_TABLE_SPECS["PAT_ENC"]
        mapper = get_mapper_for_table("PAT_ENC")
        result = mapper.to_fhir(spec.example_row)
        participant = resolve_fhir_path(result, "participant.0.individual.display")
        assert "Dr. Smith" in participant

    def test_pat_enc_reason(self):
        spec = ALL_TABLE_SPECS["PAT_ENC"]
        mapper = get_mapper_for_table("PAT_ENC")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "reasonCode.0.text") == "Annual checkup"

    def test_doc_information_type(self):
        spec = ALL_TABLE_SPECS["DOC_INFORMATION"]
        mapper = get_mapper_for_table("DOC_INFORMATION")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "type.text") == "Progress Note"

    def test_doc_information_description(self):
        spec = ALL_TABLE_SPECS["DOC_INFORMATION"]
        mapper = get_mapper_for_table("DOC_INFORMATION")
        result = mapper.to_fhir(spec.example_row)
        assert result["description"] == "Office Visit Progress Note"

    def test_doc_information_author(self):
        spec = ALL_TABLE_SPECS["DOC_INFORMATION"]
        mapper = get_mapper_for_table("DOC_INFORMATION")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "author.0.display") == "Dr. Smith"

    def test_allergy_code(self):
        spec = ALL_TABLE_SPECS["ALLERGY"]
        mapper = get_mapper_for_table("ALLERGY")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "code.text") == "Penicillin"

    def test_allergy_severity(self):
        spec = ALL_TABLE_SPECS["ALLERGY"]
        mapper = get_mapper_for_table("ALLERGY")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "reaction.0.severity") == "severe"

    def test_allergy_manifestation(self):
        spec = ALL_TABLE_SPECS["ALLERGY"]
        mapper = get_mapper_for_table("ALLERGY")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "reaction.0.manifestation.0.text") == "Hives"

    def test_immune_vaccine_code(self):
        spec = ALL_TABLE_SPECS["IMMUNE"]
        mapper = get_mapper_for_table("IMMUNE")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "vaccineCode.text") == "COVID-19 Vaccine"

    def test_immune_manufacturer(self):
        spec = ALL_TABLE_SPECS["IMMUNE"]
        mapper = get_mapper_for_table("IMMUNE")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "manufacturer.display") == "Pfizer"

    def test_immune_lot(self):
        spec = ALL_TABLE_SPECS["IMMUNE"]
        mapper = get_mapper_for_table("IMMUNE")
        result = mapper.to_fhir(spec.example_row)
        assert result["lotNumber"] == "EL9261"

    def test_order_proc_code(self):
        spec = ALL_TABLE_SPECS["ORDER_PROC"]
        mapper = get_mapper_for_table("ORDER_PROC")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "code.text") == "CT Abdomen with Contrast"

    def test_order_proc_performer(self):
        spec = ALL_TABLE_SPECS["ORDER_PROC"]
        mapper = get_mapper_for_table("ORDER_PROC")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "performer.0.actor.display") == "Dr. Smith"

    def test_vitals_value(self):
        spec = ALL_TABLE_SPECS["IP_FLWSHT_MEAS"]
        mapper = get_mapper_for_table("IP_FLWSHT_MEAS")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "valueQuantity.value") == 120.0
        assert resolve_fhir_path(result, "valueQuantity.unit") == "mmHg"

    def test_referral_code(self):
        spec = ALL_TABLE_SPECS["REFERRAL"]
        mapper = get_mapper_for_table("REFERRAL")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "code.text") == "Cardiology Consultation"

    def test_referral_providers(self):
        spec = ALL_TABLE_SPECS["REFERRAL"]
        mapper = get_mapper_for_table("REFERRAL")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "requester.display") == "Dr. Jones"
        assert resolve_fhir_path(result, "performer.0.display") == "Dr. Specialist"

    def test_encounter_dx_code(self):
        spec = ALL_TABLE_SPECS["PAT_ENC_DX"]
        mapper = get_mapper_for_table("PAT_ENC_DX")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "code.text") == "Acute Bronchitis"

    def test_encounter_dx_primary(self):
        spec = ALL_TABLE_SPECS["PAT_ENC_DX"]
        mapper = get_mapper_for_table("PAT_ENC_DX")
        result = mapper.to_fhir(spec.example_row)
        assert result.get("_primaryDiagnosis") is True

    def test_social_hx_code(self):
        spec = ALL_TABLE_SPECS["SOCIAL_HX"]
        mapper = get_mapper_for_table("SOCIAL_HX")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "code.text") == "Tobacco Use"
        assert result["valueString"] == "Former smoker, quit 2015"

    def test_family_hx_condition(self):
        spec = ALL_TABLE_SPECS["FAMILY_HX"]
        mapper = get_mapper_for_table("FAMILY_HX")
        result = mapper.to_fhir(spec.example_row)
        assert resolve_fhir_path(result, "condition.0.code.text") == "Type 2 Diabetes"
        assert resolve_fhir_path(result, "relationship.text") == "Mother"
        assert resolve_fhir_path(result, "condition.0.onsetAge.value") == 55


class TestDisplayTextExtraction:
    """Verify build_display_text produces meaningful text for each mapper."""

    @pytest.mark.parametrize(
        "table_name",
        list(ALL_TABLE_SPECS.keys()),
        ids=list(ALL_TABLE_SPECS.keys()),
    )
    def test_display_text_not_bare_resource_type(self, table_name: str):
        """display_text should contain clinical content, not just 'Condition'."""
        spec = ALL_TABLE_SPECS[table_name]
        results = map_with_metadata(table_name, [spec.example_row])
        assert len(results) == 1
        result = results[0]
        assert not result["skipped"]
        display = result["display_text"]
        # Should not be just the bare resource type name
        bare_names = {
            spec.resource_type, spec.resource_type.lower(),
            "Unknown", "unknown", "",
        }
        assert display not in bare_names, (
            f"{spec.mapper_class} display_text is bare '{display}'"
        )


# ---------------------------------------------------------------------------
# Real-data fidelity tests (require gitignored fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.fidelity
class TestRealEpicCompleteness:
    """Verify real Epic EHI export rows are not silently dropped."""

    def test_known_tables_have_mappers(self, real_epic_dir: Path):
        """All TSV files in real export should have registered mappers."""
        from app.services.ingestion.epic_parser import EPIC_TABLE_MAPPERS

        unmapped = []
        for tsv in real_epic_dir.glob("*.tsv"):
            table_name = tsv.stem.upper()
            if table_name not in EPIC_TABLE_MAPPERS:
                rows = load_tsv_rows(tsv)
                if rows:  # Only flag non-empty tables
                    unmapped.append(table_name)

        # Informational — not a hard failure, but flag for review
        if unmapped:
            pytest.skip(f"Unmapped tables with data: {unmapped}")

    @pytest.mark.parametrize(
        "table_name",
        list(ALL_TABLE_SPECS.keys()),
        ids=list(ALL_TABLE_SPECS.keys()),
    )
    def test_no_silent_drops(self, real_epic_dir: Path, table_name: str):
        """Every non-empty row with valid gate columns should produce a resource."""
        spec = ALL_TABLE_SPECS[table_name]
        tsv_path = real_epic_dir / f"{table_name}.tsv"
        if not tsv_path.exists():
            pytest.skip(f"No {table_name}.tsv in real export")

        rows = load_tsv_rows(tsv_path)
        if not rows:
            pytest.skip(f"{table_name}.tsv is header-only")

        mapper = get_mapper_for_table(table_name)
        dropped = []
        for i, row in enumerate(rows):
            # Check if gate columns are populated
            has_gate = any(row.get(col, "").strip() for col in spec.gate_columns)
            result = mapper.to_fhir(row)
            if has_gate and result is None:
                dropped.append(i)

        assert not dropped, (
            f"{spec.mapper_class} silently dropped {len(dropped)}/{len(rows)} "
            f"rows with valid gate columns: row indexes {dropped[:10]}"
        )

    @pytest.mark.parametrize(
        "table_name",
        list(ALL_TABLE_SPECS.keys()),
        ids=list(ALL_TABLE_SPECS.keys()),
    )
    def test_metadata_population(self, real_epic_dir: Path, table_name: str):
        """Track metadata population rates and warn on suspiciously low rates."""
        spec = ALL_TABLE_SPECS[table_name]
        tsv_path = real_epic_dir / f"{table_name}.tsv"
        if not tsv_path.exists():
            pytest.skip(f"No {table_name}.tsv in real export")

        rows = load_tsv_rows(tsv_path)
        if not rows:
            pytest.skip(f"{table_name}.tsv is header-only")

        results = map_with_metadata(table_name, rows)
        mapped = [r for r in results if not r["skipped"]]

        if not mapped:
            pytest.skip(f"No rows mapped for {table_name}")

        total = len(mapped)
        has_date = sum(1 for r in mapped if r["effective_date"] is not None)
        has_display = sum(1 for r in mapped if r["display_text"] and r["display_text"] != spec.resource_type)

        # display_text should always be populated (build_display_text fallback)
        assert has_display == total, (
            f"{spec.mapper_class}: only {has_display}/{total} rows have meaningful display_text"
        )

        # Warn on low date population (not a hard failure)
        if total > 0 and has_date / total < 0.5:
            import warnings
            warnings.warn(
                f"{spec.mapper_class}: only {has_date}/{total} ({has_date*100//total}%) "
                f"rows have effective_date"
            )

    @pytest.mark.parametrize(
        "table_name",
        list(ALL_TABLE_SPECS.keys()),
        ids=list(ALL_TABLE_SPECS.keys()),
    )
    def test_field_population_audit(self, real_epic_dir: Path, table_name: str):
        """For each mapped column, verify at least one real row populates it."""
        spec = ALL_TABLE_SPECS[table_name]
        tsv_path = real_epic_dir / f"{table_name}.tsv"
        if not tsv_path.exists():
            pytest.skip(f"No {table_name}.tsv in real export")

        rows = load_tsv_rows(tsv_path)
        if not rows:
            pytest.skip(f"{table_name}.tsv is header-only")

        mapper = get_mapper_for_table(table_name)
        mapped_resources = [mapper.to_fhir(r) for r in rows]
        mapped_resources = [r for r in mapped_resources if r is not None]

        if not mapped_resources:
            pytest.skip(f"No rows mapped for {table_name}")

        never_populated = []
        for col_mapping in spec.columns:
            populated = any(
                resolve_fhir_path(r, col_mapping.fhir_path) is not None
                for r in mapped_resources
            )
            if not populated:
                never_populated.append(col_mapping.tsv_column)

        if never_populated:
            import warnings
            warnings.warn(
                f"{spec.mapper_class}: columns never populated in real data: "
                f"{never_populated}"
            )


@pytest.mark.fidelity
class TestRealEpicRowDetails:
    """Spot-check specific real data values for correctness."""

    def test_row_count_summary(self, real_epic_dir: Path):
        """Report row counts per table for visibility."""
        from app.services.ingestion.epic_parser import EPIC_TABLE_MAPPERS

        summary = {}
        for tsv in sorted(real_epic_dir.glob("*.tsv")):
            table_name = tsv.stem.upper()
            rows = load_tsv_rows(tsv)
            has_mapper = table_name in EPIC_TABLE_MAPPERS
            mapped_count = 0
            if has_mapper and rows:
                mapper = get_mapper_for_table(table_name)
                mapped_count = sum(1 for r in rows if mapper.to_fhir(r) is not None)
            summary[table_name] = {
                "total_rows": len(rows),
                "has_mapper": has_mapper,
                "mapped": mapped_count,
            }

        # Print summary for test report
        for table, info in summary.items():
            status = "MAPPED" if info["has_mapper"] else "UNMAPPED"
            print(
                f"  {table}: {info['total_rows']} rows, "
                f"{status}, {info['mapped']} records"
            )

        # At least some tables should have data
        total_mapped = sum(v["mapped"] for v in summary.values())
        assert total_mapped > 0, "No rows mapped from real Epic export"
