"""FHIR bundle ingestion fidelity tests.

Tests verify that the FHIR parser correctly handles all resource types,
preserves full FHIR resources, and extracts metadata accurately.

Two categories:
1. Synthetic tests (always run) — verify supported resource types
2. Real-data fidelity tests (@pytest.mark.fidelity) — verify against
   actual FHIR bundles, skip when absent
"""
from __future__ import annotations

from collections import Counter

import pytest

from app.services.ingestion.fhir_parser import (
    SUPPORTED_RESOURCE_TYPES,
    build_display_text,
    map_fhir_resource,
)


# ---------------------------------------------------------------------------
# Synthetic tests (always available)
# ---------------------------------------------------------------------------


class TestSupportedResourceTypes:
    """Verify resource type coverage matches expectations."""

    EXPECTED_TYPES = {
        "Condition", "Observation", "MedicationRequest", "MedicationStatement",
        "AllergyIntolerance", "Procedure", "Encounter", "Immunization",
        "DiagnosticReport", "DocumentReference", "ImagingStudy",
        "ServiceRequest", "CarePlan", "Communication", "Appointment",
        "CareTeam", "ImmunizationRecommendation", "QuestionnaireResponse",
    }

    def test_all_expected_types_supported(self):
        """All expected FHIR resource types are in SUPPORTED_RESOURCE_TYPES."""
        missing = self.EXPECTED_TYPES - set(SUPPORTED_RESOURCE_TYPES.keys())
        assert not missing, f"Missing resource types: {missing}"

    def test_patient_not_supported(self):
        """Patient should not be in SUPPORTED_RESOURCE_TYPES."""
        assert "Patient" not in SUPPORTED_RESOURCE_TYPES

    def test_all_types_have_record_type(self):
        """Every supported type maps to a non-empty record_type."""
        for rt, record_type in SUPPORTED_RESOURCE_TYPES.items():
            assert record_type, f"{rt} has empty record_type"


class TestNewResourceTypes:
    """Verify the 3 newly added resource types work correctly."""

    def test_care_team_mapping(self):
        resource = {
            "resourceType": "CareTeam",
            "name": "Primary Care Team",
            "status": "active",
        }
        result = map_fhir_resource(resource)
        assert result is not None
        assert result["record_type"] == "care_team"
        assert result["display_text"] == "Primary Care Team"
        assert result["status"] == "active"

    def test_care_team_no_name(self):
        resource = {"resourceType": "CareTeam", "status": "active"}
        result = map_fhir_resource(resource)
        assert result is not None
        assert result["display_text"] == "Care Team"

    def test_immunization_recommendation_mapping(self):
        resource = {
            "resourceType": "ImmunizationRecommendation",
            "recommendation": [
                {
                    "vaccineCode": [
                        {"text": "Influenza Vaccine"}
                    ],
                    "forecastStatus": {
                        "coding": [{"code": "due"}]
                    },
                }
            ],
            "date": "2024-01-15",
        }
        result = map_fhir_resource(resource)
        assert result is not None
        assert result["record_type"] == "immunization"
        assert result["display_text"] == "Influenza Vaccine"

    def test_immunization_recommendation_no_vaccine(self):
        resource = {
            "resourceType": "ImmunizationRecommendation",
            "recommendation": [],
        }
        result = map_fhir_resource(resource)
        assert result is not None
        assert result["display_text"] == "Immunization Recommendation"

    def test_questionnaire_response_mapping(self):
        resource = {
            "resourceType": "QuestionnaireResponse",
            "questionnaire": "PHQ-9",
            "status": "completed",
            "authored": "2024-03-15",
        }
        result = map_fhir_resource(resource)
        assert result is not None
        assert result["record_type"] == "questionnaire_response"
        assert "PHQ-9" in result["display_text"]
        assert result["status"] == "completed"

    def test_questionnaire_response_no_questionnaire(self):
        resource = {
            "resourceType": "QuestionnaireResponse",
            "status": "completed",
        }
        result = map_fhir_resource(resource)
        assert result is not None
        assert result["display_text"] == "Questionnaire Response"


class TestDisplayTextForAllTypes:
    """Ensure build_display_text handles all supported types without error."""

    @pytest.mark.parametrize("resource_type", list(SUPPORTED_RESOURCE_TYPES.keys()))
    def test_bare_resource_produces_fallback(self, resource_type: str):
        """Bare resource with only resourceType should produce a non-empty display."""
        resource = {"resourceType": resource_type}
        text = build_display_text(resource, resource_type)
        assert text is not None
        assert len(text) > 0


class TestMetadataLossless:
    """Verify map_fhir_resource preserves the full resource."""

    def test_full_resource_preserved(self):
        """The fhir_resource field should contain the complete original resource."""
        resource = {
            "resourceType": "Observation",
            "id": "obs-123",
            "status": "final",
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}],
                "text": "Hemoglobin A1c",
            },
            "effectiveDateTime": "2024-01-10T10:30:00Z",
            "valueQuantity": {"value": 6.8, "unit": "%"},
            "referenceRange": [{"low": {"value": 4.0}, "high": {"value": 5.6}}],
            "interpretation": [{"coding": [{"code": "H"}]}],
            "performer": [{"display": "Dr. Lab"}],
            "note": [{"text": "Slightly elevated"}],
        }
        result = map_fhir_resource(resource)
        assert result is not None
        # Full original resource preserved
        stored = result["fhir_resource"]
        assert stored["id"] == "obs-123"
        assert stored["valueQuantity"]["value"] == 6.8
        assert stored["referenceRange"][0]["low"]["value"] == 4.0
        assert stored["note"][0]["text"] == "Slightly elevated"
        assert stored["performer"][0]["display"] == "Dr. Lab"


# ---------------------------------------------------------------------------
# Real-data fidelity tests (require gitignored fixture)
# ---------------------------------------------------------------------------


@pytest.mark.fidelity
class TestRealFhirBundleCompleteness:
    """Verify real FHIR bundle entries are not silently dropped."""

    def test_resource_type_coverage(self, real_fhir_bundle: dict):
        """Every resource type in the bundle should be supported or documented."""
        entries = real_fhir_bundle.get("entry", [])
        resource_types = Counter()
        unsupported_types = Counter()

        for entry in entries:
            resource = entry.get("resource", {})
            rt = resource.get("resourceType")
            if not rt:
                continue
            resource_types[rt] += 1
            if rt not in SUPPORTED_RESOURCE_TYPES and rt != "Patient":
                unsupported_types[rt] += 1

        # Print summary
        print(f"\nTotal entries: {len(entries)}")
        print("Resource types found:")
        for rt, count in resource_types.most_common():
            status = (
                "SUPPORTED" if rt in SUPPORTED_RESOURCE_TYPES
                else "PATIENT (skip)" if rt == "Patient"
                else "UNSUPPORTED"
            )
            print(f"  {rt}: {count} ({status})")

        if unsupported_types:
            import warnings
            warnings.warn(
                f"Unsupported resource types in real FHIR bundle: "
                f"{dict(unsupported_types)}"
            )

    def test_entry_completeness(self, real_fhir_bundle: dict):
        """Every supported entry should map to a health record."""
        entries = real_fhir_bundle.get("entry", [])
        total = 0
        mapped = 0
        skipped_patient = 0
        skipped_unsupported = 0
        errors = []

        for i, entry in enumerate(entries):
            resource = entry.get("resource", {})
            rt = resource.get("resourceType")
            if not rt:
                continue
            total += 1
            if rt == "Patient":
                skipped_patient += 1
                continue
            if rt not in SUPPORTED_RESOURCE_TYPES:
                skipped_unsupported += 1
                continue

            try:
                result = map_fhir_resource(resource)
                if result:
                    mapped += 1
                else:
                    errors.append(f"Entry {i} ({rt}): map_fhir_resource returned None")
            except Exception as e:
                errors.append(f"Entry {i} ({rt}): {e}")

        print("\nBundle summary:")
        print(f"  Total entries: {total}")
        print(f"  Mapped: {mapped}")
        print(f"  Patient (skipped): {skipped_patient}")
        print(f"  Unsupported type (skipped): {skipped_unsupported}")
        print(f"  Errors: {len(errors)}")

        expected_mappable = total - skipped_patient - skipped_unsupported
        assert mapped == expected_mappable, (
            f"Expected {expected_mappable} mapped entries, got {mapped}. "
            f"Errors: {errors[:10]}"
        )

    def test_lossless_storage(self, real_fhir_bundle: dict):
        """Full FHIR resource should be preserved in fhir_resource field."""
        entries = real_fhir_bundle.get("entry", [])
        checked = 0
        for entry in entries[:50]:  # Sample first 50
            resource = entry.get("resource", {})
            rt = resource.get("resourceType")
            if not rt or rt == "Patient" or rt not in SUPPORTED_RESOURCE_TYPES:
                continue

            result = map_fhir_resource(resource)
            if result:
                assert result["fhir_resource"] == resource, (
                    f"FHIR resource not preserved for {rt} entry"
                )
                checked += 1

        assert checked > 0, "No entries checked for lossless storage"

    def test_metadata_accuracy(self, real_fhir_bundle: dict):
        """Extracted metadata should match what's in the FHIR resource."""
        entries = real_fhir_bundle.get("entry", [])
        mismatches = []

        for i, entry in enumerate(entries):
            resource = entry.get("resource", {})
            rt = resource.get("resourceType")
            if not rt or rt == "Patient" or rt not in SUPPORTED_RESOURCE_TYPES:
                continue

            result = map_fhir_resource(resource)
            if not result:
                continue

            # Verify status consistency
            extracted_status = result["status"]
            raw_status = resource.get("status")
            if raw_status and extracted_status != raw_status:
                # Check if it came from clinicalStatus instead
                clinical = resource.get("clinicalStatus", {})
                if isinstance(clinical, dict):
                    codings = clinical.get("coding", [])
                    if codings and extracted_status != codings[0].get("code"):
                        mismatches.append(
                            f"Entry {i} ({rt}): status '{extracted_status}' != "
                            f"resource status '{raw_status}'"
                        )

        assert not mismatches, (
            f"Metadata mismatches: {mismatches[:10]}"
        )

    def test_population_rate_monitoring(self, real_fhir_bundle: dict):
        """Track and report metadata population rates."""
        entries = real_fhir_bundle.get("entry", [])
        stats = {
            "total_mapped": 0,
            "has_effective_date": 0,
            "has_status": 0,
            "has_code_system": 0,
            "has_display_text": 0,
            "has_category": 0,
        }

        for entry in entries:
            resource = entry.get("resource", {})
            rt = resource.get("resourceType")
            if not rt or rt == "Patient" or rt not in SUPPORTED_RESOURCE_TYPES:
                continue

            result = map_fhir_resource(resource)
            if not result:
                continue

            stats["total_mapped"] += 1
            if result["effective_date"]:
                stats["has_effective_date"] += 1
            if result["status"]:
                stats["has_status"] += 1
            if result["code_system"]:
                stats["has_code_system"] += 1
            if result["display_text"]:
                stats["has_display_text"] += 1
            if result["category"]:
                stats["has_category"] += 1

        total = stats["total_mapped"]
        if total == 0:
            pytest.skip("No entries mapped")

        print(f"\nPopulation rates ({total} records):")
        for field, count in stats.items():
            if field == "total_mapped":
                continue
            pct = count * 100 // total
            print(f"  {field}: {count}/{total} ({pct}%)")

        # display_text should always be populated
        assert stats["has_display_text"] == total, (
            f"display_text missing for {total - stats['has_display_text']} records"
        )

        # Warn on low rates
        if total > 10 and stats["has_effective_date"] / total < 0.3:
            import warnings
            warnings.warn(
                f"Low effective_date rate: "
                f"{stats['has_effective_date']}/{total}"
            )
