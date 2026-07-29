from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from types import SimpleNamespace
from uuid import uuid4
from datetime import datetime, timezone

from app.services.dedup.detector import (
    _apply_merge,
    _compare_records,
    _fuzzy_match,
)
from app.services.dedup.orchestrator import run_upload_dedup, DedupSummary


class FakeRecord:
    """Minimal stand-in for HealthRecord in unit tests."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid4())
        self.record_type = kwargs.get("record_type", "medication")
        self.code_value = kwargs.get("code_value")
        self.code_display = kwargs.get("code_display")
        self.display_text = kwargs.get("display_text", "Test Record")
        self.effective_date = kwargs.get("effective_date")
        self.status = kwargs.get("status")
        self.source_format = kwargs.get("source_format", "fhir_r4")
        self.source_section = kwargs.get("source_section")
        self.source_file_id = kwargs.get("source_file_id")
        self.fhir_resource = kwargs.get("fhir_resource", {})


class TestCompareRecordsUpgraded:
    """Tests for upgraded scoring with lower threshold and source_section bonus."""

    def test_threshold_at_0_5(self):
        """Records with code match (0.4) + status match (0.1) = 0.5 should pass."""
        a = FakeRecord(code_value="R14.0", status="active")
        b = FakeRecord(code_value="R14.0", status="active")
        score, reasons = _compare_records(a, b)
        assert score >= 0.5

    def test_threshold_below_0_5_rejected(self):
        """Records with only code match (0.4) and no other signals should be at boundary."""
        # Genuinely-distinct display_text (no shared tokens) so the token-set
        # fuzzy matcher contributes nothing — code match (0.4) is the only signal.
        a = FakeRecord(code_value="R14.0", status="active", display_text="Hypertension")
        b = FakeRecord(code_value="R14.0", status="resolved", display_text="Diabetes")
        score, reasons = _compare_records(a, b)
        assert score == 0.4  # only code match

    def test_source_section_bonus(self):
        """Matching source_section adds +0.15 to score."""
        a = FakeRecord(code_value="R14.0", source_section="medications")
        b = FakeRecord(code_value="R14.0", source_section="medications")
        score, reasons = _compare_records(a, b)
        assert score >= 0.55  # 0.4 code + 0.15 section
        assert reasons.get("section_match") is True

    def test_source_section_no_bonus_when_different(self):
        """Different source_section does not add bonus."""
        # Distinct display_text so only the code match (0.4) contributes.
        a = FakeRecord(
            code_value="R14.0",
            source_section="medications",
            display_text="Hypertension",
        )
        b = FakeRecord(
            code_value="R14.0", source_section="assessment", display_text="Diabetes"
        )
        score, reasons = _compare_records(a, b)
        assert score == 0.4
        assert "section_match" not in reasons

    def test_source_section_no_bonus_when_none(self):
        """None source_section does not add bonus."""
        # Distinct display_text so only the code match (0.4) contributes.
        a = FakeRecord(
            code_value="R14.0", source_section=None, display_text="Hypertension"
        )
        b = FakeRecord(
            code_value="R14.0", source_section="medications", display_text="Diabetes"
        )
        score, reasons = _compare_records(a, b)
        assert score == 0.4

    def test_exact_match_scores_above_0_95(self):
        """Exact match with all signals should score >= 0.95."""
        now = datetime.now(timezone.utc)
        a = FakeRecord(
            code_value="R14.0",
            display_text="Abdominal distension",
            effective_date=now,
            status="active",
            source_section="assessment",
        )
        b = FakeRecord(
            code_value="R14.0",
            display_text="Abdominal distension",
            effective_date=now,
            status="active",
            source_section="assessment",
        )
        score, reasons = _compare_records(a, b)
        # code(0.4) + text_exact(0.3) + date(0.2) + status(0.1) + section(0.15) = 1.0 (capped)
        assert score >= 0.95

    def test_cross_source_bonus_still_works(self):
        """Cross-source bonus (+0.1) still applies."""
        a = FakeRecord(code_value="R14.0", source_format="fhir_r4")
        b = FakeRecord(code_value="R14.0", source_format="ai_extracted")
        score, reasons = _compare_records(a, b)
        assert score >= 0.5  # 0.4 + 0.1
        assert reasons.get("cross_source") is True


class TestApplyMerge:
    """Shared merge helper sets the duplicate flags on the secondary record."""

    def test_apply_merge_sets_duplicate_flags(self):
        primary_id = uuid4()
        secondary = FakeRecord()
        secondary.is_duplicate = False
        secondary.merged_into_id = None

        _apply_merge(secondary, primary_id)

        assert secondary.is_duplicate is True
        assert secondary.merged_into_id == primary_id


class TestFuzzyMatch:
    """RapidFuzz token-set matcher (replaces the homemade Jaccard word-overlap).

    Contract relied on by ``_compare_records``: the score is in ``[0, 1]``;
    near-identical strings clear the ``> 0.8`` ``text_fuzzy_match`` gate while
    unrelated strings fall well below it. Token/dose reordering must NOT drop a
    true match below the gate — the old Jaccard scored
    "Metformin 500mg tablet" vs "metformin tablet 500 mg" at 0.4 and missed it.
    """

    def test_identical_strings_score_one(self):
        assert _fuzzy_match("Abdominal distension", "Abdominal distension") == 1.0

    def test_case_insensitive(self):
        assert _fuzzy_match("Metformin", "metformin") == 1.0

    def test_reordered_tokens_score_high(self):
        # Identical token set, reordered — an order-insensitive matcher returns 1.0.
        assert (
            _fuzzy_match("type 2 diabetes mellitus", "diabetes mellitus type 2") == 1.0
        )

    def test_reordered_with_split_dose_clears_gate(self):
        # Reordered + split dose token. The old Jaccard scored this 0.4 (a miss);
        # token_set_ratio recognizes the shared tokens and clears the 0.8 gate.
        assert _fuzzy_match("Metformin 500mg tablet", "metformin tablet 500 mg") > 0.8

    def test_unrelated_strings_score_low(self):
        assert _fuzzy_match("Lisinopril 10mg", "Atorvastatin 20mg") < 0.5
        assert _fuzzy_match("Colonoscopy", "Chest X-ray") < 0.5

    def test_distinct_labels_stay_below_gate(self):
        # Distinct clinical labels must not be flagged as a fuzzy text match.
        assert _fuzzy_match("Knee pain", "Abdominal distension") <= 0.8


class TestDateDistancePenalty:
    """Same code/text on far-apart dates are distinct time-series events
    (repeat labs/vitals, annual immunizations, separate encounters), not
    duplicates. They must score below the review threshold instead of
    flooding the dedup queue.
    """

    def test_distant_dates_score_below_review_threshold(self):
        """Identical code/text/status a year apart must not reach LLM review."""
        a = FakeRecord(
            code_value="39156-5",
            display_text="BMI",
            status="final",
            effective_date=datetime(2025, 1, 21, tzinfo=timezone.utc),
        )
        b = FakeRecord(
            code_value="39156-5",
            display_text="BMI",
            status="final",
            effective_date=datetime(2026, 2, 18, tzinfo=timezone.utc),
        )
        score, reasons = _compare_records(a, b)
        # Without a penalty this is 0.4 + 0.3 + 0.1 = 0.8 (false positive).
        assert score < 0.6, f"distant-date pair should fall below review, got {score}"
        assert reasons.get("date_distant") is True
        assert "date_proximity" not in reasons

    def test_same_day_duplicate_preserved(self):
        """A genuine same-day duplicate must still score for review/merge."""
        day = datetime(2026, 6, 7, tzinfo=timezone.utc)
        a = FakeRecord(
            code_value="39156-5", display_text="BMI", status="final", effective_date=day
        )
        b = FakeRecord(
            code_value="39156-5", display_text="BMI", status="final", effective_date=day
        )
        score, reasons = _compare_records(a, b)
        assert score >= 0.7
        assert reasons.get("date_proximity") is True
        assert "date_distant" not in reasons

    def test_missing_date_not_penalized(self):
        """When one record has no date we cannot compare — no penalty.

        Guards the real merge case: a CDA condition with an onset date vs the
        same condition from a cumulative export with a null date.
        """
        a = FakeRecord(
            code_value="E55.9",
            display_text="Vitamin D deficiency",
            effective_date=datetime(2023, 10, 5, tzinfo=timezone.utc),
        )
        b = FakeRecord(
            code_value="E55.9", display_text="Vitamin D deficiency", effective_date=None
        )
        score, reasons = _compare_records(a, b)
        assert score == pytest.approx(0.7)  # code 0.4 + text 0.3, no date signal
        assert "date_distant" not in reasons
        assert "date_proximity" not in reasons

    def test_dates_within_a_few_days_not_penalized(self):
        """A few days apart is ambiguous (same event, cross-system timestamps)
        — neither bonus nor penalty."""
        a = FakeRecord(
            code_value="R14.0",
            display_text="Abdominal distension",
            status="active",
            effective_date=datetime(2026, 2, 18, tzinfo=timezone.utc),
        )
        b = FakeRecord(
            code_value="R14.0",
            display_text="Abdominal distension",
            status="active",
            effective_date=datetime(2026, 2, 20, tzinfo=timezone.utc),
        )
        score, reasons = _compare_records(a, b)
        assert score == pytest.approx(0.8)  # code 0.4 + text 0.3 + status 0.1
        assert "date_distant" not in reasons
        assert "date_proximity" not in reasons


class TestRunUploadDedup:
    """Tests for the full dedup orchestration flow."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "processing_mode",
        ["validated_strict_local", "custom_local", "prompt_only"],
    )
    async def test_non_cloud_modes_persist_fuzzy_matches_without_loading_an_llm(
        self,
        processing_mode,
    ):
        """Non-cloud dedup keeps fuzzy pairs for manual review."""
        mock_db = AsyncMock()
        mock_db.get.return_value = FakeRecord(record_type="medication")
        auto_candidate = {
            "id": uuid4(),
            "record_a_id": uuid4(),
            "record_b_id": uuid4(),
            "similarity_score": 0.98,
            "match_reasons": {"code_match": True, "text_exact_match": True},
            "status": "pending",
            "source_upload_id": uuid4(),
        }
        fuzzy_candidate = {
            "id": uuid4(),
            "record_a_id": uuid4(),
            "record_b_id": uuid4(),
            "similarity_score": 0.72,
            "match_reasons": {"code_match": True, "text_fuzzy_match": True},
            "status": "pending",
            "source_upload_id": uuid4(),
        }

        with (
            patch(
                "app.services.dedup.orchestrator.load_llm_config",
                new_callable=AsyncMock,
            ) as mock_load_config,
            patch(
                "app.services.dedup.orchestrator.detect_upload_duplicates",
                new_callable=AsyncMock,
                return_value=([auto_candidate], [fuzzy_candidate]),
            ),
            patch(
                "app.services.dedup.orchestrator._run_llm_judge",
                new_callable=AsyncMock,
            ) as mock_judge,
            patch(
                "app.services.dedup.orchestrator._apply_auto_merges",
                new_callable=AsyncMock,
            ) as mock_apply,
            patch(
                "app.services.dedup.orchestrator._save_candidates",
                new_callable=AsyncMock,
            ) as mock_save,
        ):
            summary = await run_upload_dedup(
                uuid4(),
                uuid4(),
                uuid4(),
                mock_db,
                processing_mode=processing_mode,
            )

        mock_load_config.assert_not_awaited()
        mock_judge.assert_not_awaited()
        mock_apply.assert_awaited_once()
        assert mock_save.await_count == 2
        assert fuzzy_candidate["status"] == "pending"
        assert fuzzy_candidate["auto_resolved"] is False
        assert "llm_classification" not in fuzzy_candidate
        assert summary.total_candidates == 2
        assert summary.auto_merged == 1
        assert summary.needs_review == 1
        assert summary.dismissed == 0

    @pytest.mark.asyncio
    async def test_strict_repeated_upload_exact_merge_can_share_final_transaction(
        self,
        db_session,
    ) -> None:
        """The second strict upload is exactly merged without any LLM or early commit."""
        from app.models.patient import Patient
        from app.models.record import HealthRecord
        from app.models.uploaded_file import UploadedFile
        from app.models.user import User

        user = User(email="strict-repeat-dedup@example.com", password_hash="x")
        db_session.add(user)
        await db_session.flush()
        patient = Patient(user_id=user.id)
        db_session.add(patient)
        await db_session.flush()
        first_upload = UploadedFile(
            user_id=user.id,
            filename="first.rtf",
            mime_type="application/rtf",
            file_hash="1" * 64,
            storage_path="/private/first.rtf",
            processing_mode="validated_strict_local",
        )
        second_upload = UploadedFile(
            user_id=user.id,
            filename="second.rtf",
            mime_type="application/rtf",
            file_hash="2" * 64,
            storage_path="/private/second.rtf",
            processing_mode="validated_strict_local",
        )
        db_session.add_all([first_upload, second_upload])
        await db_session.flush()
        effective_date = datetime(2026, 7, 28, tzinfo=timezone.utc)
        first_record = HealthRecord(
            patient_id=patient.id,
            user_id=user.id,
            record_type="condition",
            fhir_resource_type="Condition",
            fhir_resource={"resourceType": "Condition"},
            source_format="ai_extracted",
            source_file_id=first_upload.id,
            effective_date=effective_date,
            status="active",
            code_value="38341003",
            display_text="Hypertension",
        )
        second_record = HealthRecord(
            patient_id=patient.id,
            user_id=user.id,
            record_type="condition",
            fhir_resource_type="Condition",
            fhir_resource={"resourceType": "Condition"},
            source_format="ai_extracted",
            source_file_id=second_upload.id,
            effective_date=effective_date,
            status="active",
            code_value="38341003",
            display_text="Hypertension",
        )
        db_session.add_all([first_record, second_record])
        await db_session.flush()

        with (
            patch(
                "app.services.dedup.orchestrator.load_llm_config",
                new_callable=AsyncMock,
            ) as mock_load_config,
            patch(
                "app.services.dedup.orchestrator._run_llm_judge",
                new_callable=AsyncMock,
            ) as mock_judge,
        ):
            summary = await run_upload_dedup(
                second_upload.id,
                patient.id,
                user.id,
                db_session,
                processing_mode="validated_strict_local",
                commit=False,
            )

        mock_load_config.assert_not_awaited()
        mock_judge.assert_not_awaited()
        assert summary.auto_merged == 1
        assert second_record.is_duplicate is True
        assert second_record.merged_into_id == first_record.id
        await db_session.rollback()
        assert await db_session.get(HealthRecord, second_record.id) is None

    @pytest.mark.asyncio
    async def test_background_dedup_forwards_persisted_processing_mode(self):
        """A restarted background pass uses the upload's immutable mode."""
        from app.services.ingestion import coordinator

        upload_id = uuid4()
        patient_id = uuid4()
        user_id = uuid4()
        upload = SimpleNamespace(
            processing_mode="validated_strict_local",
            dedup_summary=None,
            ingestion_status="dedup_scanning",
            processing_completed_at=None,
        )
        mock_db = AsyncMock()
        mock_db.get.return_value = upload

        class SessionFactory:
            def __call__(self):
                return self

            async def __aenter__(self):
                return mock_db

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with (
            patch("app.database.async_session_factory", new=SessionFactory()),
            patch(
                "app.services.dedup.orchestrator.run_upload_dedup",
                new_callable=AsyncMock,
                return_value=DedupSummary(),
            ) as mock_run,
        ):
            await coordinator._run_dedup_background(upload_id, patient_id, user_id)

        mock_run.assert_awaited_once_with(
            upload_id,
            patient_id,
            user_id,
            mock_db,
            processing_mode="validated_strict_local",
        )

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty_summary(self):
        mock_db = AsyncMock()
        with (
            patch(
                "app.services.dedup.orchestrator.load_llm_config",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.dedup.orchestrator.detect_upload_duplicates",
                new_callable=AsyncMock,
                return_value=([], []),
            ),
        ):
            summary = await run_upload_dedup(uuid4(), uuid4(), uuid4(), mock_db)

        assert isinstance(summary, DedupSummary)
        assert summary.total_candidates == 0
        assert summary.auto_merged == 0
        assert summary.needs_review == 0

    @pytest.mark.asyncio
    async def test_auto_merge_exact_matches(self):
        mock_db = AsyncMock()
        auto_candidates = [
            {
                "id": uuid4(),
                "record_a_id": uuid4(),
                "record_b_id": uuid4(),
                "similarity_score": 0.98,
                "match_reasons": {"code_match": True, "text_exact_match": True},
                "status": "pending",
                "source_upload_id": uuid4(),
            },
        ]

        with (
            patch(
                "app.services.dedup.orchestrator.load_llm_config",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.dedup.orchestrator.detect_upload_duplicates",
                new_callable=AsyncMock,
                return_value=(auto_candidates, []),
            ),
            patch(
                "app.services.dedup.orchestrator._apply_auto_merges",
                new_callable=AsyncMock,
            ) as mock_apply,
        ):
            summary = await run_upload_dedup(uuid4(), uuid4(), uuid4(), mock_db)

        assert summary.auto_merged == 1
        assert summary.needs_review == 0
        mock_apply.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_judge_called_for_fuzzy_matches(self):
        mock_db = AsyncMock()
        fuzzy_candidates = [
            {
                "id": uuid4(),
                "record_a_id": uuid4(),
                "record_b_id": uuid4(),
                "similarity_score": 0.72,
                "match_reasons": {"code_match": True, "text_fuzzy_match": True},
                "status": "pending",
                "source_upload_id": uuid4(),
            },
        ]

        mock_judgment = MagicMock()
        mock_judgment.classification = "update"
        mock_judgment.confidence = 0.85
        mock_judgment.explanation = "Dose changed"
        mock_judgment.field_diff = {
            "dosageInstruction": {"old": "500mg", "new": "1000mg"}
        }

        with (
            patch(
                "app.services.dedup.orchestrator.load_llm_config",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.dedup.orchestrator.detect_upload_duplicates",
                new_callable=AsyncMock,
                return_value=([], fuzzy_candidates),
            ),
            patch(
                "app.services.dedup.orchestrator._run_llm_judge",
                new_callable=AsyncMock,
                return_value=[mock_judgment],
            ),
            patch(
                "app.services.dedup.orchestrator._save_candidates",
                new_callable=AsyncMock,
            ),
        ):
            summary = await run_upload_dedup(uuid4(), uuid4(), uuid4(), mock_db)

        assert summary.needs_review == 1

    @pytest.mark.asyncio
    async def test_llm_duplicate_auto_merges(self):
        """LLM judge returning 'duplicate' with high confidence auto-merges."""
        mock_db = AsyncMock()
        fuzzy_candidates = [
            {
                "id": uuid4(),
                "record_a_id": uuid4(),
                "record_b_id": uuid4(),
                "similarity_score": 0.72,
                "match_reasons": {},
                "status": "pending",
                "source_upload_id": uuid4(),
            },
        ]

        mock_judgment = MagicMock()
        mock_judgment.classification = "duplicate"
        mock_judgment.confidence = 0.9
        mock_judgment.explanation = "Same record"
        mock_judgment.field_diff = None

        with (
            patch(
                "app.services.dedup.orchestrator.load_llm_config",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.dedup.orchestrator.detect_upload_duplicates",
                new_callable=AsyncMock,
                return_value=([], fuzzy_candidates),
            ),
            patch(
                "app.services.dedup.orchestrator._run_llm_judge",
                new_callable=AsyncMock,
                return_value=[mock_judgment],
            ),
            patch(
                "app.services.dedup.orchestrator._apply_auto_merges",
                new_callable=AsyncMock,
            ) as mock_apply,
            patch(
                "app.services.dedup.orchestrator._save_candidates",
                new_callable=AsyncMock,
            ),
        ):
            summary = await run_upload_dedup(uuid4(), uuid4(), uuid4(), mock_db)

        assert summary.auto_merged == 1
        mock_apply.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_distinct_auto_dismisses(self):
        """LLM judge returning 'distinct' with high confidence auto-dismisses."""
        mock_db = AsyncMock()
        fuzzy_candidates = [
            {
                "id": uuid4(),
                "record_a_id": uuid4(),
                "record_b_id": uuid4(),
                "similarity_score": 0.65,
                "match_reasons": {},
                "status": "pending",
                "source_upload_id": uuid4(),
            },
        ]

        mock_judgment = MagicMock()
        mock_judgment.classification = "distinct"
        mock_judgment.confidence = 0.92
        mock_judgment.explanation = "Different concepts"
        mock_judgment.field_diff = None

        with (
            patch(
                "app.services.dedup.orchestrator.load_llm_config",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.dedup.orchestrator.detect_upload_duplicates",
                new_callable=AsyncMock,
                return_value=([], fuzzy_candidates),
            ),
            patch(
                "app.services.dedup.orchestrator._run_llm_judge",
                new_callable=AsyncMock,
                return_value=[mock_judgment],
            ),
            patch(
                "app.services.dedup.orchestrator._save_candidates",
                new_callable=AsyncMock,
            ),
        ):
            summary = await run_upload_dedup(uuid4(), uuid4(), uuid4(), mock_db)

        assert summary.auto_merged == 0
        assert summary.needs_review == 0
        assert summary.dismissed == 1
