from __future__ import annotations

import pytest
from sqlalchemy import select

from app.middleware.encryption import decrypt_field, encrypt_field
from app.models.ai_summary import AISummaryPrompt
from app.models.llm_settings import LLMProviderConfig, UserLLMPreferences
from app.models.patient import Patient
from app.models.uploaded_file import UploadedFile
from app.models.user import User


@pytest.mark.asyncio
async def test_provider_config_roundtrip_encrypted_key(db_session):
    """An API key is stored as ciphertext and round-trips via decrypt_field."""
    user = User(login_identifier="llm-models-a@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    row = LLMProviderConfig(
        user_id=user.id,
        provider="openai",
        api_key_encrypted=encrypt_field("sk-secret"),
        base_url=None,
        model="gpt-4o-mini",
        enabled=True,
    )
    db_session.add(row)
    await db_session.commit()

    got = (
        await db_session.execute(
            select(LLMProviderConfig).where(LLMProviderConfig.user_id == user.id)
        )
    ).scalar_one()
    assert got.api_key_encrypted != b"sk-secret"  # stored as ciphertext
    assert decrypt_field(got.api_key_encrypted) == "sk-secret"
    assert got.provider == "openai"
    assert got.model == "gpt-4o-mini"
    assert got.enabled is True


@pytest.mark.asyncio
async def test_preferences_one_row_per_user(db_session):
    """Preferences persist routing overrides; unset operations stay None."""
    user = User(login_identifier="llm-models-b@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    pref = UserLLMPreferences(
        user_id=user.id,
        default_provider="anthropic",
        processing_mode="validated_strict_local",
    )
    db_session.add(pref)
    await db_session.commit()

    got = (
        await db_session.execute(
            select(UserLLMPreferences).where(UserLLMPreferences.user_id == user.id)
        )
    ).scalar_one()
    assert got.default_provider == "anthropic"
    assert got.summary_provider is None
    assert got.extraction_engine is None
    assert got.processing_mode == "validated_strict_local"


@pytest.mark.asyncio
async def test_new_processing_rows_default_prompt_only(db_session) -> None:
    """Fresh policy-bearing rows cannot opt into cloud processing by omission."""
    user = User(login_identifier="llm-models-defaults@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    patient = Patient(user_id=user.id)
    db_session.add(patient)
    await db_session.flush()

    preference = UserLLMPreferences(user_id=user.id)
    summary = AISummaryPrompt(
        user_id=user.id,
        patient_id=patient.id,
        summary_type="full",
        system_prompt="Organize the supplied records.",
        user_prompt="Create a records summary.",
        suggested_config={},
        record_count=0,
    )
    upload = UploadedFile(
        user_id=user.id,
        filename="records.json",
        mime_type="application/json",
        file_hash="f" * 64,
        storage_path="/encrypted/records.json",
    )
    db_session.add_all([preference, summary, upload])
    await db_session.flush()

    assert preference.processing_mode == "prompt_only"
    assert summary.processing_mode == "prompt_only"
    assert upload.processing_mode == "prompt_only"


def test_processing_mode_model_defaults_are_prompt_only() -> None:
    """Python and server defaults agree for every policy-bearing model."""
    for model in (UserLLMPreferences, AISummaryPrompt, UploadedFile):
        column = model.__table__.c.processing_mode

        assert column.default is not None
        assert column.default.arg == "prompt_only"
        assert column.server_default is not None
        assert str(column.server_default.arg) == "prompt_only"
