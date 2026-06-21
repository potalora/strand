from __future__ import annotations

import pytest
from sqlalchemy import select

from app.middleware.encryption import decrypt_field, encrypt_field
from app.models.llm_settings import LLMProviderConfig, UserLLMPreferences
from app.models.user import User


@pytest.mark.asyncio
async def test_provider_config_roundtrip_encrypted_key(db_session):
    """An API key is stored as ciphertext and round-trips via decrypt_field."""
    user = User(email="llm-models-a@example.com", password_hash="x")
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
    user = User(email="llm-models-b@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    pref = UserLLMPreferences(user_id=user.id, default_provider="anthropic")
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
