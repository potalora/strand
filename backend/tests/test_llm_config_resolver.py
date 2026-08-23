from __future__ import annotations

import pytest

from app.middleware.encryption import encrypt_field
from app.models.llm_settings import LLMProviderConfig, UserLLMPreferences
from app.models.user import User
from app.services.ai.llm.config import LLMConfig, load_llm_config


def test_from_settings_has_gemini_default(monkeypatch):
    """from_settings reflects the global provider + key."""
    from app.services.ai.llm import config as cfg

    monkeypatch.setattr(cfg.settings, "llm_provider", "gemini")
    monkeypatch.setattr(cfg.settings, "gemini_api_key", "g-key")
    c = LLMConfig.from_settings()
    assert c.routing["default"] == "gemini"
    assert c.providers["gemini"].api_key == "g-key"


@pytest.mark.asyncio
async def test_user_row_overrides_env(db_session, monkeypatch):
    """A user's saved provider rows + prefs override the global .env defaults."""
    from app.services.ai.llm import config as cfg

    monkeypatch.setattr(cfg.settings, "llm_provider", "gemini")
    monkeypatch.setattr(cfg.settings, "openai_api_key", "env-openai")
    user = User(login_identifier="llm-resolver-z@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    db_session.add(
        LLMProviderConfig(
            user_id=user.id,
            provider="openai",
            api_key_encrypted=encrypt_field("user-openai"),
            model="gpt-4o",
            enabled=True,
        )
    )
    db_session.add(
        UserLLMPreferences(
            user_id=user.id, default_provider="openai", summary_provider="anthropic"
        )
    )
    await db_session.commit()

    c = await load_llm_config(db_session, user.id)
    assert c.routing["default"] == "openai"  # user pref
    assert c.routing["summary"] == "anthropic"  # per-op override
    assert c.providers["openai"].api_key == "user-openai"  # decrypted user key
    assert c.providers["openai"].model == "gpt-4o"
    # provider with no user row falls back to env
    assert c.providers["gemini"].api_key == cfg.settings.gemini_api_key


@pytest.mark.asyncio
async def test_no_user_config_equals_settings(db_session):
    """A user with no saved config resolves identically to from_settings()."""
    user = User(login_identifier="llm-resolver-y@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    await db_session.commit()
    c = await load_llm_config(db_session, user.id)
    base = LLMConfig.from_settings()
    assert c.routing["default"] == base.routing["default"]
