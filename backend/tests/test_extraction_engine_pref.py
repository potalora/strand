from __future__ import annotations

import pytest

from app.models.llm_settings import UserLLMPreferences
from app.models.user import User
from app.services.ai.llm.config import LLMConfig, load_llm_config


def test_from_settings_carries_extraction_engine(monkeypatch):
    from app.services.ai.llm import config as cfg

    monkeypatch.setattr(cfg.settings, "extraction_engine", "hybrid")
    assert LLMConfig.from_settings().extraction_engine == "hybrid"


@pytest.mark.asyncio
async def test_user_pref_overrides_extraction_engine(db_session, monkeypatch):
    from app.services.ai.llm import config as cfg

    monkeypatch.setattr(cfg.settings, "extraction_engine", "hybrid")
    user = User(login_identifier="ee-pref@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    db_session.add(UserLLMPreferences(user_id=user.id, extraction_engine="local"))
    await db_session.commit()

    resolved = await load_llm_config(db_session, user.id)
    assert resolved.extraction_engine == "local"


@pytest.mark.asyncio
async def test_no_pref_falls_back_to_settings_engine(db_session, monkeypatch):
    from app.services.ai.llm import config as cfg

    monkeypatch.setattr(cfg.settings, "extraction_engine", "gemini")
    user = User(login_identifier="ee-nopref@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    await db_session.commit()

    resolved = await load_llm_config(db_session, user.id)
    assert resolved.extraction_engine == "gemini"
