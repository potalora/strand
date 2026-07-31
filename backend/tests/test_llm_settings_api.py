from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import select

from app.middleware.encryption import decrypt_field
from app.models.llm_settings import LLMProviderConfig
from app.services.ai.llm.config import load_llm_config
from app.services.local_ai.types import ProcessingMode
from tests.conftest import auth_headers


@pytest.mark.asyncio
async def test_get_settings_masks_keys(client, db_session):
    headers, uid = await auth_headers(client)
    await client.put(
        "/api/v1/settings/llm/providers/openai",
        json={"api_key": "sk-supersecretvalue", "model": "gpt-4o"},
        headers=headers,
    )
    r = await client.get("/api/v1/settings/llm", headers=headers)
    assert r.status_code == 200
    assert "sk-supersecretvalue" not in r.text  # never returned in plaintext
    openai = next(p for p in r.json()["providers"] if p["name"] == "openai")
    assert openai["has_key"] is True and openai["model"] == "gpt-4o"
    # masked: present, contains the ellipsis, and is NOT the full key
    assert openai["key_masked"]
    assert "…" in openai["key_masked"]
    assert "sk-supersecretvalue" not in openai["key_masked"]
    assert openai["source"] == "user"


@pytest.mark.asyncio
async def test_get_settings_without_saved_preference_defaults_prompt_only(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="prompt-only-default@example.com",
    )

    config = await load_llm_config(db_session, UUID(user_id))
    response = await client.get("/api/v1/settings/llm", headers=headers)

    assert config.processing_mode is ProcessingMode.PROMPT_ONLY
    assert response.status_code == 200
    assert response.json()["routing"]["processing_mode"] == "prompt_only"


@pytest.mark.asyncio
async def test_put_encrypts_key_in_db(client, db_session):
    headers, uid = await auth_headers(client)
    await client.put(
        "/api/v1/settings/llm/providers/anthropic",
        json={"api_key": "sk-ant-xyz"},
        headers=headers,
    )
    row = (
        await db_session.execute(
            select(LLMProviderConfig).where(LLMProviderConfig.provider == "anthropic")
        )
    ).scalar_one()
    assert row.api_key_encrypted != b"sk-ant-xyz"
    assert decrypt_field(row.api_key_encrypted) == "sk-ant-xyz"


@pytest.mark.asyncio
async def test_put_omitting_key_preserves_existing(client, db_session):
    headers, uid = await auth_headers(client)
    await client.put(
        "/api/v1/settings/llm/providers/openai",
        json={"api_key": "sk-original", "model": "gpt-4o"},
        headers=headers,
    )
    # Update only the model — the stored key must survive.
    await client.put(
        "/api/v1/settings/llm/providers/openai",
        json={"model": "gpt-4o-mini"},
        headers=headers,
    )
    row = (
        await db_session.execute(
            select(LLMProviderConfig).where(LLMProviderConfig.provider == "openai")
        )
    ).scalar_one()
    assert decrypt_field(row.api_key_encrypted) == "sk-original"
    assert row.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_unknown_provider_rejected(client):
    headers, uid = await auth_headers(client)
    r = await client.put(
        "/api/v1/settings/llm/providers/bogus",
        json={"api_key": "sk-1"},
        headers=headers,
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_routing_upsert_and_get(client):
    headers, uid = await auth_headers(client)
    await client.put(
        "/api/v1/settings/llm/routing",
        json={"default": "ollama", "summary": "anthropic"},
        headers=headers,
    )
    r = await client.get("/api/v1/settings/llm", headers=headers)
    routing = r.json()["routing"]
    assert routing["default"] == "ollama" and routing["summary"] == "anthropic"


@pytest.mark.asyncio
async def test_explicit_cloud_assisted_preference_remains_cloud_assisted(
    client,
) -> None:
    headers, _user_id = await auth_headers(
        client,
        email="explicit-cloud-assisted@example.com",
    )

    update = await client.put(
        "/api/v1/settings/llm/routing",
        json={"processing_mode": "cloud_assisted"},
        headers=headers,
    )
    response = await client.get("/api/v1/settings/llm", headers=headers)

    assert update.status_code == 200
    assert response.status_code == 200
    assert response.json()["routing"]["processing_mode"] == "cloud_assisted"


@pytest.mark.asyncio
async def test_delete_reverts_provider(client, db_session):
    headers, uid = await auth_headers(client)
    await client.put(
        "/api/v1/settings/llm/providers/openai",
        json={"api_key": "sk-1"},
        headers=headers,
    )
    await client.delete("/api/v1/settings/llm/providers/openai", headers=headers)
    rows = (
        (
            await db_session.execute(
                select(LLMProviderConfig).where(LLMProviderConfig.provider == "openai")
            )
        )
        .scalars()
        .all()
    )
    assert rows == []


@pytest.mark.asyncio
async def test_user_scoping(client):
    # user A saves a key; user B's GET must not see it
    ha, ua = await auth_headers(client, email="a-llm@example.com")
    await client.put(
        "/api/v1/settings/llm/providers/openai", json={"api_key": "sk-A"}, headers=ha
    )
    hb, ub = await auth_headers(client, email="b-llm@example.com")
    rb = await client.get("/api/v1/settings/llm", headers=hb)
    assert "sk-A" not in rb.text
    openai = next(p for p in rb.json()["providers"] if p["name"] == "openai")
    assert openai["has_key"] is False
