from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.middleware.encryption import decrypt_field, encrypt_field
from app.models.llm_settings import LLMProviderConfig, UserLLMPreferences
from app.schemas.llm_settings import ProviderUpdate, RoutingUpdate
from app.services.ai.llm import (
    KNOWN_PROVIDERS,
    LLMAuthError,
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMRateLimitError,
    LLMRequest,
    ProviderCreds,
    available_providers,
    load_llm_config,
)
from app.services.ai.llm import registry
from app.services.local_ai.errors import LocalPolicyError
from app.services.local_ai.policy import require_loopback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings/llm", tags=["llm-settings"])

# RoutingUpdate field -> UserLLMPreferences column.
_ROUTING_FIELDS: dict[str, str] = {
    "default": "default_provider",
    "summary": "summary_provider",
    "section": "section_provider",
    "dedup": "dedup_provider",
    "extraction": "extraction_provider",
    "vision": "vision_provider",
    "extraction_engine": "extraction_engine",
    "processing_mode": "processing_mode",
}


def _mask_key(key: str) -> str:
    """Return a masked hint for an API key — never the full value."""
    return f"{key[:3]}…{key[-4:]}" if len(key) > 8 else "…"


@router.get("")
async def get_llm_settings(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return selectable providers (with per-user key status) plus resolved routing.

    Never returns a key in plaintext — only a masked hint and a ``source`` flag
    ("user" when the user stored their own key, "env" when only the global
    ``.env`` provides one).
    """
    config = await load_llm_config(db, user_id)
    base = LLMConfig.from_settings()

    rows = (
        (
            await db.execute(
                select(LLMProviderConfig).where(LLMProviderConfig.user_id == user_id)
            )
        )
        .scalars()
        .all()
    )
    user_rows = {row.provider: row for row in rows}

    providers = available_providers(config)
    for entry in providers:
        name = entry["name"]
        row = user_rows.get(name)
        # Surface the resolved base_url/model/enabled so the UI can render the
        # local/openrouter inputs and the enabled toggle.
        creds = config.providers.get(name, ProviderCreds())
        entry["base_url"] = creds.base_url or None
        entry["enabled"] = creds.enabled
        user_key: str | None = None
        if row is not None and row.api_key_encrypted:
            try:
                user_key = decrypt_field(row.api_key_encrypted)
            except Exception:  # noqa: BLE001 - never surface key material
                logger.warning("failed to decrypt stored key for provider %s", name)
                user_key = None
        if user_key:
            entry["has_key"] = True
            entry["source"] = "user"
            entry["key_masked"] = _mask_key(user_key)
        else:
            env_key = base.providers.get(name, ProviderCreds()).api_key
            entry["has_key"] = False
            entry["source"] = "env" if env_key else "none"
            entry["key_masked"] = _mask_key(env_key) if env_key else None

    routing = dict(config.routing)
    pref = (
        await db.execute(
            select(UserLLMPreferences).where(UserLLMPreferences.user_id == user_id)
        )
    ).scalar_one_or_none()
    routing["extraction_engine"] = pref.extraction_engine if pref else None
    routing["processing_mode"] = config.processing_mode.value

    await log_audit_event(
        db,
        user_id=user_id,
        action="llm_settings.get",
        resource_type="llm_settings",
        ip_address=request.client.host if request.client else None,
    )

    return {"providers": providers, "routing": routing}


@router.put("/providers/{name}")
async def update_provider(
    name: str,
    body: ProviderUpdate,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Upsert the user's config for a single provider (API key encrypted at rest)."""
    if name not in KNOWN_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown provider")
    if name in {"ollama", "lmstudio"} and body.base_url is not None:
        try:
            require_loopback(body.base_url)
        except LocalPolicyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    row = (
        await db.execute(
            select(LLMProviderConfig).where(
                LLMProviderConfig.user_id == user_id,
                LLMProviderConfig.provider == name,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        row = LLMProviderConfig(user_id=user_id, provider=name)
        db.add(row)

    # Only a non-empty key overwrites the stored ciphertext; omitting/None/empty
    # leaves any existing key intact.
    if body.api_key:
        row.api_key_encrypted = encrypt_field(body.api_key)
    if body.base_url is not None:
        row.base_url = body.base_url
    if body.model is not None:
        row.model = body.model
    if body.enabled is not None:
        row.enabled = body.enabled

    try:
        await db.flush()
        await load_llm_config(db, user_id)
    except LocalPolicyError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="llm_settings.provider.update",
        resource_type="llm_provider_config",
        ip_address=request.client.host if request.client else None,
        details={"provider": name},
    )

    return {"ok": True, "provider": name}


@router.delete("/providers/{name}")
async def delete_provider(
    name: str,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Delete the user's row for a provider (revert to the global ``.env`` config)."""
    if name not in KNOWN_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown provider")

    row = (
        await db.execute(
            select(LLMProviderConfig).where(
                LLMProviderConfig.user_id == user_id,
                LLMProviderConfig.provider == name,
            )
        )
    ).scalar_one_or_none()
    if row is not None:
        await db.delete(row)
    try:
        await db.flush()
        await load_llm_config(db, user_id)
    except LocalPolicyError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="llm_settings.provider.delete",
        resource_type="llm_provider_config",
        ip_address=request.client.host if request.client else None,
        details={"provider": name},
    )

    return {"ok": True, "provider": name}


@router.put("/routing")
async def update_routing(
    body: RoutingUpdate,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Upsert the user's routing preferences (one row per user)."""
    pref = (
        await db.execute(
            select(UserLLMPreferences).where(UserLLMPreferences.user_id == user_id)
        )
    ).scalar_one_or_none()
    if pref is None:
        pref = UserLLMPreferences(user_id=user_id)
        db.add(pref)

    updated: list[str] = []
    for field_name, column in _ROUTING_FIELDS.items():
        value = getattr(body, field_name)
        if value is not None:
            setattr(pref, column, value)
            updated.append(field_name)

    try:
        await db.flush()
        await load_llm_config(db, user_id)
    except LocalPolicyError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="llm_settings.routing.update",
        resource_type="user_llm_preferences",
        ip_address=request.client.host if request.client else None,
        details={"fields": updated},
    )

    return {"ok": True}


@router.post("/providers/{name}/test")
async def test_provider(
    name: str,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Run one tiny live completion to validate a provider's credentials.

    Returns ``{ok: True, model}`` on success, or ``{ok: False, error_type}``
    where ``error_type`` is ``rate_limit`` / ``auth`` / ``error``. Never echoes
    the key or any internal error detail.
    """
    if name not in KNOWN_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown provider")

    config = await load_llm_config(db, user_id)

    await log_audit_event(
        db,
        user_id=user_id,
        action="llm_settings.provider.test",
        resource_type="llm_provider_config",
        ip_address=request.client.host if request.client else None,
        details={"provider": name},
    )

    try:
        provider = registry._build(name, config)
        resp = await provider.complete(
            LLMRequest(
                messages=[LLMMessage("user", "Reply: ok")],
                model="",
                max_output_tokens=16,
                temperature=0.0,
            )
        )
    except LLMRateLimitError:
        return {"ok": False, "error_type": "rate_limit"}
    except LLMAuthError:
        return {"ok": False, "error_type": "auth"}
    except LLMError:
        return {"ok": False, "error_type": "error"}
    except Exception:  # noqa: BLE001 - never surface internals/keys to the client
        logger.warning("provider connection test failed for %s", name)
        return {"ok": False, "error_type": "error"}

    return {"ok": True, "model": resp.model}
