from __future__ import annotations

from pydantic import BaseModel

from app.services.local_ai.types import ProcessingMode


class ProviderUpdate(BaseModel):
    """Partial update for a single provider's per-user config.

    Any omitted field is left unchanged. A non-empty ``api_key`` is encrypted
    and stored; ``None``/omitted/empty leaves the existing stored key intact.
    """

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    enabled: bool | None = None


class RoutingUpdate(BaseModel):
    """Partial update for a user's LLM routing preferences.

    Each field names the provider to route that operation to; omitted fields
    are left unchanged. ``extraction_engine`` selects the local/hybrid/gemini
    extraction engine.
    """

    default: str | None = None
    summary: str | None = None
    section: str | None = None
    dedup: str | None = None
    extraction: str | None = None
    vision: str | None = None
    extraction_engine: str | None = None
    processing_mode: ProcessingMode | None = None
