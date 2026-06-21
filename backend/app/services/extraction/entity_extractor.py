from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import langextract as lx

from app.config import settings
from app.services.extraction.clinical_examples import (
    CLINICAL_EXAMPLES,
    CLINICAL_EXTRACTION_PROMPT,
)

if TYPE_CHECKING:
    from app.services.ai.llm import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """A single entity extracted from clinical text."""

    entity_class: str
    text: str
    attributes: dict = field(default_factory=dict)
    start_pos: int | None = None
    end_pos: int | None = None
    confidence: float = 0.8


@dataclass
class ExtractionResult:
    """Result of entity extraction from a document."""

    source_file: str
    source_text: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    error: str | None = None


MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds

_TRANSIENT_ERROR_KEYWORDS = ("429", "resource_exhausted", "quota", "rate", "timeout", "connection")


def extract_entities(
    text: str,
    source_file: str,
    api_key: str,
    progress_callback: Callable[[str, int, int], None] | None = None,
    *,
    model_id: str | None = None,
    model_url: str | None = None,
) -> ExtractionResult:
    """Extract clinical entities from text using LangExtract.

    This is synchronous because LangExtract is synchronous.
    Use extract_entities_async() from async callers.

    LangExtract selects its backend from ``model_id`` (``^gemini`` → Gemini,
    ``^gpt-4``/``^gpt-5`` → OpenAI, ``^llama``/``^qwen``/… → Ollama). ``model_id``
    defaults to the Gemini extraction model. For Ollama pass ``model_url`` (the
    server base URL) instead of ``api_key``.

    ``progress_callback`` is optional and, when provided, is invoked once this
    chunk's extraction completes — ``(stage, completed_delta, entity_count)``.
    The worker uses it to advance section-level progress. Keeping it optional
    leaves existing callers/tests unaffected.
    """
    try:
        for attempt in range(MAX_RETRIES):
            try:
                lx_kwargs: dict = dict(
                    text_or_documents=text,
                    prompt_description=CLINICAL_EXTRACTION_PROMPT,
                    examples=CLINICAL_EXAMPLES,
                    model_id=model_id or settings.gemini_extraction_model,
                    max_char_buffer=2000,
                    max_workers=1,
                )
                # Ollama uses a local server URL (no key); others use the API key.
                if model_url:
                    lx_kwargs["model_url"] = model_url
                else:
                    lx_kwargs["api_key"] = api_key
                result = lx.extract(**lx_kwargs)
                break
            except Exception as e:
                error_str = str(e).lower()
                is_transient = any(k in error_str for k in _TRANSIENT_ERROR_KEYWORDS)
                if is_transient and attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Transient error on attempt %d for %s, retrying in %ds: %s",
                        attempt + 1, source_file, wait, e,
                    )
                    time.sleep(wait)
                    continue
                raise

        entities = []
        for extraction in result.extractions:
            start_pos = None
            end_pos = None
            if extraction.char_interval:
                start_pos = extraction.char_interval.start_pos
                end_pos = extraction.char_interval.end_pos

            attrs = extraction.attributes or {}
            confidence = 0.8  # default
            if "confidence" in attrs:
                try:
                    confidence = max(0.0, min(1.0, float(attrs["confidence"])))
                except (ValueError, TypeError):
                    pass

            entities.append(
                ExtractedEntity(
                    entity_class=extraction.extraction_class,
                    text=extraction.extraction_text,
                    attributes=attrs,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=confidence,
                )
            )

        logger.info(
            "Extracted %d entities from %s", len(entities), source_file
        )
        if progress_callback is not None:
            try:
                progress_callback("extracting_entities", 1, len(entities))
            except Exception:  # progress is best-effort, never fail extraction
                logger.debug("progress_callback raised; ignoring", exc_info=True)
        return ExtractionResult(
            source_file=source_file,
            source_text=text,
            entities=entities,
        )

    except Exception as e:
        logger.error("Entity extraction failed for %s: %s", source_file, e)
        return ExtractionResult(
            source_file=source_file,
            source_text=text,
            error=str(e),
        )


def _langextract_params(
    provider: str, config: LLMConfig | None, api_key: str
) -> tuple[str, str, str | None] | None:
    """Resolve ``(model_id, api_key, model_url)`` for a LangExtract-native provider.

    LangExtract ships native gemini/openai/ollama backends (routed by ``model_id``
    prefix), which give chunking + source-grounding the generic JSON path lacks.
    Returns ``None`` for providers with no native backend (anthropic, openrouter,
    lmstudio) so the caller falls back to the generic facade path.
    """
    creds = config.providers.get(provider) if config else None
    if provider in ("gemini", "vertex"):
        return (settings.gemini_extraction_model, api_key or settings.gemini_api_key, None)
    if provider == "openai":
        model = creds.model if creds and creds.model else settings.openai_model
        key = creds.api_key if creds and creds.api_key else settings.openai_api_key
        return (model, key, None)
    if provider == "ollama":
        model = creds.model if creds and creds.model else settings.ollama_model
        base = creds.base_url if creds and creds.base_url else settings.ollama_base_url
        return (model, "", base.removesuffix("/v1"))
    return None


async def extract_entities_async(
    text: str,
    source_file: str,
    api_key: str,
    progress_callback: Callable[[str, int, int], None] | None = None,
    config: LLMConfig | None = None,
) -> ExtractionResult:
    """Async entity extraction.

    Gemini/Vertex/OpenAI/Ollama route through LangExtract's native backend (better
    chunking + source-grounding); Anthropic/OpenRouter/LM Studio — which LangExtract
    has no native backend for — delegate to the provider-agnostic JSON extraction
    path. ``config`` carries the per-user resolved routing/credentials (``None`` =>
    global ``.env``). ``provider_name_for`` is imported lazily to avoid an import
    cycle with the LLM registry.
    """
    from app.services.ai.llm.registry import provider_name_for

    provider = provider_name_for("extraction", config)
    lx_params = _langextract_params(provider, config, api_key)
    if lx_params is not None:
        model_id, lx_key, model_url = lx_params
        return await asyncio.to_thread(
            extract_entities, text, source_file, lx_key, progress_callback,
            model_id=model_id, model_url=model_url,
        )
    from app.services.extraction.generic_entity_extractor import (
        generic_extract_entities_async,
    )

    return await generic_extract_entities_async(
        text, source_file, progress_callback, config=config
    )
