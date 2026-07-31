from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ai.llm.config import LLMConfig, ProviderCreds
from app.services.ai.llm.types import LLMRequest, LLMResponse, LLMUsage
from app.services.local_ai.types import ProcessingMode
from tests.conftest import auth_headers, create_test_patient, seed_test_records


def _cfg(default: str, providers: dict[str, ProviderCreds]) -> LLMConfig:
    """Build an LLMConfig routing every operation to ``default``."""
    routing = {"default": default}
    for op in ("summary", "section", "dedup", "extraction", "vision"):
        routing[op] = default
    return LLMConfig(routing=routing, providers=providers)


def _resp(model: str = "m", text: str = "{}") -> LLMResponse:
    return LLMResponse(
        text=text, finish_reason="stop", model=model, usage=LLMUsage(1, 1, 2), raw=None
    )


def _grounded_resp(request: LLMRequest, model: str) -> LLMResponse:
    content = request.messages[0].content
    assert isinstance(content, str)
    transport = json.loads(content.rsplit("INPUT_JSON=", 1)[1])
    fact = transport["facts"][0]
    evidence_by_id = {item["evidence_id"]: item for item in transport["evidence"]}
    evidence_id = fact["evidence_ids"][0]
    output = {
        "sections": [
            {
                "heading": "Overview",
                "claims": [
                    {
                        "fact_id": fact["fact_id"],
                        "field_paths": evidence_by_id[evidence_id]["field_paths"],
                        "evidence_ids": [evidence_id],
                    }
                ],
            }
        ],
        "uncertainties": [],
    }
    return _resp(model=model, text=json.dumps(output))


@pytest.mark.asyncio
async def test_summary_loads_user_config(client: AsyncClient, db_session: AsyncSession):
    """``generate_summary`` loads the per-user config and routes the summary
    provider through ``get_provider("summary", config)`` (no explicit provider).
    """
    from app.services.ai import summarizer

    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)
    await seed_test_records(db_session, user_id, patient.id, count=3)

    fake_provider = AsyncMock()
    fake_provider.complete.side_effect = lambda request: _grounded_resp(
        request,
        "claude-x",
    )

    cfg = _cfg("anthropic", {"anthropic": ProviderCreds(api_key="k", model="claude-x")})

    async def fake_load(db, uid):
        return cfg

    # The summarizer's GEMINI guard fires for the no-explicit-provider path; give
    # it a key so the anthropic-routed call is exercised regardless of env.
    with (
        patch.object(summarizer.settings, "gemini_api_key", "test-key"),
        patch.object(summarizer, "load_llm_config", fake_load),
        patch.object(
            summarizer, "get_provider", return_value=fake_provider
        ) as mock_get_provider,
    ):
        out = await summarizer.generate_summary(
            db_session,
            UUID(user_id),
            patient.id,
            processing_mode=ProcessingMode.CLOUD_ASSISTED,
        )

        assert out["model_used"] == "claude-x"
        # get_provider was called WITH the operation key and the loaded config object.
        args = mock_get_provider.call_args.args
        assert args[0] == "summary"
        assert args[1] is cfg


@pytest.mark.asyncio
async def test_section_parser_threads_config():
    """``parse_sections(config=...)`` forwards that exact config to get_provider."""
    from app.services.extraction import section_parser

    payload = {
        "document_type": "note",
        "primary_visit_date": None,
        "provider": None,
        "facility": None,
        "sections": [{"type": "medications", "anchor": "MEDICATIONS"}],
    }
    prov = AsyncMock()
    prov.complete.return_value = _resp(text=json.dumps(payload))
    cfg = _cfg("gemini", {"gemini": ProviderCreds(api_key="k")})

    with patch.object(section_parser, "get_provider", return_value=prov) as gp:
        await section_parser.parse_sections(
            "MEDICATIONS\nlisinopril 10mg", api_key="unused", config=cfg
        )

    assert gp.call_args.args == ("section", cfg)


@pytest.mark.asyncio
async def test_generic_extract_threads_config():
    """``generic_extract_entities_async(config=...)`` forwards it to get_provider."""
    from app.services.extraction import generic_entity_extractor as gen

    prov = AsyncMock()
    prov.complete.return_value = _resp(text=json.dumps({"entities": []}))
    cfg = _cfg("openai", {"openai": ProviderCreds(api_key="k", base_url="http://x/v1")})

    with patch.object(gen, "get_provider", return_value=prov) as gp:
        await gen.generic_extract_entities_async("text", "f.txt", config=cfg)

    assert gp.call_args.args == ("extraction", cfg)


@pytest.mark.asyncio
async def test_generic_extract_back_compat_uses_settings():
    """With ``config=None`` the call falls back to ``LLMConfig.from_settings()``."""
    from app.services.extraction import generic_entity_extractor as gen

    prov = AsyncMock()
    prov.complete.return_value = _resp(text=json.dumps({"entities": []}))

    with patch.object(gen, "get_provider", return_value=prov) as gp:
        await gen.generic_extract_entities_async("text", "f.txt")

    args = gp.call_args.args
    assert args[0] == "extraction"
    assert isinstance(args[1], LLMConfig)  # synthesized from_settings, not None


@pytest.mark.asyncio
async def test_judge_threads_config():
    """``judge_candidate_pair(config=...)`` forwards that config to get_provider."""
    from app.services.dedup import llm_judge

    payload = {
        "classification": "duplicate",
        "confidence": 0.9,
        "explanation": "same",
        "field_diff": None,
    }
    prov = AsyncMock()
    prov.complete.return_value = _resp(text=json.dumps(payload))
    cfg = _cfg("anthropic", {"anthropic": ProviderCreds(api_key="k")})

    with patch.object(llm_judge, "get_provider", return_value=prov) as gp:
        out = await llm_judge.judge_candidate_pair(
            {"resourceType": "Condition"},
            {"resourceType": "Condition"},
            "condition",
            api_key="unused",
            config=cfg,
        )

    assert out.classification == "duplicate"
    assert gp.call_args.args == ("dedup", cfg)


@pytest.mark.asyncio
async def test_judge_batch_threads_config_into_pairs():
    """``judge_candidates_batch(config=...)`` threads the config down to each pair."""
    from app.services.dedup import llm_judge

    cfg = _cfg("anthropic", {"anthropic": ProviderCreds(api_key="k")})
    pairs = [
        ({"resourceType": "Condition"}, {"resourceType": "Condition"}, "condition")
    ]

    async def fake_pair(fhir_a, fhir_b, record_type, api_key, config=None):
        assert config is cfg
        return llm_judge.JudgmentResult.error_fallback()

    with patch.object(llm_judge, "judge_candidate_pair", fake_pair):
        results = await llm_judge.judge_candidates_batch(pairs, "unused", config=cfg)

    assert len(results) == 1
