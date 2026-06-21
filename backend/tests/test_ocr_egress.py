"""W12 — vision OCR minimum-necessary egress (HIPAA DEID-01/DEID-06, SEC-PHI-03).

Scanned-document OCR sends the raw (un-redacted) document bytes to a vision LLM.
Before W12 the fallback chain re-sent the SAME document to EVERY configured
provider in turn on refusal/error, so one upload could transmit a patient's full
document to multiple third-party cloud processors (anthropic -> openai -> gemini
-> openrouter -> vertex -> ...). That violates minimum-necessary.

New contract:
  * The user-configured vision provider is tried first.
  * A CLOUD provider that authenticated and then refused/errored on the content
    already received the document -> we STOP. The same document is NOT re-sent to
    a different cloud vendor.
  * A genuine capability gap (a LOCAL ollama/lmstudio model that can't do vision)
    keeps the document on-machine, so it MAY fall back exactly once to the
    documented Gemini cloud fallback.
  * Net effect: at most ONE third-party cloud provider receives the document
    (the chosen cloud provider, OR — for a local capability gap — the single
    Gemini fallback). Never the whole chain.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.ai.llm.config import LLMConfig
from app.services.ai.llm.types import LLMRateLimitError, LLMResponse, LLMUsage
from app.services.extraction import text_extractor

_CLOUD = {"gemini", "vertex", "anthropic", "openai", "openrouter"}


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, finish_reason="stop", model="m", usage=LLMUsage(1, 1, 2), raw=None)


def _provider(text: str = "", *, error: Exception | None = None) -> AsyncMock:
    """A mock vision provider that returns ``text`` (``""`` = refused/blocked) or raises."""
    m = AsyncMock()
    if error is not None:
        m.complete.side_effect = error
    else:
        m.complete.return_value = _resp(text)
    return m


def _received(m: AsyncMock) -> bool:
    """True iff this provider was actually sent the document (its complete() ran)."""
    return m.complete.await_count > 0


def _cloud_egress(candidates: list) -> int:
    """Count distinct CLOUD providers that received the document bytes."""
    return sum(1 for name, m in candidates if name in _CLOUD and _received(m))


# --- a CLOUD provider that refuses/errors must NOT fan out to other clouds ---


@pytest.mark.asyncio
async def test_cloud_refusal_does_not_fan_out_to_other_clouds(tmp_path):
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake")
    anthropic = _provider("")  # authenticated but blocked/empty on this content
    openai = _provider("OPENAI WOULD HAVE READ IT")
    gemini = _provider("GEMINI WOULD HAVE READ IT")
    candidates = [("anthropic", anthropic), ("openai", openai), ("gemini", gemini)]
    trace: list = []
    import unittest.mock as _m
    with _m.patch.object(text_extractor, "_vision_candidates", return_value=candidates):
        out = await text_extractor.extract_text_from_tiff(
            tiff, api_key="k", config=LLMConfig.from_settings(), trace=trace)

    assert out == ""  # the chosen provider refused; we do NOT recover via another cloud
    assert _received(anthropic)
    assert not _received(openai), "document must NOT be re-sent to a 2nd cloud vendor"
    assert not _received(gemini), "document must NOT be re-sent to a 3rd cloud vendor"
    assert _cloud_egress(candidates) == 1
    # Trace still reflects the single refusal -> an unreadable warning notice.
    assert trace == [{"provider": "anthropic", "status": "refused"}]
    assert text_extractor.build_ocr_notice(trace)["type"] == "ocr_unreadable"


@pytest.mark.asyncio
async def test_cloud_error_does_not_fan_out_to_other_clouds(tmp_path):
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake")
    openai = _provider(error=LLMRateLimitError("429"))  # authenticated; errored on content
    anthropic = _provider("ANTHROPIC WOULD HAVE READ IT")
    gemini = _provider("GEMINI WOULD HAVE READ IT")
    candidates = [("openai", openai), ("anthropic", anthropic), ("gemini", gemini)]
    import unittest.mock as _m
    with _m.patch.object(text_extractor, "_vision_candidates", return_value=candidates):
        with pytest.raises(LLMRateLimitError):
            await text_extractor.extract_text_from_tiff(tiff, api_key="k")

    assert _received(openai)
    assert not _received(anthropic), "an errored cloud provider must NOT re-send to another cloud"
    assert not _received(gemini)
    assert _cloud_egress(candidates) == 1


@pytest.mark.asyncio
async def test_total_cloud_egress_is_bounded_to_at_most_two(tmp_path):
    """The headline invariant: even with a long configured chain, a refusal never
    fans the document across more than the bounded set of cloud processors."""
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake")
    candidates = [
        ("anthropic", _provider("")),
        ("openai", _provider("")),
        ("gemini", _provider("")),
        ("openrouter", _provider("")),
        ("vertex", _provider("")),
    ]
    import unittest.mock as _m
    with _m.patch.object(text_extractor, "_vision_candidates", return_value=candidates):
        out = await text_extractor.extract_text_from_tiff(tiff, api_key="k")
    assert out == ""
    assert _cloud_egress(candidates) <= 2
    assert _cloud_egress(candidates) == 1  # chosen cloud refused -> no cloud fallback


@pytest.mark.asyncio
async def test_chosen_cloud_success_sends_to_only_one_provider(tmp_path):
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake")
    anthropic = _provider("READABLE OCR TEXT")
    openai = _provider("SHOULD NOT BE CALLED")
    candidates = [("anthropic", anthropic), ("openai", openai)]
    import unittest.mock as _m
    with _m.patch.object(text_extractor, "_vision_candidates", return_value=candidates):
        out = await text_extractor.extract_text_from_tiff(tiff, api_key="k")
    assert out == "READABLE OCR TEXT"
    assert _received(anthropic) and not _received(openai)
    assert _cloud_egress(candidates) == 1


# --- a genuine capability gap (local model) DOES fall back once to Gemini ---


@pytest.mark.asyncio
async def test_capability_gap_local_provider_falls_back_once_to_gemini(tmp_path):
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake")
    ollama = _provider("")  # on-machine model that can't do vision -> capability gap
    gemini = _provider("GEMINI OCR")
    anthropic = _provider("SHOULD NOT BE CALLED")
    candidates = [("ollama", ollama), ("gemini", gemini), ("anthropic", anthropic)]
    trace: list = []
    import unittest.mock as _m
    with _m.patch.object(text_extractor, "_vision_candidates", return_value=candidates):
        out = await text_extractor.extract_text_from_tiff(
            tiff, api_key="k", config=LLMConfig.from_settings(), trace=trace)

    assert out == "GEMINI OCR"
    assert _received(ollama)  # tried on-machine first (no third-party egress)
    assert _received(gemini)  # the single documented capability fallback
    assert not _received(anthropic), "capability fallback must hit AT MOST one cloud (Gemini)"
    assert _cloud_egress(candidates) == 1  # only Gemini, not the rest of the chain
    assert trace == [
        {"provider": "ollama", "status": "refused"},
        {"provider": "gemini", "status": "ok"},
    ]
    assert text_extractor.build_ocr_notice(trace)["type"] == "ocr_fallback"


@pytest.mark.asyncio
async def test_capability_gap_local_error_falls_back_once_to_gemini(tmp_path):
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake")
    ollama = _provider(error=LLMRateLimitError("local server unreachable"))
    gemini = _provider("GEMINI OCR")
    candidates = [("ollama", ollama), ("gemini", gemini)]
    import unittest.mock as _m
    with _m.patch.object(text_extractor, "_vision_candidates", return_value=candidates):
        out = await text_extractor.extract_text_from_tiff(tiff, api_key="k")
    assert out == "GEMINI OCR"
    assert _received(ollama) and _received(gemini)
    assert _cloud_egress(candidates) == 1


@pytest.mark.asyncio
async def test_capability_gap_fallback_is_bounded_to_one_cloud(tmp_path):
    """Even a local capability gap fans out to AT MOST the single Gemini fallback."""
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake")
    candidates = [
        ("ollama", _provider("")),       # local capability gap
        ("gemini", _provider("")),       # documented fallback also blocks
        ("anthropic", _provider("")),    # must NOT be reached
        ("openai", _provider("")),       # must NOT be reached
    ]
    import unittest.mock as _m
    with _m.patch.object(text_extractor, "_vision_candidates", return_value=candidates):
        out = await text_extractor.extract_text_from_tiff(tiff, api_key="k")
    assert out == ""
    assert _cloud_egress(candidates) <= 2
    assert _cloud_egress(candidates) == 1  # only Gemini got it; anthropic/openai did not
    names_received = {n for n, m in candidates if _received(m)}
    assert names_received == {"ollama", "gemini"}
