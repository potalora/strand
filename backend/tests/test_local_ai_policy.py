from __future__ import annotations

from typing import cast

import pytest

from app.services.local_ai.errors import LocalPolicyError
from app.services.local_ai.policy import assert_processing_route, require_loopback
from app.services.local_ai.types import ProcessingMode


def test_strict_local_rejects_every_network_endpoint() -> None:
    with pytest.raises(LocalPolicyError, match="embedded worker"):
        assert_processing_route(
            ProcessingMode.VALIDATED_STRICT_LOCAL,
            "https://generativelanguage.googleapis.com",
        )


def test_strict_local_requires_the_embedded_worker_route() -> None:
    assert_processing_route(ProcessingMode.VALIDATED_STRICT_LOCAL, None)


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434/v1", "http://localhost:1234/v1", "http://[::1]:8000"],
)
def test_custom_local_accepts_only_loopback(url: str) -> None:
    require_loopback(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://192.168.1.9:11434",
        "ftp://127.0.0.1:11434",
        "http://0.0.0.0:11434",
        "http://localhost.evil.example:11434",
        "http:///v1",
    ],
)
def test_custom_local_rejects_non_loopback_or_invalid_urls(url: str) -> None:
    with pytest.raises(LocalPolicyError, match="loopback"):
        require_loopback(url)


def test_custom_local_requires_an_explicit_loopback_endpoint() -> None:
    with pytest.raises(LocalPolicyError, match="requires a loopback endpoint"):
        assert_processing_route(ProcessingMode.CUSTOM_LOCAL, None)


def test_prompt_only_forbids_a_provider_endpoint() -> None:
    with pytest.raises(LocalPolicyError, match="cannot call a provider"):
        assert_processing_route(ProcessingMode.PROMPT_ONLY, "http://127.0.0.1:11434")


def test_cloud_assisted_permits_explicit_cloud_route() -> None:
    assert_processing_route(
        ProcessingMode.CLOUD_ASSISTED,
        "https://generativelanguage.googleapis.com",
    )


def test_unknown_processing_mode_is_rejected_fail_closed() -> None:
    unknown_mode = cast(ProcessingMode, "unrecognized_mode")

    with pytest.raises(LocalPolicyError, match="Unsupported processing mode"):
        assert_processing_route(unknown_mode, None)


@pytest.mark.parametrize(
    "url",
    ["http://localhost:not-a-port", "http://[::1]:70000", "http://[::1"],
)
def test_custom_local_rejects_invalid_url_syntax_or_port_range(url: str) -> None:
    with pytest.raises(LocalPolicyError, match="valid HTTP loopback URL"):
        require_loopback(url)
