"""Fail-closed routing policy for local AI processing modes."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlparse

from app.services.local_ai.errors import LocalPolicyError
from app.services.local_ai.types import ProcessingMode


def require_loopback(endpoint: str) -> None:
    """Require an explicit HTTP(S) endpoint on a loopback interface."""
    if not isinstance(endpoint, str):
        raise LocalPolicyError(
            "Custom local endpoint must be a valid HTTP loopback URL"
        )
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise LocalPolicyError(
            "Custom local endpoint must be a valid HTTP loopback URL"
        ) from exc
    if parsed.scheme not in {"http", "https"} or not host:
        raise LocalPolicyError("Custom local endpoint must be an HTTP loopback URL")
    if host == "localhost":
        return
    try:
        if ip_address(host).is_loopback:
            return
    except ValueError as exc:
        raise LocalPolicyError(
            "Custom local endpoint must resolve to loopback"
        ) from exc
    raise LocalPolicyError("Custom local endpoint must use loopback")


def assert_processing_route(mode: ProcessingMode, endpoint: str | None) -> None:
    """Reject routes that violate the explicitly selected processing mode."""
    if mode is ProcessingMode.VALIDATED_STRICT_LOCAL:
        if endpoint is not None:
            raise LocalPolicyError("Validated local jobs use only the embedded worker")
        return
    if mode is ProcessingMode.CUSTOM_LOCAL:
        if endpoint is None:
            raise LocalPolicyError("Custom local mode requires a loopback endpoint")
        require_loopback(endpoint)
        return
    if mode is ProcessingMode.CLOUD_ASSISTED:
        return
    if mode is ProcessingMode.PROMPT_ONLY:
        if endpoint is not None:
            raise LocalPolicyError("Prompt-only mode cannot call a provider")
        return
    raise LocalPolicyError("Unsupported processing mode")
