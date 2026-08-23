from __future__ import annotations

import json

import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from app.middleware.encryption import blind_index
from app.schemas.auth import LoginRequest, RegisterRequest


def test_identifier_normalization_is_exact_lower_not_unicode_caseless() -> None:
    assert RegisterRequest.model_validate(
        {"login_identifier": "  Alice  ", "password": "SecurePass123!"}
    ).login_identifier == "Alice"
    assert blind_index("Alice") == blind_index("alice")
    assert blind_index("Straße") != blind_index("STRASSE")
    assert blind_index("Å") != blind_index("A\u030a")


def test_matching_legacy_alias_is_accepted() -> None:
    request = LoginRequest.model_validate(
        {
            "login_identifier": " Existing@Example.com ",
            "email": "existing@example.com",
            "password": "SecurePass123!",
        }
    )
    assert request.login_identifier == "Existing@Example.com"


@pytest.mark.parametrize(
    ("request_type", "password"),
    [
        (RegisterRequest, "SecurePass123!"),
        (LoginRequest, "any-password"),
    ],
)
def test_explicit_null_legacy_alias_is_rejected_by_schema(
    request_type: type[RegisterRequest] | type[LoginRequest], password: str
) -> None:
    with pytest.raises(ValidationError):
        request_type.model_validate(
            {
                "login_identifier": "schema-identifier-sentinel-null-alias",
                "email": None,
                "password": password,
            }
        )


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("  two words  ", "two words"),
        (" printable-😀 ", "printable-😀"),
        ("x" * 255, "x" * 255),
    ],
)
def test_printable_identifier_boundaries(identifier: str, expected: str) -> None:
    request = RegisterRequest.model_validate(
        {"login_identifier": identifier, "password": "SecurePass123!"}
    )
    assert request.login_identifier == expected


@pytest.mark.parametrize("identifier", ["   ", "x" * 256, "line\nbreak"])
def test_invalid_identifier_boundaries(identifier: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RegisterRequest.model_validate(
            {"login_identifier": identifier, "password": "SecurePass123!"}
        )
    assert exc_info.value.errors()[0]["loc"] == ("login_identifier",)


def _assert_content_free_response(
    response_content: bytes,
    response_json: dict[str, object],
    sentinels: list[str],
    captured_logs: str,
) -> None:
    assert response_content == b'{"detail":"Invalid authentication request."}'
    assert response_json == {"detail": "Invalid authentication request."}
    assert set(response_json) == {"detail"}
    serialized = response_content.decode("utf-8")
    for sentinel in sentinels:
        assert sentinel not in serialized
        assert sentinel.strip().lower() not in serialized
        assert blind_index(sentinel) not in serialized
        assert sentinel not in captured_logs
    for framework_key in ("input", "ctx", "loc"):
        assert framework_key not in serialized
    assert json.loads(serialized) == {"detail": "Invalid authentication request."}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {
            "login_identifier": "identifier-sentinel-conflict",
            "email": "alias-sentinel-conflict",
            "password": "PasswordSentinel1!",
        },
        {
            "login_identifier": "identifier\nsentinel-control",
            "password": "PasswordSentinel2!",
        },
        {
            "login_identifier": ["identifier-sentinel-type"],
            "password": "PasswordSentinel3!",
        },
        {
            "login_identifier": "identifier-sentinel-password",
            "password": "password-sentinel-no-complexity",
        },
        {
            "login_identifier": "register-identifier-sentinel-null-alias",
            "email": None,
            "password": "RegisterPasswordSentinelNull1!",
        },
    ],
)
async def test_register_422_is_complete_and_content_free(
    client: AsyncClient, payload: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 422
    sentinels = [
        value for value in payload.values() if isinstance(value, str)
    ] + ["identifier-sentinel-type"]
    _assert_content_free_response(
        response.content, response.json(), sentinels, caplog.text
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {
            "login_identifier": "login-identifier-sentinel-conflict",
            "email": "login-alias-sentinel-conflict",
            "password": "LoginPasswordSentinel1!",
        },
        {
            "login_identifier": ["login-identifier-sentinel-type"],
            "password": "LoginPasswordSentinel2!",
        },
        {
            "login_identifier": "login-identifier-sentinel-password",
            "password": ["login-password-sentinel-type"],
        },
        {
            "login_identifier": "login-identifier-sentinel-null-alias",
            "email": None,
            "password": "LoginPasswordSentinelNull1!",
        },
    ],
)
async def test_login_422_is_complete_and_content_free(
    client: AsyncClient, payload: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    response = await client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 422
    sentinels = [
        value for value in payload.values() if isinstance(value, str)
    ] + ["login-identifier-sentinel-type", "login-password-sentinel-type"]
    _assert_content_free_response(
        response.content, response.json(), sentinels, caplog.text
    )


@pytest.mark.asyncio
async def test_login_malformed_json_422_is_complete_and_content_free(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    identifier_sentinel = "login-identifier-sentinel-malformed-json"
    password_sentinel = "login-password-sentinel-malformed-json"
    malformed_json = (
        f'{{"login_identifier":"{identifier_sentinel}",'
        f'"password":"{password_sentinel}"'
    )

    response = await client.post(
        "/api/v1/auth/login",
        content=malformed_json,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    _assert_content_free_response(
        response.content,
        response.json(),
        [identifier_sentinel, password_sentinel],
        caplog.text,
    )


@pytest.mark.asyncio
async def test_refresh_keeps_default_validation_shape(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/refresh", json={})
    assert response.status_code == 422
    assert response.json() != {"detail": "Invalid authentication request."}
    assert isinstance(response.json()["detail"], list)
    assert response.json()["detail"][0]["loc"] == ["body", "refresh_token"]
