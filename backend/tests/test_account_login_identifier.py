from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from asyncpg.exceptions import UniqueViolationError
from httpx import AsyncClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.encryption import blind_index
from app.middleware.rate_limit import login_limiter
from app.models.audit import AuditLog
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest, UserResponse


def test_identifier_normalization_is_exact_lower_not_unicode_caseless() -> None:
    assert (
        RegisterRequest.model_validate(
            {"login_identifier": "  Alice  ", "password": "SecurePass123!"}
        ).login_identifier
        == "Alice"
    )
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
        for sensitive_value in (
            sentinel,
            sentinel.strip().lower(),
            blind_index(sentinel),
        ):
            assert sensitive_value not in serialized
            assert sensitive_value not in captured_logs
    for framework_key in ("input", "ctx", "loc"):
        assert framework_key not in serialized
        assert framework_key not in captured_logs
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
    sentinels = [value for value in payload.values() if isinstance(value, str)] + [
        "identifier-sentinel-type"
    ]
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
    sentinels = [value for value in payload.values() if isinstance(value, str)] + [
        "login-identifier-sentinel-type",
        "login-password-sentinel-type",
    ]
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
        f'{{"login_identifier":"{identifier_sentinel}","password":"{password_sentinel}"'
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


def test_user_model_and_response_expose_canonical_identifier_only() -> None:
    assert "login_identifier" in User.__mapper__.attrs
    assert "login_identifier_hmac" in User.__mapper__.attrs
    assert "email" not in User.__mapper__.attrs
    assert "email_hmac" not in User.__mapper__.attrs
    response_fields = UserResponse.model_json_schema()["properties"]
    assert response_fields["email"]["deprecated"] is True


@pytest.mark.asyncio
async def test_non_email_registration_login_and_response_alias(
    client: AsyncClient,
) -> None:
    payload = {
        "login_identifier": "  Pedro Health  ",
        "password": "SecurePass123!",
        "display_name": "Pedro",
    }
    registered = await client.post("/api/v1/auth/register", json=payload)
    assert registered.status_code == 201
    assert registered.json()["login_identifier"] == "Pedro Health"
    assert registered.json()["email"] == "Pedro Health"

    login = await client.post(
        "/api/v1/auth/login",
        json={
            "login_identifier": "pedro health",
            "password": "SecurePass123!",
        },
    )
    assert login.status_code == 200
    assert login.json()["access_token"]


@pytest.mark.asyncio
async def test_legacy_email_payload_and_existing_account_still_work(
    client: AsyncClient,
) -> None:
    legacy = {"email": "Legacy@Example.com", "password": "SecurePass123!"}
    assert (await client.post("/api/v1/auth/register", json=legacy)).status_code == 201
    assert (await client.post("/api/v1/auth/login", json=legacy)).status_code == 200


@pytest.mark.asyncio
async def test_duplicate_normalized_identifier_is_generic(client: AsyncClient) -> None:
    first = {"login_identifier": "Alice", "password": "SecurePass123!"}
    second = {"login_identifier": " alice ", "password": "SecurePass123!"}
    assert (await client.post("/api/v1/auth/register", json=first)).status_code == 201
    duplicate = await client.post("/api/v1/auth/register", json=second)
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "Account identifier is unavailable."}
    assert "alice" not in duplicate.text.lower()


@pytest.mark.asyncio
async def test_unicode_values_declared_distinct_can_both_register(
    client: AsyncClient,
) -> None:
    for identifier in ("Straße", "STRASSE", "Å", "A\u030a"):
        response = await client.post(
            "/api/v1/auth/register",
            json={"login_identifier": identifier, "password": "SecurePass123!"},
        )
        assert response.status_code == 201


@pytest.mark.asyncio
async def test_unknown_wrong_password_and_disabled_are_same_generic_401(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    identifier = "private account name"
    registered = await client.post(
        "/api/v1/auth/register",
        json={"login_identifier": identifier, "password": "SecurePass123!"},
    )
    assert registered.status_code == 201

    unknown = await client.post(
        "/api/v1/auth/login",
        json={
            "login_identifier": "unknown private account",
            "password": "SecurePass123!",
        },
    )
    wrong_password = await client.post(
        "/api/v1/auth/login",
        json={"login_identifier": identifier, "password": "WrongPass123!"},
    )

    user = await db_session.get(User, UUID(registered.json()["id"]))
    assert user is not None
    user.is_active = False
    await db_session.commit()
    disabled = await client.post(
        "/api/v1/auth/login",
        json={"login_identifier": identifier, "password": "SecurePass123!"},
    )

    expected = {"detail": "Invalid account identifier or password."}
    for response in (unknown, wrong_password, disabled):
        assert response.status_code == 401
        assert response.json() == expected
        assert identifier not in response.text


@pytest.mark.asyncio
async def test_active_lockout_response_remains_unchanged(client: AsyncClient) -> None:
    identifier = "active lockout account"
    assert (
        await client.post(
            "/api/v1/auth/register",
            json={"login_identifier": identifier, "password": "SecurePass123!"},
        )
    ).status_code == 201

    for _ in range(5):
        failed = await client.post(
            "/api/v1/auth/login",
            json={"login_identifier": identifier, "password": "WrongPass123!"},
        )
        assert failed.status_code == 401
        assert failed.json() == {"detail": "Invalid account identifier or password."}

    login_limiter._requests.clear()
    locked = await client.post(
        "/api/v1/auth/login",
        json={"login_identifier": identifier, "password": "SecurePass123!"},
    )
    assert locked.status_code == 401
    assert locked.json() == {
        "detail": "Account is temporarily locked. Please try again later."
    }


@pytest.mark.asyncio
async def test_expired_lockout_resets_before_next_failure(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    identifier = "expired lockout account"
    registered = await client.post(
        "/api/v1/auth/register",
        json={"login_identifier": identifier, "password": "SecurePass123!"},
    )
    assert registered.status_code == 201
    user_id = UUID(registered.json()["id"])
    user = await db_session.get(User, user_id)
    assert user is not None
    user.failed_login_attempts = 5
    user.locked_until = datetime.now(timezone.utc) - timedelta(minutes=1)
    user.last_failed_login_at = datetime.now(timezone.utc) - timedelta(minutes=20)
    await db_session.commit()

    failed = await client.post(
        "/api/v1/auth/login",
        json={"login_identifier": identifier, "password": "WrongPass123!"},
    )
    assert failed.status_code == 401
    assert failed.json() == {"detail": "Invalid account identifier or password."}

    db_session.expire_all()
    reloaded = await db_session.get(User, user_id)
    assert reloaded is not None
    assert reloaded.failed_login_attempts == 1
    assert reloaded.locked_until is None


def _wrapped_unique_violation(constraint_name: str) -> IntegrityError:
    driver_error = UniqueViolationError.new(
        {
            "C": "23505",
            "M": "synthetic unique violation",
            "n": constraint_name,
        }
    )
    dbapi_error = RuntimeError("synthetic asyncpg adapter error")
    dbapi_error.__cause__ = driver_error
    return IntegrityError("synthetic insert", {}, dbapi_error)


@pytest.mark.asyncio
async def test_commit_time_identifier_duplicate_is_generic_and_rolls_back(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rollback = db_session.rollback
    rollback = AsyncMock(wraps=original_rollback)
    commit = AsyncMock(
        side_effect=_wrapped_unique_violation("ix_users_login_identifier_hmac")
    )
    monkeypatch.setattr(db_session, "commit", commit)
    monkeypatch.setattr(db_session, "rollback", rollback)

    response = await client.post(
        "/api/v1/auth/register",
        json={
            "login_identifier": "concurrent account sentinel",
            "password": "SecurePass123!",
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Account identifier is unavailable."}
    assert "concurrent account sentinel" not in response.text
    rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_commit_time_unrelated_integrity_error_is_reraised_after_rollback(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rollback = db_session.rollback
    rollback = AsyncMock(wraps=original_rollback)
    integrity_error = _wrapped_unique_violation("unrelated_constraint")
    monkeypatch.setattr(
        db_session,
        "commit",
        AsyncMock(side_effect=integrity_error),
    )
    monkeypatch.setattr(db_session, "rollback", rollback)

    with pytest.raises(IntegrityError) as exc_info:
        await client.post(
            "/api/v1/auth/register",
            json={
                "login_identifier": "unrelated integrity sentinel",
                "password": "SecurePass123!",
            },
        )

    assert exc_info.value is integrity_error
    rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_me_returns_canonical_and_deprecated_alias(client: AsyncClient) -> None:
    identifier = "profile account name"
    registered = await client.post(
        "/api/v1/auth/register",
        json={"login_identifier": identifier, "password": "SecurePass123!"},
    )
    assert registered.status_code == 201
    login = await client.post(
        "/api/v1/auth/login",
        json={"login_identifier": identifier, "password": "SecurePass123!"},
    )
    assert login.status_code == 200

    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    assert response.status_code == 200
    assert response.json()["login_identifier"] == identifier
    assert response.json()["email"] == identifier


@pytest.mark.asyncio
async def test_successful_login_audit_details_are_none(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    identifier = "audit private identifier"
    assert (
        await client.post(
            "/api/v1/auth/register",
            json={"login_identifier": identifier, "password": "SecurePass123!"},
        )
    ).status_code == 201
    assert (
        await client.post(
            "/api/v1/auth/login",
            json={"login_identifier": identifier, "password": "SecurePass123!"},
        )
    ).status_code == 200

    login_logs = (
        (
            await db_session.execute(
                select(AuditLog).where(AuditLog.action == "user.login")
            )
        )
        .scalars()
        .all()
    )
    assert login_logs
    assert all(log.details is None for log in login_logs)
    assert identifier not in json.dumps(
        [{"action": log.action, "details": log.details} for log in login_logs]
    )
