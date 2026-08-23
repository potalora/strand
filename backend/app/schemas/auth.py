from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


def normalize_login_identifier(value: str) -> str:
    trimmed = value.strip()
    if not 1 <= len(trimmed) <= 255 or not trimmed.isprintable():
        raise ValueError("invalid login identifier")
    return trimmed


class _LoginIdentifierRequest(BaseModel):
    login_identifier: str = Field(
        validation_alias=AliasChoices("login_identifier", "email")
    )

    @model_validator(mode="before")
    @classmethod
    def validate_matching_aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if "login_identifier" in value and "email" in value:
            canonical = value["login_identifier"]
            legacy = value["email"]
            if not isinstance(canonical, str) or not isinstance(legacy, str):
                raise ValueError("invalid login identifier aliases")
            if canonical.strip().lower() != legacy.strip().lower():
                raise ValueError("invalid login identifier aliases")
        return value

    @field_validator("login_identifier")
    @classmethod
    def validate_login_identifier(cls, value: str) -> str:
        return normalize_login_identifier(value)


class RegisterRequest(_LoginIdentifierRequest):
    password: str = Field(..., min_length=8, max_length=128)
    display_name: str | None = None

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        """Require uppercase, lowercase, digit, and special character."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;':\",./<>?\\`~]", v):
            raise ValueError("Password must contain at least one special character")
        return v


class LoginRequest(_LoginIdentifierRequest):
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    """Optional logout body — supplying the refresh token revokes it server-side."""

    refresh_token: str | None = None


class UserResponse(BaseModel):
    id: UUID
    login_identifier: str
    email: str = Field(deprecated=True)
    display_name: str | None
    is_active: bool
    created_at: datetime

    @classmethod
    def from_user(cls, user: Any) -> Self:
        return cls(
            id=user.id,
            login_identifier=user.login_identifier,
            email=user.login_identifier,
            display_name=user.display_name,
            is_active=user.is_active,
            created_at=user.created_at,
        )
