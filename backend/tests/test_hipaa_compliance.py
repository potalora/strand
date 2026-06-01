"""HIPAA Compliance Tests — verifying all audit findings are remediated."""
from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient

from tests.conftest import auth_headers, create_test_patient, seed_test_records


# ===========================================================================
# C1: Token revocation on logout
# ===========================================================================


@pytest.mark.asyncio
async def test_revoked_token_rejected_after_logout(client: AsyncClient):
    """After logout, the same token should be rejected (401)."""
    headers, user_id = await auth_headers(client, "revoke@test.com")
    token = headers["Authorization"].split(" ")[1]

    # Logout
    resp = await client.post("/api/v1/auth/logout", headers=headers)
    assert resp.status_code == 204

    # Attempt to use the same token
    resp = await client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_revoked_after_use(client: AsyncClient):
    """After refreshing, the old refresh token should be revoked."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "refresh-revoke@test.com", "password": "SecurePass123!"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "refresh-revoke@test.com", "password": "SecurePass123!"},
    )
    old_refresh = login.json()["refresh_token"]

    # First refresh should succeed
    resp = await client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 200

    # Second use of same refresh token should fail (revoked)
    resp = await client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_jwt_contains_jti(client: AsyncClient):
    """Tokens should contain a JTI claim for revocation tracking."""
    from app.middleware.auth import decode_token

    await client.post(
        "/api/v1/auth/register",
        json={"email": "jti@test.com", "password": "SecurePass123!"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "jti@test.com", "password": "SecurePass123!"},
    )
    data = login.json()
    access_payload = decode_token(data["access_token"])
    refresh_payload = decode_token(data["refresh_token"])

    assert "jti" in access_payload
    assert "jti" in refresh_payload
    assert access_payload["jti"] != refresh_payload["jti"]


# ===========================================================================
# C2: Rate limiting + account lockout
# ===========================================================================


@pytest.mark.asyncio
async def test_login_rate_limiting(client: AsyncClient):
    """Login endpoint should reject after too many rapid requests."""
    from app.middleware.rate_limit import login_limiter

    # Reset the limiter state for this test
    login_limiter._requests.clear()

    await client.post(
        "/api/v1/auth/register",
        json={"email": "ratelimit@test.com", "password": "SecurePass123!"},
    )

    for i in range(login_limiter.max_requests):
        await client.post(
            "/api/v1/auth/login",
            json={"email": "ratelimit@test.com", "password": "SecurePass123!"},
        )

    # Next request should be rate limited
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "ratelimit@test.com", "password": "SecurePass123!"},
    )
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_account_lockout_after_failed_attempts(client: AsyncClient):
    """Account should lock after 5 failed login attempts."""
    from app.middleware.rate_limit import login_limiter
    login_limiter._requests.clear()

    await client.post(
        "/api/v1/auth/register",
        json={"email": "lockout@test.com", "password": "SecurePass123!"},
    )

    # 5 failed attempts
    for _ in range(5):
        await client.post(
            "/api/v1/auth/login",
            json={"email": "lockout@test.com", "password": "WrongPass1!"},
        )

    login_limiter._requests.clear()  # Clear rate limit to isolate lockout test

    # Even correct password should fail now (account locked)
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "lockout@test.com", "password": "SecurePass123!"},
    )
    assert resp.status_code == 401
    assert "locked" in resp.json()["detail"].lower()


# ===========================================================================
# C7: Security headers
# ===========================================================================


@pytest.mark.asyncio
async def test_security_headers_present(client: AsyncClient):
    """All required security headers should be present on responses."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert resp.headers.get("Cache-Control") == "no-store"
    assert "Content-Security-Policy" in resp.headers


# ===========================================================================
# C8: CORS restriction
# ===========================================================================


@pytest.mark.asyncio
async def test_cors_not_wildcard(client: AsyncClient):
    """CORS should not use wildcard methods."""
    resp = await client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    allowed_methods = resp.headers.get("Access-Control-Allow-Methods", "")
    assert "*" not in allowed_methods


# ===========================================================================
# H1: Password complexity
# ===========================================================================


@pytest.mark.asyncio
async def test_password_requires_uppercase(client: AsyncClient):
    """Password must contain at least one uppercase letter."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "noupper@test.com", "password": "securepass123!"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_password_requires_digit(client: AsyncClient):
    """Password must contain at least one digit."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "nodigit@test.com", "password": "SecurePass!!"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_password_requires_special(client: AsyncClient):
    """Password must contain at least one special character."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "nospecial@test.com", "password": "SecurePass123"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_password_requires_lowercase(client: AsyncClient):
    """Password must contain at least one lowercase letter."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "nolower@test.com", "password": "SECUREPASS123!"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_valid_complex_password_accepted(client: AsyncClient):
    """A password meeting all complexity requirements should be accepted."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "complex@test.com", "password": "SecurePass123!"},
    )
    assert resp.status_code == 201


# ===========================================================================
# H3: Config hardening
# ===========================================================================


def test_config_rejects_default_secret_in_production():
    """Production mode should reject default JWT secret."""
    import os
    from app.config import Settings

    with pytest.raises(Exception):
        Settings(
            app_env="production",
            jwt_secret_key="change-me-in-production",
            database_encryption_key="",
        )


def test_config_accepts_default_secret_in_development():
    """Development mode should allow default JWT secret."""
    from app.config import Settings

    s = Settings(app_env="development", jwt_secret_key="change-me-in-production")
    assert s.jwt_secret_key == "change-me-in-production"


# ===========================================================================
# H5: No plaintext email in audit log
# ===========================================================================


@pytest.mark.asyncio
async def test_login_audit_does_not_log_email(client: AsyncClient, db_session):
    """Login audit event should only log email domain, not full email."""
    from sqlalchemy import select, text
    from app.models.audit import AuditLog

    await client.post(
        "/api/v1/auth/register",
        json={"email": "auditcheck@test.com", "password": "SecurePass123!"},
    )
    await client.post(
        "/api/v1/auth/login",
        json={"email": "auditcheck@test.com", "password": "SecurePass123!"},
    )

    result = await db_session.execute(
        select(AuditLog).where(AuditLog.action == "user.login")
    )
    logs = result.scalars().all()

    for log in logs:
        if log.details:
            assert "auditcheck@test.com" not in str(log.details)
            if "email_domain" in log.details:
                assert log.details["email_domain"] == "test.com"


# ===========================================================================
# H6: PHI scrubber covers additional identifiers
# ===========================================================================


def test_phi_scrubber_removes_fax():
    """PHI scrubber should remove fax numbers."""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "Contact fax: 555-123-4567 for records"
    scrubbed, report = scrub_phi(text)
    assert "555-123-4567" not in scrubbed
    assert "[FAX]" in scrubbed or "[PHONE]" in scrubbed


def test_phi_scrubber_removes_vin():
    """PHI scrubber should remove vehicle identification numbers."""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "Vehicle ID: 1HGBH41JXMN109186"
    scrubbed, report = scrub_phi(text)
    assert "1HGBH41JXMN109186" not in scrubbed


def test_phi_scrubber_removes_device_id():
    """PHI scrubber should remove device identifiers."""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "Device serial: ABC123-DEF456"
    scrubbed, report = scrub_phi(text)
    assert "ABC123-DEF456" not in scrubbed


def test_phi_scrubber_removes_health_plan_number():
    """PHI scrubber should remove health plan numbers."""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "Member number: HPN12345"
    scrubbed, report = scrub_phi(text)
    assert "HPN12345" not in scrubbed


def test_phi_scrubber_word_boundary_short_names():
    """Short name parts (<=3 chars) should use word boundaries to avoid false positives."""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "The patient named Li has diabetes."
    scrubbed, report = scrub_phi(text, patient_names=["Li"])
    assert "[PATIENT]" in scrubbed
    # But "lipid" should NOT be scrubbed
    text2 = "Check the lipid panel results."
    scrubbed2, _ = scrub_phi(text2, patient_names=["Li"])
    assert "lipid" in scrubbed2


def test_phi_scrubber_generalizes_slash_dates():
    """MM/DD/YYYY dates are generalized to MM/YYYY (day dropped), matching the
    'Month DD, YYYY' handling — a lab report's DOB/collection dates must not leak.
    (enable_ner=False isolates the regex layer.)"""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "DOB: 07/31/1996  Collected: 02/16/2026  Reported: 2/3/2026"
    scrubbed, report = scrub_phi(text, enable_ner=False)
    assert "07/31/1996" not in scrubbed
    assert "07/1996" in scrubbed
    assert "02/2026" in scrubbed
    assert "2/2026" in scrubbed
    assert report.get("dates_generalized", 0) >= 3


def test_phi_scrubber_redacts_account_and_accession_numbers():
    """Account and lab-accession numbers (e.g. 'Account No: 235410324',
    'Lab Accession: 87414853') are identifiers and must be redacted."""
    from app.services.ai.phi_scrubber import scrub_phi
    # Plain, markdown-bold (as OCR sometimes emits), and shorthand variants.
    text = (
        "Account No: 235410324\n"
        "**Lab Accession:** 87414853\n"
        "Acct #998877"
    )
    scrubbed, _ = scrub_phi(text, enable_ner=False)
    assert "235410324" not in scrubbed
    assert "87414853" not in scrubbed
    assert "998877" not in scrubbed
    assert "[ACCOUNT]" in scrubbed


def test_phi_scrubber_preserves_clinical_numbers():
    """De-identification must not destroy lab values, ranges, or percentages."""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "Anti-CdtB Ab 1.24; reference 0.00 - 1.55; 96% - 100% PPV; IBS-D"
    scrubbed, _ = scrub_phi(text, enable_ner=False)
    assert "1.24" in scrubbed
    assert "0.00 - 1.55" in scrubbed
    assert "96% - 100%" in scrubbed
    assert "IBS-D" in scrubbed


def test_phi_scrubber_redacts_street_address():
    """Street addresses (incl. suite/unit continuation) are geographic PHI.
    (enable_ner=False isolates the regex layer.)"""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "Address: 275 Post Rd E, Ste. 10, Unit 310; also 1234 Elm Street, Apt 5B"
    scrubbed, _ = scrub_phi(text, enable_ner=False)
    assert "275 Post Rd" not in scrubbed
    assert "1234 Elm Street" not in scrubbed
    assert "[LOCATION]" in scrubbed


def test_phi_scrubber_street_regex_preserves_clinical_text():
    """The street-address regex must not eat dosing / vitals / lab lines."""
    from app.services.ai.phi_scrubber import scrub_phi
    text = "Take 2 tablets by mouth daily; 5 mg PO; BP 120/80; 3 episodes per week"
    scrubbed, _ = scrub_phi(text, enable_ner=False)
    assert "2 tablets" in scrubbed
    assert "5 mg" in scrubbed
    assert "120/80" in scrubbed
    assert "3 episodes" in scrubbed


# ===========================================================================
# C5: Path traversal prevention
# ===========================================================================


def test_safe_file_path_prevents_traversal():
    """File path helper should prevent path traversal via filename."""
    from app.api.upload import _safe_file_path

    upload_dir = Path("/tmp/claude/test_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Should not raise and should be inside upload_dir
    safe = _safe_file_path(upload_dir, uuid4(), "../../etc/passwd")
    assert str(safe).startswith(str(upload_dir.resolve()))
    assert "etc" not in str(safe)
    assert "passwd" not in str(safe)


def test_safe_file_path_preserves_extension():
    """File path helper should preserve the original extension."""
    from app.api.upload import _safe_file_path

    upload_dir = Path("/tmp/claude/test_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe = _safe_file_path(upload_dir, uuid4(), "my_data.json")
    assert safe.suffix == ".json"


# ===========================================================================
# M1: Magic byte validation
# ===========================================================================


def test_magic_bytes_pdf():
    """PDF magic bytes should be validated."""
    from app.api.upload import _validate_magic_bytes
    assert _validate_magic_bytes(b"%PDF-1.4 ...", ".pdf") is True
    assert _validate_magic_bytes(b"not a pdf file", ".pdf") is False


def test_magic_bytes_rtf():
    """RTF magic bytes should be validated."""
    from app.api.upload import _validate_magic_bytes
    assert _validate_magic_bytes(b"{\\rtf1 ...", ".rtf") is True
    assert _validate_magic_bytes(b"not rtf content", ".rtf") is False


def test_magic_bytes_tiff():
    """TIFF magic bytes should be validated (both LE and BE)."""
    from app.api.upload import _validate_magic_bytes
    assert _validate_magic_bytes(b"\x49\x49\x2a\x00", ".tif") is True  # LE
    assert _validate_magic_bytes(b"\x4d\x4d\x00\x2a", ".tiff") is True  # BE
    assert _validate_magic_bytes(b"not a tiff", ".tif") is False


# ===========================================================================
# C3: Audit logging on data-access endpoints
# ===========================================================================


@pytest.mark.asyncio
async def test_records_list_creates_audit_log(client: AsyncClient, db_session):
    """Records list endpoint should create an audit log entry."""
    from sqlalchemy import select
    from app.models.audit import AuditLog

    headers, user_id = await auth_headers(client, "audit-records@test.com")
    await client.get("/api/v1/records", headers=headers)

    result = await db_session.execute(
        select(AuditLog).where(AuditLog.action == "records.list")
    )
    logs = result.scalars().all()
    assert len(logs) >= 1


@pytest.mark.asyncio
async def test_timeline_creates_audit_log(client: AsyncClient, db_session):
    """Timeline endpoint should create an audit log entry."""
    from sqlalchemy import select
    from app.models.audit import AuditLog

    headers, user_id = await auth_headers(client, "audit-timeline@test.com")
    await client.get("/api/v1/timeline", headers=headers)

    result = await db_session.execute(
        select(AuditLog).where(AuditLog.action == "timeline.view")
    )
    logs = result.scalars().all()
    assert len(logs) >= 1


@pytest.mark.asyncio
async def test_dashboard_overview_creates_audit_log(client: AsyncClient, db_session):
    """Dashboard overview should create an audit log entry."""
    from sqlalchemy import select
    from app.models.audit import AuditLog

    headers, user_id = await auth_headers(client, "audit-dashboard@test.com")
    await client.get("/api/v1/dashboard/overview", headers=headers)

    result = await db_session.execute(
        select(AuditLog).where(AuditLog.action == "dashboard.overview")
    )
    logs = result.scalars().all()
    assert len(logs) >= 1
