"""W19 — transport security (HIPAA CRYPTO-04, SEC-API-01/-05).

Covers four hardening changes, all gated so local dev/tests (http on 127.0.0.1)
are unaffected:

1. HSTS is emitted unconditionally in production (behind a TLS-terminating proxy
   the app only ever observes ``http``, so the old ``scheme == "https"`` gate
   never fired) — SEC-API-01.
2. In production the app installs ``HTTPSRedirectMiddleware`` +
   ``TrustedHostMiddleware``; in dev it must NOT (a redirect would break the
   whole http test suite) — CRYPTO-04.
3. CORS origins are ``.strip()``-ed and the wildcard origin can never be paired
   with ``allow_credentials=True`` — SEC-API-05.
4. A non-loopback production ``DATABASE_URL`` without ``sslmode`` surfaces an
   advisory warning (non-breaking).
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.config import Settings
from app.main import build_cors_config
from app.middleware.security_headers import HSTS_VALUE, should_emit_hsts

VALID_KEY = "ab" * 32
STRONG_SECRET = "x" * 48


# ---------------------------------------------------------------------------
# 1. HSTS gating (pure function — assertable without a live https stack)
# ---------------------------------------------------------------------------


def test_hsts_emitted_in_production_regardless_of_scheme():
    """SEC-API-01: production emits HSTS even on an observed http scheme."""
    assert should_emit_hsts(is_production=True, scheme="http", forwarded_proto=None) is True


def test_hsts_not_emitted_in_dev_over_http():
    """Dev behavior preserved: plain http does not emit HSTS."""
    assert should_emit_hsts(is_production=False, scheme="http", forwarded_proto=None) is False


def test_hsts_emitted_in_dev_over_https():
    """Dev behavior preserved: an actual https request still emits HSTS."""
    assert should_emit_hsts(is_production=False, scheme="https", forwarded_proto=None) is True


def test_hsts_honors_forwarded_proto_in_dev():
    """An https X-Forwarded-Proto counts as https in non-production too."""
    assert should_emit_hsts(is_production=False, scheme="http", forwarded_proto="https") is True


# ---------------------------------------------------------------------------
# 1b. HSTS via the real middleware stack (integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hsts_header_present_when_production_forced(client: AsyncClient, monkeypatch):
    """Force the middleware to see production and assert the header appears over http."""
    import app.middleware.security_headers as sh

    class _Stub:
        is_production = True

    monkeypatch.setattr(sh, "settings", _Stub())
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.headers.get("Strict-Transport-Security") == HSTS_VALUE


@pytest.mark.asyncio
async def test_no_hsts_header_in_dev_over_http(client: AsyncClient):
    """The real (dev) app must not emit HSTS over the http test transport."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert "Strict-Transport-Security" not in resp.headers


@pytest.mark.asyncio
async def test_hsts_emitted_in_dev_behind_https_proxy(client: AsyncClient):
    """Portless scenario: an https ``X-Forwarded-Proto`` makes the dev app emit HSTS.

    portless fronts the loopback backend at ``https://api.medtimeline.localhost``
    and injects ``X-Forwarded-Proto: https``. The dev app must then emit HSTS even
    though it terminated plain http itself — the whole point of issue #57. This is
    the http-vs-https assertion pair: same dev app, header present here, absent in
    ``test_no_hsts_header_in_dev_over_http``.
    """
    resp = await client.get(
        "/api/v1/health", headers={"X-Forwarded-Proto": "https"}
    )
    assert resp.status_code == 200
    assert resp.headers.get("Strict-Transport-Security") == HSTS_VALUE


# ---------------------------------------------------------------------------
# 2. HTTPSRedirect / TrustedHost gating (production-only)
# ---------------------------------------------------------------------------


def test_dev_app_has_no_redirect_or_trustedhost_middleware():
    """Dev must NOT install the redirect/host middleware — it would 301 the suite."""
    from app.main import app

    classes = {m.cls.__name__ for m in app.user_middleware}
    assert "HTTPSRedirectMiddleware" not in classes
    assert "TrustedHostMiddleware" not in classes


def test_production_app_installs_transport_middleware(monkeypatch):
    """Production create_app() wires HTTPSRedirect + TrustedHost."""
    import app.main as main_mod

    class _Stub:
        is_production = True
        gemini_api_key = "set"
        app_env = "production"
        cors_origins = "https://app.example.com"
        allowed_hosts_list = ["app.example.com"]

        @staticmethod
        def database_ssl_warning() -> None:
            return None

    monkeypatch.setattr(main_mod, "settings", _Stub())
    prod_app = main_mod.create_app()
    classes = {m.cls.__name__ for m in prod_app.user_middleware}
    assert "HTTPSRedirectMiddleware" in classes
    assert "TrustedHostMiddleware" in classes


# ---------------------------------------------------------------------------
# 3. CORS hygiene (SEC-API-05)
# ---------------------------------------------------------------------------


def test_cors_strips_whitespace():
    origins, allow_credentials = build_cors_config("http://a.com , http://b.com")
    assert origins == ["http://a.com", "http://b.com"]
    assert allow_credentials is True


def test_cors_drops_empty_origins():
    origins, _ = build_cors_config("http://a.com,,  ,")
    assert origins == ["http://a.com"]


def test_cors_wildcard_disables_credentials():
    """A '*' origin must never be paired with credentials (browser rejects it)."""
    origins, allow_credentials = build_cors_config("*")
    assert origins == ["*"]
    assert allow_credentials is False


def test_cors_wildcard_among_others_disables_credentials():
    origins, allow_credentials = build_cors_config("http://a.com, *")
    assert "*" in origins
    assert allow_credentials is False


def test_cors_normal_keeps_credentials():
    origins, allow_credentials = build_cors_config("http://localhost:3000")
    assert origins == ["http://localhost:3000"]
    assert allow_credentials is True


def test_cors_accepts_portless_https_origin():
    """The portless https origin parses cleanly and keeps credentials enabled.

    ``scripts/run-https.sh`` sets ``CORS_ORIGINS`` to the portless frontend origin
    so the browser app at ``https://medtimeline.localhost`` can call the api
    subdomain cross-origin with credentials.
    """
    origins, allow_credentials = build_cors_config(
        "https://medtimeline.localhost, http://localhost:3000"
    )
    assert "https://medtimeline.localhost" in origins
    assert allow_credentials is True


# ---------------------------------------------------------------------------
# 4. DB sslmode advisory (CRYPTO-04, non-breaking)
# ---------------------------------------------------------------------------


def _prod_settings(database_url: str) -> Settings:
    return Settings(
        app_env="production",
        jwt_secret_key=STRONG_SECRET,
        database_encryption_key=VALID_KEY,
        database_url=database_url,
    )


def test_db_ssl_warning_for_nonloopback_production_without_sslmode():
    s = _prod_settings("postgresql+asyncpg://user:pw@db.internal:5432/medtimeline")
    assert s.database_ssl_warning() is not None


def test_db_ssl_warning_none_for_loopback():
    s = _prod_settings("postgresql+asyncpg://localhost:5432/medtimeline")
    assert s.database_ssl_warning() is None


def test_db_ssl_warning_none_when_sslmode_set():
    s = _prod_settings(
        "postgresql+asyncpg://user:pw@db.internal:5432/medtimeline?sslmode=require"
    )
    assert s.database_ssl_warning() is None


def test_db_ssl_warning_none_in_dev():
    s = Settings(
        app_env="development",
        database_url="postgresql+asyncpg://db.internal:5432/medtimeline",
    )
    assert s.database_ssl_warning() is None


def test_allowed_hosts_list_parses_and_defaults():
    s = Settings(app_env="development", allowed_hosts="a.com, b.com")
    assert s.allowed_hosts_list == ["a.com", "b.com"]
    # Empty / blank falls back to permissive wildcard.
    s2 = Settings(app_env="development", allowed_hosts="  ")
    assert s2.allowed_hosts_list == ["*"]
