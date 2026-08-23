# General Account Login Identifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each task gets a fresh implementer and one independent task reviewer who returns both spec-compliance and code-quality verdicts. Critical or Important findings require a fix and re-review before the next task; Task 3 also gets a targeted migration-security review.

**Goal:** Let Strand users register and sign in with a private account name instead of an email address while preserving existing accounts and all authentication, encryption, privacy, and authorization behavior.

**Architecture:** Replace the canonical encrypted `email` identity with one encrypted `login_identifier` and its unique HMAC blind index. Keep the legacy `email` JSON key and response field as deprecated API compatibility aliases, sanitize complete register/login 422 responses at a route-local boundary, and rename existing database columns without rewriting ciphertext or HMAC bytes. JWT ownership remains UUID-based and unchanged.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2 async, Alembic, PostgreSQL 16, pytest, Next.js 16, React 19, TypeScript, Playwright, ESLint.

## Global Constraints

- The approved specification is `docs/superpowers/specs/2026-08-23-account-login-identifier-design.md`; do not broaden beyond it.
- Canonical backend/API name: `login_identifier`; user-facing label: `Account name`.
- Preserve legacy `email` request compatibility and the deprecated `email` response alias.
- Normalize lookup/uniqueness by exactly `value.strip().lower()`; do not add NFKC or `casefold()`.
- Require a trimmed identifier of 1–255 Unicode code points whose complete value satisfies `str.isprintable()`.
- The complete register/login HTTP 422 body is exactly `{"detail":"Invalid authentication request."}` and contains no submitted identifier, alias, password, Pydantic input/context, normalized value, blind index, or derived fragment.
- Preserve registration/login rate limits, five-attempt lockout, bcrypt rules, UUID JWT subjects, refresh rotation/revocation, idle timeout, logout, user scoping, and content-free authored errors.
- Preserve AES-256-GCM encrypted identifiers and keyed HMAC-SHA256 lookup/uniqueness.
- Upgrade renames columns/index only. Downgrade decrypts only in process with `DATABASE_ENCRYPTION_KEY`, logs/persists no plaintext, and fails before any schema mutation when legacy email compatibility cannot be proven.
- Do not use real medical data, live cloud/provider credentials, networked AI, or model downloads.
- Start each command block from the worktree root unless that block explicitly
  states another working directory; do not rely on directory state from a prior
  block.
- Public prose must pass the `humanizer` workflow before acceptance.
- Implementer, fixer, and reviewer subagents must never stage, commit, push,
  open PRs, or change GitHub state. The issue #67 worktree root alone owns Git
  mutations and integration.
- Do not branch from the current `b49b66f` snapshot. The execution preflight
  below must wait for PRs #68 and #69 to merge, move this detached worktree to
  the resulting fresh `origin/main`, revalidate the auth/migration baseline,
  and only then create `codex/issue-67-login-identifier`.
- The worktree root creates one focused review commit for each completed Task
  1–6. That commit supplies the SDD `BASE..HEAD` review range and is not accepted
  until one independent task reviewer returns both a clean spec-compliance
  verdict and an approved code-quality verdict. Critical or Important fixes are
  made by a fresh fixer and folded into the same root-owned task commit before a
  fresh package and re-review.
- Record every accepted task range and all Minor findings in
  `.superpowers/sdd/progress.md`. Task 3 alone adds a separate targeted security
  reviewer for migration/downgrade key handling, fail-before-DDL ordering, and
  plaintext/log privacy. Keep the broad whole-branch review in Task 7.
- Do not squash or restage the accepted task commits in Task 7. The parent root
  may choose GitHub's squash-merge behavior after independent PR review.
- Do not merge the PR; leave it for independent root review and sequential merge.

## File Structure and Responsibilities

- `backend/app/api/auth_validation.py` — route-local, content-free request-validation response boundary for register/login only.
- `backend/app/schemas/auth.py` — canonical/legacy request resolution, exact identifier validation, unchanged password validation, additive user response contract.
- `backend/app/models/user.py` — canonical encrypted ORM fields and automatic HMAC synchronization.
- `backend/app/services/auth_service.py` — canonical availability/authentication lookup, race-safe duplicate handling, unchanged lockout/token behavior.
- `backend/app/api/auth.py` — rate-limited routes, generic errors, compatibility response construction, identifier-free login audit event.
- `backend/alembic/versions/d6e7f8a9b0c1_general_login_identifier.py` — byte-preserving upgrade renames and preflighted downgrade.
- `backend/tests/test_account_login_identifier.py` — focused canonical/legacy/Unicode/422/privacy/API tests.
- `backend/tests/test_account_login_identifier_migration.py` — migration ordering, byte preservation, downgrade privacy, and create-all parity.
- Existing backend auth/encryption tests — preserve lockout, JWT, audit, and encrypted-at-rest regressions.
- Existing backend fixture modules that construct `User` directly — mechanical ORM keyword rename only.
- `frontend/src/lib/login-identifier.ts` — deterministic Unicode-code-point masking helper.
- `frontend/src/lib/login-identifier.unit.spec.ts` — server-free mask tests.
- Auth pages, API types, current-user consumers, and E2E helpers/specs — canonical UI/API behavior and legacy-account coverage.
- `scripts/e2e_full_v2.py` — canonical synthetic account setup without plaintext identifier SQL lookup; update statically but do not run its real-data/provider workflow.
- `README.md`, `docs/backend-handoff.md`, `docs/operations-backup-restore.md` — truthful public API, encryption, recovery, and downgrade wording.

## Execution Preflight and SDD Ledger

This preflight is a hard gate, not Task 1 implementation. The worktree root
performs it only after the orchestrating root approves this revised plan.

- [ ] **Preflight 1: Wait for the overlapping prerequisite PRs**

Run:

```bash
test "$(gh pr view 68 --repo potalora/strand --json state --jq '.state')" = "MERGED"
test "$(gh pr view 69 --repo potalora/strand --json state --jq '.state')" = "MERGED"
gh pr view 68 --repo potalora/strand --json number,title,mergedAt,mergeCommit,url
gh pr view 69 --repo potalora/strand --json number,title,mergedAt,mergeCommit,url
```

Expected: both tests pass and both PRs have non-null `mergedAt` and
`mergeCommit`. If either PR is still open, closed without merge, or awaiting the
parent root's sequential merge, stop without branching or implementing and
report the wait condition to the orchestrating root.

- [ ] **Preflight 2: Prove the detached worktree contains only the approved documents**

Run:

```bash
test -z "$(git branch --show-current)"
test "$(git rev-parse HEAD)" = "b49b66f654de1c911f2dc9caf15c203ccf0e132f"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
issue67_expected_untracked=$'docs/superpowers/plans/2026-08-23-account-login-identifier.md\ndocs/superpowers/specs/2026-08-23-account-login-identifier-design.md'
issue67_actual_untracked="$(git ls-files --others --exclude-standard | LC_ALL=C sort)"
test "$issue67_actual_untracked" = "$issue67_expected_untracked"
```

Expected: every assertion passes. Any tracked, staged, or additional untracked
issue file is a scope conflict; stop for root review rather than stashing,
deleting, or resetting it.

- [ ] **Preflight 3: Fetch and inspect the fresh mainline before moving the worktree**

Run:

```bash
git fetch origin
git log --oneline --decorate b49b66f..origin/main
git diff --name-only b49b66f..origin/main -- \
  backend/app/api/auth.py \
  backend/app/api/auth_validation.py \
  backend/app/models/user.py \
  backend/app/schemas/auth.py \
  backend/app/services/auth_service.py \
  backend/app/middleware/encryption.py \
  backend/alembic/versions \
  'frontend/src/app/(auth)/login/page.tsx' \
  'frontend/src/app/(auth)/register/page.tsx' \
  frontend/src/types/api.ts \
  frontend/e2e/helpers/api-client.ts \
  frontend/e2e/helpers/auth.ts \
  frontend/e2e/helpers/browser-login.ts
```

Expected: the log includes the parent root's merges of #68 and #69. The targeted
diff-name command prints nothing; changes to `README.md` and the Admin page from
those PRs are expected and must be preserved by editing their fresh versions.
If any auth, identifier, encryption, migration, auth-page, API-type, or auth-
helper path in the targeted command changed, stop for a root plan update before
moving the worktree.

- [ ] **Preflight 4: Move safely to fresh `origin/main` and revalidate the migration head**

Run:

```bash
issue67_expected_untracked=$'docs/superpowers/plans/2026-08-23-account-login-identifier.md\ndocs/superpowers/specs/2026-08-23-account-login-identifier-design.md'
issue67_spec_sha="$(shasum -a 256 docs/superpowers/specs/2026-08-23-account-login-identifier-design.md | awk '{print $1}')"
issue67_plan_sha="$(shasum -a 256 docs/superpowers/plans/2026-08-23-account-login-identifier.md | awk '{print $1}')"
git switch --detach origin/main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test "$issue67_spec_sha" = "$(shasum -a 256 docs/superpowers/specs/2026-08-23-account-login-identifier-design.md | awk '{print $1}')"
test "$issue67_plan_sha" = "$(shasum -a 256 docs/superpowers/plans/2026-08-23-account-login-identifier.md | awk '{print $1}')"
issue67_actual_untracked="$(git ls-files --others --exclude-standard | LC_ALL=C sort)"
test "$issue67_actual_untracked" = "$issue67_expected_untracked"
test "$(cd backend && uv run --no-sync alembic heads)" = "c5d6e7f8a9b0 (head)"
```

Expected: the detached worktree now equals fresh `origin/main`, both approved
documents are byte-identical and still the only untracked files, and Alembic has
exactly the planned `c5d6e7f8a9b0` head. `git switch` must be allowed to refuse
an overwrite. Do not use reset, checkout-overwrite, stash, or cleanup to force
the move. A different or multiple migration head requires a root plan update.

- [ ] **Preflight 5: Create the issue branch and commit the approved planning baseline**

Run:

```bash
test -z "$(git branch --list codex/issue-67-login-identifier)"
test -z "$(git ls-remote --heads origin codex/issue-67-login-identifier)"
test "$(gh pr list --repo potalora/strand --state all --head codex/issue-67-login-identifier --json number --jq 'length')" = "0"
git switch -c codex/issue-67-login-identifier
git add -- \
  docs/superpowers/specs/2026-08-23-account-login-identifier-design.md \
  docs/superpowers/plans/2026-08-23-account-login-identifier.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs(auth): plan general login identifiers"
test -z "$(git status --porcelain=v1)"
```

Expected: the new issue branch begins at the fresh post-#68/#69 mainline and has
one root-owned planning commit containing only the approved spec and plan.

- [ ] **Preflight 6: Initialize and use the durable SDD ledger**

Using `apply_patch`, create the ignored file `.superpowers/sdd/progress.md` with
this initial content:

```markdown
# Issue #67 SDD Progress

Branch: `codex/issue-67-login-identifier`
Plan: `docs/superpowers/plans/2026-08-23-account-login-identifier.md`

- Task 1: pending
- Task 2: pending
- Task 3: pending
- Task 4: pending
- Task 5: pending
- Task 6: pending
- Task 7: pending
- Minor findings: none
```

Before every dispatch or resumed session, read both the ledger and `git log`:

```bash
cat .superpowers/sdd/progress.md
git log --oneline --decorate origin/main..HEAD
```

Each task below provides its exact task-brief command, base variable, staging
allowlist, commit message, and review-package command. After the implementer
reports its tests and self-review, the worktree root stages only that task's
allowlist and creates its sole root-owned review commit. The package always uses
the base recorded before implementer dispatch—never a derived `HEAD~1`—and one
independent reviewer receives the task brief, implementer report, package, and
binding global constraints.

The reviewer must return both verdicts: spec compliance and code quality. If it
reports a Critical or Important finding, dispatch one fresh fixer with the full
finding set and covering tests; the worktree root stages the same task allowlist
and amends that task's review commit, regenerates the package with the original
base and new head, and re-dispatches the reviewer. No next task starts until both
verdicts pass. After acceptance, use `apply_patch` to replace that task's ledger
line with a completion entry containing the actual seven-character base SHA,
the actual seven-character head SHA, and `review clean`; append any Minor
findings for the final reviewer.

The candidate review commit is created before reviewer dispatch because the SDD
package is commit-range based; it becomes an accepted task commit only after the
combined gate passes. Implementer, fixer, and reviewer subagents never commit.

---

### Task 1: Canonical Auth Input and Content-Free 422 Boundary

**Files:**
- Create: `backend/app/api/auth_validation.py`
- Create: `backend/tests/test_account_login_identifier.py`
- Modify: `backend/app/schemas/auth.py`
- Modify: `backend/app/api/auth.py`
- Test: `backend/tests/test_account_login_identifier.py`

**Interfaces:**
- Consumes: existing `blind_index(value: str) -> str`, FastAPI `APIRoute`, Pydantic request validation, existing auth service functions.
- Produces: `normalize_login_identifier(value: str) -> str`, request attribute `body.login_identifier: str`, `ContentFreeAuthValidationRoute`, and `INVALID_AUTH_REQUEST_BODY`.

- [ ] **Step 0: Prepare the Task 1 SDD handoff**

Run:

```bash
issue67_task1_base="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/task-brief \
  docs/superpowers/plans/2026-08-23-account-login-identifier.md 1
```

Using `apply_patch`, mark Task 1 `in progress` in the ledger with the actual full
base SHA. Dispatch one fresh implementer with the Task 1 brief and report path;
explicitly forbid staging, commits, pushes, and GitHub mutations.

- [ ] **Step 1: Add failing schema and route-boundary tests**

Add focused tests that declare the exact contract before production changes:

```python
from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

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
    with pytest.raises(ValueError):
        RegisterRequest.model_validate(
            {"login_identifier": identifier, "password": "SecurePass123!"}
        )


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
    ],
)
async def test_register_422_is_complete_and_content_free(
    client: AsyncClient, payload: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 422
    assert response.content == b'{"detail":"Invalid authentication request."}'
    assert response.json() == {"detail": "Invalid authentication request."}
    assert set(response.json()) == {"detail"}
    serialized = response.content.decode("utf-8")
    sentinels = [
        value
        for value in payload.values()
        if isinstance(value, str)
    ] + ["identifier-sentinel-type"]
    for sentinel in sentinels:
        assert sentinel not in serialized
        assert sentinel.strip().lower() not in serialized
        assert blind_index(sentinel) not in serialized
        assert sentinel not in caplog.text
    for framework_key in ("input", "ctx", "loc"):
        assert framework_key not in serialized
    assert json.loads(serialized) == {"detail": "Invalid authentication request."}


@pytest.mark.asyncio
async def test_refresh_keeps_default_validation_shape(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/refresh", json={})
    assert response.status_code == 422
    assert response.json() != {"detail": "Invalid authentication request."}
    assert isinstance(response.json()["detail"], list)
    assert response.json()["detail"][0]["loc"] == ["body", "refresh_token"]
```

Add the equivalent parameterized assertion for `/api/v1/auth/login`, including
malformed JSON/type, conflicting aliases, and a password sentinel. For malformed
JSON sent as raw content, assert the same exact response bytes and that its raw
identifier/password sentinels do not occur in `response.content` or
`caplog.text`.

- [ ] **Step 2: Run the focused tests and confirm the intended failures**

Run:

```bash
cd backend
uv run --no-sync pytest tests/test_account_login_identifier.py -q
```

Expected: FAIL because `login_identifier`, alias resolution, and the generic route-local 422 response do not exist.

- [ ] **Step 3: Implement exact request normalization and legacy alias resolution**

In `backend/app/schemas/auth.py`, replace the email-only request base with this contract while retaining the existing password validator body:

```python
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
        canonical = value.get("login_identifier")
        legacy = value.get("email")
        if canonical is not None and legacy is not None:
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


class LoginRequest(_LoginIdentifierRequest):
    password: str
```

Keep the current four-part password-complexity validator on `RegisterRequest` unchanged.

- [ ] **Step 4: Implement the route-local validation response boundary**

Create `backend/app/api/auth_validation.py`:

```python
from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

INVALID_AUTH_REQUEST_BODY = {"detail": "Invalid authentication request."}
_CONTENT_FREE_ROUTE_NAMES = frozenset({"auth_register", "auth_login"})


class ContentFreeAuthValidationRoute(APIRoute):
    """Sanitize register/login validation without changing other route errors."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()
        if self.name not in _CONTENT_FREE_ROUTE_NAMES:
            return original

        async def content_free_handler(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError:
                return JSONResponse(status_code=422, content=INVALID_AUTH_REQUEST_BODY)

        return content_free_handler
```

In `backend/app/api/auth.py`, construct the router with
`route_class=ContentFreeAuthValidationRoute` and give the two protected routes
explicit inclusion-stable names. Replace the current registration decorator
with:

```python
@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    name="auth_register",
)
```

Replace the current login decorator with:

```python
@router.post("/login", response_model=TokenResponse, name="auth_login")
```

The function definitions and bodies remain directly beneath their respective
decorators.

FastAPI clones routes when `auth.router` is included under the `/api/v1`
router. Route names are preserved through that inclusion; effective paths are
not. Do not match `self.path`. Pass `body.login_identifier` to
registration/authentication, change the generic login error to `Invalid account
identifier or password`, and remove `email_domain` from the successful login
audit event. Do not move either rate-limit check. Do not sanitize validation
errors for refresh, logout, `/me`, or any non-register/login route.

- [ ] **Step 5: Run focused request/privacy tests and existing password tests**

Run:

```bash
cd backend
uv run --no-sync pytest tests/test_account_login_identifier.py -q
uv run --no-sync ruff check app/api/auth_validation.py app/api/auth.py app/schemas/auth.py tests/test_account_login_identifier.py
```

Expected: all selected tests pass. The old audit-domain test is intentionally
updated and run with the focused privacy suite in Task 2.

- [ ] **Step 6: Create the root-owned Task 1 review commit and run the combined gate**

After the implementer report contains the failing-then-passing test evidence and
self-review, run:

```bash
issue67_task1_base="$(git rev-parse HEAD)"
git add -- \
  backend/app/api/auth.py \
  backend/app/api/auth_validation.py \
  backend/app/schemas/auth.py \
  backend/tests/test_account_login_identifier.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat(auth): accept general login identifiers"
issue67_task1_head="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/review-package \
  "$issue67_task1_base" "$issue67_task1_head"
```

The worktree root verifies the base equals the SHA recorded before dispatch and
that the cached list contains only the four Task 1 paths. Dispatch one
independent task reviewer with the brief, implementer report, review package,
and global constraints. It must return both spec-compliance and code-quality
verdicts, including the route-name stability and refresh-unsanitized regression.
For Critical or Important findings, use one fresh fixer, rerun covering tests,
amend this root-owned commit, regenerate the same-base package, and re-review.
After both verdicts pass, record the accepted range and Minor findings in the
ledger. Do not start Task 2 earlier.

---

### Task 2: Canonical Encrypted User Model, Auth Service, and API Responses

**Files:**
- Modify: `backend/app/models/user.py`
- Modify: `backend/app/services/auth_service.py`
- Modify: `backend/app/schemas/auth.py`
- Modify: `backend/app/api/auth.py`
- Modify: `backend/tests/conftest.py`
- Modify: `backend/tests/test_auth.py`
- Modify: `backend/tests/test_auth_hardening.py`
- Modify: `backend/tests/test_hipaa_compliance.py`
- Modify: `backend/tests/test_at_rest_encryption.py`
- Modify: `backend/tests/test_account_login_identifier.py`

**Interfaces:**
- Consumes: Task 1 `body.login_identifier`, exact normalization, and sanitized validation route.
- Produces: ORM fields `User.login_identifier` and `User.login_identifier_hmac`; service signatures using `login_identifier`; additive `UserResponse.login_identifier` plus deprecated `UserResponse.email`; `UserResponse.from_user(user)`.

- [ ] **Step 0: Prepare the Task 2 SDD handoff**

Run:

```bash
issue67_task2_base="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/task-brief \
  docs/superpowers/plans/2026-08-23-account-login-identifier.md 2
```

Using `apply_patch`, mark Task 2 `in progress` in the ledger with the actual full
base SHA. Dispatch one fresh implementer with the Task 2 brief and report path;
explicitly forbid all Git and GitHub mutations.

- [ ] **Step 1: Add failing canonical API, uniqueness, Unicode, and audit tests**

Extend `backend/tests/test_account_login_identifier.py` with these behaviors:

```python
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
```

Also add exact tests for unknown identifier, wrong password, and disabled account sharing the same generic 401; active/expired lockout behavior remaining unchanged; commit-time `IntegrityError` returning the same generic 409 after rollback; `/auth/me` returning both fields; and `user.login` audit rows having `details is None`.

- [ ] **Step 2: Run focused tests and confirm model/response failures**

Run:

```bash
cd backend
uv run --no-sync pytest tests/test_account_login_identifier.py tests/test_auth.py tests/test_at_rest_encryption.py -q
```

Expected: FAIL because the model, service, and response still expose `email` as the canonical field.

- [ ] **Step 3: Rename the canonical ORM fields and listener**

In `backend/app/models/user.py`, define only canonical storage fields:

```python
login_identifier: Mapped[str] = mapped_column(EncryptedText, nullable=False)
login_identifier_hmac: Mapped[str] = mapped_column(
    String(64), unique=True, index=True, nullable=False
)


def _sync_login_identifier_hmac(mapper, connection, target: User) -> None:
    if target.login_identifier is not None:
        target.login_identifier_hmac = blind_index(target.login_identifier)


event.listen(User, "before_insert", _sync_login_identifier_hmac)
event.listen(User, "before_update", _sync_login_identifier_hmac)
```

Update comments to say the value is a private login identifier. Do not add an
ORM `email` synonym or property.

- [ ] **Step 4: Rename service inputs/lookups and make duplicate races safe**

In `backend/app/services/auth_service.py`, use `login_identifier` throughout:

```python
async def register_user(
    db: AsyncSession,
    login_identifier: str,
    password: str,
    display_name: str | None = None,
) -> User:
    login_identifier = normalize_login_identifier(login_identifier)
    identifier_hmac = blind_index(login_identifier)
    existing = await db.execute(
        select(User).where(User.login_identifier_hmac == identifier_hmac)
    )
    if existing.scalar_one_or_none():
        raise ValueError("Account identifier is unavailable.")

    user = User(
        login_identifier=login_identifier,
        login_identifier_hmac=identifier_hmac,
        password_hash=hash_password(password),
        display_name=display_name,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise ValueError("Account identifier is unavailable.") from None
    await db.refresh(user)
    return user
```

Import `normalize_login_identifier` from `app.schemas.auth`; do not duplicate or
silently extend its normalization behavior in the service.

`authenticate_user` first applies the same imported
`normalize_login_identifier`, then queries `User.login_identifier_hmac ==
blind_index(login_identifier)`. It raises `Invalid account identifier or
password` for missing users and bad passwords. Preserve every existing line of
lockout, disabled-user mapping, refresh-family cleanup, token creation, and
commit ordering.

- [ ] **Step 5: Add the canonical and deprecated response fields explicitly**

In `backend/app/schemas/auth.py`, replace the response schema with an explicit
constructor so ORM attribute mapping cannot silently miss the alias:

```python
from typing import Any, Self


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
```

Use `UserResponse.from_user(user)` in both registration and `/auth/me` routes.

- [ ] **Step 6: Update focused fixtures and privacy/encryption assertions**

In `backend/tests/conftest.py`, keep the widely used test-helper parameter name
`email` for caller compatibility, but send canonical JSON:

```python
json={
    "login_identifier": email,
    "password": "SecurePass123!",
    "display_name": "Test",
}
```

Update `test_auth.py` to assert both response fields. Update
`test_at_rest_encryption.py` to read `login_identifier` and
`login_identifier_hmac`, assert raw ciphertext lacks the sentinel, and retain
the exact lower/strip blind-index assertions. Replace the old HIPAA audit-domain
test with `assert log.details is None` and assert the identifier/domain does not
appear in serialized audit rows.

- [ ] **Step 7: Run focused backend regression tests**

Run:

```bash
cd backend
uv run --no-sync pytest tests/test_account_login_identifier.py tests/test_auth.py tests/test_auth_hardening.py tests/test_hipaa_compliance.py tests/test_at_rest_encryption.py tests/test_token_revocation.py -q
uv run --no-sync ruff check app/api/auth.py app/api/auth_validation.py app/models/user.py app/schemas/auth.py app/services/auth_service.py tests/test_account_login_identifier.py tests/test_auth.py tests/test_auth_hardening.py tests/test_hipaa_compliance.py tests/test_at_rest_encryption.py
```

Expected: all selected tests pass. JWT, refresh, rate-limit, and lockout tests
must remain green without relaxed assertions.

- [ ] **Step 8: Create the root-owned Task 2 review commit and run the combined gate**

After the implementer report contains focused test evidence and self-review,
run:

```bash
issue67_task2_base="$(git rev-parse HEAD)"
git add -- \
  backend/app/api/auth.py \
  backend/app/models/user.py \
  backend/app/schemas/auth.py \
  backend/app/services/auth_service.py \
  backend/tests/conftest.py \
  backend/tests/test_account_login_identifier.py \
  backend/tests/test_at_rest_encryption.py \
  backend/tests/test_auth.py \
  backend/tests/test_auth_hardening.py \
  backend/tests/test_hipaa_compliance.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat(auth): preserve encrypted identifier behavior"
issue67_task2_head="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/review-package \
  "$issue67_task2_base" "$issue67_task2_head"
```

Verify the base against the ledger and the cached list against this exact Task 2
allowlist. Dispatch one independent reviewer for both spec compliance and code
quality, with explicit attention to duplicate races, generic 401/409 behavior,
lockout/JWT/refresh invariants, audit privacy, and encrypted blind-index lookup.
Critical or Important findings use a fresh fixer, covering-test evidence, a
root-owned amend, a regenerated same-base package, and re-review. Record the
accepted range and Minor findings in the ledger before Task 3.

---

### Task 3: Byte-Preserving Alembic Migration and Conditional Downgrade

**Files:**
- Create: `backend/alembic/versions/d6e7f8a9b0c1_general_login_identifier.py`
- Create: `backend/tests/test_account_login_identifier_migration.py`
- Test: `backend/tests/test_account_login_identifier_migration.py`
- Reference only: `backend/alembic/versions/c4d5e6f7a8b9_encrypt_phi_columns_at_rest.py`

**Interfaces:**
- Consumes: current head `c5d6e7f8a9b0`, `decrypt_field(bytes) -> str`, `TypeAdapter(EmailStr)`.
- Produces: revision `d6e7f8a9b0c1`, canonical database columns/index, generic preflight failure constant.

- [ ] **Step 0: Prepare the Task 3 SDD handoff**

Run:

```bash
issue67_task3_base="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/task-brief \
  docs/superpowers/plans/2026-08-23-account-login-identifier.md 3
```

Using `apply_patch`, mark Task 3 `in progress` in the ledger with the actual full
base SHA. Dispatch one fresh implementer with the Task 3 brief and report path;
explicitly forbid all Git and GitHub mutations.

- [ ] **Step 1: Add failing migration unit and integration tests**

The test module must load the new revision with `importlib`, use a literal
disposable database guard named `medtimeline_issue67_migrations_ci`, and assert:

```python
from collections.abc import Iterator


DOWNGRADE_BLOCKED = (
    "Downgrade blocked: account identifiers are not compatible with the legacy schema."
)


class FakeScalarResult:
    def __init__(self, values: list[bytes]) -> None:
        self._values = values

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._values)


class FakeResult:
    def __init__(self, values: list[bytes]) -> None:
        self._values = values

    def scalars(self) -> FakeScalarResult:
        return FakeScalarResult(self._values)


class FakeConnection:
    def __init__(self, values: list[bytes]) -> None:
        self._values = values

    def execute(self, statement: object) -> FakeResult:
        del statement
        return FakeResult(self._values)


def test_downgrade_preflight_failure_is_content_free_and_precedes_ddl(
    migration_module, monkeypatch, caplog
) -> None:
    sentinel = "not-an-email-identifier-sentinel"
    ddl_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fake_connection = FakeConnection([encrypt_field(sentinel)])
    monkeypatch.setattr(migration_module.op, "get_bind", lambda: fake_connection)
    monkeypatch.setattr(
        migration_module.op,
        "alter_column",
        lambda *args, **kwargs: ddl_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        migration_module.op,
        "execute",
        lambda *args, **kwargs: ddl_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="Downgrade blocked") as exc_info:
        migration_module.downgrade()

    assert ddl_calls == []
    assert sentinel not in str(exc_info.value)
    assert sentinel not in caplog.text
```

Add the same ordering/privacy assertions for a missing/wrong key or decryption
failure. The disposable-DB test must upgrade to `c5d6e7f8a9b0`, insert a real
encrypted legacy-email row, capture raw ciphertext/HMAC bytes, upgrade to
`d6e7f8a9b0c1`, and verify the renamed columns contain byte-identical values.
Against that migrated row, exercise login once with canonical
`login_identifier` JSON and once with legacy `email` JSON. For successful
email-only downgrade, capture both bytes before and after the inverse renames
and assert they remain identical. For refused non-email and wrong-key
downgrades, assert the canonical column/index names still exist, the legacy
names do not, no `UPDATE` statement was issued, and ciphertext/HMAC bytes are
unchanged.

- [ ] **Step 2: Run migration tests and confirm the revision is missing**

Run:

```bash
cd backend
uv run --no-sync pytest tests/test_account_login_identifier_migration.py -q
```

Expected: FAIL because the revision and its downgrade preflight do not exist.
The disposable database tests may skip until the literal database URL is
provided; unit ordering/privacy tests must fail rather than skip.

- [ ] **Step 3: Implement the migration with preflight before DDL**

Create the revision with these essential operations:

```python
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from cryptography.exceptions import InvalidTag
from pydantic import EmailStr, TypeAdapter, ValidationError

from app.middleware.encryption import decrypt_field

revision = "d6e7f8a9b0c1"
down_revision = "c5d6e7f8a9b0"
branch_labels = None
depends_on = None

DOWNGRADE_BLOCKED = (
    "Downgrade blocked: account identifiers are not compatible with the legacy schema."
)
_EMAIL_ADAPTER = TypeAdapter(EmailStr)


def upgrade() -> None:
    op.alter_column("users", "email", new_column_name="login_identifier")
    op.alter_column(
        "users", "email_hmac", new_column_name="login_identifier_hmac"
    )
    op.execute(
        "ALTER INDEX ix_users_email_hmac "
        "RENAME TO ix_users_login_identifier_hmac"
    )


def _assert_legacy_email_compatible() -> None:
    connection = op.get_bind()
    ciphertext_values = connection.execute(
        sa.text("SELECT login_identifier FROM users")
    ).scalars()
    try:
        for ciphertext in ciphertext_values:
            plaintext = decrypt_field(bytes(ciphertext))
            _EMAIL_ADAPTER.validate_python(plaintext)
    except (InvalidTag, RuntimeError, ValueError, ValidationError):
        raise RuntimeError(DOWNGRADE_BLOCKED) from None


def downgrade() -> None:
    _assert_legacy_email_compatible()
    op.execute(
        "ALTER INDEX ix_users_login_identifier_hmac "
        "RENAME TO ix_users_email_hmac"
    )
    op.alter_column(
        "users", "login_identifier_hmac", new_column_name="email_hmac"
    )
    op.alter_column("users", "login_identifier", new_column_name="email")
```

Do not broaden the exception clause or log the exception, plaintext, row ID, or
account count. No update statement is permitted in downgrade preflight.

- [ ] **Step 4: Prove upgrade, downgrade, and create-all parity in a literal disposable database**

After validating that the target name is exactly
`medtimeline_issue67_migrations_ci`, recreate the disposable local database and
run:

```bash
dropdb --if-exists medtimeline_issue67_migrations_ci
createdb medtimeline_issue67_migrations_ci
cd backend
DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_issue67_migrations_ci uv run --no-sync alembic heads
DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_issue67_migrations_ci uv run --no-sync pytest tests/test_account_login_identifier_migration.py -q
```

Expected: head is `d6e7f8a9b0c1`; byte-preserving upgrade passes; email-only
downgrade passes; non-email and wrong-key downgrade cases fail before DDL with
the generic message; model `create_all` and Alembic head expose the same user
column names, nullability, type family, unique index, and index name.

- [ ] **Step 5: Run migration formatting and static checks**

Run:

```bash
cd backend
uv run --no-sync ruff check alembic/versions/d6e7f8a9b0c1_general_login_identifier.py tests/test_account_login_identifier_migration.py
uv run --no-sync ruff format --check alembic/versions/d6e7f8a9b0c1_general_login_identifier.py tests/test_account_login_identifier_migration.py
```

Expected: both commands pass.

- [ ] **Step 6: Create the root-owned Task 3 review commit and run both review gates**

After the implementer report contains migration/unit/integration evidence and
self-review, run:

```bash
issue67_task3_base="$(git rev-parse HEAD)"
git add -- \
  backend/alembic/versions/d6e7f8a9b0c1_general_login_identifier.py \
  backend/tests/test_account_login_identifier_migration.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat(db): rename encrypted login identifier columns"
issue67_task3_head="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/review-package \
  "$issue67_task3_base" "$issue67_task3_head"
```

Verify the base against the ledger and the cached list against these two paths.
First dispatch one independent task reviewer for both spec-compliance and code-
quality verdicts. Then dispatch a separate targeted security reviewer with the
same brief, report, and package. The security review is the sole per-task
exception to the one-reviewer rule and is limited to encryption-key handling,
in-process-only decryption, content-free failure/logging, byte preservation, and
proof that every refusal happens before DDL or identifier writes.

Any Critical or Important finding from either reviewer uses one fresh fixer,
covering migration tests, a root-owned amend, and a regenerated same-base
package. Re-run both Task 3 reviews after the amended diff. Record the accepted
range and any Minor findings only after both gates pass; do not start Task 4
earlier.

---

### Task 4: Backend Fixture Migration and Broad Regression Parity

**Files:**
- Modify mechanical ORM constructor keywords in:
  - `backend/tests/test_dedup_orchestrator.py`
  - `backend/tests/test_encounter_enrichment.py`
  - `backend/tests/test_extraction_engine_pref.py`
  - `backend/tests/test_llm_config_resolver.py`
  - `backend/tests/test_llm_settings_models.py`
  - `backend/tests/test_local_ai_extraction_checkpoint.py`
  - `backend/tests/test_local_ai_log_privacy.py`
  - `backend/tests/test_local_ai_models.py`
  - `backend/tests/test_local_engine_integration.py`
  - `backend/tests/test_patient_demographics.py`
  - `backend/tests/test_processing_mode_snapshot.py`
  - `backend/tests/test_provider_wiring.py`
  - `backend/tests/test_records_perf.py`
  - `backend/tests/test_strict_local_failure_stages.py`
  - `backend/tests/test_strict_local_pipeline.py`
  - `backend/tests/test_unstructured_upload.py`
  - `backend/tests/test_upload_progress_cancel.py`
- Preserve historical schema references in:
  - `backend/alembic/versions/9fbdd669ed1b_initial_schema.py`
  - `backend/alembic/versions/c4d5e6f7a8b9_encrypt_phi_columns_at_rest.py`
  - pre-issue-67 seed SQL in `backend/tests/test_local_ai_migrations.py`

**Interfaces:**
- Consumes: Task 2 canonical ORM fields and Task 3 migration head.
- Produces: current-head tests and fixtures that construct `User` only with canonical ORM keywords, while historical migration tests retain historical column names at the correct revision.

- [ ] **Step 0: Prepare the Task 4 SDD handoff**

Run:

```bash
issue67_task4_base="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/task-brief \
  docs/superpowers/plans/2026-08-23-account-login-identifier.md 4
```

Using `apply_patch`, mark Task 4 `in progress` in the ledger with the actual full
base SHA. Dispatch one fresh implementer with the Task 4 brief and report path;
explicitly forbid all Git and GitHub mutations.

- [ ] **Step 1: Run the broad backend suite to capture canonical-field failures**

Run:

```bash
cd backend
uv run --no-sync pytest -m "not slow and not fidelity and not local_model and not hardware" -q
```

Expected: failures identify remaining current-head `User(email=...)`,
`email_hmac`, or current-schema SQL references. Record the exact failing list;
do not weaken assertions or skip tests.

- [ ] **Step 2: Apply the bounded mechanical fixture rewrite**

In only the listed current-head fixture/test files, apply these exact rewrites
within `User(...)` constructor calls:

```text
email=VALUE       -> login_identifier=VALUE
email_hmac=VALUE  -> login_identifier_hmac=VALUE
```

Do not rename clinical email test data, API legacy-compatibility payloads, PHI
scrubber email patterns, historical Alembic columns, or the pre-issue-67 raw SQL
seed in `test_local_ai_migrations.py`. Update `test_records_perf.py` current-head
raw inserts to canonical column names.

- [ ] **Step 3: Run a semantic search guard**

Run:

```bash
rg -n -U 'User\([\s\S]{0,260}?email(_hmac)?\s*=' backend/tests backend/app -g '!backend/.venv/**'
rg -n 'User\.email(_hmac)?|users\.email(_hmac)?' backend/app backend/tests -g '!backend/.venv/**'
```

Expected: no current model/service/test constructor references remain. Any hit
must be classified as a deliberate legacy API or historical migration fixture;
do not blindly replace it.

- [ ] **Step 4: Re-run focused and broad backend suites**

Run:

```bash
cd backend
uv run --no-sync pytest tests/test_account_login_identifier.py tests/test_account_login_identifier_migration.py tests/test_auth.py tests/test_auth_hardening.py tests/test_hipaa_compliance.py tests/test_at_rest_encryption.py tests/test_token_revocation.py -q
uv run --no-sync pytest -m "not slow and not fidelity and not local_model and not hardware" -q
uv run --no-sync ruff check app tests alembic/versions/d6e7f8a9b0c1_general_login_identifier.py
uv run --no-sync ruff format --check app tests alembic/versions/d6e7f8a9b0c1_general_login_identifier.py
```

Expected: all selected and broad offline tests pass. Any existing unrelated
baseline failure must be independently reproduced at the original commit and
reported; it must not be hidden or repaired outside issue #67.

- [ ] **Step 5: Create the root-owned Task 4 review commit and run the combined gate**

After the implementer report contains focused/broad regression evidence and
self-review, run:

```bash
issue67_task4_base="$(git rev-parse HEAD)"
git add -- \
  backend/tests/test_dedup_orchestrator.py \
  backend/tests/test_encounter_enrichment.py \
  backend/tests/test_extraction_engine_pref.py \
  backend/tests/test_llm_config_resolver.py \
  backend/tests/test_llm_settings_models.py \
  backend/tests/test_local_ai_extraction_checkpoint.py \
  backend/tests/test_local_ai_log_privacy.py \
  backend/tests/test_local_ai_models.py \
  backend/tests/test_local_engine_integration.py \
  backend/tests/test_patient_demographics.py \
  backend/tests/test_processing_mode_snapshot.py \
  backend/tests/test_provider_wiring.py \
  backend/tests/test_records_perf.py \
  backend/tests/test_strict_local_failure_stages.py \
  backend/tests/test_strict_local_pipeline.py \
  backend/tests/test_unstructured_upload.py \
  backend/tests/test_upload_progress_cancel.py
git diff --cached --name-only
git diff --cached --check
git commit -m "test(auth): use canonical user identifier fields"
issue67_task4_head="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/review-package \
  "$issue67_task4_base" "$issue67_task4_head"
```

Verify the base against the ledger and the cached list against this exact
mechanical allowlist. One independent reviewer returns both spec-compliance and
code-quality verdicts, with emphasis on accidental historical migration,
clinical-email, or compatibility-fixture renames. Critical or Important
findings use a fresh fixer, covering tests, a root-owned amend, fresh same-base
package, and re-review. Record the clean range and Minor findings before Task 5.

---

### Task 5: Frontend Contract, UI, Masking, and Auth E2E

**Files:**
- Create: `frontend/src/lib/login-identifier.ts`
- Create: `frontend/src/lib/login-identifier.unit.spec.ts`
- Modify: `frontend/src/types/api.ts`
- Modify: `frontend/src/app/(auth)/register/page.tsx`
- Modify: `frontend/src/app/(auth)/login/page.tsx`
- Modify: `frontend/src/components/retro/RetroNav.tsx`
- Modify: `frontend/src/app/(dashboard)/admin/page.tsx`
- Modify: `frontend/e2e/helpers/api-client.ts`
- Modify: `frontend/e2e/helpers/browser-login.ts`
- Modify: `frontend/e2e/helpers/auth.ts`
- Modify: `frontend/e2e/helpers/test-data.ts`
- Modify: `frontend/e2e/auth-register.spec.ts`
- Modify: `frontend/e2e/auth-login.spec.ts`
- Modify: `frontend/e2e/admin-system.spec.ts`
- Modify: `frontend/e2e/setup.spec.ts`
- Modify matching `/auth/me` success mocks in:
  - `frontend/e2e/admin-consolidation.spec.ts`
  - `frontend/e2e/admin-dedup.spec.ts`
  - `frontend/e2e/background-processing-summary.spec.ts`
  - `frontend/e2e/extraction-terminal-state.spec.ts`
  - `frontend/e2e/llm-settings.spec.ts`
  - `frontend/e2e/local-model-pack-settings.spec.ts`
  - `frontend/e2e/multi-provider-summary.spec.ts`
  - `frontend/e2e/record-extraction-evidence.spec.ts`
  - `frontend/e2e/strict-local-upload-progress.spec.ts`
  - `frontend/e2e/summary-processing-modes.spec.ts`
  - `frontend/e2e/upload-extraction-ux.spec.ts`
  - `frontend/e2e/upload-ocr-notices.spec.ts`
  - `frontend/e2e/walkthrough-defects.spec.ts`

**Interfaces:**
- Consumes: canonical `login_identifier` request/response contract and deprecated `email` response alias.
- Produces: account-name form fields and hints, canonical TypeScript types/helpers, exact Unicode-code-point mask, non-email E2E coverage.

- [ ] **Step 0: Prepare the Task 5 SDD handoff**

Run:

```bash
issue67_task5_base="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/task-brief \
  docs/superpowers/plans/2026-08-23-account-login-identifier.md 5
```

Using `apply_patch`, mark Task 5 `in progress` in the ledger with the actual full
base SHA. Dispatch one fresh implementer with the Task 5 brief and report path;
explicitly forbid all Git and GitHub mutations.

- [ ] **Step 1: Add failing unit and browser assertions**

Create the unit test:

```typescript
import { expect, test } from "@playwright/test";
import { maskLoginIdentifier } from "./login-identifier";

test("masks account names by Unicode code point", () => {
  expect(maskLoginIdentifier("a")).toBe("•");
  expect(maskLoginIdentifier("ab")).toBe("••");
  expect(maskLoginIdentifier("alice")).toBe("al•••");
  expect(maskLoginIdentifier("😀xray")).toBe("😀x•••");
});
```

Update auth E2E expectations before UI code:

```typescript
await page.locator("#loginIdentifier").fill("e2e account name");
await expect(page.getByLabel("Account name")).toHaveAttribute(
  "autocomplete",
  "username"
);
await expect(page.getByText(/does not need to be an email address/i)).toBeVisible();
```

Registration must succeed with a non-email identifier. Login must succeed with
both that account name and one pre-registered email-shaped legacy account.
`admin-system.spec.ts` must assert the label `Account name` and the exact masked
value, not an `@test.com` domain. Preserve the existing empty-input, duplicate,
wrong-password, loading-state, and post-auth navigation assertions; do not
replace them with only the new happy paths.

- [ ] **Step 2: Run unit/type checks and focused E2E to confirm failures**

Run:

```bash
cd frontend
npm ci
dropdb --if-exists medtimeline_issue67_e2e
createdb medtimeline_issue67_e2e
test ! -e /tmp/strand-issue67-no-real-fixtures
npx playwright test --config playwright.unit.config.ts
npx tsc --noEmit
env REAL_MEDICAL_FIXTURES_DIR=/tmp/strand-issue67-no-real-fixtures E2E_LOCAL_ONLY=1 E2E_DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_issue67_e2e npx playwright test e2e/auth-register.spec.ts e2e/auth-login.spec.ts e2e/admin-system.spec.ts --workers=1
```

Expected: lockfile installation succeeds; unit test fails because the helper is
missing; type/E2E checks fail on the old email-only contract and selectors. The
local-only network guard must be active; no external provider mode is permitted.

- [ ] **Step 3: Implement canonical types and masking helper**

In `frontend/src/types/api.ts`:

```typescript
export interface UserResponse {
  id: string;
  login_identifier: string;
  /** @deprecated Compatibility alias; this value is not necessarily an email. */
  email: string;
  display_name: string | null;
  is_active: boolean;
  created_at: string;
}

export interface RegisterRequest {
  login_identifier: string;
  password: string;
  display_name?: string;
}

export interface LoginRequest {
  login_identifier: string;
  password: string;
}
```

Create `frontend/src/lib/login-identifier.ts`:

```typescript
export function maskLoginIdentifier(value: string): string {
  const codePoints = Array.from(value);
  if (codePoints.length <= 2) return "•".repeat(codePoints.length);
  return `${codePoints.slice(0, 2).join("")}•••`;
}
```

- [ ] **Step 4: Implement registration/login UI and canonical requests**

Both pages use state named `loginIdentifier` and submit
`{ login_identifier: loginIdentifier, password }`. Use:

```tsx
<input
  id="loginIdentifier"
  className="auth-input"
  type="text"
  value={loginIdentifier}
  onChange={(event) => setLoginIdentifier(event.target.value)}
  required
  autoComplete="username"
  autoCapitalize="none"
  spellCheck={false}
/>
```

Registration label: `Account name`. Login label: `Account name or existing
email`. Include both approved hints verbatim before the humanizer pass:

```text
Used only to sign in to this Strand instance. It does not need to be an email address.
Leading and trailing spaces are ignored. Capitalization of A–Z does not matter; visually similar Unicode text can still be different.
```

- [ ] **Step 5: Update current-user consumers and E2E helpers**

Use `Array.from(user.login_identifier)[0]` for an avatar fallback initial so a
supplementary-plane Unicode code point is not split. In Admin, import
`maskLoginIdentifier`, label the field `Account name`, and delete `maskEmail`.

Change `ApiClient.register`, `ApiClient.login`, `getTestAuth`, and
`browserLogin` parameters to `loginIdentifier`; send the canonical JSON key and
use selector `#loginIdentifier`. Add `testIdentifier()` and
`uniqueIdentifier()` helpers for new auth tests; retain `testEmail()` only where
an email-shaped legacy compatibility case is intentional. `setup.spec.ts`
asserts `me.login_identifier` and `me.email === me.login_identifier`. Add a
matching `login_identifier` to every listed successful `/auth/me` mock while
retaining the equal deprecated `email` alias; failure-only mocks and comments
need no mechanical edit.

- [ ] **Step 6: Run frontend unit, type, lint, build, and focused auth E2E**

Run:

```bash
cd frontend
test ! -e /tmp/strand-issue67-no-real-fixtures
npx playwright test --config playwright.unit.config.ts
npx tsc --noEmit
npm run lint
NEXT_TELEMETRY_DISABLED=1 npm run build
env REAL_MEDICAL_FIXTURES_DIR=/tmp/strand-issue67-no-real-fixtures E2E_LOCAL_ONLY=1 E2E_DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_issue67_e2e npx playwright test e2e/auth-register.spec.ts e2e/auth-login.spec.ts e2e/auth-guard.spec.ts e2e/auth-refresh.spec.ts e2e/auth-logout-refresh-race.spec.ts e2e/admin-system.spec.ts --workers=1
```

Expected: all commands pass. E2E uses only loopback services, the dedicated
synthetic database, and the network-denied local-only profile.

- [ ] **Step 7: Run the frontend identity-semantics guard and classify every hit**

First run hard no-hit guards for stale canonical consumers and email-only auth
forms/helpers:

```bash
! rg -n '\b(user|me)\.email\b' frontend/src -g '*.ts' -g '*.tsx'
! rg -n '#email|id="email"|type="email"|autoComplete="email"' \
  'frontend/src/app/(auth)' frontend/e2e/helpers \
  frontend/e2e/auth-register.spec.ts frontend/e2e/auth-login.spec.ts
! rg -n -U 'interface (RegisterRequest|LoginRequest)[\s\S]{0,220}\bemail\s*:' \
  frontend/src/types/api.ts
! rg -n -U '(register|login|browserLogin)\([\s\S]{0,180}\bemail\b' \
  frontend/e2e/helpers
! rg -n -U 'JSON\.stringify\(\{[\s\S]{0,180}\bemail\b' \
  frontend/e2e/helpers
```

Then inventory all remaining email semantics in current application code, auth
helpers/specs, and `/auth/me` mocks:

```bash
rg -n '\bemail\b' \
  frontend/src \
  frontend/e2e/helpers \
  frontend/e2e/auth-register.spec.ts \
  frontend/e2e/auth-login.spec.ts \
  frontend/e2e/admin-system.spec.ts \
  frontend/e2e/setup.spec.ts \
  $(rg -l '/auth/me' frontend/e2e -g '*.spec.ts')
```

Expected: every hard guard has no matches. Add a file-and-line classification
of every inventory hit to the implementer report. Each remaining hit must be
exactly one of: the deprecated `UserResponse.email` alias declared, mirrored,
or equality-tested alongside `login_identifier`; an explicit legacy email-
shaped compatibility case; or an unrelated clinical/contact email field. A
canonical auth request, UI consumer, helper parameter, selector, or success
mock that still relies only on `email` is a failure. If a real stale consumer
requires a file outside Task 5's enumerated allowlist, stop for a root plan
update before editing it.

- [ ] **Step 8: Create the root-owned Task 5 review commit and run the combined gate**

After the implementer report contains frontend test evidence, the semantic-hit
classification, and self-review, run:

```bash
issue67_task5_base="$(git rev-parse HEAD)"
git add -- \
  'frontend/src/app/(auth)/login/page.tsx' \
  'frontend/src/app/(auth)/register/page.tsx' \
  'frontend/src/app/(dashboard)/admin/page.tsx' \
  frontend/src/components/retro/RetroNav.tsx \
  frontend/src/lib/login-identifier.ts \
  frontend/src/lib/login-identifier.unit.spec.ts \
  frontend/src/types/api.ts \
  frontend/e2e/helpers/api-client.ts \
  frontend/e2e/helpers/auth.ts \
  frontend/e2e/helpers/browser-login.ts \
  frontend/e2e/helpers/test-data.ts \
  frontend/e2e/admin-consolidation.spec.ts \
  frontend/e2e/admin-dedup.spec.ts \
  frontend/e2e/admin-system.spec.ts \
  frontend/e2e/auth-login.spec.ts \
  frontend/e2e/auth-register.spec.ts \
  frontend/e2e/background-processing-summary.spec.ts \
  frontend/e2e/extraction-terminal-state.spec.ts \
  frontend/e2e/llm-settings.spec.ts \
  frontend/e2e/local-model-pack-settings.spec.ts \
  frontend/e2e/multi-provider-summary.spec.ts \
  frontend/e2e/record-extraction-evidence.spec.ts \
  frontend/e2e/setup.spec.ts \
  frontend/e2e/strict-local-upload-progress.spec.ts \
  frontend/e2e/summary-processing-modes.spec.ts \
  frontend/e2e/upload-extraction-ux.spec.ts \
  frontend/e2e/upload-ocr-notices.spec.ts \
  frontend/e2e/walkthrough-defects.spec.ts
git diff --cached --name-only
git diff --cached --check
git commit -m "feat(web): use account names for authentication"
issue67_task5_head="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/review-package \
  "$issue67_task5_base" "$issue67_task5_head"
```

Verify the base against the ledger and the cached paths against the Task 5
allowlist. One independent reviewer returns both spec-compliance and code-
quality verdicts, including UI truthfulness, accessibility, Unicode masking,
mock parity, E2E evidence, and the semantic-hit classification. Critical or
Important findings use a fresh fixer, covering tests, a root-owned amend, fresh
same-base package, and re-review. Record the accepted range and Minor findings
before Task 6.

---

### Task 6: Operator Script, Documentation, and Humanized Public Prose

**Files:**
- Modify: `scripts/e2e_full_v2.py`
- Modify: `README.md`
- Modify: `docs/backend-handoff.md`
- Modify: `docs/operations-backup-restore.md`
- Modify public copy already changed in:
  - `frontend/src/app/(auth)/register/page.tsx`
  - `frontend/src/app/(auth)/login/page.tsx`

**Interfaces:**
- Consumes: final canonical API/database contract, migration rollback boundary, approved copy facts.
- Produces: truthful public wording and a synthetic account setup script that no longer queries encrypted identity plaintext.

- [ ] **Step 0: Prepare the Task 6 SDD handoff**

Run:

```bash
issue67_task6_base="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/task-brief \
  docs/superpowers/plans/2026-08-23-account-login-identifier.md 6
```

Using `apply_patch`, mark Task 6 `in progress` in the ledger with the actual full
base SHA. Dispatch one fresh implementer with the Task 6 brief and report path;
explicitly forbid all Git and GitHub mutations.

- [ ] **Step 1: Add documentation/search regression expectations**

Before editing prose, run and preserve the hits that must change:

```bash
rg -n 'account email|users\.email|email_hmac|duplicate email|"email": "user@example.com"|#email|type="email"' README.md docs/backend-handoff.md docs/operations-backup-restore.md scripts/e2e_full_v2.py frontend/src/app/\(auth\)
```

Expected: hits identify current public/API/storage wording and script payloads.
Historical Alembic files are intentionally outside this scan.

- [ ] **Step 2: Update the standalone script without running real-data/provider work**

Replace the auth setup with canonical synthetic identity and API-derived user
ID:

```python
login_identifier = f"e2e-v2-{stamp}"
with short() as client:
    registration = client.post(
        f"{BASE}/auth/register",
        json={
            "login_identifier": login_identifier,
            "password": pw,
            "display_name": "E2E V2",
        },
    )
    registration.raise_for_status()
    uid = registration.json()["id"]
    token = client.post(
        f"{BASE}/auth/login",
        json={"login_identifier": login_identifier, "password": pw},
    ).json()["access_token"]
results["login_identifier"] = login_identifier
log(f"fresh user {login_identifier}")
```

Delete the plaintext `SELECT id FROM users WHERE email=...` lookup. Do not run
the rest of this script because it intentionally uses private fixtures and a
networked summary provider outside this task's authorization.

- [ ] **Step 3: Update API, privacy, encryption, backup, and downgrade prose**

Make these facts explicit:

- README says account login identifiers—not account emails—are encrypted at
  rest and that no email is needed for account authentication.
- Backend handoff examples use canonical `login_identifier`, document legacy
  `email` as deprecated input/output compatibility, document exact generic
  422/409/401 bodies, and change the status table to duplicate identifier.
- Backend handoff states that lookup uses exactly `value.strip().lower()`, not
  full Unicode caseless or canonical matching, and includes the approved
  `Alice`/`alice`, `Straße`/`STRASSE`, and precomposed/decomposed `Å` examples.
- Backup/restore lists `users.login_identifier` and
  `users.login_identifier_hmac`, says a failed sign-in may indicate identifier
  decryption/key failure, and states that downgrade after an emailless account
  requires restoring the pre-migration backup.
- Auth pages claim only that account authentication does not require/send email;
  they do not claim the entire network-capable web process is local-only.

- [ ] **Step 4: Invoke and apply the humanizer workflow**

Read `/Users/potalora/.codex/skills/humanizer/SKILL.md` completely, then review
all changed public prose in the three documents and two auth pages. Remove
formulaic, promotional, repetitive, or over-polished wording without weakening
the exact security/privacy caveats or normalization rules.

- [ ] **Step 5: Run prose/script/static checks**

Run:

```bash
uv run --project backend --no-sync ruff check scripts/e2e_full_v2.py
python3 -m py_compile scripts/e2e_full_v2.py
rg -n 'account email|users\.email|email_hmac|duplicate email|#email|type="email"' README.md docs/backend-handoff.md docs/operations-backup-restore.md scripts/e2e_full_v2.py frontend/src/app/\(auth\)
git diff --check
```

Expected: Ruff, compilation, and whitespace checks pass. Remaining search hits
must be only clearly labeled legacy compatibility examples; no UI or current
storage wording calls arbitrary identifiers email.

- [ ] **Step 6: Create the root-owned Task 6 review commit and run the combined gate**

After the implementer report contains humanizer influence, prose/static checks,
and self-review, run:

```bash
issue67_task6_base="$(git rev-parse HEAD)"
git add -- \
  README.md \
  docs/backend-handoff.md \
  docs/operations-backup-restore.md \
  'frontend/src/app/(auth)/login/page.tsx' \
  'frontend/src/app/(auth)/register/page.tsx' \
  scripts/e2e_full_v2.py
git diff --cached --name-only
git diff --cached --check
git commit -m "docs(auth): explain private account identifiers"
issue67_task6_head="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/review-package \
  "$issue67_task6_base" "$issue67_task6_head"
```

Verify the base against the ledger and the cached paths against this Task 6
allowlist. One independent reviewer returns both spec-compliance and code-
quality verdicts, including factual normalization, privacy/recovery wording,
humanizer restraint, and the operator script's no-private-data boundary.
Critical or Important findings use a fresh fixer, covering checks, a root-owned
amend, fresh same-base package, and re-review. Record the accepted range and
Minor findings before Task 7.

---

### Task 7: Broad Review, Fresh Verification, and Unmerged PR

**Files:**
- Review all issue #67 files from Tasks 1–6.
- Do not modify unrelated files.

**Interfaces:**
- Consumes: all task deliverables and task-scoped review approvals.
- Produces: broad independent review, final verification evidence, the reviewed
  multi-commit `codex/issue-67-login-identifier` branch, and one unmerged PR
  closing issue #67.

- [ ] **Step 1: Build the whole-branch review package and run a broad independent review**

Using `apply_patch`, mark Task 7 `in progress` in the ledger. Run:

```bash
issue67_merge_base="$(git merge-base origin/main HEAD)"
issue67_review_head="$(git rev-parse HEAD)"
/Users/potalora/.codex/skills/subagent-driven-development/scripts/review-package \
  "$issue67_merge_base" "$issue67_review_head"
```

Invoke `requesting-code-review` with the approved spec, this plan, the SDD
ledger including all Minor findings, and the generated whole-branch package.
Use a capable independent reviewer and explicit axes: API compatibility,
route-name-scoped sanitized 422 behavior, normalization truthfulness,
encryption/HMAC preservation, migration preflight ordering/privacy,
JWT/lockout/rate-limit invariants, frontend semantic classification, UI copy,
and test gaps.

Critical or Important findings block shipping. Dispatch one fresh fix subagent
with the complete finding set and covering tests; it must not stage or commit.
The worktree root stages only the specifically changed paths from the union of
the explicit Preflight and Task 1–6 allowlists, creates one focused Task 7 review
fix commit, regenerates the whole-branch package from the unchanged merge base,
and requests re-review. Do not amend accepted Task 1–6 commits. Record Minor
residuals and the clean whole-branch verdict in the ledger.

- [ ] **Step 2: Invoke verification-before-completion and run fresh backend gates**

Run from a clean test environment without provider credentials:

```bash
dropdb --if-exists medtimeline_issue67_migrations_ci
createdb medtimeline_issue67_migrations_ci
cd backend
env GEMINI_API_KEY= OPENAI_API_KEY= ANTHROPIC_API_KEY= OPENROUTER_API_KEY= VERTEX_PROJECT= uv run --no-sync pytest tests/test_account_login_identifier.py tests/test_account_login_identifier_migration.py tests/test_auth.py tests/test_auth_hardening.py tests/test_hipaa_compliance.py tests/test_at_rest_encryption.py tests/test_token_revocation.py -q
env GEMINI_API_KEY= OPENAI_API_KEY= ANTHROPIC_API_KEY= OPENROUTER_API_KEY= VERTEX_PROJECT= uv run --no-sync pytest -m "not slow and not fidelity and not local_model and not hardware" -q
uv run --no-sync ruff check app tests alembic/versions/d6e7f8a9b0c1_general_login_identifier.py
uv run --no-sync ruff format --check app tests alembic/versions/d6e7f8a9b0c1_general_login_identifier.py
DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_issue67_migrations_ci uv run --no-sync alembic heads
DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_issue67_migrations_ci uv run --no-sync pytest tests/test_account_login_identifier_migration.py -q
```

Expected: all focused/broad tests and static checks pass; Alembic reports only
`d6e7f8a9b0c1 (head)`.

- [ ] **Step 3: Run fresh frontend gates and local-only auth E2E**

Run:

```bash
dropdb --if-exists medtimeline_issue67_e2e
createdb medtimeline_issue67_e2e
test ! -e /tmp/strand-issue67-no-real-fixtures
cd frontend
npx playwright test --config playwright.unit.config.ts
npx tsc --noEmit
npm run lint
NEXT_TELEMETRY_DISABLED=1 npm run build
env REAL_MEDICAL_FIXTURES_DIR=/tmp/strand-issue67-no-real-fixtures E2E_LOCAL_ONLY=1 E2E_DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_issue67_e2e npx playwright test e2e/auth-register.spec.ts e2e/auth-login.spec.ts e2e/auth-guard.spec.ts e2e/auth-refresh.spec.ts e2e/auth-logout-refresh-race.spec.ts e2e/admin-system.spec.ts --workers=1
```

Expected: all commands pass with loopback-only services and synthetic data.

Recreate the same literal database once more, re-check that the fixture guard
path does not exist, and run the broader local-only browser suite because the
canonical auth helpers are shared by nearly every authenticated flow:

```bash
dropdb --if-exists medtimeline_issue67_e2e
createdb medtimeline_issue67_e2e
test ! -e /tmp/strand-issue67-no-real-fixtures
cd frontend
env REAL_MEDICAL_FIXTURES_DIR=/tmp/strand-issue67-no-real-fixtures E2E_LOCAL_ONLY=1 E2E_DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_issue67_e2e npx playwright test --workers=1
```

Expected: the full local-only suite passes or skips only tests whose private
fixture directories are absent. The network-denied profile remains active, and
the nonexistent override prevents `.env.test.local` from exposing real medical
fixture paths to the run.

- [ ] **Step 4: Audit the clean multi-commit branch, ledger, and semantic guards**

Run:

```bash
issue67_merge_base="$(git merge-base origin/main HEAD)"
test -z "$(git status --porcelain=v1)"
git log --reverse --oneline "$issue67_merge_base"..HEAD
git diff --stat "$issue67_merge_base"..HEAD
git diff --check "$issue67_merge_base"..HEAD
git diff --name-only "$issue67_merge_base"..HEAD
cat .superpowers/sdd/progress.md
rg -n -U 'User\([\s\S]{0,260}?email(_hmac)?\s*=' backend/tests backend/app -g '!backend/.venv/**'
! rg -n '\b(user|me)\.email\b' frontend/src -g '*.ts' -g '*.tsx'
! rg -n '#email|id="email"|type="email"|autoComplete="email"' \
  'frontend/src/app/(auth)' frontend/e2e/helpers \
  frontend/e2e/auth-register.spec.ts frontend/e2e/auth-login.spec.ts
```

Expected: the worktree is clean; the branch log contains the root-owned planning
baseline and accepted Task 1–6 commits, plus a Task 7 fix commit only if broad
review required one. Every Task 1–6 ledger line names its accepted range and
review verdict. The branch diff contains exactly the union of the explicit
Preflight and Task 1–6 allowlists—issue #67 plan/spec, backend auth/model/
migration/tests, bounded fixtures, frontend auth/account files/tests, script,
and three documents. No secret, real fixture, generated build output, database
file, stale canonical email consumer, or unrelated change is present.

- [ ] **Step 5: Verify the reviewed branch is still based on current main**

Run:

```bash
git fetch origin
test "$(git merge-base origin/main HEAD)" = "$(git rev-parse origin/main)"
test "$(git branch --show-current)" = "codex/issue-67-login-identifier"
test -z "$(git status --porcelain=v1)"
git log --reverse --format='%H %s' origin/main..HEAD
```

Expected: `origin/main` is still the exact branch base, the worktree is clean,
and the log preserves the root-owned planning and task commits reviewed through
the SDD ledger. Do not restage, squash, amend, or replace them with a final
aggregate commit. If `origin/main` advanced after branch creation, stop for root
integration guidance; rebasing would change every accepted range and requires
regenerating the whole-branch package plus proportionate re-review/verification.

- [ ] **Step 6: Push and open one unmerged PR linked to issue #67**

Run:

```bash
git push -u origin codex/issue-67-login-identifier
gh pr create --repo potalora/strand --base main --head codex/issue-67-login-identifier --title "Allow account registration without an email address" --body-file /tmp/strand-issue-67-pr.md
```

The root agent must create `/tmp/strand-issue-67-pr.md` with a concise summary,
migration/compatibility notes, exact verification results, residual risks, and
`Closes #67`. Do not merge, enable auto-merge, comment on the issue separately,
or alter issue metadata.

- [ ] **Step 7: Report evidence to the orchestrating root task**

Report the branch, ordered commit list, per-task ledger ranges, PR URL, exact
test commands/counts/results, migration head and downgrade evidence, humanizer
impact, broad review verdict, and any residual risks. State explicitly that the
task commits were not squashed and the PR remains unmerged for independent root
review and sequential merge.
