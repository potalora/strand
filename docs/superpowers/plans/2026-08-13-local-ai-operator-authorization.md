# Local AI operator authorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict web-based mutation and detailed inspection of the machine-global validated local model pack to explicitly configured machine-operator accounts, while leaving authenticated users able to read pack readiness and manage their own strict-local jobs.

**Architecture:** Parse `LOCAL_AI_OPERATOR_USER_IDS` once during settings construction as a fail-closed comma-separated UUID allowlist. A dependency layers authorization on the existing authenticated-user dependency so missing/revoked credentials retain `401`, while authenticated non-operators receive `403`. Pack lifecycle routes and detailed operation reads use that dependency; the status payload tells the UI whether the signed-in user can manage the pack, and non-operators receive readiness without lifecycle-operation detail.

**Tech Stack:** Python 3.11, FastAPI dependencies, Pydantic Settings v2, pytest/pytest-asyncio/httpx, Next.js 16, TypeScript, Playwright unit tests, Ruff, ESLint.

## Global Constraints

- Work only on Track A: global pack authorization. Do not alter Track B clinical extraction/bounded-inference code, Track C user-owned job lifecycle/upload identity code, or Track D worker-runtime attestation/release evidence.
- Validated strict-local routing must still branch before any cloud-capable provider is constructed and must never fall back to cloud.
- `LOCAL_AI_OPERATOR_USER_IDS` is a comma-separated UUID allowlist. An empty value deliberately grants no web pack-mutation authority; malformed non-empty input must abort application startup.
- Do not promote the first registered user or infer authority from a user-owned job, upload, or patient row. Local CLI maintenance remains an operating-system-owner action outside web-account authorization.
- Preserve the existing content-free response contract: never add document text, prompt text, model output, patient identity, paths, or raw exceptions to status, lifecycle, progress, or failure responses.
- Keep status authenticated and available to every authenticated user. Pack lifecycle mutation and `GET /local-ai/operations/{operation_id}` require a machine operator. User-owned `/local-ai/jobs` list/get/cancel/retry routes retain their current owner scoping and do not require operator authority.
- Existing advisory-lock, active-job `409`, audit actor, and lifecycle behavior remain unchanged after authorization succeeds.
- Tests use deterministic synthetic manifests and files only. Do not download models, call a provider, use private medical fixtures, or run live cloud tests.
- Subagents must not commit. The root agent owns review and verification. Every commit block is a
  proposed checkpoint only; do not stage or commit unless Pedro explicitly authorizes it.
- Keep unrelated worktrees and untracked files untouched. Start this plan in a Codex-managed
  **Worktree** task based on `codex/pr62-pr63-remediation-planning`; retain it as
  `codex/local-ai-operator-authorization` only when the implementation is ready for root review.

---

## File structure

| File | Responsibility |
| --- | --- |
| `backend/app/config.py` | Validate the raw operator allowlist at settings startup and expose its parsed UUID set. |
| `backend/app/dependencies.py` | Provide one reusable predicate and FastAPI dependency that preserves authentication semantics and denies non-operators. |
| `backend/app/schemas/local_ai.py` | Add the bounded authorization capability bit to the existing pack-status contract. |
| `backend/app/api/local_ai.py` | Apply operator authorization to every machine-global lifecycle/detail route and make status capability-aware without changing job routes. |
| `backend/tests/test_config_hardening.py` | Exercise valid, empty, malformed, and separator-invalid configuration at settings construction. |
| `backend/tests/test_local_ai_api.py` | Prove every global lifecycle/detail endpoint returns `401` or `403` correctly, operators retain successful behavior, and status stays generally readable. |
| `frontend/src/types/local-ai.ts` | Add `can_manage_pack` to the typed status boundary. |
| `frontend/src/types/local-ai.unit.spec.ts` | Update the server-status fixture so route/type tests stay contract-complete. |
| `frontend/src/components/admin/ValidatedLocalPackCard.tsx` | Show pack readiness to all authenticated users; hide lifecycle controls and show fixed operator-management copy to non-operators. |
| `frontend/src/components/admin/ValidatedLocalPackCard.unit.spec.ts` | Test the exported control-visibility decision and fixed non-operator copy without a live backend. |
| `frontend/e2e/local-model-pack-settings.spec.ts` | Route-mock a non-operator response and prove the rendered card exposes readiness but no lifecycle controls or operation polling. |
| `.env.example` | Document the empty-by-default operator allowlist and its comma-separated UUID format. |
| `docs/backend-handoff.md` | Specify capability-aware status, operator-only lifecycle/detail endpoints, the exact `401`/`403` split, and the CLI boundary. |
| `README.md` | Explain the administrator-facing local-pack ownership boundary in plain language. |

## Interfaces

```python
# backend/app/config.py
class Settings(BaseSettings):
    local_ai_operator_user_ids: str = ""

    @property
    def local_ai_operator_ids(self) -> frozenset[UUID]:
        pass

# backend/app/dependencies.py
def is_local_ai_operator(user_id: UUID) -> bool:
    pass

async def require_local_ai_operator(
    user_id: UUID = Depends(get_authenticated_user_id),
) -> UUID:
    raise NotImplementedError

# backend/app/schemas/local_ai.py
class LocalPackStatusResponse(_StrictResponse):
    can_manage_pack: bool

# backend/app/api/local_ai.py
async def get_local_pack_status(
    user_id: UUID = Depends(get_authenticated_user_id),
) -> LocalPackStatusResponse:
    raise NotImplementedError

async def get_local_pack_operation(
    operation_id: UUID,
    _operator_id: UUID = Depends(require_local_ai_operator),
) -> LocalPackOperationResponse:
    raise NotImplementedError
```

The API applies `Depends(require_local_ai_operator)` to exactly these global routes:

| Method | Path | Existing handler |
| --- | --- | --- |
| `POST` | `/api/v1/local-ai/install` | `install_local_pack` |
| `GET` | `/api/v1/local-ai/operations/{operation_id}` | `get_local_pack_operation` |
| `POST` | `/api/v1/local-ai/operations/{operation_id}/resume` | `resume_local_pack_operation` |
| `POST` | `/api/v1/local-ai/operations/{operation_id}/retry` | `retry_local_pack_operation` |
| `POST` | `/api/v1/local-ai/verify` | `verify_local_pack` |
| `POST` | `/api/v1/local-ai/update` | `update_local_pack` |
| `POST` | `/api/v1/local-ai/rollback` | `rollback_local_pack` |
| `DELETE` | `/api/v1/local-ai/models/{role}` | `remove_local_model` |
| `DELETE` | `/api/v1/local-ai` | `remove_local_pack` |

`GET /api/v1/local-ai/status` continues to use `get_authenticated_user_id`. Its `can_manage_pack` is true exactly when `is_local_ai_operator(user_id)` is true. To avoid exposing detailed lifecycle progress through the status shortcut, it returns `operation=None` to a non-operator even if an operation exists; its state still truthfully reports `downloading` or `verifying`.

### Task 1: Fail-closed settings and reusable operator dependency

**Files:**
- Modify: `backend/app/config.py:3-8,19-70,210-230`
- Modify: `backend/app/dependencies.py:3-50`
- Modify: `backend/tests/test_config_hardening.py:1-81`
- Test: `backend/tests/test_local_ai_api.py:1-36` (add the reusable operator-header fixture only; route assertions belong to Task 2)

**Consumes:** The existing `Settings` Pydantic-v2 model, singleton `settings`, `get_authenticated_user_id() -> UUID`, and HTTP `401` behavior in `backend/app/dependencies.py:23-50`.

**Produces:** `Settings.local_ai_operator_user_ids: str`, `Settings.local_ai_operator_ids: frozenset[UUID]`, `is_local_ai_operator(user_id: UUID) -> bool`, and `require_local_ai_operator(user_id: UUID = Depends(get_authenticated_user_id)) -> UUID`. Later route work relies on this dependency and must not duplicate allowlist parsing.

- [ ] **Step 1: Write failing settings tests for empty, valid, and malformed allowlists**

Add these tests to `backend/tests/test_config_hardening.py`; reuse the existing `STRONG_SECRET` constant so production construction reaches the new validation.

```python
from uuid import uuid4

from app.config import Settings


def test_local_ai_operator_allowlist_is_empty_by_default() -> None:
    configured = Settings(app_env="development")

    assert configured.local_ai_operator_ids == frozenset()


def test_local_ai_operator_allowlist_parses_comma_separated_uuids() -> None:
    first, second = uuid4(), uuid4()
    configured = Settings(
        app_env="development",
        local_ai_operator_user_ids=f" {first}, {second} ",
    )

    assert configured.local_ai_operator_ids == frozenset({first, second})


@pytest.mark.parametrize(
    "value",
    ("not-a-uuid", "550e8400-e29b-41d4-a716-446655440000,"),
)
def test_local_ai_operator_allowlist_rejects_malformed_nonempty_values(value: str) -> None:
    with pytest.raises(ValueError, match="LOCAL_AI_OPERATOR_USER_IDS"):
        Settings(app_env="development", local_ai_operator_user_ids=value)


def test_local_ai_operator_allowlist_is_parsed_once_at_settings_construction() -> None:
    configured_operator, later_raw_value = uuid4(), uuid4()
    configured = Settings(
        app_env="development",
        local_ai_operator_user_ids=str(configured_operator),
    )
    parsed = configured.local_ai_operator_ids

    configured.local_ai_operator_user_ids = str(later_raw_value)

    assert configured.local_ai_operator_ids is parsed
    assert configured.local_ai_operator_ids == frozenset({configured_operator})
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/test_config_hardening.py -k local_ai_operator`

Expected: FAIL because `Settings` has no `local_ai_operator_ids` property and malformed operator values are currently accepted as ignored/unknown configuration.

- [ ] **Step 3: Implement one validated parser and the authorization dependency**

In `backend/app/config.py`, import `PrivateAttr` and `UUID`, add one pure parser, and store the parsed set in a private immutable attribute during settings construction. Do not re-parse the raw environment string in a dependency: the approved boundary is a startup-validated allowlist. Keep the existing production-secret validator intact; the new validator runs in development too because a malformed authorization boundary must never start permissively.

```python
from pydantic import PrivateAttr
from uuid import UUID


def _parse_local_ai_operator_ids(raw: str) -> frozenset[UUID]:
    if raw.strip() == "":
        return frozenset()
    values = [value.strip() for value in raw.split(",")]
    if any(not value for value in values):
        raise ValueError(
            "LOCAL_AI_OPERATOR_USER_IDS must be empty or a comma-separated list of UUIDs"
        )
    try:
        return frozenset(UUID(value) for value in values)
    except ValueError as exc:
        raise ValueError(
            "LOCAL_AI_OPERATOR_USER_IDS must be empty or a comma-separated list of UUIDs"
        ) from exc


class Settings(BaseSettings):
    local_ai_operator_user_ids: str = ""
    _local_ai_operator_ids: frozenset[UUID] = PrivateAttr(default_factory=frozenset)

    @model_validator(mode="after")
    def validate_local_ai_operator_user_ids(self) -> "Settings":
        self._local_ai_operator_ids = _parse_local_ai_operator_ids(
            self.local_ai_operator_user_ids
        )
        return self

    @property
    def local_ai_operator_ids(self) -> frozenset[UUID]:
        return self._local_ai_operator_ids
```

In `backend/app/dependencies.py`, import `settings` and add the predicate plus dependency immediately after `get_authenticated_user_id`. Do not accept an optional ID here: the nested dependency must be what preserves the existing authentication and token-revocation checks.

```python
from app.config import settings


def is_local_ai_operator(user_id: UUID) -> bool:
    """Return whether this authenticated user can manage the machine-global pack."""
    return user_id in settings.local_ai_operator_ids


async def require_local_ai_operator(
    user_id: UUID = Depends(get_authenticated_user_id),
) -> UUID:
    """Require configured machine-operator authority for pack management."""
    if not is_local_ai_operator(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local model pack management requires a machine operator.",
        )
    return user_id
```

At the top of `backend/tests/test_local_ai_api.py`, add a fixture for later positive route tests. It changes only the in-process settings singleton for that test and never writes `.env`.

```python
@pytest_asyncio.fixture
async def local_ai_operator_headers(client, monkeypatch: pytest.MonkeyPatch):
    headers, user_id = await auth_headers(client, email="local-ai-operator@example.com")
    monkeypatch.setattr(settings, "_local_ai_operator_ids", frozenset({UUID(user_id)}))
    return headers, UUID(user_id)
```

Add `import pytest_asyncio` alongside the existing pytest imports. The fixture changes only the already-parsed private set on the in-process singleton; `monkeypatch` cleanup restores the fail-closed default after every test. The preceding direct parser test proves the returned `frozenset` remains unchanged if the source raw string is subsequently reassigned, so authorization cannot silently start reparsing mutable configuration.

- [ ] **Step 4: Run the settings and dependency lint checks to verify they pass**

Run: `cd backend && uv run pytest -q tests/test_config_hardening.py -k local_ai_operator && uv run ruff check app/config.py app/dependencies.py tests/test_config_hardening.py tests/test_local_ai_api.py`

Expected: the four new settings tests PASS and Ruff reports `All checks passed!`.

- [ ] **Step 5: Root-agent review and prepare the proposed authorization checkpoint**

Verify the diff contains only the four files in this task. If Pedro separately authorizes a
commit, the root agent may then run:

```bash
git add backend/app/config.py backend/app/dependencies.py \
  backend/tests/test_config_hardening.py backend/tests/test_local_ai_api.py
git commit -m "feat(local-ai): add machine operator authorization"
```

Expected if authorized: one new commit containing only settings/dependency/test-fixture work. No
subagent commits.

### Task 2: Gate every global lifecycle and operation-detail endpoint

**Files:**
- Modify: `backend/app/schemas/local_ai.py:91-109`
- Modify: `backend/app/api/local_ai.py:16-19,592-703,933-1203`
- Modify: `backend/tests/test_local_ai_api.py:1119-1719`

**Consumes:** `require_local_ai_operator(user_id: UUID = Depends(get_authenticated_user_id)) -> UUID` and `is_local_ai_operator(user_id: UUID) -> bool` from Task 1; existing `_queue_operation`, `_restart_operation`, `_acquire_pack_mutation_db_guard`, and audit calls.

**Produces:** `LocalPackStatusResponse.can_manage_pack: bool`; operator-gated global lifecycle routes; status that is readable by any authenticated user but suppresses `operation` detail for non-operators. The frontend in Task 3 consumes only the additive boolean and existing nullable operation field.

- [ ] **Step 1: Write failing API regression tests for authorization matrix and status capability**

Add one parametrized test that creates an ordinary authenticated account, configures a different UUID as the sole operator, and asserts every global endpoint returns `403` before it reaches any mutation code. Use a syntactically valid random operation UUID because authorization must run before operation lookup.

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("post", "/api/v1/local-ai/install"),
        ("get", "/api/v1/local-ai/operations/00000000-0000-0000-0000-000000000001"),
        ("post", "/api/v1/local-ai/operations/00000000-0000-0000-0000-000000000001/resume"),
        ("post", "/api/v1/local-ai/operations/00000000-0000-0000-0000-000000000001/retry"),
        ("post", "/api/v1/local-ai/verify"),
        ("post", "/api/v1/local-ai/update"),
        ("post", "/api/v1/local-ai/rollback"),
        ("delete", "/api/v1/local-ai/models/summary"),
        ("delete", "/api/v1/local-ai"),
    ),
)
async def test_non_operator_cannot_access_any_global_pack_endpoint(
    client, monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    headers, _user_id = await auth_headers(client, email="ordinary-pack-user@example.com")
    monkeypatch.setattr(settings, "_local_ai_operator_ids", frozenset({uuid4()}))

    response = await getattr(client, method)(path, headers=headers)

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Local model pack management requires a machine operator."
    }
```

Add the representative unauthenticated test and the status tests below. `status` must remain readable and truthful about readiness while not supplying a detail object to the ordinary user.

```python
@pytest.mark.asyncio
async def test_global_pack_endpoint_preserves_401_without_credentials(client) -> None:
    response = await client.post("/api/v1/local-ai/install")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_status_is_readable_but_hides_operation_detail_from_non_operator(
    client, monkeypatch: pytest.MonkeyPatch, local_ai_paths
) -> None:
    headers, _user_id = await auth_headers(client, email="status-only-user@example.com")
    monkeypatch.setattr(settings, "_local_ai_operator_ids", frozenset({uuid4()}))

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    assert response.json()["can_manage_pack"] is False
    assert response.json()["operation"] is None
```

Convert `test_install_returns_document_free_operation` to use `local_ai_operator_headers`, then add this positive capability assertion to `test_status_reflects_exact_validation_receipt_without_fake_memory` after configuring that fixture:

```python
assert body["can_manage_pack"] is True
```

Also update every existing successful lifecycle call in this test module (`install`, `verify`, `update`, `rollback`, `resume`, `retry`, role removal, and pack removal) to request `local_ai_operator_headers` instead of calling `auth_headers` directly. This preserves their established `202`, `200`, `204`, lock, `409`, and audit assertions under the new authorization boundary rather than weakening the new allowlist for tests.

- [ ] **Step 2: Run the focused API tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/test_local_ai_api.py -k "non_operator or preserves_401 or status_is_readable or install_returns_document_free_operation"`

Expected: FAIL because non-operators currently reach lifecycle handlers, the status JSON has no `can_manage_pack`, and existing successful lifecycle tests have not yet configured an operator.

- [ ] **Step 3: Implement the additive response contract and apply the single dependency consistently**

In `backend/app/schemas/local_ai.py`, make capability explicit and bounded; it is authorization state, not an inferred frontend policy.

```python
class LocalPackStatusResponse(_StrictResponse):
    """Current pack readiness plus this user's management capability."""

    platform: PackPlatform
    compatible: bool
    enabled: bool
    can_manage_pack: bool
    state: PackState
    # Keep the existing status_reason through operation declarations unchanged.
```

In `backend/app/api/local_ai.py`, import the two Task-1 symbols and compute the one capability bit once in the status handler. Use that exact value in both return paths, including the missing-manifest failure path.

```python
from app.dependencies import (
    get_authenticated_user_id,
    is_local_ai_operator,
    require_local_ai_operator,
)


@router.get("/status", response_model=LocalPackStatusResponse)
async def get_local_pack_status(
    user_id: UUID = Depends(get_authenticated_user_id),
) -> LocalPackStatusResponse:
    can_manage_pack = is_local_ai_operator(user_id)
    # Existing manifest/state computation follows unchanged.
    return LocalPackStatusResponse(
        platform=platform_name,  # type: ignore[arg-type]
        compatible=compatible,
        enabled=settings.local_ai_enabled,
        can_manage_pack=can_manage_pack,
        state=pack_state,  # type: ignore[arg-type]
        status_reason=status_reason,
        active_revision=active_revision,
        available_revision=manifest.pack_revision,
        models=models,
        operation=(
            _operation_response(operation_store, latest)
            if can_manage_pack and latest
            else None
        ),
    )
```

Update the earlier missing-manifest return in the same handler too, so its required response field cannot be omitted and an operator can still observe a persisted operation:

```python
except LocalAIError:
    return LocalPackStatusResponse(
        platform=platform_name,  # type: ignore[arg-type]
        compatible=compatible,
        enabled=settings.local_ai_enabled,
        can_manage_pack=can_manage_pack,
        state="failed",
        active_revision=None,
        available_revision=None,
        models=[],
        operation=(
            _operation_response(operation_store, latest)
            if can_manage_pack and latest
            else None
        ),
    )
```

Apply `Depends(require_local_ai_operator)` to each route in the interface table. Retain a named `user_id` for mutations because it remains the audit actor; name the unused values `_operator_id` only on the operation-detail `GET`. For example:

```python
async def install_local_pack(
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID = Depends(require_local_ai_operator),
    db: AsyncSession = Depends(get_db),
) -> LocalPackOperationCreated:
    return await _queue_operation(
        action="install",
        background_tasks=background_tasks,
        request=request,
        user_id=user_id,
        db=db,
    )


async def get_local_pack_operation(
    operation_id: UUID,
    _operator_id: UUID = Depends(require_local_ai_operator),
) -> LocalPackOperationResponse:
    operation_store = _operations()
    operation = operation_store.get(str(operation_id))
    if operation is None:
        raise HTTPException(status_code=404, detail="Operation not found.")
    return _operation_response(operation_store, operation)
```

Make the same dependency replacement at `backend/app/api/local_ai.py:1043-1049`, `1064-1070`, `1086-1091`, `1106-1111`, `1126-1131`, `1145-1150`, and `1177-1181`. Do not add it to `GET /status` or any `/jobs` route. Do not move Track C's job cancellation/retry code while editing nearby lines.

- [ ] **Step 4: Run focused backend regression tests to verify the behavior passes**

Run: `cd backend && uv run pytest -q tests/test_local_ai_api.py -k "non_operator or preserves_401 or status_is_readable or install_returns_document_free_operation or operation_status_is_persisted or resume_and_retry or remove_refuses" && uv run ruff check app/api/local_ai.py app/schemas/local_ai.py tests/test_local_ai_api.py`

Expected: all selected tests PASS; the ordinary account receives `403` for all nine global endpoints, no credential receives `401`, the configured operator retains lifecycle behavior/audit identity, and Ruff reports `All checks passed!`.

- [ ] **Step 5: Root-agent review and prepare the proposed API checkpoint**

Verify only the Task-2 files would be included. If Pedro separately authorizes a commit, the root
agent may then run:

```bash
git add backend/app/api/local_ai.py backend/app/schemas/local_ai.py \
  backend/tests/test_local_ai_api.py
git commit -m "fix(local-ai): restrict global pack lifecycle to operators"
```

Expected if authorized: one reviewable commit that does not include Track B/C/D changes. No
subagent commits.

### Task 3: Surface capability in the typed UI and hide management controls

**Files:**
- Modify: `frontend/src/types/local-ai.ts:67-82`
- Modify: `frontend/src/types/local-ai.unit.spec.ts:18-62`
- Modify: `frontend/src/components/admin/ValidatedLocalPackCard.tsx:44-308`
- Create: `frontend/src/components/admin/ValidatedLocalPackCard.unit.spec.ts`
- Modify: `frontend/e2e/local-model-pack-settings.spec.ts`

**Consumes:** The additive backend field `LocalPackStatusResponse.can_manage_pack: bool` from Task 2; the existing `useLocalPackOperation()` hook and nullable `LocalPackStatus.operation`.

**Produces:** `LocalPackStatus.can_manage_pack: boolean`, `LOCAL_PACK_OPERATOR_COPY`, `canManageValidatedLocalPack(status)`, and a card that renders model/readiness status to everyone but only renders lifecycle-control controls for a machine operator.

- [ ] **Step 1: Write failing type and presentation tests**

Update the typed `PACK_STATUS` fixture in `frontend/src/types/local-ai.unit.spec.ts` to include `can_manage_pack: true`; this makes a missing API-contract update a compile-time failure.

Create `frontend/src/components/admin/ValidatedLocalPackCard.unit.spec.ts` with a server-free assertion of the same values the component uses:

```typescript
import { expect, test } from "@playwright/test";
import {
  canManageValidatedLocalPack,
  LOCAL_PACK_OPERATOR_COPY,
} from "./ValidatedLocalPackCard";

test("non-operators see status but not local-pack lifecycle controls", () => {
  expect(canManageValidatedLocalPack({ can_manage_pack: false })).toBe(false);
  expect(LOCAL_PACK_OPERATOR_COPY).toBe(
    "This model pack is managed by the machine operator. You can review its status here."
  );
});

test("operators may use local-pack lifecycle controls", () => {
  expect(canManageValidatedLocalPack({ can_manage_pack: true })).toBe(true);
});
```

Also add a rendered regression to the existing
`frontend/e2e/local-model-pack-settings.spec.ts` harness. Extend its `setup()`
helper with a final `canManagePack = true` argument and include
`can_manage_pack: canManagePack` in every mocked `/local-ai/status` response.
The non-operator case must return `operation: null`, then verify that the
visible card retains its readiness text and fixed copy while every lifecycle
control is absent:

```typescript
test("non-operator sees readiness without local-pack controls", async ({ page }) => {
  await setup(
    page, "not_installed", undefined, 0, undefined, undefined, undefined,
    "feature_disabled", undefined, "completed", false
  );
  await page.goto("/admin");

  const card = page
    .getByRole("heading", { name: "Validated local pack" })
    .locator("..").locator("..").locator("..");
  await expect(card).toContainText("Optional download");
  await expect(card).toContainText("This model pack is managed by the machine operator.");
  for (const name of [
    /install local pack/i, /verify again/i, /install verified update/i,
    /roll back/i, /resume operation/i, /retry operation/i, /remove pack/i,
    /retry status check/i,
  ]) {
    await expect(card.getByRole("button", { name })).toHaveCount(0);
  }
});
```

Make the helper's route throw if this case receives `GET /local-ai/operations/`.
That confirms `operation: null` prevents detailed-operation polling for an
account that is not allowed to inspect operation detail.

- [ ] **Step 2: Run the frontend unit tests to verify they fail**

Run: `cd frontend && npx playwright test --config playwright.unit.config.ts src/components/admin/ValidatedLocalPackCard.unit.spec.ts src/types/local-ai.unit.spec.ts && npx playwright test e2e/local-model-pack-settings.spec.ts --grep "non-operator sees readiness"`

Expected: FAIL because the status type lacks `can_manage_pack`, the card does not export the tested presentation boundary, and the rendered card still exposes lifecycle controls to a non-operator.

- [ ] **Step 3: Implement the typed capability and one conditional UI boundary**

In `frontend/src/types/local-ai.ts`, add the required backend field rather than deriving it from local state:

```typescript
export interface LocalPackStatus {
  platform: "apple_silicon" | "linux_cpu" | "linux_cuda" | "linux_rocm" | "unsupported";
  compatible: boolean;
  enabled: boolean;
  can_manage_pack: boolean;
  state: PackState;
  status_reason: LocalPackStatusReason | null;
  active_revision: string | null;
  available_revision: string | null;
  models: LocalModelArtifact[];
  operation: LocalPackOperation | null;
}
```

In `frontend/src/components/admin/ValidatedLocalPackCard.tsx`, export the fixed copy and pure decision at module scope, then use it around the existing lifecycle button container. Do not disable controls into view: non-operators must not see install, verify, update, rollback, resume, retry-operation, remove-pack, or polling-retry controls.

```typescript
export const LOCAL_PACK_OPERATOR_COPY =
  "This model pack is managed by the machine operator. You can review its status here.";

export function canManageValidatedLocalPack(
  status: Pick<LocalPackStatus, "can_manage_pack">
): boolean {
  return status.can_manage_pack;
}
```

Replace the current controls section beginning at `ValidatedLocalPackCard.tsx:221` with this shape, preserving all existing button conditions inside it verbatim:

```tsx
{canManageValidatedLocalPack(status) ? (
  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
    {status.state === "not_installed" && (
      <button type="button" className="btn" disabled={!status.compatible || busy}
        onClick={() => void start("install")}>
        <Download size={14} /> Install local pack
      </button>
    )}
    {status.state === "ready" && (
      <button type="button" className="btn ghost sm" disabled={busy}
        onClick={() => void start("verify")}>
        <RefreshCw size={14} /> Verify again
      </button>
    )}
    {status.state === "update_available" && (
      <>
        <button type="button" className="btn" disabled={busy}
          onClick={() => void start("update")}>
          <Download size={14} /> Install verified update
        </button>
        <button type="button" className="btn ghost sm" disabled={busy}
          onClick={() => void start("rollback")}>
          <RotateCcw size={14} /> Roll back
        </button>
      </>
    )}
    {operation?.state === "paused" && (
      <button type="button" className="btn" disabled={busy}
        onClick={() => void restart("resume")}>
        Resume operation
      </button>
    )}
    {operation?.state === "failed" && operation.retryable && (
      <button type="button" className="btn" disabled={busy}
        onClick={() => void restart("retry")}>
        Retry operation
      </button>
    )}
    {pollingInterrupted && operation && ["queued", "running"].includes(operation.state) && (
      <button type="button" className="btn" disabled={busy} onClick={retryPolling}>
        Retry status check
      </button>
    )}
    {artifactsMayBeInstalled && !busy && (
      <button type="button" className="btn ghost sm" onClick={() => void remove()}>
        <Trash2 size={14} /> Remove pack
      </button>
    )}
  </div>
) : (
  <p className="muted" style={{ fontSize: 13, lineHeight: 1.55, margin: "14px 0 0" }}>
    {LOCAL_PACK_OPERATOR_COPY}
  </p>
)}
```

Keep `useLocalPackOperation.ts` unchanged: it can still load readiness and its existing errors are correct for operator actions. The server-side `operation: null` for non-operators means it cannot poll detailed operation data accidentally.

- [ ] **Step 4: Run frontend focused checks to verify they pass**

Run: `cd frontend && npx playwright test --config playwright.unit.config.ts src/components/admin/ValidatedLocalPackCard.unit.spec.ts src/types/local-ai.unit.spec.ts && npx playwright test e2e/local-model-pack-settings.spec.ts --grep "non-operator sees readiness" && npx tsc --noEmit && npm run lint`

Expected: both unit specs PASS, TypeScript accepts the required capability bit in all local-status fixtures, and ESLint finishes without errors.

- [ ] **Step 5: Root-agent review and prepare the proposed UI checkpoint**

Verify only the Task-3 frontend files would be included. If Pedro separately authorizes a commit,
the root agent may then run:

```bash
git add frontend/src/types/local-ai.ts frontend/src/types/local-ai.unit.spec.ts \
  frontend/src/components/admin/ValidatedLocalPackCard.tsx \
  frontend/src/components/admin/ValidatedLocalPackCard.unit.spec.ts \
  frontend/e2e/local-model-pack-settings.spec.ts
git commit -m "feat(local-ai): hide pack controls for non-operators"
```

Expected if authorized: one frontend-only commit. No subagent commits.

### Task 4: Document the machine-operator boundary and run the integration gate

**Files:**
- Modify: `.env.example:82-111`
- Modify: `docs/backend-handoff.md:566-690`
- Modify: `README.md:99-104,176-187`

**Consumes:** The exact environment key, API matrix, response field, and UI copy established in Tasks 1-3.

**Produces:** Operator-facing setup documentation that makes empty-allowlist fail-closed behavior, configuration validation, `401`/`403` semantics, all global lifecycle routes, general readiness access, and local CLI ownership explicit without publishing any real user UUID.

- [ ] **Step 1: Write the documentation assertions as a focused content check**

Before prose edits, run this command to establish that the old documents lack the authorization contract:

```bash
rg -n "LOCAL_AI_OPERATOR_USER_IDS|can_manage_pack|machine operator" \
  .env.example docs/backend-handoff.md README.md
```

Expected: no `LOCAL_AI_OPERATOR_USER_IDS` setting and no complete machine-operator authorization contract in the three target files.

- [ ] **Step 2: Add exact configuration and API-contract prose**

Add this commented block directly after `LOCAL_AI_ENABLED=false` in `.env.example`. Keep the value empty; do not add a real account identifier.

```dotenv
# Comma-separated web-account UUIDs allowed to mutate the machine-global local pack.
# Empty is fail-closed: all authenticated users can read readiness but none can manage it.
# Invalid non-empty values prevent backend startup. Local CLI maintenance remains an OS-owner task.
LOCAL_AI_OPERATOR_USER_IDS=
```

In `docs/backend-handoff.md`'s validated-local section, state all of the following in direct prose and update the status JSON with the new field immediately after `enabled`:

```json
"can_manage_pack": false,
```

The exact API documentation requirements are:

- `LOCAL_AI_OPERATOR_USER_IDS` is a comma-separated UUID allowlist; an empty value denies all web mutation and a malformed non-empty value stops startup.
- Every authenticated user can call `GET /local-ai/status` and read readiness. `can_manage_pack` identifies whether controls may be shown; non-operators receive `operation: null` even while the summary state accurately reports lifecycle activity.
- `POST /install`, `POST /verify`, `POST /update`, `POST /rollback`, `GET /operations/{operation_id}`, `POST /operations/{operation_id}/resume`, `POST /operations/{operation_id}/retry`, `DELETE /models/{role}`, and `DELETE /local-ai` require an operator.
- A missing or revoked credential receives `401`; an authenticated account absent from the allowlist receives `403` with `Local model pack management requires a machine operator.` Existing active-job/lifecycle conflicts still return `409` only after authorization succeeds.
- `/local-ai/jobs`, `/local-ai/jobs/{job_id}`, and their retry/cancel routes remain owner-scoped rather than operator-scoped.
- Browser authorization does not constrain an operating-system owner using the documented local CLI maintenance commands.

In `README.md`, add one concise paragraph near the validated strict-local explanation: users can see whether the pack is ready, while only account UUIDs set by the machine owner in `LOCAL_AI_OPERATOR_USER_IDS` can install, verify, update, roll back, resume/retry operations, or remove it. Mention that an empty setting intentionally permits no web maintenance.

- [ ] **Step 3: Humanize the public-facing README prose**

Use the `humanizer` skill on the added `README.md` paragraph. It must retain the exact configuration key and security semantics while avoiding marketing language, canned framing, and em-dash-heavy prose. Keep the backend handoff precise and API-oriented.

- [ ] **Step 4: Run focused docs, complete automated verification, and the scope check**

Run:

```bash
rg -n "LOCAL_AI_OPERATOR_USER_IDS|can_manage_pack|machine operator" \
  .env.example docs/backend-handoff.md README.md
cd backend && uv run pytest -q tests/test_config_hardening.py tests/test_local_ai_api.py
cd ../frontend && npx playwright test --config playwright.unit.config.ts \
  src/components/admin/ValidatedLocalPackCard.unit.spec.ts src/types/local-ai.unit.spec.ts
npx playwright test e2e/local-model-pack-settings.spec.ts --grep "non-operator sees readiness"
npx tsc --noEmit
npm run lint
npm run build
cd ../backend && uv run pytest -m "not slow and not fidelity and not local_model and not hardware" -q
```

Expected: the documentation search finds every new contract term; focused backend and frontend tests PASS; TypeScript, ESLint, and production build PASS; the ordinary backend CI marker suite PASSes without provider calls, model downloads, private fixtures, or hardware/local-model markers.

- [ ] **Step 5: Root-agent final review and prepare the proposed documentation checkpoint**

Confirm `git diff --check` is clean and `git status --short` lists only the three documentation
files. If Pedro separately authorizes a commit, the root agent may then run:

```bash
git add .env.example docs/backend-handoff.md README.md
git commit -m "docs(local-ai): document pack operator authorization"
```

Expected if authorized: one documentation-only commit. No subagent commits.

## Spec coverage review

- Machine-global pack versus user-owned jobs: Tasks 1-2 use a dedicated UUID allowlist only for pack routes and leave `/local-ai/jobs` owner-scoped.
- Empty fail-closed configuration and malformed startup failure: Task 1 validates both at `Settings` construction; Task 4 documents them.
- `401` versus `403`: Task 2 tests representative unauthenticated access and all nine non-operator endpoints; Task 4 documents the exact response semantics.
- All lifecycle mutation/restart/removal endpoints and operation detail: the interface table and Task 2 enumerate all nine current routes.
- Readiness for every authenticated user with operator-only operation detail: Task 2 returns `can_manage_pack` for status readers and suppresses `operation` for non-operators; Task 3 renders status plus fixed non-operator copy.
- Existing successful operator behavior and audit actor: Task 2 migrates successful lifecycle tests to the configured operator fixture rather than bypassing the dependency.
- `.env.example`, backend contract, README, focused tests, full test/type/lint/build verification, and no private/provider/model work: Task 4 covers each item.
- Parallel independence: no task modifies extraction scheduling/grounding, user-job cancellation/retry/hydration, upload responses, worker source/locks, manifests, migrations, or release evidence.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-13-local-ai-operator-authorization.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
