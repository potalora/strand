# User-Managed Multi-Provider LLM Configuration — Design Spec

**Date:** 2026-06-21
**Status:** Approved (brainstorming complete)
**Builds on:** PR #41 (merged) — the provider-agnostic `services/ai/llm/` facade + read-only `GET /summary/providers` + Summarize selector.
**Decisions locked:** Merge #41 first (done) → phased PRs. API keys stored **encrypted per-user in the DB**, reusing the app's AES-256-GCM field encryption.

## 0. Goal

Turn LLM provider configuration from operator-only `.env` into a **user-managed feature**: a settings pane in Admin → System where the user enters and saves API keys, configures providers (key / base URL / model), picks the default and per-operation provider, tests connections, and switches live. Extend the facade so **input is type-agnostic** (text + images + PDFs route to any vision-capable provider, cloud or local), and surface the **entity-extraction engine/provider** (including LangExtract's native gemini/openai/ollama backends) in the same pane.

Providers in scope (unchanged set): `gemini`, `vertex`, `openai`, `anthropic`, `openrouter`, `ollama`, `lmstudio`.

This is one cohesive feature delivered as **three phased PRs**:
- **Part 1 — Settings pane + encrypted per-user config** (foundation; everything plugs into it).
- **Part 2 — Multimodal vision across all providers.**
- **Part 3 — Extraction engine/provider exposure.**

## 1. Corrected provider facts (from grounding checks)

- **OpenAI takes PDF natively** via the Chat Completions `file` content part (base64 `file_data`) or the Files API — no rasterizer needed. (Earlier table was wrong.)
- **LangExtract is NOT Gemini-bound** — the installed package ships `providers/{gemini,openai,ollama,router,registry}`. Entity extraction can run on Gemini, OpenAI, or local Ollama through LangExtract's own backends, selected by model id.
- **Local + OpenRouter vision is real** — Ollama / LM Studio serve modern vision models (llama3.2-vision, qwen2.5-VL, minicpm-v, moondream, …) over the OpenAI image API; OpenRouter proxies vision models. Images work directly; the only spot that may want page→image rasterization is a local/OpenRouter model that can't read PDF — an optional later enhancement, not in scope now.

## 2. Existing infrastructure (confirmed)

- **Encryption** (`app/middleware/encryption.py`): app-side AES-256-GCM. `encrypt_field(str)->bytes` (12-byte nonce + ciphertext), `decrypt_field(bytes)->str`. Key from `DATABASE_ENCRYPTION_KEY`. Pattern: `LargeBinary` column. Reuse verbatim for API keys.
- **No per-user settings table exists** — Preferences are frontend-only Zustand. This feature adds the first server-persisted per-user config.
- **`user_id` availability at LLM call sites**: summarizer (in scope), section_parser (one level up in `_run_gemini_extraction_engine`), entity_extractor (in scope at both upload engines), llm_judge (two levels up in `run_upload_dedup`). All reachable.
- **Registry** (`services/ai/llm/registry.py`): global-settings-driven, caches by provider name. Must accept per-user config and cache without cross-user leakage.
- **Alembic head**: `e1f2a3b4c5d6` (new migrations chain from it).
- **Admin System tab**: `frontend/src/app/(dashboard)/admin/page.tsx`, `SystemTab` (≈1919-2185); new card slots between Preferences (~2112) and "Your data" (~2114).

---

## PART 1 — Settings pane + encrypted per-user config

### 1.1 Data model (2 new tables, migration chains from `e1f2a3b4c5d6`)

**`llm_provider_configs`** — one row per (user, provider):
| column | type | notes |
|---|---|---|
| id | UUID PK | |
| user_id | UUID FK→users | indexed; owner scope |
| provider | str | one of the 7 known providers |
| api_key_encrypted | LargeBinary, null | `encrypt_field(key)`; null = use `.env`/none |
| base_url | str, null | override (openai-compat/local/openrouter) |
| model | str, null | default model for this provider |
| enabled | bool, default true | |
| created_at / updated_at | timestamptz | |

Unique constraint `(user_id, provider)`.

**`user_llm_preferences`** — one row per user (routing):
| column | type | notes |
|---|---|---|
| id | UUID PK | |
| user_id | UUID FK→users | unique |
| default_provider | str, null | null = fall back to `settings.llm_provider` |
| summary_provider / section_provider / dedup_provider / extraction_provider / vision_provider | str, null | per-operation overrides; null = fall back to default |
| extraction_engine | str, null | `gemini`/`local`/`hybrid` override (Part 3) |
| created_at / updated_at | timestamptz | |

Test DB note: `conftest.py` uses `create_all` (creates tables, does NOT alter) — fine for brand-new tables; still ship the Alembic migration and run it on dev + `medtimeline_test`.

### 1.2 Config resolution (`services/ai/llm/config.py`, new)

```python
@dataclass
class ProviderCreds:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    enabled: bool = True

@dataclass
class LLMConfig:
    routing: dict[str, str]              # {"default":..., "summary":..., "vision":..., ...}
    providers: dict[str, ProviderCreds]  # provider_name -> creds
    @classmethod
    def from_settings(cls) -> "LLMConfig": ...      # global .env (back-compat)

async def load_llm_config(db, user_id) -> LLMConfig:
    """User DB rows decrypted, with per-field fallback to global settings."""
```

Resolution per field: **user DB row → global `settings` → hard default**. A user who never opened the pane behaves exactly as today (all fall back to `.env`).

### 1.3 Registry changes

- `get_provider(operation: str | None = None, config: LLMConfig | None = None) -> LLMProvider`: when `config is None`, use `LLMConfig.from_settings()` (back-compat). Resolve provider name from `config.routing[operation] or config.routing["default"]`. Build from `config.providers[name]`.
- `provider_name_for(operation, config)` and `_build(name, config)` take the config.
- **Cache key** changes from `name` to `(name, base_url, model, sha256(api_key))` so (a) no cross-user key leakage and (b) a settings change naturally busts the cache. Provider SDK clients are reused when creds are identical.
- `available_providers(config)` reflects the resolved config (configured = has key OR local server), still **never returns keys**.

### 1.4 Call-site threading

Load `LLMConfig` once at each request/job entry point (where `db`+`user_id` exist) and pass the object down:
- **summarizer.generate_summary**: load `config = await load_llm_config(db, user_id)`; `get_provider("summary", config)`. The explicit per-request `provider`/`model` override still wins.
- **upload engines** (`_run_gemini_extraction_engine`, `_run_local_extraction_engine`): load config once; thread into `parse_sections(text, config=...)` and `extract_entities_async(..., config=...)`.
- **dedup** (`run_upload_dedup` → `_run_llm_judge` → `judge_candidates_batch` → `judge_candidate_pair`): load config at `run_upload_dedup`, thread `config` down (replaces the `api_key` thread).

`api_key` params already on these functions become legacy/no-op (kept for signature stability where cheap, or removed where clean — decided per function in the plan).

### 1.5 API (`api/llm_settings.py`, prefix `/settings/llm`, all user-scoped + audited; keys NEVER logged/returned)

- `GET /settings/llm` → `{ providers: [{name, is_local, supports_vision, configured, has_key, key_masked, base_url, model, enabled, source: "user"|"env"|"none"}], routing: {default, summary, section, dedup, extraction, vision, extraction_engine} }`. `key_masked` = e.g. `"sk-…AB12"` or `null`.
- `PUT /settings/llm/providers/{name}` body `{api_key?, base_url?, model?, enabled?}` → upsert. Non-empty `api_key` → `encrypt_field` + store. Omitted `api_key` leaves the stored key intact. `api_key: ""` (explicit empty) → leave intact (use DELETE to clear).
- `DELETE /settings/llm/providers/{name}` → delete the user's row for that provider (revert to `.env`/none).
- `PUT /settings/llm/routing` body any subset of routing fields → upsert preferences.
- `POST /settings/llm/providers/{name}/test` → resolve the user's creds, run one tiny `complete()` (or a vision probe), return `{ok: bool, model: str, error_type?: str}`. No key in the response; a 429/quota → `{ok:false, error_type:"rate_limit"}` with a note that auth succeeded.

### 1.6 Frontend (Admin → System "AI providers" card)

Invoke the **frontend-design** skill before building (existing "Reimagined" theme; this is new UI). Native to the theme (`card-surface pad`, `.selectbox`, existing field styles). Contents:
- **Routing**: default-provider select + an "Advanced" disclosure with per-operation selects (summary, extraction, vision, dedup, section). Each option shows `name · model`; disabled if not configured (except local).
- **Per-provider rows**: name + cloud/local + vision badge; masked key display; a password input + **Save**; **Clear**; **Test** (shows ok/error inline); base-URL + model inputs for local/openrouter; enabled toggle.
- **Copy**: one line per operation explaining what it does + the cloud-vs-local data note ("cloud providers receive de-identified records; local providers keep data on this machine"). Keep the AI disclaimer elsewhere intact.
- `lib/api.ts`: `getLlmSettings`, `saveProvider`, `clearProvider`, `saveRouting`, `testProvider`.

### 1.7 Testing (Part 1)

- Model/migration: table creation + unique constraints; `decrypt_field(encrypt_field(k)) == k` round-trip on the column.
- `load_llm_config`: user-row override beats settings; missing fields fall back; no user → `from_settings()`.
- Registry: `get_provider(op, config)` picks user provider; cache keyed so two users with different keys get different clients; settings change busts cache; back-compat (no config → settings).
- API: GET masks keys (assert no full key, no `sk-…` beyond mask, in body); PUT encrypts (assert ciphertext in DB ≠ plaintext); DELETE reverts; routing upsert; test endpoint returns ok for a mocked provider; **all endpoints user-scoped** (user B cannot read/modify user A's config) and audited.
- De-id unchanged: a summarizer test with a user-configured non-Gemini provider still scrubs PHI before send.
- E2E (mocked): the AI-providers card renders, saving a key persists (masked on reload), switching default updates routing, Test shows a result.

---

## PART 2 — Multimodal vision across all providers

### 2.1 Content parts (`services/ai/llm/types.py`)

`LLMMessage.content` becomes `str | list[ContentPart]`:
```python
@dataclass
class TextPart:     text: str
@dataclass
class ImagePart:    data: bytes; mime: str          # image/png, image/tiff, image/jpeg
@dataclass
class DocumentPart: data: bytes; mime: str           # application/pdf
ContentPart = TextPart | ImagePart | DocumentPart
```
Plain-string content stays valid (wrapped as a single TextPart internally). Back-compat: existing text callers unchanged.

### 2.2 Provider mapping (each provider's `complete` handles parts)

- **Gemini**: `types.Part.from_bytes(data, mime)` for image + PDF; text as string parts.
- **Anthropic**: image block (base64) + document block (base64 PDF) + text block. (`supports_vision=True` already; PDF document blocks supported.)
- **OpenAI-compat**: image → `{"type":"image_url","image_url":{"url":"data:<mime>;base64,…"}}`; PDF → `{"type":"file","file":{"filename","file_data":"data:application/pdf;base64,…"}}` for endpoints that support it (OpenAI does; OpenRouter/local depend on model). Capability resolved from the selected vision provider/model.
- Capability gating: a provider/model that can't accept a part type raises `LLMBadRequestError`; the OCR caller falls back to the configured fallback (Gemini if a key exists) and logs it.

### 2.3 Vision provider selection + OCR routing

- The pane's `vision_provider` (Part 1 routing) selects who does OCR. Default falls back to the global provider, then Gemini.
- `text_extractor`: keep local `pdfplumber` first for text PDFs; for scanned PDFs / TIFFs, build a multimodal `LLMRequest` (DocumentPart for PDF, ImagePart for TIFF) and call `get_provider("vision", config).complete(...)`. Drop the hard "GEMINI-only" guard; replace with "vision provider not configured / can't handle this input → fall back to Gemini, else clear error".
- Local vision requires the user to have pulled/loaded a vision model; the pane notes this and Test surfaces failures.

### 2.4 Testing (Part 2)

- Each vision provider maps an ImagePart + DocumentPart to the correct SDK shape (mocked SDK; assert the request payload structure).
- `text_extractor` routes a scanned PDF / TIFF through the configured vision provider; falls back to Gemini when the provider can't.
- Live smoke (gated): a small image OCR via Gemini + Anthropic (and OpenAI if quota), skipped otherwise; local vision smoke skipped unless a vision model is loaded.
- Back-compat: text-only operations send a single TextPart and are byte-identical in behavior.

---

## PART 3 — Extraction engine/provider exposure

### 3.1 LangExtract native backends

- `entity_extractor`: when the resolved extraction provider is `gemini`/`openai`/`ollama`, use LangExtract's **native backend** (set `model_id` + key/`model_url` for that backend) with the existing few-shot examples. For `anthropic`/`openrouter`/`lmstudio` (not LangExtract backends), use the `generic_entity_extractor` facade path (already built). Selection driven by `config.routing["extraction"]` + `extraction_engine`.
- `EXTRACTION_ENGINE` (`gemini`/`local`/`hybrid`) stays the on-device-vs-cloud switch; `local`/`hybrid` keep medspaCy/scispaCy; the cloud escalation/path uses the resolved extraction provider.

### 3.2 Settings exposure

- The pane surfaces `extraction_engine` (on-device / hybrid / cloud) and the extraction provider, with copy making the trade-off explicit: **local/hybrid keep data on-device; cloud sends de-identified text out.**
- `available_providers`/settings GET include extraction capability so the UI can disable cloud extraction copy when `local` is chosen.

### 3.3 Testing (Part 3)

- Extraction provider routing: gemini/openai/ollama → LangExtract backend selected (mock `lx.extract`/factory, assert model_id/backend); others → generic path.
- `extraction_engine=local` keeps on-device path (no provider call).
- Settings round-trip for `extraction_engine` + `extraction_provider`.

---

## 4. Security / HIPAA (whole feature)

- API keys: encrypted at rest (AES-256-GCM), per-user, **never** returned in plaintext (masked), **never** logged (audit logs record action + provider name only), decrypted only server-side at call time.
- De-identification still runs before EVERY provider call, every operation, every provider — unchanged invariant (Absolute Rule #2). New tests assert it across user-configured non-Gemini providers.
- User-scoped: every settings query filters by `user_id`; user B can never read/modify/test user A's providers.
- `.env` remains a valid fallback (operator default); a user with no DB config behaves exactly as today.
- Cloud providers receive only de-identified text/records; the pane makes "this leaves your machine" explicit when a cloud provider/operation is selected.

## 5. Out of scope (YAGNI)

- PDF→image rasterization for local/OpenRouter models that lack native PDF (optional later; images already work).
- Per-team/shared org config (single-user ownership model stands).
- Streaming, cost/budget accounting beyond existing token counts.
- Bedrock/Azure (reachable later via the same seams).

## 6. Rollout / sequencing

Three PRs, foundation-first: **Part 1** (model + config resolver + registry + API + pane), then **Part 2** (multimodal), then **Part 3** (extraction exposure). Each ships green tests and updates `.env.example` / `CLAUDE.local.md`. Default behavior with no user config = today's behavior (Gemini, byte-for-byte).
