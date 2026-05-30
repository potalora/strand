# Entity Extraction Per-Call Overhead: Context Caching Investigation

**Date:** 2026-05-29
**Task:** Task 5 — Investigate Gemini context caching feasibility for the static extraction prompt prefix
**Scope:** Investigation only. No production code changes were made.

---

## 1. Measured Static Prompt Cost

| Metric | Value | Estimate |
|--------|-------|----------|
| `CLINICAL_EXTRACTION_PROMPT` chars | 3,145 | ~786 tokens |
| Number of few-shot examples | 4 | — |
| Examples total chars (all 4) | 15,577 | ~3,894 tokens |
| **Combined static prefix** | **18,722 chars** | **~4,681 tokens** |

The static prefix (system prompt + all four few-shot examples) accounts for the bulk of each call's input. A typical 128-char chunk adds only ~32 tokens on top of this, meaning the static prefix is >99% of per-call token weight for small chunks.

---

## 2. LangExtract Version and Caching API Support

**Installed version:** `langextract 1.1.1`

The `lx.extract` function signature (from `langextract/extraction.py`) accepts:

```
text_or_documents, prompt_description, examples, model_id, api_key,
language_model_type, format_type, max_char_buffer, temperature,
fence_output, use_schema_constraints, batch_length, max_workers,
additional_context, resolver_params, language_model_params, debug,
model_url, extraction_passes, config, model, ...
```

The `model` parameter (a pre-configured `BaseLanguageModel`) takes precedence over all others including `config`. The `config` parameter (a `ModelConfig` dataclass) allows passing `provider_kwargs` into the Gemini provider constructor.

**Caching-related API surface in LangExtract:** None. A search of the public API (`dir(lx)`), the `GeminiLanguageModel.__init__` allowlist (`_API_CONFIG_KEYS`), and the `infer()` call path (`_process_single_prompt`) finds no `cached_content`, `cache_id`, or context-caching parameter. The only keys forwarded to `generate_content` are: `response_mime_type`, `response_schema`, `safety_settings`, `system_instruction`, `tools`, `stop_sequences`, `candidate_count`. Context caching is not among them.

---

## 3. Google-Genai Client Caching Support

**Installed version:** `google-genai 1.63.0`

The SDK exposes a `caches` resource on the client:

```python
client = genai.Client(api_key=...)
# Available: client.caches.create / get / list / update / delete
client.caches.create(
    model="...",
    config=CreateCachedContentConfig(...)  # contents + ttl + system_instruction
)
```

The `create` call returns a `CachedContent` object with a `name` field (e.g., `cachedContents/abc123`). That name is then passed as `cached_content` in subsequent `generate_content` calls. The Gemini API requires a minimum of 1,024 tokens to cache; our static prefix at ~4,681 tokens comfortably clears that threshold.

---

## 4. Feasibility Assessment

### Option A: Use LangExtract with context caching
**Not feasible without a LangExtract patch.**

LangExtract's `GeminiLanguageModel._process_single_prompt` calls:

```python
response = self._client.models.generate_content(
    model=self.model_id, contents=prompt, config=config
)
```

The `config` dict is built from a fixed allowlist (`_API_CONFIG_KEYS`) that does not include `cached_content`. Even if we pass `cached_content` through `language_model_params`, it is silently dropped before the API call. There is no hook, subclass point, or plugin mechanism in v1.1.1 that would allow injecting `cached_content` into the `generate_content` call without modifying LangExtract source.

### Option B: Bypass LangExtract for extraction calls
**Feasible, but a significant scope change.**

The extraction pipeline in `services/extraction/entity_extractor.py` currently delegates entirely to `lx.extract`. Replacing it with a direct `google-genai` `generate_content` call (passing `cached_content=cache_name`) would enable context caching, but requires:

1. Replicating LangExtract's chunking logic (`max_char_buffer`, tokenization).
2. Replicating or replacing its prompt assembly (`PromptTemplateStructured` + few-shot formatting).
3. Replicating its output resolution/alignment (JSON parsing, entity alignment to source spans).
4. Losing LangExtract's schema-constrained structured output (`GeminiSchema`) which currently guarantees valid entity JSON shapes.

This is essentially rewriting the extraction layer, well outside Task 5 scope.

---

## 5. Feasibility Verdict

**VERDICT: NOT FEASIBLE WITHOUT BYPASSING LANGEXTRACT**

Context caching via `google-genai`'s `client.caches` is technically sound — the static prefix (~4,681 tokens) exceeds the 1,024-token minimum and would remain identical across all 18 per-document chunks. The TTL could be set to cover a single document's extraction session.

However, LangExtract 1.1.1 provides no mechanism to inject a `cached_content` reference into its internal `generate_content` calls. Workarounds that avoid full bypass — such as subclassing `GeminiLanguageModel` — would still hit the same allowlist wall unless the subclass overrides `_process_single_prompt` entirely, effectively rewriting the hot path anyway.

**Recommendation:** Defer to Phase 2b. The prerequisite is either (a) an upstream LangExtract release that adds `cached_content` passthrough (watch `_API_CONFIG_KEYS` expansion in future releases), or (b) a decision to drop LangExtract and own the extraction call path directly. Neither is a minimal change.

**Note:** Nothing in this investigation alters which entities are produced. Any Phase 2b implementation must produce identical entity sets — context caching only affects where the static prompt tokens are computed, not the prompt content itself.

---

## 6. Files Inspected (No Changes Made)

- `backend/app/services/extraction/clinical_examples.py` — prompt size measurement
- `backend/.venv/lib/python3.11/site-packages/langextract/extraction.py` — extract() signature
- `backend/.venv/lib/python3.11/site-packages/langextract/providers/gemini.py` — GeminiLanguageModel, _API_CONFIG_KEYS, _process_single_prompt
- `backend/.venv/lib/python3.11/site-packages/langextract/factory.py` — ModelConfig, create_model
- `backend/.venv/lib/python3.11/site-packages/langextract/core/base_model.py` — BaseLanguageModel
- `google.genai.caches.Caches.create` — signature confirmed via inspection
