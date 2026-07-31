# Bounded local inference implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent expensive predictable summary failures and bound NuExtract
retry/split amplification while preserving strict grounding and fail-closed
privacy.

**Architecture:** Validate and size the complete reference-only summary before
model loading, then use the installed MLX JSON-schema logits processor for one
bounded generation. Extend the existing content-free progress protocol with
allowlisted counters. Replace NuExtract's 256-attempt recursion allowance with
independent token, attempt, split, and fragment-depth budgets.

**Tech Stack:** Python 3.11, Pydantic v2, MLX-VLM 0.5.0, llguidance 1.7.6,
JSON Lines worker protocol, pytest.

## Global constraints

- Qwen remains summary-only and never repairs or infers ingestion facts.
- Accepted summary claims retain deterministic fact/field/evidence validation.
- Strict-local inference never constructs a provider or falls back to cloud.
- No partial invalid model output is persisted as a clinical summary.
- Progress/logs contain stable stages and bounded integer counters only.
- Summary generation has one constrained attempt; there is no blind second
  4096-token retry.
- NuExtract request limits are exactly 16,384 generated tokens, 12 attempts,
  7 runtime splits, and fragment depth 3.
- Subagents must not commit. The root agent stages and commits each reviewed
  task.

---

### Task 1: Preflight grounded summary evidence and output fit

**Files:**
- Modify: `backend/app/services/local_ai/summary_projection.py`
- Modify: `backend/app/services/local_ai/grounded_summary.py`
- Modify: `backend/app/services/ai/summarizer.py`
- Modify: `backend/tests/test_summary_projection.py`
- Modify: `backend/tests/test_grounded_local_summary.py`
- Modify: `backend/tests/test_summarization.py`

**Interfaces:**
- Produces: `build_maximal_reference_document(summary_input)`.
- Produces: `required_summary_output_tokens(processor, summary_input)`.
- Consumes: immutable projected facts/evidence/uncertainties.

- [ ] **Step 1: Write failing missing-qualifier and no-model tests**

Use the existing fixture that projects `status="active"` without `/status`
evidence:

```python
with pytest.raises(
    LocalValidationError,
    match="safety qualifier lacks evidence support",
):
    project_summary_records([record], evidence_by_record)
```

Add `/status` evidence and assert the projection succeeds. Patch
`local_model_manager.run` in the service test and assert it is not awaited when
projection/preflight fails.

- [ ] **Step 2: Run tests and verify RED**

```bash
cd backend
uv run pytest tests/test_summary_projection.py \
  tests/test_grounded_local_summary.py tests/test_summarization.py -q
```

- [ ] **Step 3: Validate projected safety qualifiers**

After mapped evidence is assembled, require exact support:

```python
SAFETY_QUALIFIER_PATHS = frozenset(
    {"/assertion", "/relationship", "/status"}
)

def validate_projected_qualifier_evidence(
    fact: Mapping[str, object],
    evidence: Sequence[Mapping[str, object]],
) -> None:
    content = fact["content"]
    required = {
        path
        for path in SAFETY_QUALIFIER_PATHS
        if path.removeprefix("/") in content
    }
    required.update(
        path
        for path in content
        if isinstance(path, str) and path.startswith("/statuses/")
    )
    supported = {
        str(path)
        for item in evidence
        for path in item.get("field_paths", [])
    }
    if required - supported:
        raise LocalValidationError(
            "Summary fact safety qualifier lacks evidence support."
        )
```

Adapt the real helper to the current fact content shape and mapped evidence IDs;
do not add evidence or omit a qualifier to make validation pass.

- [ ] **Step 4: Build the deterministic maximal reference document**

Return the same strict schema consumed by post-generation validation. Every
fact appears exactly once under its allowed heading with every field path
supported by linked evidence. Every uncertainty appears exactly once.

The token budget is:

```python
required = count_tokens(processor, compact_json(maximal_document)) + 128
summary_cap = max(256, required)
if summary_cap > min(manifest_max_output_tokens, 4096):
    raise LocalValidationError(
        "Strict-local summary reference output exceeds the validated limit."
    )
```

Persist `preflight_projection` and `preflight_output_fit` before
`local_model_manager.run`. Include the computed cap in the worker payload.

- [ ] **Step 5: Verify GREEN**

Run Step 2 and assert the no-model test remains green.

- [ ] **Step 6: Root review and commit**

```bash
git add backend/app/services/local_ai/summary_projection.py \
  backend/app/services/local_ai/grounded_summary.py \
  backend/app/services/ai/summarizer.py \
  backend/tests/test_summary_projection.py \
  backend/tests/test_grounded_local_summary.py \
  backend/tests/test_summarization.py
git commit -m "fix(local-ai): preflight grounded summaries"
```

---

### Task 2: Constrain Qwen reference selection to one generation

**Files:**
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/common.py`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/qwen_summary.py`
- Modify: `workers/local_ai/apple_mlx/tests/test_protocol.py`
- Modify: `backend/tests/test_local_ai_summary_worker_contract.py`
- Modify: `backend/tests/test_local_ai_offline_venv.py`

**Interfaces:**
- Produces: optional `json_schema` input for `generate_content`.
- Produces: dynamic reference-only JSON schema from validated IDs/paths.
- Consumes: Task 1 exact `max_output_tokens`.

- [ ] **Step 1: Write failing validate-before-load and one-attempt tests**

Tests must prove:

```python
with patch("local_ai_mlx_worker.qwen_summary.load_role_from_payload") as load:
    with pytest.raises(WorkerInputError):
        run_summary(malformed_payload)
    load.assert_not_called()

generate = Mock(return_value='{"bad":true}')
with pytest.raises(GenerationError):
    run_summary(valid_payload, loaded=loaded, generate_fn=generate)
assert generate.call_count == 1
assert generate.call_args.kwargs["json_schema"]["type"] == "object"
```

Add an offline-venv test importing the locked
`build_json_schema_logits_processor`.

- [ ] **Step 2: Run worker tests and verify RED**

```bash
cd workers/local_ai/apple_mlx
uv run pytest tests/test_protocol.py tests/test_offline_loading.py -q
cd ../../../backend
uv run pytest tests/test_local_ai_summary_worker_contract.py \
  tests/test_local_ai_offline_venv.py -q
```

- [ ] **Step 3: Add optional JSON-schema generation**

Extend the real `generate_content` signature:

```python
def generate_content(
    *,
    model: object,
    processor: object,
    prompt: str,
    images: list[object],
    max_tokens: int,
    temperature: float,
    do_sample: bool,
    input_token_limit: int,
    enable_thinking: bool,
    json_schema: Mapping[str, object] | None = None,
    **kwargs: object,
) -> GeneratedText:
    logits_processors: list[object] | None = None
    if json_schema is not None:
        from mlx_vlm.structured import build_json_schema_logits_processor

        tokenizer = getattr(processor, "tokenizer", processor)
        logits_processors = [
            build_json_schema_logits_processor(
                tokenizer,
                json_schema,
            )
        ]
    generation_options: dict[str, object] = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "verbose": False,
    }
    if logits_processors is not None:
        generation_options["logits_processors"] = logits_processors
    stream = stream_generate(
        model,
        processor,
        formatted,
        image=images or None,
        **generation_options,
    )
```

Use the installed MLX-VLM API signature discovered in the locked environment;
do not add a dependency or change model artifacts.

- [ ] **Step 4: Build a bounded dynamic schema and remove the retry loop**

Validate payload before loading:

```python
safe_input = _validated_input(payload)
selected = loaded or load_role_from_payload("summary", payload)
```

The schema enumerates allowed headings, fact IDs, field paths, evidence IDs,
and uncertainty IDs. Call `generate_fn` once with the exact Task 1 cap and
`json_schema=schema`, then retain `_validated_output`.

- [ ] **Step 5: Verify GREEN**

Run Step 2. Existing malformed-link, unsupported-field, unsupported-heading,
and uncertainty tests must still pass.

- [ ] **Step 6: Root review and commit**

```bash
git add workers/local_ai/apple_mlx/src workers/local_ai/apple_mlx/tests \
  backend/tests/test_local_ai_summary_worker_contract.py \
  backend/tests/test_local_ai_offline_venv.py
git commit -m "perf(local-ai): constrain summary reference output"
```

---

### Task 3: Preserve safe worker categories and progress counters

**Files:**
- Modify: `backend/app/services/local_ai/protocol.py`
- Modify: `backend/app/services/local_ai/model_manager.py`
- Modify: `backend/app/services/ai/summarizer.py`
- Modify: `backend/app/api/upload.py`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/__main__.py`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/qwen_summary.py`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py`
- Modify: `backend/tests/test_local_ai_protocol.py`
- Modify: `backend/tests/test_local_ai_model_manager.py`
- Modify: `backend/tests/test_local_ai_api.py`
- Modify: `backend/tests/test_local_ai_log_privacy.py`
- Modify: `workers/local_ai/apple_mlx/tests/test_protocol.py`

**Interfaces:**
- Produces: allowlisted progress counters from worker to `LocalAIJob.progress`.
- Produces: `LocalWorkerError.category` using an allowlisted terminal category.

- [ ] **Step 1: Write failing protocol and privacy tests**

Add legal frames containing:

```json
{
  "role": "summary",
  "stage": "generating",
  "attempt": 1,
  "attempt_limit": 1,
  "input_tokens": 2048,
  "output_tokens": 512,
  "output_token_limit": 1024
}
```

Reject negative, decreasing, over-bound, unknown-key, and string counters.
Assert a worker `invalid_structured_output` reaches the safe job failure code
without returning its message. Seed PHI in the worker error text and assert it
is absent from logs/API.

- [ ] **Step 2: Run tests and verify RED**

```bash
cd backend
uv run pytest tests/test_local_ai_protocol.py \
  tests/test_local_ai_model_manager.py tests/test_local_ai_api.py \
  tests/test_local_ai_log_privacy.py -q
cd ../workers/local_ai/apple_mlx
uv run pytest tests/test_protocol.py -q
```

- [ ] **Step 3: Extend progress frames with fixed counters**

Allow only:

```python
PROGRESS_COUNTERS = frozenset(
    {
        "current",
        "total",
        "activity",
        "attempt",
        "attempt_limit",
        "input_tokens",
        "output_tokens",
        "output_token_limit",
        "splits_used",
        "split_limit",
    }
)
```

Stages are fixed codes such as `loading`, `generating`, `validating`, and
existing extraction stages. The manager forwards visible counters and liveness
without embedding prompts or output.

- [ ] **Step 4: Persist summary and extraction telemetry**

Pass a summary progress callback into `local_model_manager.run`. Merge only
allowlisted values into the isolated job progress transaction. Preserve the
worker terminal category on `LocalWorkerError` and map it to the stored safe
failure code.

- [ ] **Step 5: Verify GREEN**

Run Step 2.

- [ ] **Step 6: Root review and commit**

```bash
git add backend/app/services/local_ai backend/app/services/ai/summarizer.py \
  backend/app/api/upload.py backend/tests workers/local_ai/apple_mlx
git commit -m "feat(local-ai): report bounded inference progress"
```

---

### Task 4: Bound NuExtract amplification

**Files:**
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py`
- Modify: `workers/local_ai/apple_mlx/tests/test_protocol.py`
- Modify: `backend/tests/test_local_ai_summary_worker_contract.py`
- Modify: `docs/operations-strict-local-ai.md`

**Interfaces:**
- Produces: `_ExtractionWorkBudget` with token, attempt, split, and depth
  accounting.
- Consumes: Task 3 progress counters.

- [ ] **Step 1: Write failing budget tests**

Add tests for:

1. successful one-retry under-cap malformed JSON;
2. immediate shallow split after output-limit exhaustion;
3. `work_token_limit` after 16,384 charged tokens;
4. `work_attempt_limit` on attempt 13;
5. `work_split_limit` on split 8;
6. `fragment_depth_limit` beyond depth 3;
7. successful depth-3 fragmentation;
8. each call receives `min(4096, remaining_generated_tokens)`.

- [ ] **Step 2: Run tests and verify RED**

```bash
cd workers/local_ai/apple_mlx
uv run pytest tests/test_protocol.py -q
```

- [ ] **Step 3: Implement independent accounting**

Use exact constants:

```python
MAX_EXTRACTION_GENERATED_TOKENS = 16_384
MAX_EXTRACTION_GENERATION_ATTEMPTS = 12
MAX_EXTRACTION_RUNTIME_SPLITS = 7
MAX_EXTRACTION_FRAGMENT_DEPTH = 3
```

The budget:

```python
@dataclass
class _ExtractionWorkBudget:
    generated_tokens: int = 0
    attempts: int = 0
    runtime_splits: int = 0

    def next_output_cap(self) -> int:
        remaining = MAX_EXTRACTION_GENERATED_TOKENS - self.generated_tokens
        if remaining <= 0:
            raise GenerationError(
                "Local extraction exceeded its token work limit.",
                category="work_token_limit",
            )
        return min(EXTRACTION_OUTPUT_CAP, remaining)

    def charge_generation(self, generated_tokens: int) -> None:
        self.attempts += 1
        if self.attempts > MAX_EXTRACTION_GENERATION_ATTEMPTS:
            raise GenerationError(
                "Local extraction exceeded its attempt work limit.",
                category="work_attempt_limit",
            )
        self.generated_tokens += generated_tokens
        if self.generated_tokens > MAX_EXTRACTION_GENERATED_TOKENS:
            raise GenerationError(
                "Local extraction exceeded its token work limit.",
                category="work_token_limit",
            )
```

Add bounded split/depth methods. Track fragment depth internally without adding
it to source-visible extraction content. Charge exact
`GeneratedText.generation_tokens`; otherwise charge the tokenizer count, and
use the requested cap when neither is dependable.

- [ ] **Step 4: Keep retry/split semantics narrow**

Under-cap JSON parse failure gets one retry. Output-limit failure splits
immediately. Only `invalid_structured_output`, `output_limit`, or formatted
input-limit failures may split. Other runtime failures propagate unchanged.

- [ ] **Step 5: Verify GREEN and worker contracts**

```bash
cd workers/local_ai/apple_mlx
uv run pytest tests/test_protocol.py tests/test_offline_loading.py -q
cd ../../../backend
uv run pytest tests/test_local_ai_summary_worker_contract.py \
  tests/test_strict_local_pipeline.py -q
```

- [ ] **Step 6: Root review and commit**

```bash
git add workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py \
  workers/local_ai/apple_mlx/tests/test_protocol.py \
  backend/tests/test_local_ai_summary_worker_contract.py \
  docs/operations-strict-local-ai.md
git commit -m "perf(local-ai): bound extraction retry work"
```
