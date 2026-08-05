# ADR: Strand Local AI — Simplified Extraction + OCR Engine

- **Status:** Proposed (pending Pedro's sign-off)
- **Date:** 2026-08-05
- **Supersedes:** current `codex/strict-local-ai` architecture (merged 2026-07-31)
- **Scope:** entity extraction from unstructured documents + OCR for scanned files/images on a local ~16GB machine. Structured exports (FHIR/Epic/CDA/XDM) stay on deterministic mappers — no model ever touches them.

---

## 1. Context

The strict-local branch delivered a working local pipeline plus a trust platform
(manifests, pack promotion, release evidence, fidelity suite, bespoke worker
protocol, four processing modes). It is ~12.3k lines across 35 modules plus a
separate MLX worker package and ~40 dedicated test files.

Three constraints now dominate:

1. **The model field moves fast.** Any specific model choice is temporary. The
   system must absorb model swaps as routine maintenance, not projects.
2. **Maintenance cost is real.** Every custom protocol, gate, and release ritual
   is standing cost paid forever.
3. **Solo builder.** No delicate systems. Fewer moving parts, each either
   deterministic (never churns) or disposable (swappable in one config edit).

The product need is exactly two things: entity extraction from structured
exports (already deterministic), and OCR + entity extraction from scans/images.

## 2. Design principles (from the constraints above)

- **P1 — Models are cattle.** Every model sits behind one of exactly two
  operations: `ocr(page_image) -> markdown` or `chat(messages) -> json`.
  Swapping a model = edit one registry entry + pass the eval gate. Zero code
  change on a model swap.
- **P2 — Invest in what outlives models.** The durable assets are the
  deterministic layer: the evidence schema, the validator's hard gates, the
  evidence-lineage model, the fidelity/trap-fixture harness, the egress tests.
  Model glue is disposable by design.
- **P3 — One runtime, one protocol.** Chat-shaped models are served through an
  OpenAI-compatible loopback server (already supported by the pluggable LLM
  layer). OCR is one small HTTP endpoint. No bespoke worker protocol.
- **P4 — No release engineering.** The model manifest is a JSON file checked
  into the repo, edited like code, versioned by git. Pull verifies sha256. No
  candidate packs, promotion scripts, release evidence, or validation receipts.
- **P5 — Two modes.** `local` (default, everything on-machine) and
  `cloud-opt-in` (explicit, per-page, scrubbed). The four-mode routing matrix
  collapses to this.
- **P6 — Never silently wrong.** Anything the local stack can't handle with
  confidence goes to quarantine ("needs review"), optionally escalating to
  per-page cloud opt-in. Quarantine is a first-class outcome, not a failure.

## 3. Pipeline

```
file
 └─ triage (deterministic)
     ├─ structured export ──────────────► existing mappers (no model)
     └─ unstructured doc
         ├─ text-layer PDF / RTF ───────► direct text extraction
         └─ scanned page / image
             ├─ preprocess (deskew, DPI check, blur flag)
             ├─ ocr.primary ──── good ──► markdown + text
             └─ ocr.escalation ─ good ──► markdown + text
 └─ extract: chat model + ClinicalDocumentExtraction schema (per doc/chunk)
 └─ validate: hard gates (deterministic) + soft scores
 └─ reconcile: intra-doc dedup, temporal consistency, terminology (RxNorm/ICD-10/LOINC)
 └─ FHIR R4

 low-confidence page at any point ──► quarantine ──► optional per-page cloud opt-in
                                                      (3-layer scrubber → egress receipt)
```

Escalation ladder is two local rungs (primary → escalation) then quarantine.
No third local rung — the marginal model in the middle is maintenance debt.

## 4. Model registry (the entire model surface)

One file, e.g. `backend/strand/models.json`, checked in:

```json
{
  "schema_version": 1,
  "roles": {
    "ocr.primary":     { "model": "lightonocr-1b",            "runtime": "mlx-or-transformers", "quant": "4bit", "sha256": "...", "eval_score": null },
    "ocr.escalation":  { "model": "qwen3-vl-4b-instruct",     "runtime": "mlx-vlm",             "quant": "4bit", "sha256": "...", "eval_score": null },
    "extract":         { "model": "qwen3-4b-instruct-2507",   "runtime": "openai-loopback",     "quant": "4bit", "sha256": "...", "eval_score": null },
    "summary":         { "model": null, "note": "phase 2" }
  }
}
```

Rules:
- `eval_score` is filled by the fidelity harness; a model swap that regresses
  the score does not ship.
- Runtimes are limited to a fixed small set (openai-loopback, mlx-vlm,
  transformers). Adding a new runtime is the only model change that costs code.
- `strand models pull` downloads + verifies. `strand models status` reports.
  That is the entire pack machinery.

## 5. Selected stack (benchmark-driven choices)

| Role | Model | Size (4-bit) | Why |
|---|---|---|---|
| ocr.primary | **LightOnOCR-1B** (Apache 2.0, LightOn/FR) | ~0.7GB | Only candidate with a head-to-head win on actual scanned medical records (cleanest markdown, correct table + checkbox semantics, fastest). SOTA claim on OlmOCR-Bench. |
| ocr.escalation | **Qwen3-VL-4B-Instruct** (Apache 2.0) | ~3GB | OCRBench 88.1, DocVQA 95.3 at 4B params; open weights run locally, so vendor policy is satisfied. Hard pages only. |
| extract | **Qwen3-4B-Instruct-2507** (Apache 2.0) | ~2.5GB | Ranked #1 of 12 SLMs in a multi-task comparison (beats Qwen3-8B, Llama-3.1-8B, Llama-3.2-3B); strongest fine-tuning base for later distillation on Strand's trap fixtures; IFEval ~82–90% depending on variant. Thinking mode OFF, temp 0. |
| summary | deferred to phase 2 (candidate: keep Qwen3.5-9B) | ~5.5GB | Not in the core need; keeps v1 small. |

**Retired:** OvisOCR2 (superseded as primary; may return as fallback), NuExtract3
(its image-direct value prop is covered by Qwen3-VL-4B on the escalation rung;
text-to-text extraction keeps every fact groundable against real OCR text, which
is what makes the evidence-span invariant enforceable).

**Known risk on the extractor:** one independent lab found Qwen3-4B can exhaust
its output token budget with verbose generation before completing JSON (0%
structured-output success in their harness). Mitigations: thinking mode off,
generous max_tokens, strict output template, validator-with-retry loop (already
exists). Fallback if the fidelity gate disagrees: Gemma-3-4B-it (100%
structured-output success in the same lab). This is exactly the kind of thing
the eval gate exists to settle — not taste.

## 6. Memory budget (M4, 16GB, sequential residency)

Never two big models resident at once. Load → process → unload.

- ocr.primary 0.7GB, ocr.escalation ~3GB, extract ~2.5GB, summary 5.5GB (phase 2)
- OS + app + Postgres ≈ 4–5GB
- Comfortable headroom on every rung, even with summary added later.

## 7. What gets deleted from the branch

| Delete | Replaced by |
|---|---|
| Old extraction engine + `adapters.py` bridge (LangExtract, medspaCy/scispaCy as rival extractor, gemini/local/hybrid EXTRACTION_ENGINE) | `ClinicalDocumentExtraction` becomes the canonical shape, mapped straight to FHIR. medspaCy ConText reborn as the deterministic assertion layer (present/absent/possible/historical/family/someone-else) feeding the same schema. |
| Bespoke JSONL worker protocol, `fake_worker`, contract-test sprawl | OpenAI-compatible loopback server + one `/ocr` endpoint. |
| `model_manager` process-lease/watchdog complexity | Loopback server lifecycle (start/health/stop). |
| pack_verifier, release_evidence, validation_receipt, promote/lock scripts, candidate-pack CLI | `strand models pull/status` + one checked-in manifest. |
| CUSTOM_LOCAL and PROMPT_ONLY processing modes | Two modes: `local`, `cloud-opt-in`. |
| Hard-reject heuristics in the validator (negation regexes, unit-truncation, token-sequence grounding) as pipeline blockers | Soft scores + warnings measured by the harness. Only provably-unsafe checks stay hard: evidence span exists in source, numeric literal present, unit present, date parses, patient-identity leakage, duplicate signatures. |
| `grounded_summary.py` (2,737 lines) from v1 | Phase 2, rebuilt thin if/when needed. |
| RawExtractionCheckpoint tier | Two tiers: OCR + extraction. |

## 8. What is kept, unchanged

- Fail-closed egress tests (`test_e2e_local_network_guard`, strict-egress suite) — the trust story.
- Evidence lineage data model (verbatim + page + span per fact).
- Fidelity/trap-fixture harness — now also the model-swap acceptance gate.
- OCR + extraction checkpoints (long OCR runs must resume).
- Encryption at rest, decrypt-to-memory; models receive bytes/plaintext, never ciphertext paths.
- Deterministic mappers for all structured formats.
- Three-layer de-id scrubber, now as the pre-egress step for per-page cloud opt-in.

## 9. Acceptance gate

`strand eval` runs the fidelity corpus (synthetic traps + private corpus) and
emits one composite per role: OCR (downstream extraction F1 as the primary
metric, CER as diagnostic) and extraction (F1 vs answer keys, PHI-gate
false-negative rate). Rules:

1. A model swap ships only if composite ≥ incumbent and PHI gate is unchanged.
2. Cloud and local outputs pass through the *same* validator — cloud is a
   backend, not a bypass.

## 10. Sequencing

- **Phase 0 — verification spike (before any commitment).** Verify
  LightOnOCR-1B loads and runs on Apple Silicon / 16GB (MLX or transformers).
  Its reference path is vLLM-on-Linux; if the Apple path is broken, OvisOCR2
  stays primary and nothing else changes. One afternoon, throwaway script.
- **Phase 1 — deletion + interface.** Delete §7 items; introduce `models.json`
  + the two operations; wire loopback serving.
- **Phase 2 — gate.** Run the fidelity suite on the new stack; record
  baselines; ship if green.
- **Phase 3 — quarantine UX + per-page cloud opt-in** (scrubber → consent →
  egress receipt).
- **Phase 4 (optional) — summaries.**

## 11. Net effect

~12.3k lines → target ~3–4k for the whole strict-local path. One pipeline, two
modes, two model operations, one manifest file. Model churn is confined to a
config edit plus an eval run; everything durable (schema, gates, evidence,
harness) is exactly where the investment already is.

## Sources (accessed 2026-08-05)

- Independent scanned-medical-records OCR comparison (LightOnOCR vs MinerU-Diffusion vs LiteParse vs Chandra) — Medium, "Which Small vLLM OCR Model Is the Best For Private Use"
- Open-weight OCR/DocAI leaderboard overview — presenc.ai research, 2026
- 12-SLM structured-output + quality comparison — local-model-lab (GitHub); Distillabs 12-SLM benchmark (Qwen3-4B-Instruct-2507 #1)
- Qwen2.5-VL / Qwen3-VL benchmark tables — llm-stats comparisons, arXiv:2502.13923
