# Design — OSS Adoption Across Ingestion & Extraction

**Date:** 2026-06-20
**Status:** Design approved; implementation plan intentionally **not** written yet (to be authored in a later session).
**Companion:** `docs/oss-landscape-analysis.md` (the analysis this design acts on).
**Scope:** Adopt mature, permissively-licensed OSS to retire duplicated custom code and add capability across five workstreams — *without* violating any Absolute Rule.

> This is a **design/architecture spec**, not a step-by-step implementation plan. It defines *what* changes, *where* it plugs in, *how* it rolls out safely, and *what decisions remain open*. The granular task breakdown (writing-plans output) is deliberately deferred.

---

## 1. Goal & non-goals

**Goal.** Shed maintenance burden and gain capability by replacing three duplicative custom subsystems with OSS, fixing one dependency-hygiene issue, and landing two quick wins — all behind reversible flags, all license-clean.

**Non-goals.**
- No change to the app's product posture (record organizer + de-identified AI prompts; **never** diagnoses/advice).
- No new cloud AI provider (Rule 3); Gemini remains the only external AI. New libraries are **local** (not providers).
- No re-architecture of storage, the dedup LLM-judge, Epic EHI mappers, the CDA converter, or the bundled-terminology design — the analysis found these justified. They are explicitly **kept**.

---

## 2. The five workstreams

| ID | Workstream | Adopts | Replaces / fixes | License |
|----|------------|--------|------------------|---------|
| **WS-A** | Clinical NLP | medspaCy + scispaCy **or** GLiNER-biomed | Hand-rolled negation/section/precision logic; slow per-section Gemini fan-out | MIT / Apache-2.0 |
| **WS-B** | PHI de-id hardening | Microsoft Presidio + a clinical de-id NER model | Hand-maintained regex layer; the open city/location Safe Harbor gap | MIT |
| **WS-C** | Fuzzy matching | RapidFuzz | Homemade Jaccard in dedup; exact-only terminology lookup | MIT |
| **WS-D** | FHIR structural validation | `fhir.resources` (already declared) | No structural validation today; **complements** the existing terminology-code validation (not a duplicate) | n/a |
| **WS-E** | Quick wins | Synthea | CLAUDE.md license wording; tiny/synthetic test data | Apache-2.0 |

---

## 3. Guiding principles (apply to every workstream)

1. **TDD, project-style.** Define expected output → write the unit test first → test exhaustively → compare real-vs-expected. Each new component lands test-first.
2. **Feature-verification order.** API/script-level proof first (status codes, payloads, DB state), then a manual frontend check. Never skip to UI.
3. **Reversible by flag + shadow-compare.** Every adoption sits behind a config flag (the existing `phi_ner_enabled` pattern). The new path first runs in **shadow mode** — compute both old and new, log the diffs, keep the old output as authoritative — until validated on real data, *then* flip the default. This is the core de-risking mechanism for anything that changes extraction or de-id behavior.
4. **License gate (Rule 17).** Only MIT/Apache/BSD enter runtime deps. Explicitly rejected: MedCAT (Elastic 2.0), Zingg (AGPL), PhysioNet `deid` (GPL), CLAMP (non-commercial), NLM-Scrubber (closed).
5. **Local-first preserved / strengthened.** WS-A's local NER path *removes* a Gemini round-trip (and its de-id requirement) rather than adding network dependence.
6. **Respect known gotchas.** Test DB is `create_all`-managed (manual `ALTER` for new columns, ship the Alembic migration too); dev uvicorn runs without `--reload` (restart after backend edits); run pytest via `.venv`; fast suite is `-m "not slow"`; literal `/records` subpaths precede `/{record_id}`; NER fail-open must stay non-latching.

---

## 4. Sequencing

Value-ordered, with a groundwork phase first so later phases test against realistic data.

```
Phase 0 ─ Groundwork & quick wins (WS-E + WS-D scaffolding)
            └─ enables ──► realistic fixtures for testing Phases 1–2
Phase 1 ─ Clinical NLP (WS-A): medspaCy ──► then local NER fast-path
Phase 2 ─ PHI de-id hardening (WS-B)
Phase 3 ─ Fuzzy matching (WS-C)
Phase 4 ─ FHIR validation / hygiene (WS-D decision + impl)
```

**Dependencies & ordering notes.**
- Phases are largely independent and may be reordered or dropped, **except**: Phase 0's Synthea fixtures support testing Phases 1–2, and within Phase 1 **medspaCy precedes** the NER fast-path (the NER stage feeds medspaCy's assertion/section/postprocess layer).
- WS-B and WS-C are independent of WS-A and could run in parallel if desired.
- WS-D is fully independent; its only coupling is conceptual (FHIR correctness).

---

## 5. Per-workstream design

### WS-A · Clinical NLP adoption

**Objective.** Replace ~1,000 LOC of hand-rolled negation/section/precision logic with validated OSS, and cut the ~6.5-min/note latency by running common-entity extraction locally, reserving Gemini for hard cases.

**Current state.**
- `services/extraction/section_parser.py` (227) — custom Gemini section anchoring + chunker.
- `services/extraction/entity_validator.py` (407) — custom precision guards (mentioned-not-performed, value-only fragments, lifestyle-as-observation, negation, etc.).
- Negation/family-history guards embedded in the `clinical_examples.py` prompt + `entity_extractor.py`.
- Extraction runs LangExtract per section, sequentially, via Gemini → latency is round-trip-bound.

**Target design.**
1. **`ClinicalContext` stage** built on medspaCy components:
   - `medspacy.section_detection` → replaces the custom section parser.
   - `medspacy.context` (ConText) → replaces negated-finding + family/experiencer guards; negated conditions still map to FHIR `inactive` clinicalStatus (preserve current behavior).
   - `medspacy.postprocess` → replaces `entity_validator` drop-rules (procedure/observation/social-history/medication validators, supplement allowlist, PHI-placeholder checks).
2. **Local NER fast-path** — new module `services/extraction/local_ner.py`:
   - Engine: **scispaCy** (`en_ner_bc5cdr_md`, chemicals/diseases) or **GLiNER-biomed** (zero-shot, CPU, ~seconds). Final choice is an open decision (§7); design supports either behind one interface.
   - Map recognized spans to codes using the **existing** bundled RxNorm/ICD-10 indexes (`terminology.py`) — no UMLS dependency introduced.
   - **Escalation policy:** sections/spans below a confidence threshold, or section types the local model handles poorly, escalate to LangExtract/Gemini. Everything else stays local.
3. **Pipeline wiring.** Insert the local-NER → medspaCy stages into `_process_unstructured` ahead of the existing Gemini path; the precision validator becomes medspaCy `postprocess`.

**Config flag.** `EXTRACTION_ENGINE = gemini | local | hybrid` (default `gemini`; ship `hybrid` only after shadow validation). medspaCy assertion/section can ride its own sub-flag if we want to adopt it independently of the NER fast-path.

**Testing strategy.**
- Unit: ConText negation, section segmentation, and each postprocess drop-rule against expected outputs migrated from `test_entity_validator` / `test_extraction_prompt_guards`.
- Parity/shadow: run `local`/`hybrid` and `gemini` on the same real notes; assert entity-set parity within tolerance and record latency. Synthea + retained real fixtures (Phase 0) provide the corpus.
- Perf target: confirm typical note drops from ~6.5 min toward seconds for the local path.

**Rollback.** Flip `EXTRACTION_ENGINE=gemini`. medspaCy/NER code remains dormant.

**Privacy note.** The local path never leaves the device, so it requires **no** PHI de-id round-trip — a strict reduction in exposure surface for everything it handles.

---

### WS-B · PHI de-id hardening (Presidio)

**Objective.** Retire the duplicated regex layer, keep the two layers that are genuinely differentiated, and **close the documented city/location Safe Harbor gap** — with a measurable PHI-recall benchmark.

**Current state.** `phi_scrubber.py` (184, ~23 regexes), `patient_phi.py` (94, decrypt-known-identity), `phi_ner.py` (172, spaCy PERSON + eponym allowlist). City/location names pass through (general GPE NER rejected because it mislabels drugs).

**Target design.**
1. **Layer 1 → Presidio.** Replace hand-written patterns with `presidio-analyzer` predefined recognizers (SSN, phone, email, IP, URL, medical license, dates). Re-express address/account/accession as Presidio **custom recognizers** where coverage is missing.
2. **Layer 2 → kept, re-homed.** The decrypt-patient-identity-and-match logic (the documented defense against the `name_encrypted`-NULL leak) becomes a Presidio **deny-list / ad-hoc recognizer** fed the decrypted name/MRN/DOB. Behavior preserved; this is a unique strength, not outsourced.
3. **Layer 3 → kept.** The eponym/clinical-suffix intelligence is re-homed as Presidio context + allowlist around the `PERSON` recognizer so `Crohn's`/`Hodgkin`/`Gastroenterology` still survive.
4. **Close the city gap.** Add a `LOCATION` pass driven by a **clinical** model via Presidio's `TransformersNlpEngine` — `StanfordAIMI/stanford-deidentifier-base` (MIT, F1 98.9 on i2b2-2014) or `obi/deid_roberta_i2b2` (MIT) — which labels locations *without* tagging `Rifaximin` as a place. **Not** the default `en_core_web_lg` GPE.
5. **Benchmark harness.** Add an i2b2-style held-out PHI-**recall** measurement (recall is the de-id metric that matters — a miss is a breach) so we can compare before/after; the custom scrubber currently has no measured F1.

**Config flag.** `PHI_ENGINE = legacy | presidio`; the clinical `LOCATION` pass behind `PHI_LOCATION_NER_ENABLED`. Keep NER fail-open **non-latching**.

**Testing strategy.** Port every existing scrubber regression (SSN/phone/MRN/address/date-generalization, eponym survival, known-patient leak) to the Presidio path; add location-redaction tests with drug-name negatives (`Rifaximin`, `Crohn's` must survive); run the recall benchmark.

**Rollback.** `PHI_ENGINE=legacy`.

---

### WS-C · Fuzzy matching (RapidFuzz)

**Objective.** Replace homemade string similarity with a fast, maintained library in the two places it matters, preserving correctness guarantees.

**Current state.** `dedup/detector.py::_fuzzy_match` is a homemade Jaccard word-overlap; `terminology.py` lookups are exact/normalized only (the one capability the bundled-index approach lacks vs. OSS linkers).

**Target design.**
- `terminology.py`: add a **high-threshold** RapidFuzz fallback after exact/normalized lookup fails. Threshold tuned so the **"never emit a wrong code; unknown stays uncoded"** guarantee holds — below threshold returns `None`, never a guess.
- `dedup/detector.py`: swap `_fuzzy_match` for `rapidfuzz.fuzz.token_set_ratio` (or `token_sort_ratio`), keeping the existing additive-weight scoring and integer-percent banding (don't disturb the documented float-imprecision banding logic).

**Config flag.** Terminology fuzzy fallback behind `TERMINOLOGY_FUZZY_ENABLED` (default off until threshold validated). Dedup swap is a like-for-like internal change, guarded by tests rather than a runtime flag.

**Testing strategy.** Terminology: assert known fuzzy hits code correctly and near-misses stay uncoded (guardrail tests). Dedup: re-run `TestDateDistancePenalty`, band-filter, and resolve-bulk regressions to confirm scoring/banding unchanged at boundaries.

**Rollback.** Disable the terminology flag; revert the dedup matcher (trivial, isolated).

---

### WS-D · FHIR structural validation (complements the existing terminology-validation layer)

**Context — two validation layers, not one.** The codebase *already* validates the **terminology/codes** inside resources and recently added a periodic-refresh system for it. That is a **different and complementary** layer from `fhir.resources`, which validates resource **structure**. They do not overlap:

| Layer | Question it answers | Status | How it stays current |
|-------|---------------------|--------|----------------------|
| **Terminology / code** | "Is this a real ICD-10 / RxNorm / LOINC code?" | **Exists** — bundled indexes built & validated via `simple-icd-10-cm` (MIT), RxNav, NLM Clinical Tables; wired at runtime through `entity_to_fhir._resolve_coding → terminology.lookup_*` and Epic `OrderMedMapper` | **Periodic, fail-open** `refresh_medication_index` / `schedule_medication_refresh` (RxNorm churn), fired non-blocking in `main.py` lifespan |
| **FHIR structural** | "Is this a well-formed FHIR R4B resource — required fields, types, cardinality?" | **Absent** — `fhir-resources` / `fhirpathpy` declared in `pyproject.toml` but never imported (only `fhir_converter` is used, for CDA) | Version-pinned spec — updated by bumping the library, **not** a network refresh |

So adopting `fhir.resources` does **not** duplicate the recently-added medication/condition validators. It fills the one validation gap they don't cover, and it extends the established "validate against open standards, keep current, fail open" direction to resource structure.

**Objective.** Add the missing structural-validation layer using the already-declared `fhir.resources`, in fail-open log-only mode, alongside the existing terminology validation.

**Decision (revised per your steer).** Lead with **Option A — use `fhir.resources`.** Option B (delete the deps) is retained only as a fallback if structural validation proves too noisy against intentionally-partial AI resources.

**Target design (Option A).**
- Add a structural-validation step at the two points resources are produced: after `entity_to_fhir` builds an AI resource, and after `fhir_parser` maps an incoming bundle resource — validate against the `fhir.resources` R4B model in **log-only mode**. Failures are logged as drift signals; the resource still ingests (AI/hand-built/partial resources must never be rejected).
- Run it **alongside** `_resolve_coding`, not instead of it: terminology validation (codes) + structural validation (shape) become two facets of one pre-insert "resource quality" gate.
- Mirror the **fail-open, non-latching** posture already used by the medication refresh and PHI-NER — a validation/library hiccup must never block ingestion.
- *Optional, lower priority:* use `fhirpathpy` to replace some hand-rolled dict traversal in `fhir_parser` where it improves clarity.

**On "periodic updates."** The terminology layer needs periodic refresh (RxNorm/ICD churn) and already has it. FHIR **structure** does not — R4B is a pinned spec, so `fhir.resources` stays current via a version bump, not a network refresh. The existing medication-refresh design is the **precedent for the fail-open posture** this work copies, not a system this work plugs into.

**Config flag.** `FHIR_VALIDATION = off | log | strict` (default `log` after validation; `strict` never applied to AI resources). Fail-open like the existing refresh/NER.

**Testing strategy.**
- A malformed resource (missing required field) is logged but still ingested in `log` mode; a valid resource passes silently.
- Representative AI-built **partial** resources don't produce false-failure noise (tune the required-field subset / log level).
- Confirm `fhir.resources` R4B models cover the resource types actually built (the 18 `SUPPORTED_RESOURCE_TYPES` + extraction outputs).
- Use Synthea fixtures (Phase 0) as known-valid inputs.

**Rollback.** `FHIR_VALIDATION=off` (or, if Option B is ever chosen, remove the unused deps — the inventory already verified zero imports).

---

### WS-E · Quick wins

**Objective.** Two low-risk, high-clarity improvements.

1. **CLAUDE.md license wording.** Correct the terminology gotcha: **CPT** is *genuinely* license-restricted (AMA, paid — exclusion permanent); **SNOMED CT** is *free* under the UMLS Affiliate License (US + member countries) but carries redistribution/reporting obligations that make *bundling* impractical. Two different reasons — current wording conflates them. Note the legitimate future path: an **opt-in, operator-licensed** SNOMED feature (UMLS/Athena-sourced), never a bundled default.
2. **Synthea fixtures (Apache-2.0).** Add a synthetic-data generator producing FHIR R4 bundles + C-CDA for test fixtures; wire into `conftest.py` / fidelity fixtures. Addresses the e2e "tiny dataset" gap (`docs/e2e-gap-analysis.md`) and gives Phases 1–2 a realistic, shareable corpus without real PHI. Keep real user fixtures gitignored (Rule 10).

**Testing strategy.** Synthea: a smoke test that generated bundles ingest cleanly through the FHIR parser and produce expected record counts/types.

**Rollback.** Both are additive; remove the generator or revert the doc edit.

---

## 6. New dependencies (summary)

| Package | Workstream | License | Runtime weight | Notes |
|---------|------------|---------|----------------|-------|
| `medspacy` | WS-A | MIT | spaCy component | Same framework as existing `en_core_web_md` |
| `scispacy` + `en_ner_bc5cdr_md` **or** `gliner` (biomed) | WS-A | Apache-2.0 | model download (CPU) | One chosen in §7; map spans to existing indexes (no UMLS) |
| `presidio-analyzer` (+`presidio-anonymizer`) | WS-B | MIT | analyzer engine | Reuses spaCy; adds optional transformers backend |
| clinical de-id model (`StanfordAIMI/stanford-deidentifier-base` or `obi/deid_roberta_i2b2`) | WS-B | MIT | PyTorch model | Only for the `LOCATION` pass |
| `rapidfuzz` | WS-C | MIT | tiny, C-backed | Drop-in fuzzy |
| `fhir.resources` / `fhirpathpy` (already declared) | WS-D | — | — | **Use** for structural validation (log-only); complements terminology validation — not a duplicate of the recently-added code validators |
| Synthea (external generator) | WS-E | Apache-2.0 | build-time only | Not a runtime dep |

All permissive; none triggers Rule 17. Transformers/PyTorch (WS-B location model) is the heaviest addition — acceptable for a local app, and gated behind its own flag.

---

## 7. Open decisions (resolve during implementation)

1. **WS-A NER engine:** scispaCy (`en_ner_bc5cdr_md`) vs. GLiNER-biomed (zero-shot, possibly broader entity coverage, CPU-fast). Decide after a small accuracy/latency bake-off on real notes.
2. **WS-A scope of medspaCy:** adopt assertion + sections + postprocess together, or stage them.
3. **WS-B location model:** Stanford deidentifier (best benchmark) vs. OBI RoBERTa (lighter). Both MIT.
4. **WS-D:** Direction confirmed — use `fhir.resources` for structural validation in **log-only** mode (Option A). Remaining call: the exact required-field subset / log level that avoids false-failure noise on partial AI resources. Option B (delete) kept only as fallback.
5. **Thresholds:** WS-A escalation confidence; WS-C terminology fuzzy cutoff. Set empirically against fixtures, preserving "never a wrong code."

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| New extraction/de-id path changes outputs subtly | Shadow-compare on real data before flipping any default; flags allow instant rollback |
| Local NER under-extracts vs. Gemini | `hybrid` mode escalates low-confidence sections; parity tests gate the default flip |
| Presidio `LOCATION` re-introduces drug-as-place errors | Use a *clinical* model (not GPE); add drug-name negative tests |
| Transformers/PyTorch weight | Gate behind `PHI_LOCATION_NER_ENABLED`; warm-load like the existing NER |
| Terminology fuzzy emits a wrong code | High threshold; below-threshold returns `None` (guardrail tests) |
| Test DB missing new columns | Manual `ALTER` on `medtimeline_test` + ship Alembic migration (known gotcha) |
| Stale server behavior during testing | Restart uvicorn after backend edits (no `--reload`) |

---

## 9. Explicitly deferred

- The **granular implementation plan** (ordered, checkable steps per phase) — to be authored in a later session via writing-plans, combined with the user's other planned work.
- Any actual code changes — this session produced design only.
