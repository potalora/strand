# Strict-Local Medical Model Pack — Design Spec

- **Date:** 2026-07-26
- **Status:** Approved
- **Target baseline:** Apple M4 MacBook Air with 16 GB unified memory
- **Cross-platform target:** Linux with 16 GB system RAM; CPU-only supported,
  GPU acceleration optional

## 1. Goal

Add an optional, version-pinned local model pack that can OCR, ingest, extract,
and summarize medical records without sending document content to a cloud
provider.

The default validated pack has three distinct roles:

1. **OvisOCR2** — page image to faithful Markdown.
2. **NuExtract3** — OCR content, and selected source pages when needed, to
   schema-constrained clinical JSON with evidence.
3. **Qwen3.5-9B** — validated structured clinical data to a grounded final
   summary.

The feature must fit and run reliably on the 16 GB Apple baseline. It must also
support CPU-only Linux through a platform-neutral backend contract, with
optional NVIDIA or AMD acceleration.

The product promise is stronger than “a local model is selected.” In
**strict-local mode**, the complete document-processing job is local and
fail-closed. A missing, incompatible, or crashed model produces a visible local
error; it never triggers a cloud fallback.

## 2. Motivation

The application currently de-identifies health data before cloud AI calls, but
de-identification is not equivalent to keeping medical data on-device. OCR in
particular must see the original pixels, before identifiers can be reliably
removed. A cloud OCR provider therefore receives the raw medical document.

The current code also contains behavior that conflicts with a strict-local
promise:

- `backend/app/services/extraction/text_extractor.py` may fall back from a
  selected local vision provider to Gemini.
- `backend/app/api/upload.py::_resolve_extraction_engine` may degrade a missing
  local clinical-NLP installation to Gemini.
- The hybrid extraction engine may escalate difficult sections to a configured
  provider.
- Scanned PDFs are currently passed to providers as PDF bytes; local
  page-oriented OCR models need local PDF-to-image rasterization.

Those behaviors remain reasonable in explicitly cloud-assisted modes, but they
must be unreachable from strict-local jobs.

## 3. Locked decisions

1. **16 GB is both the minimum and recommended Apple memory profile.**
2. **The validated Apple runtime is embedded MLX**, not a required external
   Ollama or LM Studio installation.
3. **Only one heavyweight model is resident at a time.**
4. **OvisOCR2 performs primary OCR.**
5. **NuExtract3 performs ingestion-time structured extraction.**
6. **Qwen3.5-9B is summary-only.** It does not OCR pages, perform primary fact
   extraction, or resolve ingestion uncertainty.
7. **No second OCR sentinel or full OCR ensemble is included.**
8. **NuExtract3 may inspect the original local page image selectively** when
   OCR text is insufficient, without invoking another OCR model.
9. **Strict-local mode is fail-closed across the entire pipeline.**
10. **Linux CPU-only operation is supported.** Compatible NVIDIA and AMD GPUs
    are optional accelerators, not requirements.
11. **Ollama and LM Studio remain advanced bring-your-own-model options** in a
    visibly unverified custom-local mode. They do not inherit the validated
    pack's containment claim.
12. **Model downloads and document processing are separate trust boundaries.**
    The downloader may access Hugging Face; document-processing workers may not.
13. **QAT is not a selection requirement.** End-to-end medical-document
    fidelity and measured resource use determine approval.
14. **The pack is optional.** Existing cloud and prompt-only modes remain
    available as separate, explicit user choices.

## 4. Model selection

These are the current candidates as of 2026-07-26. Production manifests will
pin exact revisions and hashes after the artifacts pass the application fixture
suite.

### 4.1 OCR: OvisOCR2

- **Base model:** [`ATH-MaaS/OvisOCR2`](https://huggingface.co/ATH-MaaS/OvisOCR2)
- **Apple candidate:** [`sahilchachra/ovisocr2-int4-mlx`](https://huggingface.co/sahilchachra/ovisocr2-int4-mlx)
- **Linux candidate:** [`Abiray/OvisOCR2-GGUF`](https://huggingface.co/Abiray/OvisOCR2-GGUF), subject to upstream multimodal compatibility testing
- **Linux fallback:** official safetensors through a CPU/GPU Transformers worker

Rationale:

- Compact, approximately 0.8B page-level parser.
- Produces Markdown in reading order, including text, formulas, tables, and
  visual-region references.
- Reports 96.58 on OmniDocBench v1.6.
- The candidate Apple int4 artifact is approximately 622 MB.
- Small enough to leave ample memory for rasterization and application state.

Decode behavior:

- Temperature `0`.
- Greedy/non-sampling generation.
- One page at a time.
- Bounded output tokens per page.
- Generative OCR is not promised to be bitwise deterministic across runtimes,
  even at temperature `0`; critical fact-set stability is a release gate.

### 4.2 Structured extraction: NuExtract3

- **Base model:** [`numind/NuExtract3`](https://huggingface.co/numind/NuExtract3)
- **Apple candidate:** [`numind/NuExtract3-mlx-4bits`](https://huggingface.co/numind/NuExtract3-mlx-4bits)
- **Linux candidate:** [`numind/NuExtract3-GGUF`](https://huggingface.co/numind/NuExtract3-GGUF), initially `Q4_K_M`

Rationale:

- Qwen3.5-based document-understanding model specialized for structured JSON
  extraction.
- Accepts text, images, or both with a JSON template.
- Supports extractive `verbatim-string` fields and null output for missing
  values.
- Its published structured-extraction result is materially stronger than the
  compared general-purpose small models, although the benchmark is currently
  internal and is not medical-specific.

NuExtract3 is invoked through a new internal extraction backend rather than
pretending MLX is a native LangExtract provider. The backend maps NuExtract3
output into the application's existing extracted-entity, evidence, and FHIR
mapping contracts. LangExtract remains available for cloud and compatible
advanced-provider modes; strict-local mode does not make a second LangExtract
model call after NuExtract3.

Default extraction policy:

- Process OCR Markdown first.
- Use `verbatim-string` for medication names, dosages, lab values, units,
  dates, allergies, and other safety-sensitive source fields.
- Normalize dates, units, and codes only after retaining the original
  verbatim value.
- Provide the original page image only for deterministic escalation cases:
  tables, handwriting, unresolved required fields, or malformed OCR structure.
- Return `null`, `[]`, or an explicit unresolved status when evidence is
  insufficient.

### 4.3 Summarization: Qwen3.5-9B

- **Base model:** [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Apple candidate:** [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit)
- **Linux candidate:** [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), initially `Q4_K_M`

Rationale:

- Strong current small-model instruction following, long-context performance,
  and multilingual capability.
- The Apple 4-bit artifact is approximately 5.98 GB.
- The Linux `Q4_K_M` artifact is approximately 5.68 GB.
- It leaves substantially more operational headroom than experimental 27B
  ultra-low-bit models.

Hard role boundary:

- Qwen receives validated clinical JSON and bounded supporting excerpts.
- It does not receive the raw upload or perform primary ingestion.
- It does not replace missing NuExtract3 fields with guesses.
- It may perform hierarchical summary passes, but every pass remains within the
  summary stage and uses the same validated evidence set.

Initial context target is 16K–32K tokens, determined by measured memory and
fidelity. The advertised 262K context is not a default target on a 16 GB
machine.

### 4.4 Models not selected as defaults

- **Baidu Unlimited-OCR:** viable, but larger and less attractive than OvisOCR2
  for the current 16 GB MLX target.
- **PaddleOCR-VL-1.6:** close OCR benchmark competitor and a useful evaluation
  challenger, particularly for warped or photographed pages, but its official
  macOS path currently favors Docker and its MLX path is less mature.
- **GLM-OCR:** efficient with mature Ollama support, but the selected cascade
  provides stronger separation between OCR and extraction.
- **Gemma 4 QAT 26B-A4B:** genuine first-party QAT, but the official Q4_0 GGUF
  is approximately 14.4 GB before context and runtime overhead. It is not a
  responsible 16 GB default.
- **Gemma 4 E4B QAT:** feasible, but QAT preserves the base model's capability;
  it does not automatically make the smaller base a better summarizer than
  Qwen3.5-9B.
- **Ternary Bonsai 27B:** technically interesting, but its MLX path is
  approximately 9.2 GB peak at 4K context and depends on custom low-bit
  kernels. It remains an experimental advanced option, not the validated pack.

## 5. User-facing processing modes

### 5.1 Validated strict local

- Requires a validated pack/runtime profile.
- Performs OCR, extraction, validation, and summarization locally.
- Never instantiates or calls a cloud provider for the job.
- Rejects non-local provider URLs.
- Fails visibly when a required local capability is missing.
- Displays the exact models and revisions used.

### 5.2 Custom local

- Allows an explicitly configured loopback Ollama, LM Studio, or compatible
  endpoint.
- Keeps application routing fail-closed and rejects non-loopback endpoints.
- Is labelled **Custom local (unverified)** because the application cannot
  prove that an arbitrary external runtime or model has equivalent process,
  artifact, logging, or egress isolation.
- Never silently switches to validated strict local, cloud assisted, or another
  custom model.

### 5.3 Cloud assisted

- Existing cloud-capable behavior remains available as a separate mode.
- The UI states that medical or de-identified content may leave the machine.
- Cloud fallback is allowed only inside this explicitly selected mode.

### 5.4 Prompt only

- Existing prompt-building behavior remains available.
- The user manually executes the prompt elsewhere.
- It is not represented as strict local because the eventual destination is
  outside the application's control.

Changing modes is an explicit user action. A running strict-local job cannot
change mode or fall through to another mode.

## 6. End-to-end data flow

```text
Optional model download
  -> pinned artifact cache
  -> checksum/revision validation

Strict-local document job
  -> local format validation
  -> local PDF/TIFF rasterization
  -> OvisOCR2 page Markdown
  -> section/page checkpoint
  -> unload OvisOCR2
  -> NuExtract3 schema extraction
  -> deterministic schema/evidence/normalization validation
  -> persist structured clinical records and provenance
  -> unload NuExtract3
  -> Qwen3.5-9B grounded summarization
  -> validate summary evidence references
  -> persist final summary
  -> unload Qwen3.5-9B
  -> remove temporary page/model-input artifacts
```

### 6.1 OCR output contract

Each page result includes:

- Upload identifier.
- Page number.
- Source-image dimensions and rasterization settings.
- Markdown content.
- Tables/formulas preserved in a stable representation.
- Detected warnings such as empty output, repetition, truncation, or malformed
  structure.
- Model id, revision, quantization, decode settings, and processing version.
- Content hash for resume and audit integrity.

### 6.2 Extraction output contract

The clinical schema includes:

- Normalized value used by the application.
- Original verbatim source value.
- Entity type.
- Page and section reference.
- Supporting OCR excerpt.
- Source-image reference when image escalation was used.
- Confidence or unresolved status.
- Normalization/coding method and version.
- No unsupported inferred clinical fact.

The existing FHIR mapping layer remains downstream. NuExtract3 does not write
FHIR resources directly.

### 6.3 Summary input and output contract

The summary model receives:

- Validated structured clinical JSON.
- Bounded evidence excerpts.
- Requested summary type and date/category filters.
- Existing no-diagnosis/no-treatment-advice constraints.

The summary output is typed data rather than unconstrained rendered Markdown.
Each factual statement carries the ids of the validated clinical facts and
evidence records that support it. The server rejects unknown ids and factual
entries without support, then renders the accepted structure.

The rendered summary includes:

- The existing medical disclaimer.
- Source references for factual claims.
- Explicit uncertainty or missing information.
- No ingestion-time field creation or correction.

## 7. Platform-neutral runtime architecture

The application depends on role contracts, not MLX-specific calls:

```python
class OCRBackend(Protocol):
    async def parse_page(self, request: OCRPageRequest) -> OCRPageResult: ...

class ExtractionBackend(Protocol):
    async def extract(self, request: ExtractionRequest) -> ClinicalExtraction: ...

class SummaryBackend(Protocol):
    async def summarize(self, request: SummaryRequest) -> GroundedSummary: ...
```

Requests and results are serializable and contain no provider SDK objects.
Platform adapters implement these contracts.

### 7.1 Apple Silicon

- Embedded MLX workers are the validated default.
- Workers run natively on the macOS host so Metal can use unified memory.
- The model manager spawns each worker as an isolated child process.
- Worker communication uses process pipes or an owner-only local IPC transport,
  never a LAN-bound public model server.

### 7.2 Linux CPU

- 16 GB system RAM is the supported baseline.
- `llama.cpp` with pinned GGUF artifacts is preferred where the exact
  architecture passes fidelity tests.
- Transformers/safetensors is the local fallback for models whose GGUF
  multimodal path is not validated.
- CPU processing is expected to be slower and is presented as background work
  with page-level progress and resume.
- No silent GPU, cloud, or alternate-model substitution occurs.

### 7.3 Linux GPU acceleration

- NVIDIA CUDA and AMD ROCm/HIP/Vulkan are optional acceleration profiles.
- A hardware/runtime combination is advertised only after it passes the full
  fixture, memory, and egress suite.
- GPU acceleration does not change schemas, prompts, privacy behavior, or
  release thresholds.
- Intel and other accelerator profiles can be added later through the same
  backend contracts.

### 7.4 Advanced external runtimes

Ollama and LM Studio remain supported as advanced loopback-only endpoints where
the selected model actually supports the task. They are labelled **unverified**
unless the exact model/runtime combination passes the same acceptance suite.

The default pack never assumes that the presence of a GGUF file proves upstream
Ollama, LM Studio, or llama.cpp multimodal compatibility.

## 8. Model manifest and artifact management

Every validated downloadable artifact is described by an application-shipped,
version-controlled manifest entry. If a future release distributes catalog
updates independently of the application, those catalog entries must be
cryptographically signed before the application trusts them.

- Logical role.
- Platform and architecture.
- Hugging Face repository.
- Exact revision.
- Required filenames.
- SHA-256 hashes.
- Quantization method and bit width.
- Runtime and minimum compatible version.
- License and attribution.
- Download and expected resident size.
- Context/output limits.
- Decode defaults.
- Validation status and fixture-suite version.

Downloads:

- Run outside document-processing jobs.
- Receive no upload identifiers, document bytes, prompts, or clinical output.
- Support progress, retry, and resume.
- Verify hashes before atomic activation.
- Leave incomplete downloads inactive.
- Never replace an active model during a job.
- Preserve the previous validated revision for rollback.

The downloader treats model repositories as untrusted data:

- It fetches only manifest-listed weights, tokenizer/config files, and
  explicitly approved data assets.
- It rejects pickle-based weights, arbitrary executables, and repository Python
  code.
- Runtime `trust_remote_code` behavior is disabled.
- Any architecture-specific model or preprocessing code is reviewed, pinned,
  and shipped with the application/runtime adapter rather than executed from
  the downloaded repository.
- File count, individual size, aggregate size, and archive expansion are
  bounded before activation.

Users may remove one model or the complete pack without deleting medical
records.

## 9. Model lifecycle and the 16 GB budget

The host model manager owns a global local-model execution queue. On the
validated 16 GB profile, concurrent jobs do not load different models
simultaneously.

Lifecycle:

```text
READY -> LOADING -> RUNNING -> UNLOADING -> READY
                    |             |
                    +-> ERROR <----+
```

- OvisOCR2 processes pages and exits before NuExtract3 loads.
- NuExtract3 finishes and exits before Qwen3.5-9B loads.
- Process termination is used to ensure Metal/RAM is reclaimed.
- Page and section checkpoints allow resume without retaining the earlier model.
- Cancellation terminates the active worker and cleans temporary inputs.
- A summary request arriving during ingestion queues behind the active model
  stage.

Preflight checks:

- Model presence and hash.
- Runtime compatibility.
- Available disk.
- Expected peak memory.
- Context and output limits.
- Strict-local provider policy.

The acceptance target is no memory-pressure termination and no sustained swap
thrashing on the 16 GB M4. Exact safe resident limits are set from real
measurements rather than estimated solely from artifact size.

## 10. Strict-local privacy boundary

### 10.1 Provider routing

For a strict-local job:

- Cloud provider objects are not constructed.
- Cloud API keys are ignored even when configured.
- Existing Gemini OCR fallback is disabled.
- Existing local/hybrid-to-Gemini extraction degradation is disabled.
- Hybrid cloud escalation is disabled.
- Non-loopback custom endpoints are rejected.
- Model download code is unavailable to the processing worker.

### 10.2 Worker isolation

- Embedded model workers load only local, pre-verified artifact paths.
- They communicate over pipes or owner-only local IPC.
- They do not bind a public HTTP port.
- Rasterizers and model workers run with bounded inputs, page dimensions,
  output sizes, execution time, and temporary storage.
- Core dumps are disabled for processes that handle document content where the
  target packaging/runtime permits it.
- Outbound networking is denied at the packaging/runtime layer where the target
  platform provides a dependable mechanism.
- Application-level egress guards and tests remain mandatory even where an
  OS-level sandbox is available.

The release claim is based on verified application behavior and supported
packaging profiles. The UI must not imply that an arbitrary unverified custom
runtime has equivalent OS-level isolation.

### 10.3 Logging and telemetry

Validated strict-local jobs produce local operational diagnostics only. Remote
analytics events and automatic crash-report uploads are disabled for the job
and its workers rather than being permitted after redaction.

Application logs, analytics, and crash diagnostics must not include:

- Document text or page images.
- Prompts.
- Extracted clinical values.
- Summaries.
- Patient identifiers.

Local non-content diagnostics may include model revision, stage, duration, page
number, error category, memory measurements, and result hashes.

Diagnostic sharing is a separate, explicit user action and the exported bundle
is structurally redacted before it can leave the machine.

### 10.4 Storage boundary

Temporary rasterized pages and transient model inputs are removed after
success, cancellation, and recoverable crash cleanup.

Durable OCR text, extracted records, provenance, and summaries continue to
follow the application's medical-record retention and deletion rules. This
feature does **not** by itself solve the repository's separately documented
at-rest encryption gaps.

## 11. Validation and deterministic guards

Deterministic validators run after NuExtract3 and before Qwen:

- JSON/schema parsing.
- Allowed entity and enum values.
- Required field relationships.
- Date parsing and valid timezone/range checks.
- Unit/value separation.
- Verbatim evidence presence.
- Page/section reference validity.
- Duplicate/repetition detection.
- Existing negation, mentioned-not-performed, family-history, terminology, and
  FHIR mapping safeguards.

Validation never asks Qwen to repair missing clinical facts. A failed field is
rejected or marked unresolved.

A bounded local retry may correct malformed syntax using the same NuExtract3
worker and source evidence. Exhausted retries fail the extraction stage locally.

## 12. Installation and user experience

The Local AI settings area:

- Detects Apple Silicon or Linux and optional accelerators.
- Recommends only artifacts for the current platform.
- Shows model role, repository, revision, license, download size, expected
  memory, and validation status.
- Provides install, resume, retry, update, rollback, and remove controls.
- Clearly separates **Validated local pack** from **Custom local model**.

Strict-local job progress displays:

- Preflight.
- Rasterization.
- OCR page `N / total`.
- Extraction section/page progress.
- Validation.
- Summarization.
- Model loading and unloading.

Failures display:

- Failed stage.
- Local model/runtime involved.
- Retryability.
- Preserved checkpoint.
- Confirmation that cloud fallback was not attempted.

The UI does not promise instant processing. Large records may take minutes on
the M4 baseline and substantially longer on Linux CPU.

## 13. Cancellation, retry, and resume

- Page OCR checkpoints are keyed by upload hash, page number, rasterization
  version, and model manifest.
- Extraction checkpoints are keyed by OCR-content hash, schema version, prompt
  version, and model manifest.
- Summary checkpoints are invalidated when validated clinical input changes.
- Retrying a failed stage does not repeat completed valid work.
- Changing a model revision or schema invalidates only the dependent
  checkpoints.
- Cancellation is checked between pages, sections, and summary passes.
- A stale/crashed worker is terminated before a replacement worker starts.

## 14. Auditability

The durable processing audit records:

- Job privacy mode.
- Model roles, repositories, revisions, and quantizations.
- Runtime/backend versions.
- Prompt/schema/validator versions.
- Stage timestamps and non-content hashes.
- Page and evidence references.
- Unresolved or rejected fields.
- Cancellation, retry, and resume events.

It does not record raw prompts or document content in the operational log.
Clinical content stored as part of the user's record remains governed by the
normal database authorization and retention model.

## 15. Testing and release gates

### 15.1 Fixture tiers

- Public synthetic records committed for CI.
- Adversarial generated pages covering decimal points, dosage units, dates,
  negation, tables, handwriting, skew, repeated headers, illumination, and poor
  scans.
- Real private medical fixtures resolved through
  `REAL_MEDICAL_FIXTURES_DIR`, never committed.
- Manually annotated golden OCR, entity, evidence, and summary outputs.

### 15.2 Hard privacy and correctness gates

- Zero external network requests during strict-local processing.
- Zero cloud fallback when a model is absent, incompatible, crashed, or
  malformed.
- 100% schema-valid accepted extraction output after bounded retries.
- Every accepted extracted clinical fact has valid source evidence.
- Zero unsupported summary facts in the release acceptance corpus.
- Zero document/PHI content in logs, telemetry, or crash diagnostics.
- Temporary input cleanup passes on success, cancellation, and simulated crash.
- Model files exactly match the approved manifest.

### 15.3 Quality targets

- At least 99% exact recognition of critical numeric tokens: dosage, lab value,
  date, and unit.
- At least 98% precision and 95% recall for critical extracted fields.
- No silent page omission, repetition loop, or fabricated line in the release
  corpus.
- Three repeated runs produce the same critical fact set, allowing only
  non-material Markdown formatting differences.
- MLX and Linux variants produce materially equivalent extracted facts.

Missing a target keeps the artifact in preview/unverified status. Targets are
not relaxed merely to ship a model revision.

### 15.4 Resource and lifecycle gates

- Complete the reference workflow on the 16 GB M4 without memory-pressure
  termination or sustained swap thrashing.
- Keep every worker below its measured safe resident-memory budget.
- Complete the Linux CPU reference workload within documented bounded times and
  the 16 GB system-RAM profile.
- Release memory after every worker exits.
- Resume without repeating completed pages or sections.

### 15.5 Test layers

1. Unit tests for manifests, routing policy, schemas, validators, checkpoints,
   and worker state transitions.
2. Worker contract tests for MLX, llama.cpp, and Transformers adapters.
3. Integration tests for the full validated-strict-local upload pipeline with
   all cloud keys present, plus separate custom-local routing tests.
4. Egress tests that intercept or deny external sockets and assert no attempt.
5. Fidelity tests on synthetic and private medical fixtures.
6. Repeated-run and cross-quantization parity tests.
7. Memory/throughput benchmarks on the exact 16 GB M4 and supported Linux
   profiles.
8. API verification first, followed by manual frontend verification.

## 16. Rollout

1. Add platform-neutral contracts, manifest handling, and strict-local routing
   policy behind a disabled feature flag.
2. Implement and validate the Apple MLX workers on the exact 16 GB M4.
3. Integrate OCR, NuExtract3 extraction, deterministic validation, and
   summary-only Qwen in that order.
4. Add model management and strict-local progress/error UI.
5. Run privacy, fidelity, and resource gates; label passing Apple revisions
   validated.
6. Implement Linux CPU adapters and optional accelerator profiles.
7. Validate each Linux artifact/runtime independently before advertising it.
8. Retain cloud and prompt-only modes as explicit alternatives throughout.

## 17. Out of scope

- Supporting machines below 16 GB as validated profiles.
- Diagnoses, treatment recommendations, or clinical decision support.
- Fine-tuning models during the initial implementation.
- Sending raw documents to a summary model.
- Using Qwen3.5-9B to repair or infer ingestion facts.
- A second OCR sentinel or mandatory OCR ensemble.
- Treating every Hugging Face quantization as trusted or compatible.
- Automatically updating model revisions without validation and rollback.
- Advertising arbitrary Ollama/LM Studio models as validated.
- Windows certification in the first rollout.
- Solving the application's broader encryption-at-rest remediation in this
  feature.

## 18. Known risks and open implementation measurements

- OvisOCR2's community GGUF multimodal path must be validated against upstream
  llama.cpp; the official safetensors worker is the Linux fallback.
- The NuExtract3 benchmark is not medical-specific and its published evaluation
  is not yet an independently open fixture suite.
- Quantization may alter OCR characters or extraction behavior despite strong
  general benchmarks.
- The fanless M4 Air may throttle during long multi-page jobs.
- Exact context, batching, resident-memory, and throughput settings require
  measurement on the target machine.
- Containerized macOS operation needs a host-native MLX companion and a private
  IPC integration; it cannot run Metal MLX inside the Linux backend container.
- OS-level outbound-network isolation differs across packaging targets, so only
  profiles that pass the egress suite receive the strict-local validation label.

These are release measurements, not reasons to weaken the locked privacy or
role-separation decisions.

## 19. Success criteria

The feature is successful when a user with the validated optional pack can:

1. Disconnect from the internet after installation.
2. Upload a supported medical document.
3. See local page OCR, structured clinical extraction, and a grounded summary.
4. Inspect page-level evidence and unresolved fields.
5. Confirm the exact local models used.
6. Complete the workflow within the 16 GB target profile.
7. Verify through application behavior and tests that no document content was
   sent to a cloud provider.
