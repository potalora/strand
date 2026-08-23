# Backend Engineering Handoff: AI-Enabled Clinical Uploads/Exports API Contract

This document defines the complete API contract between the AI-Enabled Clinical Uploads/Exports frontend and backend. All endpoints are prefixed with `/api/v1`. The frontend expects JSON responses with the schemas defined below.

---

## Authentication

All authenticated endpoints require a `Bearer` token in the `Authorization` header. The frontend stores JWT tokens in localStorage via Zustand and attaches them automatically through the `ApiClient` class (`src/lib/api.ts`).

### Account login identifier

New clients send `login_identifier`, the private value used to sign in. No email
address is required. The deprecated `email` field is still accepted in register
and login requests and returned in user responses as an alias for
`login_identifier`. It does not mean that the value is an email address. If a
request supplies both fields, they must match under the rule below or the API
returns the generic 422 response.

Strand trims the value for storage. Lookup and uniqueness use exactly:

```python
value.strip().lower()
```

This is not Unicode `casefold()` or Unicode canonical-equivalence matching.
For example:

- ` Alice ` and `alice` both produce `alice` and identify the same account.
- `Straße` produces `straße`, while `STRASSE` produces `strasse`; they remain
  distinct.
- Precomposed `Å` (`U+00C5`) produces `å` (`U+00E5`). Decomposed `A` plus a
  combining ring (`U+0041 U+030A`) produces `a` plus a combining ring
  (`U+0061 U+030A`), so the two forms remain distinct.

The identifier is private. `users.login_identifier` contains app-layer
AES-256-GCM ciphertext. `users.login_identifier_hmac` contains the unique,
indexed keyed HMAC-SHA256 blind index used for exact lookup; it is not a
plaintext identifier.

Auth validation and credential failures return these complete generic bodies:

| Condition | Response |
|-----------|----------|
| Invalid payload, invalid identifier or password shape, or conflicting aliases | HTTP 422 `{"detail":"Invalid authentication request."}` |
| Duplicate or concurrently claimed identifier | HTTP 409 `{"detail":"Account identifier is unavailable."}` |
| Unknown identifier, wrong password, or disabled account | HTTP 401 `{"detail":"Invalid account identifier or password."}` |

An active temporary lockout keeps its existing 401 response instead of the
generic invalid-credentials body.

### POST `/auth/register`

Create a new user account.

**Request:**
```json
{
  "login_identifier": "pedro",
  "password": "SecurePass123!",
  "display_name": "Pedro"
}
```

Deprecated request spelling:

```json
{
  "email": "existing@example.com",
  "password": "SecurePass123!"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "login_identifier": "pedro",
  "email": "pedro",
  "display_name": "Pedro",
  "is_active": true,
  "created_at": "timestamp"
}
```

The deprecated `email` response alias repeats `login_identifier`.

### POST `/auth/login`

Authenticate and receive JWT tokens.

**Request:**
```json
{
  "login_identifier": "pedro",
  "password": "SecurePass123!"
}
```

The deprecated `email` request spelling shown under registration is also
accepted for login.

**Response (200):**
```json
{
  "access_token": "jwt-string",
  "refresh_token": "jwt-string",
  "token_type": "bearer"
}
```

### POST `/auth/refresh`

Refresh the access token using a refresh token. The old refresh token is revoked and a new token pair is issued.

**Request:**
```json
{
  "refresh_token": "jwt-string"
}
```

**Response (200):**
```json
{
  "access_token": "jwt-string",
  "refresh_token": "jwt-string",
  "token_type": "bearer"
}
```

**Errors:** `401` (invalid or expired refresh token)

### POST `/auth/logout`

Revoke the current session. Requires auth header.

**Response (204):** No content.

### GET `/auth/me`

Get the authenticated user's profile. Used by the Admin > SYS tab.

**Response (200):**
```json
{
  "id": "uuid",
  "login_identifier": "pedro",
  "email": "pedro",
  "display_name": "Pedro",
  "is_active": true,
  "created_at": "timestamp"
}
```

The response has the same fields as registration.

---

## Dashboard

### GET `/dashboard/overview`

Primary data source for the Home pane. Returns aggregate statistics and recent records.

**Response (200):**
```json
{
  "total_records": 347,
  "total_patients": 1,
  "total_uploads": 3,
  "records_by_type": {
    "condition": 42,
    "observation": 128,
    "medication": 65,
    "encounter": 30,
    "immunization": 12,
    "procedure": 20,
    "document": 15,
    "allergy": 8,
    "imaging": 5,
    "diagnostic_report": 22
  },
  "recent_records": [
    {
      "id": "uuid",
      "record_type": "observation",
      "display_text": "Blood pressure: 120/80 mmHg",
      "effective_date": "2024-01-25T14:00:00Z",
      "created_at": "2024-02-01T10:30:00Z"
    }
  ],
  "date_range_start": "2019-03-15T00:00:00Z",
  "date_range_end": "2024-01-25T14:00:00Z"
}
```

**Notes:**
- `recent_records` should return the 10 most recently created records
- `records_by_type` keys are the `record_type` field values from `health_records`
- `date_range_start` and `date_range_end` are the min/max `effective_date` across all records
- All data MUST be scoped to the authenticated user (`user_id` filter)

### GET `/dashboard/labs`

Lab-specific dashboard data. Used by the Admin > LABS tab.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | Page number (1-indexed) |
| `page_size` | int | 20 | Items per page (max 100) |

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "display_text": "Glucose [Mass/volume] in Blood",
      "effective_date": "2024-01-15T08:00:00Z",
      "value": 95,
      "unit": "mg/dL",
      "reference_low": 70,
      "reference_high": 100,
      "interpretation": "N",
      "code_display": "Glucose",
      "code_value": "2345-7"
    }
  ],
  "total": 128,
  "page": 1,
  "page_size": 20
}
```

**Interpretation codes:** `N` (normal), `H` (high), `HH` (critical high), `L` (low), `LL` (critical low), `A` (abnormal), `AA` (critical abnormal).

**Notes:**
- Extract from `health_records` where `record_type = 'observation'` and the FHIR resource contains lab-relevant category codes
- `value` can be numeric or string; `unit` is the UCUM unit from the FHIR Observation
- `reference_low`/`reference_high` come from `referenceRange` in the FHIR resource

### GET `/dashboard/patients`

List patients belonging to the current user.

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "fhir_id": "Patient/12345",
      "gender": "female"
    }
  ],
  "total": 1
}
```

---

## Records

### GET `/records`

Paginated, filterable record list. Primary data source for Admin > ALL tab and type-specific tabs.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | Page number (1-indexed) |
| `page_size` | int | 20 | Items per page (max 100) |
| `record_type` | string | - | Filter by record type (e.g., `medication`, `condition`) |
| `search` | string | - | Full-text search on `display_text` and `code_display` |

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "patient_id": "uuid",
      "record_type": "medication",
      "fhir_resource_type": "MedicationRequest",
      "fhir_resource": { "resourceType": "MedicationRequest" },
      "source_format": "fhir_r4",
      "effective_date": "2024-01-15T00:00:00Z",
      "status": "active",
      "category": ["medication"],
      "code_system": "http://www.nlm.nih.gov/research/umls/rxnorm",
      "code_value": "197361",
      "code_display": "Lisinopril 10 MG Oral Tablet",
      "display_text": "Lisinopril 10 MG — Take once daily",
      "created_at": "2024-02-01T10:30:00Z"
    }
  ],
  "total": 347,
  "page": 1,
  "page_size": 20
}
```

### GET `/records/:id`

Single record detail. Used by RecordDetailSheet and deep-link record detail page.

**Response (200):** Same schema as a single item from the `items` array above.

**Errors:** `404` (record not found or belongs to different user)

### DELETE `/records/:id`

Soft-delete a record (sets `deleted_at` timestamp). **Never hard-delete.**

**Response (204):** No content.

---

## Timeline

### GET `/timeline`

Date-ordered event list for the Timeline pane. Lighter than `/records` — doesn't include full FHIR resources.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `record_type` | string | - | Filter by type |
| `limit` | int | 200 | Max events to return |

**Response (200):**
```json
{
  "events": [
    {
      "id": "uuid",
      "record_type": "observation",
      "display_text": "Blood pressure: 120/80 mmHg",
      "effective_date": "2024-01-25T14:00:00Z",
      "code_display": "Blood pressure panel",
      "category": ["vital-signs"]
    }
  ],
  "total": 347
}
```

**Notes:**
- Events should be ordered by `effective_date DESC` (newest first)
- `total` is the total count matching filters (before limit is applied)
- Frontend groups events by month/year for display

---

## Upload & Ingestion

### POST `/upload`

Upload a file for ingestion. Accepts multipart form data.

**Request:** `multipart/form-data` with field `file` (JSON, XML, or ZIP)

**Response (200):**
```json
{
  "upload_id": "uuid",
  "status": "completed",
  "records_inserted": 142,
  "errors": []
}
```

**Notes:**
- For small files: process synchronously and return results immediately
- For large files (>5s processing): return `202 Accepted` with `upload_id` and `status: "processing"`, then process in background
- File size limit: 500MB per file, 5GB for Epic exports
- Supported MIME types: `application/json`, `application/zip`, `text/xml`, `application/xml`
- Validate file integrity before processing

#### Standalone CDA XML Upload

The upload endpoint auto-detects CDA XML documents (`.xml` files containing `<ClinicalDocument>` in the first 500 bytes). When detected:

1. **CDA-to-FHIR conversion** -- The document is converted to FHIR R4 resources via `python-fhir-converter`
2. **Bulk insertion** -- Records are inserted in configurable batches (default 100)
3. **DB dedup** -- Standard upload-scoped dedup runs against existing database records

No special endpoint needed -- uses the existing `POST /upload` endpoint.

#### IHE XDM Package Ingestion

The upload endpoint automatically detects IHE XDM packages (ZIP files containing `METADATA.XML`). When detected:

1. **Manifest parsing** -- `METADATA.XML` provides document inventory, SHA-1 hashes, and patient demographics
2. **Format prioritization** -- CDA XML documents are parsed for structured data; PDF/HTML files in the same package are skipped (logged as "structured preferred")
3. **CDA-to-FHIR conversion** -- Each CDA XML document is converted to FHIR R4 resources via `python-fhir-converter`
4. **Cross-document dedup** -- Identical records across multiple CDA documents (e.g., same allergy in 6 documents) are collapsed before insertion
5. **DB dedup** -- Standard upload-scoped dedup runs against existing database records

No new API endpoints -- uses the existing `POST /upload` endpoint. The coordinator auto-detects the format.

### GET `/upload/:id/status`

Poll for ingestion progress on large imports. Frontend polls every 2 seconds.

**Response (200):**
```json
{
  "upload_id": "uuid",
  "ingestion_status": "processing",
  "ingestion_progress": {
    "current_file": "MEDICATIONS.tsv",
    "file_index": 12,
    "total_files": 47,
    "records_ingested": 8400,
    "records_failed": 3
  },
  "ingestion_errors": [
    {
      "file": "ORDER_PROC.tsv",
      "row": 445,
      "error": "invalid date format"
    }
  ],
  "record_count": 8400,
  "total_file_count": 47,
  "processing_started_at": "2024-02-01T10:30:00Z",
  "processing_completed_at": null
}
```

**Status values:** `pending`, `processing`, `dedup_scanning`, `dedup_processing`, `completed`, `completed_with_merges`, `awaiting_review`, `failed`, `partial`

For strict-local unstructured jobs, `local_run.models` contains the captured OCR
and extraction model identities only. Summary-model provenance is reported on
the summary response, not on ingestion status.

**Frontend behavior:**
- Polls while status is `pending` or `processing`
- Stops polling when `completed`, `failed`, or `partial`
- Shows partial results (records already committed) even during processing

### GET `/upload/:id/errors`

Get row-level ingestion errors for a specific upload.

**Response (200):**
```json
{
  "errors": [
    {
      "file": "ORDER_PROC.tsv",
      "row": 445,
      "error": "invalid date format"
    }
  ]
}
```

### GET `/upload/history`

List upload history with record counts.

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "filename": "records.json",
      "ingestion_status": "completed",
      "record_count": 142,
      "file_size_bytes": 524288,
      "created_at": "2024-02-01T10:30:00Z"
    }
  ],
  "total": 3
}
```

### POST `/upload/unstructured`

Upload a PDF, RTF, or TIFF for AI-powered text extraction and entity extraction. Processing happens in the background.

**Request:** `multipart/form-data` with:

- `file`: PDF, RTF, TIF, or TIFF
- `processing_mode` (optional): `validated_strict_local` or `cloud_assisted`.
  When omitted, the user's saved ingestion mode is captured.

**Response (202):**
```json
{
  "upload_id": "uuid",
  "filename": "record.pdf",
  "status": "processing",
  "file_type": "pdf"
}
```

**Notes:**
- Allowed extensions: `.pdf`, `.rtf`, `.tif`, `.tiff`
- Magic byte validation ensures content matches claimed file type
- The processing mode, model manifest, and schema version are immutable once
  the upload row is created.
- `validated_strict_local` uses the contained OvisOCR2 and NuExtract3 path. It
  branches before provider configuration and fails without cloud fallback.
- The self-hosted backend still receives, decrypts, and prepares the upload.
  PDF/TIFF rasterization and RTF decoding happen locally in that
  network-capable process; model inference runs in the separate macOS
  network-denied worker. Strict-local is an application-enforced no-cloud
  route, not a machine-wide data-loss-prevention boundary around the backend.
- For cloud-assisted PDF and TIFF OCR, the selected vision provider receives
  the original, unredacted document or pages. OCR happens before Strand has
  text to scrub. Returned text is de-identified before downstream external
  extraction calls. RTF text is extracted locally before that scrub.
- If the selected vision provider is a cloud service and refuses or fails,
  Strand does not resend the document to a second cloud provider. A loopback
  vision provider that cannot read the document may fall back once to Gemini,
  which then receives the original document.
- Accepted entities are evidence-validated, mapped to FHIR, and auto-confirmed.
- File size limit: 500MB

### POST `/upload/unstructured-batch`

Upload multiple unstructured files for concurrent processing.

**Request:** `multipart/form-data` with:

- `files`: multiple PDF, RTF, or TIFF files
- `processing_mode` (optional): one captured `validated_strict_local` or
  `cloud_assisted` mode for the whole batch

**Response (202):**
```json
{
  "uploads": [
    {
      "upload_id": "uuid",
      "filename": "record.pdf",
      "status": "processing",
      "file_type": "pdf"
    }
  ],
  "rejected": [
    {
      "filename": "notes.txt",
      "code": "unsupported_type"
    }
  ],
  "total": 1
}
```

**Notes:**
- The request is best effort. Valid files are accepted even when another file
  is rejected.
- `total` is the number of accepted uploads, not the number of submitted files.
- `rejected` contains at most 50 entries. Its stable codes are
  `missing_filename`, `unsupported_type`, `file_too_large`, and
  `invalid_signature`. It does not include raw validation details or local
  paths.
- Each file is processed independently in the background
- Strict-local batch admission is atomic with model-pack maintenance. A pack
  update, verification, rollback, or removal cannot race a newly admitted job.

### GET `/upload/:id/extraction`

Get extracted text and entities for an unstructured upload.

**Response (200):**
```json
{
  "upload_id": "uuid",
  "status": "awaiting_confirmation",
  "extracted_text_preview": "First 500 characters of extracted text...",
  "entities": [
    {
      "entity_class": "medication",
      "text": "Lisinopril 10mg",
      "attributes": { "dosage": "10mg", "frequency": "daily" },
      "start_pos": 120,
      "end_pos": 135,
      "confidence": 0.92
    }
  ],
  "error": null
}
```

**Entity classes:** `medication`, `condition`, `procedure`, `lab_result`, `vital_sign`, `allergy`, `provider`

**Notes:**
- `extracted_text_preview` is the first 500 characters of the extracted text
- `entities` may be empty if extraction is still processing or if no entities were found
- `error` contains a user-safe error message if extraction failed

### POST `/upload/:id/confirm-extraction`

Confirm extracted entities and create FHIR health records.

**Request:**
```json
{
  "patient_id": "uuid",
  "confirmed_entities": [
    {
      "entity_class": "medication",
      "text": "Lisinopril 10mg",
      "attributes": { "dosage": "10mg" },
      "start_pos": 120,
      "end_pos": 135,
      "confidence": 0.92
    }
  ]
}
```

**Response (200):**
```json
{
  "upload_id": "uuid",
  "records_created": 5,
  "status": "completed"
}
```

**Notes:**
- The user reviews extracted entities in the UI and submits only the confirmed ones
- Each confirmed entity is mapped to a FHIR resource and stored as a `HealthRecord` with `ai_extracted=true`
- The upload status transitions from `awaiting_confirmation` to `completed`

---

## Validated local AI

All endpoints below require authentication and are prefixed with
`/api/v1/local-ai`.

`LOCAL_AI_OPERATOR_USER_IDS` is a comma-separated UUID allowlist for browser
accounts that may manage the machine-global pack. An empty value denies all web
pack maintenance, and a malformed non-empty value stops backend startup. A
missing or revoked credential receives `401`. An authenticated account absent
from the allowlist receives `403` with `Local model pack management requires a
machine operator.`

The v2 candidate profile is native Apple Silicon with at least 16 GB unified
memory; 16 GB is both the minimum and recommended baseline. Its locked 9.02
GiB pack uses OvisOCR2 for OCR, NuExtract3 for clinical extraction, and
Qwen3.5-9B for final summarization. The static candidate references are
`catalog-v2.json` and `apple-m4-16gb-v2.lock.json`.

This Track D state has no verified v2 release evidence. The file
`apple-m4-16gb-v2.release.json` is written only after the separately authorized
benchmark, synthetic fidelity, promotion, and post-promotion verification
gates pass. Historical v1 metrics and evidence remain diagnostic history; they
do not validate the v2 candidate.

Strict local remains opt-in through `LOCAL_AI_ENABLED=false`. Once enabled, a
pack is `ready` only when the active files match the exact immutable manifest,
a sealed runtime-validation receipt exists for that manifest, and the bound
benchmark and fidelity evidence passes revalidation.

Schema-v2 manifests also carry `worker_identity_scheme` and
`worker_bundle_sha256`. The `local-ai-worker-bundle.v1` digest covers the fixed
`local-ai-mlx-worker=local_ai_mlx_worker.__main__:main` entry point,
`pyproject.toml`, `uv.lock`, and the effective worker `.py` tree. Launcher,
virtual-environment, CPython, and import-surface checks are fail-closed
preconditions but are not hashed. This identity does not attest the operating
system owner or root of trust.

The native macOS worker is launched under a fixed OS network-deny profile. An
owner-only cross-process lock remains held through worker-group cleanup, so
separate backend processes cannot load two model roles at once. A parent-death
watchdog terminates the worker group and releases that lease if the backend
exits unexpectedly.

### GET `/local-ai/status`

Returns the platform, compatibility, available and active pack revisions, the
three role identities, and at most one current lifecycle operation. Artifact
download bytes are reported separately from expected resident memory. Expected
memory is taken from the release evidence when the selected profile has
measurements; it remains `null` for an unmeasured future profile.
Every authenticated user can read this endpoint. `can_manage_pack` states
whether the account may use the lifecycle controls. A non-operator receives
`operation: null` even while `state` still reports `downloading` or `verifying`.

```json
{
  "platform": "apple_silicon",
  "compatible": true,
  "enabled": false,
  "can_manage_pack": false,
  "state": "not_installed",
  "status_reason": null,
  "active_revision": null,
  "available_revision": "apple-m4-16gb-v2",
  "models": [
    {
      "role": "ocr",
      "repository": "sahilchachra/ovisocr2-int4-mlx",
      "revision": "1e9cea98871c19b2349a5d2df36fb6c4c38a1237",
      "quantization": "int4",
      "runtime": "mlx-vlm 0.5.0",
      "license": "apache-2.0",
      "download_bytes": 652031947,
      "expected_memory_bytes": null,
      "installed": false,
      "validated": false
    },
    {
      "role": "extraction",
      "repository": "numind/NuExtract3-mlx-4bits",
      "revision": "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
      "quantization": "4bit",
      "runtime": "mlx-vlm 0.5.0",
      "license": "apache-2.0",
      "download_bytes": 3054403529,
      "expected_memory_bytes": null,
      "installed": false,
      "validated": false
    },
    {
      "role": "summary",
      "repository": "mlx-community/Qwen3.5-9B-MLX-4bit",
      "revision": "938d8919941c6e7efd3c7150eff7fe9d12afa631",
      "quantization": "4bit",
      "runtime": "mlx-vlm 0.5.0",
      "license": "apache-2.0",
      "download_bytes": 5977073021,
      "expected_memory_bytes": null,
      "installed": false,
      "validated": false
    }
  ],
  "operation": null
}
```

`not_installed` still reports the available locked pack and its three model
artifacts. `installed` and `validated` remain `false` until that exact pack is
present and passes validation. `expected_memory_bytes` remains `null` until
matching release evidence provides measured values.

`state` is one of `not_installed`, `downloading`, `verifying`, `preview`,
`ready`, `update_available`, or `failed`. When `state` is `preview`,
`status_reason` is either `feature_disabled` or `release_evidence_missing`.
For every other state it is `null`. The `enabled` field reports
`LOCAL_AI_ENABLED` directly. A queued, running, or paused maintenance operation
takes precedence over `ready`; a terminal maintenance failure does not hide an
otherwise validated active pack.

### Pack lifecycle

Every route in this table requires a machine operator:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/local-ai/install` | Download, hash-check, test offline, and activate the locked pack |
| `POST` | `/local-ai/verify` | Re-run exact offline runtime validation |
| `POST` | `/local-ai/update` | Stage and validate the available locked revision before activation |
| `POST` | `/local-ai/rollback` | Activate the previously validated exact manifest |
| `GET` | `/local-ai/operations/{operation_id}` | Read bounded, document-free progress |
| `POST` | `/local-ai/operations/{operation_id}/resume` | Resume a paused operation from a clean stage |
| `POST` | `/local-ai/operations/{operation_id}/retry` | Retry a retryable failed operation from a clean stage |
| `DELETE` | `/local-ai/models/{role}` | Remove one role artifact |
| `DELETE` | `/local-ai` | Remove all model artifacts and activation state |

Install, verify, update, and rollback return `202`:

```json
{
  "operation_id": "uuid",
  "state": "queued"
}
```

Operation status contains only action, state, current role, byte counts, a
bounded server-owned message, and retryability. It never contains filenames,
document text, prompts, evidence, patient identifiers, or model output.

Only one lifecycle operation may be non-terminal. Pack mutation and
strict-local job admission share a database advisory lock. Mutations return
`409` while a clinical local-AI job is queued or processing, but only after
operator authorization succeeds.

Browser authorization does not restrict an operating-system owner using the
documented local CLI maintenance commands.

### Local processing jobs

These routes remain owner-scoped. They do not require machine-operator
authority:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/local-ai/jobs` | List owner-scoped, content-free job status |
| `GET` | `/local-ai/jobs/{job_id}` | Read one owner-scoped job |
| `POST` | `/local-ai/jobs/{job_id}/retry` | Requeue one failed, retryable ingestion or summary job |
| `POST` | `/local-ai/jobs/{job_id}/cancel` | Persist cancellation and terminate the active worker |

`GET /local-ai/jobs` accepts `kind`, `active_only`, and
`include_retryable_failed`. The retryable-failure flag is additive only when
`active_only=true`; it has no effect when `active_only=false`. When enabled,
the query adds exact owner-scoped failures whose stored retryability field is
the JSON boolean `true`. Manual ZIP children, non-retryable failures,
completed or cancelled jobs, and jobs owned by another account are excluded
before the 50-row limit.

Jobs report target UUIDs, processing mode, kind, status, stage, cancellation
state, timestamps, and bounded progress/failure taxonomy. Progress exposes only
stable model-role and numeric counters; failure exposes only stage, code,
model-role, retryability, checkpoint, and fallback flags. Stored failure
messages, clinical payloads, prompts, excerpts, evidence, manifests, and model
output are excluded. Retry is owner-scoped and accepts only failed, retryable
ingestion or summary jobs. Ingestion retry resets the paired upload and job
together while keeping strict-local checkpoints. Summary retry preserves its
stored prompt scope and immutable model snapshot. All other job states return
`409`.

The schema-v2 runtime identity is checked before strict-local admission and
again immediately before each worker spawn. Admission-time drift is rejected
before provider construction, release-evidence loading, or creation of a new
job or upload snapshot. The public failure boundary is content-free and does
not expose local paths, source filenames, manifest bytes, or digests.

Queued or processing schema-v1 ingestion snapshots can make only the legacy
terminal transition to `failed` with code `runtime_identity_required`; the
paired upload is failed in the same transaction. Schema-v1 summary jobs use
the same code but remain job-only. These failures are non-retryable and leave
the stored manifest unchanged. After repairing and revalidating the worker,
the client must submit new schema-v2 work rather than retry or rewrite the
legacy snapshot. Deployment must follow the no-active-job preflight documented
in [Strict-local AI operations](operations-strict-local-ai.md).

### GET `/records/{record_id}/evidence`

This literal route is declared before `/records/{record_id}`. It returns
bounded evidence only when the authenticated user owns the record, upload, and
strict-local provenance. Cross-user and missing lookups both return `404`.

The response includes the captured processing mode, extraction schema version,
OCR/extraction model identities, evidence spans, and unresolved or rejected
field names. A survivor can list more than one identity per role when it
inherits evidence from archived strict-local uploads processed by an older
model pack. The list is bounded by the evidence-source limit. Summary-model
provenance is not mixed into ingestion evidence.

See [Strict-local AI operations](operations-strict-local-ai.md) for the
download/processing network boundary and the v2 release-gate status.

---

## AI Summary

Summary execution is explicit:

- `validated_strict_local` uses Qwen3.5-9B after ingestion. It receives only
  validated facts and evidence IDs and never receives the uploaded document.
- `custom_local` uses an enabled loopback Ollama or LM Studio route and is
  labelled unverified.
- `cloud_assisted` uses the configured external provider after de-identification.
- `prompt_only` calls `build-prompt` and does not call a model.

### POST `/summary/build-prompt`

Build a de-identified reference-selection prompt from health records. The
external model can select only server-issued fact, field, evidence, and
uncertainty references. It cannot supply clinical prose. This endpoint does not
call an AI service.

**Request:**
```json
{
  "patient_id": "uuid",
  "summary_type": "full",
  "category": null,
  "date_from": null,
  "date_to": null,
  "output_format": "natural_language",
  "record_ids": null,
  "record_types": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | UUID | Required. Patient to summarize. |
| `summary_type` | string | `full`, `category`, `date_range`, or `single_record` |
| `category` | string? | Filter by record category (e.g., `medication`) |
| `date_from` | datetime? | Start of date range |
| `date_to` | datetime? | End of date range |
| `output_format` | string | `natural_language`, `json`, or `both`; defaults to `natural_language` |
| `record_ids` | UUID[]? | Specific records to include |
| `record_types` | string[]? | Filter by record types (e.g., `["medication", "condition"]`) |

**`summary_type` values:** `full`, `category`, `date_range`, `single_record`

A grounded prompt accepts only one of `category`, `record_ids`, or
`record_types`. A date range needs both bounds and cannot be combined with one
of those selectors.

**Response (200):**
```json
{
  "id": "uuid",
  "summary_type": "full",
  "system_prompt": "Return exactly one JSON object with keys sections and uncertainties...",
  "user_prompt": "Select references from the de-identified registry... INPUT_JSON={...}",
  "target_model": "gemini-3.5-flash",
  "suggested_config": {
    "temperature": 0,
    "max_output_tokens": 4096,
    "thinking_level": "low",
    "response_format": "json"
  },
  "record_count": 47,
  "de_identification_report": {
    "names_scrubbed": 12,
    "dates_generalized": 8,
    "mrns_removed": 3,
    "addresses_removed": 2
  },
  "copyable_payload": "...single string ready to paste into Google AI Studio...",
  "processing_mode": "prompt_only",
  "model_provenance": null,
  "generated_at": "2024-02-01T10:30:00Z"
}
```

**De-identification behavior (best effort):**
- Before downstream cloud-assisted text calls, including summary generation,
  the app applies structured regex filters, known-patient identifier
  substitutions, and optional spaCy PERSON NER. These layers reduce PII
  exposure but do not guarantee removal of every HIPAA identifier category or
  constitute certified Safe Harbor de-identification.
- Cloud-assisted PDF and TIFF OCR is the exception: the selected vision
  provider receives the original document or pages before text exists to
  scrub. Returned OCR text is scrubbed before downstream external extraction.
- Regex filters replace recognized forms of SSNs, phone and fax numbers,
  emails, IP addresses, URLs, ZIP codes, street addresses, VINs, and labeled
  account, accession, license, device, biometric, and health-plan identifiers.
  Known patient names, MRNs, dates of birth, and addresses are substituted when
  those demographics are populated.
- Recognized full dates in supported month-name, ISO, and slash formats are
  reduced to the four-digit year. Other date formats may remain unchanged.
- Generic city and location names may remain. Location NER is intentionally
  disabled because the general model can misclassify clinical terms as places.
- Free-text name redaction depends on the configured spaCy PERSON model. The
  pass is skipped when disabled. If the model is missing or fails to load, the
  pass fails open for the current call; the regex and known-patient filters
  still run.
- `de_identification_report` records replacement types and counts, never the
  matched values.
- `copyable_payload` is the combined system and user prompt.

**Prompt constraints (MUST be embedded in `system_prompt`):**
- Return only the exact reference JSON schema.
- Do not emit clinical free text, headings, or uncertainty text.
- Select only linked facts, fields, evidence, and server-owned uncertainties.
- Do not provide diagnoses, treatment recommendations, medical advice, or clinical decision support.

The saved prompt includes an owner-scoped digest of the exact grounding
registry and the exact selected record IDs. Those internal snapshot fields are
not returned. If the records or evidence change before a response is pasted,
the paste fails closed and the user must build a new prompt.

### GET `/summary/prompts`

List previously built prompts.

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "summary_type": "full",
      "system_prompt": "...",
      "user_prompt": "...",
      "target_model": "mlx-community/Qwen3.5-9B-MLX-4bit@938d8919941c6e7efd3c7150eff7fe9d12afa631",
      "suggested_config": {},
      "record_count": 47,
      "de_identification_report": null,
      "copyable_payload": "...",
      "processing_mode": "validated_strict_local",
      "model_provenance": {
        "processing_mode": "validated_strict_local",
        "manifest_sha256": "64-character SHA-256",
        "pack_revision": "apple-m4-16gb-v2",
        "model": {
          "role": "summary",
          "repository": "mlx-community/Qwen3.5-9B-MLX-4bit",
          "revision": "938d8919941c6e7efd3c7150eff7fe9d12afa631",
          "quantization": "4bit",
          "runtime": {
            "name": "mlx-vlm",
            "version": "0.5.0"
          }
        }
      },
      "generated_at": "2024-02-01T10:30:00Z"
    }
  ]
}
```

`processing_mode` and `model_provenance` are nullable for older rows. The API
revalidates stored provenance before returning it and omits malformed,
mismatched, PHI-shaped, or secret-shaped model identity. Clients must not infer
identity from `target_model`.

### GET `/summary/prompts/:id`

Get a single prompt detail, including any stored response.

**Response (200):**
```json
{
  "id": "uuid",
  "summary_type": "full",
  "system_prompt": "...",
  "user_prompt": "...",
  "target_model": "mlx-community/Qwen3.5-9B-MLX-4bit@938d8919941c6e7efd3c7150eff7fe9d12afa631",
  "suggested_config": {},
  "record_count": 47,
  "de_identification_report": null,
  "copyable_payload": "...",
  "response_text": "Previously pasted or generated response, if any",
  "response_format": "natural_language",
  "typed_response": {
    "sections": [],
    "uncertainties": []
  },
  "processing_mode": "validated_strict_local",
  "model_provenance": {
    "processing_mode": "validated_strict_local",
    "manifest_sha256": "64-character SHA-256",
    "pack_revision": "apple-m4-16gb-v2",
    "model": {
      "role": "summary",
      "repository": "mlx-community/Qwen3.5-9B-MLX-4bit",
      "revision": "938d8919941c6e7efd3c7150eff7fe9d12afa631",
      "quantization": "4bit",
      "runtime": {
        "name": "mlx-vlm",
        "version": "0.5.0"
      }
    }
  },
  "generated_at": "2024-02-01T10:30:00Z"
}
```

**Errors:** `404` (prompt not found or belongs to different user)

### POST `/summary/paste-response`

User pastes back the reference-only JSON returned by the external model. The
server rebuilds the original owner-scoped grounding registry, verifies its
snapshot digest, validates every reference and evidence link, and renders the
clinical text itself. Free-form text is rejected and is never stored.

**Request:**
```json
{
  "prompt_id": "uuid",
  "response_text": "{\"sections\":[{\"heading\":\"Overview\",\"claims\":[{\"fact_id\":\"fact1_...\",\"field_paths\":[\"/name\"],\"evidence_ids\":[\"evidence1_...\"]}]}],\"uncertainties\":[]}"
}
```

**Response (200):**
```json
{
  "id": "uuid",
  "prompt_id": "uuid",
  "response_pasted_at": "2024-02-01T11:00:00Z",
  "typed_response": {
    "sections": [],
    "uncertainties": []
  },
  "natural_language": "Server-rendered markdown, or null for JSON-only output",
  "json_data": null
}
```

`natural_language` and `json_data` follow the `output_format` saved when the
prompt was built. Errors: `400` for malformed, free-text, unknown, or unlinked
references; `404` for a missing or other-user prompt; `409` for an already
completed prompt, a legacy prompt without a grounding snapshot, or a prompt
whose records or evidence changed.

### POST `/summary/generate`

Generate an AI summary in one of the three live execution modes. The request
must identify `processing_mode`; the default remains `cloud_assisted` for API
backward compatibility. The frontend does not silently choose that default
while settings are unresolved.

**Request:**
```json
{
  "patient_id": "uuid",
  "summary_type": "full",
  "category": null,
  "date_from": null,
  "date_to": null,
  "output_format": "natural_language",
  "custom_system_prompt": null,
  "custom_user_prompt": null,
  "processing_mode": "validated_strict_local"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | UUID | Required. Patient to summarize. |
| `summary_type` | string | `full`, `category`, `date_range`, or `single_record` |
| `output_format` | string | `natural_language`, `json`, or `both` |
| `custom_system_prompt` | string? | Additional instructions, at most 4,096 characters; server safety rules remain in force |
| `custom_user_prompt` | string? | Additional user preferences, at most 4,096 characters; server safety rules remain in force |
| `processing_mode` | string | `validated_strict_local`, `custom_local`, or `cloud_assisted` |
| `provider` | string? | Optional only for cloud-assisted explicit routing; forbidden for strict and custom local |
| `model` | string? | Optional only for cloud-assisted explicit routing; forbidden for strict and custom local |

Validated strict-local summaries are accepted after the prompt and job commit,
then run in the server-owned background runner. They return `202` without
waiting for model inference:

**Response (202, validated strict-local):**
```json
{
  "id": "summary-prompt-uuid",
  "job_id": "local-job-uuid",
  "processing_mode": "validated_strict_local",
  "kind": "summary",
  "status": "queued",
  "stage": "queued",
  "created_at": "2024-02-01T10:30:00Z"
}
```

Custom-local and cloud-assisted summaries remain synchronous and return `200`:

**Response (200, custom-local or cloud-assisted):**
```json
{
  "id": "uuid",
  "processing_mode": "cloud_assisted",
  "model_provenance": {
    "processing_mode": "cloud_assisted",
    "provider": "gemini",
    "model": "gemini-3.5-flash"
  },
  "typed_response": {
    "sections": [],
    "uncertainties": []
  },
  "natural_language": "The patient's records show...",
  "json_data": null,
  "record_count": 47,
  "duplicate_warning": null,
  "de_identification_report": null,
  "model_used": "gemini-3.5-flash",
  "generated_at": "2024-02-01T10:30:00Z"
}
```

**Notes:**
- `typed_response` is present for completed custom-local and cloud-assisted
  summaries. It has exactly `sections` and `uncertainties`. It appears on the
  strict-local prompt after its background job completes. Legacy and prompt-only
  history detail may return it as null.
- `natural_language` contains the text summary (null when `output_format` is `json`)
- `json_data` contains structured output (null when `output_format` is `natural_language`)
- `duplicate_warning` is included only when a synchronous response excluded
  duplicates. Strict-local job status remains content-free while it runs.
- Strict-local summaries accept only validated fact and evidence IDs, reject
  unsupported claims, and append the server-owned no-medical-advice disclaimer.
- Custom-local and cloud-assisted providers receive a de-identified reference
  registry and can return only linked references. The server renders the text.
- Prompt-only paste uses the same reference validation and server renderer.
  Free text is not persisted.

**Errors:** `400` (no records found, invalid request), `404` (patient not found)

### GET `/summary/responses`

List stored AI responses (both pasted and API-generated).

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "summary_type": "full",
      "record_count": 47,
      "response_text": "First 200 characters of the response...",
      "response_pasted_at": "2024-02-01T11:00:00Z"
    }
  ],
  "total": 5
}
```

**Notes:**
- `response_text` is truncated to 200 characters in the list view
- Use `GET /summary/prompts/:id` to retrieve the full response

---

## Upload Review (Dedup)

Per-upload dedup review. These endpoints let the user review auto-merged and pending dedup candidates for a specific upload, resolve them in bulk, or undo auto-merges.

### GET `/upload/:id/review`

Get dedup review data for an upload.

**Response (200):**
```json
{
  "upload": {
    "id": "uuid",
    "filename": "export.json",
    "uploaded_at": "2026-04-05T10:00:00Z",
    "record_count": 200,
    "status": "awaiting_review",
    "dedup_summary": {
      "total_candidates": 15,
      "auto_merged": 12,
      "needs_review": 3,
      "dismissed": 0,
      "by_type": {
        "medication": 5,
        "condition": 7,
        "observation": 3
      }
    }
  },
  "auto_merged": [
    {
      "candidate_id": "uuid",
      "primary": {
        "id": "uuid",
        "display_text": "Metformin 500mg",
        "record_type": "medication",
        "fhir_resource": { "resourceType": "MedicationRequest" }
      },
      "secondary": {
        "id": "uuid",
        "display_text": "Metformin 500mg",
        "record_type": "medication",
        "fhir_resource": { "resourceType": "MedicationRequest" }
      },
      "similarity_score": 0.98,
      "llm_classification": "duplicate",
      "llm_confidence": 0.95,
      "llm_explanation": "Same medication, same dose",
      "merged_at": "2026-04-05T10:01:00Z"
    }
  ],
  "needs_review": {
    "medication": [
      {
        "candidate_id": "uuid",
        "primary": {
          "id": "uuid",
          "display_text": "Metformin 500mg",
          "record_type": "medication",
          "fhir_resource": { "resourceType": "MedicationRequest" }
        },
        "secondary": {
          "id": "uuid",
          "display_text": "Metformin 1000mg",
          "record_type": "medication",
          "fhir_resource": { "resourceType": "MedicationRequest" }
        },
        "similarity_score": 0.72,
        "llm_classification": "update",
        "llm_confidence": 0.85,
        "llm_explanation": "Same medication with dose increase from 500mg to 1000mg",
        "field_diff": {
          "dosageInstruction": { "old": "500mg daily", "new": "1000mg daily" }
        }
      }
    ],
    "condition": []
  }
}
```

**Notes:**
- `auto_merged` contains candidates that were automatically resolved (heuristic score >= 0.95 or LLM duplicate with high confidence)
- `needs_review` is keyed by `record_type` for category-grouped display
- `field_diff` is present for `update` classifications — shows which FHIR fields changed
- Only cloud-assisted dedup may use the LLM judge. In every other processing
  mode, scores at or above 0.95 may auto-merge, while fuzzy matches stay pending
  for manual review without loading provider configuration or calling a model.
- User-scoped: only returns data for uploads owned by the authenticated user

### POST `/upload/:id/review/resolve`

Bulk resolve dedup candidates.

**Request:**
```json
{
  "resolutions": [
    { "candidate_id": "uuid", "action": "merge" },
    { "candidate_id": "uuid", "action": "update", "field_overrides": ["clinicalStatus", "dosageInstruction"] },
    { "candidate_id": "uuid", "action": "dismiss" },
    { "candidate_id": "uuid", "action": "keep_both" }
  ]
}
```

**Actions:**
| Action | Effect |
|--------|--------|
| `merge` | Keep primary, mark secondary as `is_duplicate=true` |
| `update` | Apply selected fields from secondary to primary (all changed fields if `field_overrides` omitted), mark secondary as duplicate |
| `dismiss` | Not a duplicate — set candidate status to `dismissed` |
| `keep_both` | Related but distinct — set candidate status to `dismissed`, no record changes |

**Response (200):**
```json
{
  "resolved": 4,
  "remaining": 0
}
```

**Notes:**
- When `remaining` reaches 0, upload status transitions to `completed`
- All merge/update actions create provenance records
- `update` with `field_overrides` enables cherry-picking specific FHIR fields to accept
- Protected fields (`resourceType`, `_extraction_metadata`, `id`, `meta`) are never overwritten

### POST `/upload/:id/review/undo-merge`

Undo an auto-merged or manually merged candidate.

**Request:**
```json
{
  "candidate_id": "uuid"
}
```

**Response (200):**
```json
{
  "status": "undone",
  "candidate_id": "uuid"
}
```

**Errors:** `400` (candidate is not merged), `404` (candidate or upload not found)

**Notes:**
- Restores the secondary record (clears `is_duplicate`)
- Reverts field changes on primary using stored `previous_values` from `merge_metadata`
- Resets candidate status to `pending`
- If upload was `completed`, transitions back to `awaiting_review`

---

## Deduplication

> **Note:** These are the legacy global dedup endpoints (Admin Console > Dedup tab). For per-upload dedup review, see [Upload Review](#upload-review-dedup) above.

### GET `/dedup/candidates`

List deduplication candidates (paginated).

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | Page number (1-indexed) |
| `limit` | int | 20 | Items per page |

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "similarity_score": 0.92,
      "match_reasons": {
        "same_code": true,
        "same_date": true,
        "similar_text": false
      },
      "status": "pending",
      "record_a": {
        "id": "uuid",
        "display_text": "Lisinopril 10 MG Oral Tablet",
        "record_type": "medication",
        "source_format": "fhir_r4",
        "effective_date": "2024-01-15T00:00:00Z"
      },
      "record_b": {
        "id": "uuid",
        "display_text": "LISINOPRIL 10MG TAB",
        "record_type": "medication",
        "source_format": "epic_ehi",
        "effective_date": "2024-01-15T00:00:00Z"
      }
    }
  ],
  "total": 5
}
```

**Notes:**
- Only `pending` candidates are returned (server-side filter)
- `record_a` and `record_b` can be `null` if a record was deleted

### POST `/dedup/scan`

Trigger a deduplication scan across all records.

**Response (200):**
```json
{
  "candidates_found": 5
}
```

### POST `/dedup/merge`

Merge two duplicate records (keep primary, archive secondary).

**Request:**
```json
{
  "candidate_id": "uuid"
}
```

**Response (200):**
```json
{
  "status": "merged",
  "primary_record_id": "uuid",
  "archived_record_id": "uuid"
}
```

### POST `/dedup/dismiss`

Dismiss a candidate pair as not duplicates.

**Request:**
```json
{
  "candidate_id": "uuid"
}
```

**Response (200):**
```json
{
  "status": "dismissed"
}
```

---

## Record Types

The frontend recognizes these `record_type` values and assigns distinct visual styling to each:

| `record_type` | Label | Short Code | FHIR Resource |
|---------------|-------|------------|---------------|
| `condition` | Conditions | COND | Condition |
| `observation` | Labs & Vitals | OBS | Observation |
| `medication` | Medications | MED | MedicationRequest / MedicationStatement |
| `encounter` | Encounters | ENC | Encounter |
| `immunization` | Immunizations | IMMUN | Immunization |
| `procedure` | Procedures | PROC | Procedure |
| `document` | Documents | DOC | DocumentReference |
| `allergy` | Allergies | ALRG | AllergyIntolerance |
| `imaging` | Imaging | IMG | ImagingStudy |
| `diagnostic_report` | Diagnostic Reports | DIAG | DiagnosticReport |

Additional types the frontend handles gracefully (with default styling):
- `service_request`, `communication`, `appointment`, `care_plan`, `care_team`, `immunization_recommendation`, `questionnaire_response`, `family_member_history`

Any unknown `record_type` values render with a neutral gray badge.

---

## Error Response Format

All error responses MUST follow this format:

```json
{
  "detail": "Human-readable error message"
}
```

The frontend reads `response.json().detail` for error display. **Never expose stack traces, internal errors, or PII in error responses.**

**Standard HTTP Status Codes:**
| Code | Usage |
|------|-------|
| 200 | Success |
| 201 | Created (registration) |
| 202 | Accepted (async processing started) |
| 204 | No content (logout, delete) |
| 400 | Validation error |
| 401 | Authentication required or invalid token |
| 403 | Forbidden (accessing another user's data) |
| 404 | Resource not found |
| 409 | Conflict (duplicate account login identifier) |
| 413 | File too large |
| 422 | Unprocessable entity; auth payload errors use the exact generic body above |
| 500 | Internal server error |

---

## CORS Requirements

The backend MUST allow CORS from the frontend origin:

```
Access-Control-Allow-Origin: http://localhost:3000
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type, Accept
Access-Control-Allow-Credentials: true
```

---

## Security Requirements

1. **User-scoped data:** Every database query MUST filter by `user_id`. A user must never see another user's records.
2. **JWT validation:** Verify token signature, expiration, and issuer on every authenticated request.
3. **Audit logging:** Log every data access and mutation to the `audit_log` table.
4. **Soft deletes only:** Never hard-delete health records. Use `deleted_at` timestamps.
5. **Explicit AI modes:** Validated strict local branches before provider
   construction and fails closed. Custom-local summaries and downstream
   cloud-assisted text calls use the de-identification path. Cloud-assisted
   PDF and TIFF OCR sends the original document to the selected vision
   provider first. Prompt-only makes no model call.
6. **API key management:** Provider keys are encrypted at rest or supplied
   through the environment. Never hardcode, log, or return them.
7. **Input validation:** Strict Pydantic validation on all inputs. Sanitize file uploads.
8. **Password hashing:** bcrypt with cost >= 12.

---

## Testing Requirements

For each endpoint, ensure:

1. **Auth guard:** Unauthenticated requests return 401
2. **User isolation:** User A cannot access User B's records
3. **Pagination:** Page boundaries work correctly, total counts are accurate
4. **Filtering:** Record type and search filters return correct subsets
5. **Soft delete:** Deleted records don't appear in list/timeline queries
6. **De-identification:** Test each documented regex and known-patient
   substitution, supported date formats reducing to year only, and the
   documented fail-open PERSON-NER and city/location limitations. These tests
   verify implemented behavior; they are not proof that a payload contains no
   PII.
7. **File upload:** Reject oversized files, invalid MIME types, and malformed content
8. **Error format:** All error responses use `{"detail": "..."}` format
