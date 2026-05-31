# Unstructured Idempotency + Dedup Integration — Design (Phase 2a)

**Date:** 2026-05-29
**Status:** Approved (design); pending implementation plan
**Branch:** `feat/unstructured-idempotency` (stacked on `feat/idempotent-incremental-ingestion` — depends on Phase 1's `content_hash`)
**Predecessor:** `docs/superpowers/specs/2026-05-29-idempotent-incremental-ingestion-design.md` (Phase 1, structured)

## Problem

Phase 1 made *structured* ingestion (FHIR/CDA/Epic) idempotent via a stable-id gate. The
*unstructured* / AI-extracted path was not covered:

- Extracted records insert via direct `db.add()` (`api/upload.py:733`, `:1045`), with
  `source_format="ai_extracted"`, `ai_extracted=true`, and **no** `external_id` /
  `source_system` / `content_hash`.
- `uploaded_files.file_hash` is computed (`api/upload.py:842`) but **never checked** — so
  re-uploading the same PDF re-extracts and duplicates records.
- Entity extraction is **non-deterministic** (Gemini / LangExtract, no `temperature=0`), so
  re-extracting the same document yields different entity boundaries/text — hashing the full
  extracted resource is not a reliable cross-run identity key.
- The existing content/fuzzy/LLM dedup runs post-confirm, but a re-extraction lands in a new
  `source_file_id` partition, so it does not reliably collapse against the prior set.

This is **Phase 2a**. Extraction *quality* improvements for non-standard documents
(transcripts, phone notes) are **Phase 2b** (separate spec).

## Goal & Behavior Contract

| Scenario | Behavior |
|----------|----------|
| Same file (identical `file_hash`) re-uploaded by a user who already extracted it | **Skip** re-extraction; record a `duplicate_file` upload pointing at the original; surface existing records. No Gemini cost. |
| Prior upload of that hash **failed** (produced no records) | Not a duplicate — extract normally. |
| Extraction explicitly re-triggered on a file that already has records | **Replace**: soft-delete that file's prior extracted records, insert the fresh set, run dedup. |
| New, different document whose facts overlap existing records | Unchanged — the **existing fuzzy/LLM dedup** handles it post-confirm. |
| Any extracted record inserted | Carries a populated **`content_hash`** (uniform with structured records). |

**Design choice (robustness over non-determinism):** the two *file-level* guards (hash-skip,
replace-on-reextract) are deterministic and do the heavy lifting. AI-extracted records stay
**no-identity** (no synthetic `external_id`) — a content-signature derived from
non-deterministic extracted text would cause both missed matches and false collapses.
Cross-document fuzzy overlap remains the existing dedup pipeline's job.

## Architecture (Approach A)

Three small, isolated changes plus reuse of existing dedup:

1. **File-hash skip** at the unstructured upload endpoints.
2. **Replace-on-reextract** in the extraction insert path (soft-delete prior, insert fresh).
3. **`content_hash` population** in the entity→FHIR dict builder.

No rerouting of the extraction insert through Phase 1's `idempotent_insert_records`: that
path's `_build_row` does not carry `source_section`/encounter links and returns dicts (not
ORM objects), which would break the post-insert encounter-linking at `upload.py:742-773`.
A one-line `content_hash` addition achieves uniformity without that risk.

```
upload (PDF/RTF/TIFF) ──► compute file_hash
   │
   ├─ hash already produced records for this user? ──YES──► record duplicate_file, surface existing, STOP
   │
   └─ NO ─► extract ─► [replace: soft-delete file's prior extracted records]
                       ─► build dicts (now with content_hash) ─► db.add ─► encounter-link
                       ─► background dedup (unchanged)
```

## Components

### `api/upload.py` — file-hash skip
After `file_hash` is computed in `upload_unstructured` (and the batch endpoint), query for a
non-deleted `UploadedFile` for the **same user** with the same `file_hash` whose
`ingestion_status` is in the produced-records set:
`{"completed", "completed_with_merges", "awaiting_review", "awaiting_confirmation"}`.
If found:
- still create the new `UploadedFile` row (audit/history),
- set `ingestion_status = "duplicate_file"`,
- set `ingestion_progress = {"duplicate_of": <original_upload_id>, "record_count": <N from original>}`,
- do **not** enqueue extraction; return a response indicating the duplicate and the original id.
If not found (or the only prior is `failed`), proceed with normal extraction.

### `api/upload.py` — replace-on-reextract
Immediately before inserting freshly extracted records for a file (auto-confirm path
`upload.py:712`, and manual confirm endpoint `upload.py:1035`), soft-delete the file's prior
live extracted records:
```sql
UPDATE health_records SET deleted_at = now()
WHERE source_file_id = :upload_id AND ai_extracted = true AND deleted_at IS NULL
```
First extraction → no-op; re-extraction → clean replace. Write one `provenance` row with
`action = "reextraction_replace"` recording the count replaced. Never hard-delete (rule 6).

### `services/extraction/entity_to_fhir.py` — content_hash
In `entity_to_health_record_dict` (`entity_to_fhir.py:38`), after `fhir_resource` is built,
set `record_dict["content_hash"] = content_hash(fhir_resource)` (import from
`app.services.ingestion.content_hash`). The existing `db.add()` + encounter-linking flow is
otherwise unchanged. No `external_id`/`source_system`.

### Cross-document overlap — unchanged
`_run_dedup_background` / `run_upload_dedup` already run post-confirm for unstructured uploads
(`upload.py:782`, `:1055`). Left exactly as-is; it is the mechanism for fuzzy overlap between a
new document and existing records.

## Data Model

**No migration.** Reuses Phase 1's `content_hash` column. The `duplicate_file` linkage lives in
the existing `ingestion_progress` JSONB. `duplicate_file` is a new value of the free-text
`ingestion_status` column (no enum to change).

## Error Handling & Safety

- **Never hard-delete** (rule 6): replace = soft-delete via `deleted_at`.
- **User-scoped** (rule 11): the `file_hash` lookup filters by `user_id`.
- Skip only fires when a prior upload of that hash reached a records-producing status; failed
  priors re-extract.
- Replace + insert run in one transaction per file; a failure rolls back without leaving the
  file with zero records (the prior set's soft-delete rolls back too).
- The duplicate-file short-circuit must not leak PHI: the response carries only ids/counts.

## Test Strategy (expected-output-first)

Every unit gets its expected output defined and a failing test written before implementation.

- **Unit (no Gemini; patch extraction to return fixed entities, per the existing
  `tests/test_unstructured_upload.py` pattern that patches `_process_unstructured` /
  `start_extraction_worker`):**
  - file-hash skip: a `completed` prior upload with hash X → new upload of X yields
    `duplicate_file`, 0 new records, `ingestion_progress.duplicate_of` set; a `failed` prior
    upload of X → extracts normally.
  - replace-on-reextract: a file with 3 prior live extracted records, re-extraction yielding 2
    → prior 3 have `deleted_at` set, 2 live remain, exactly 1 `provenance` row with
    `action="reextraction_replace"`.
  - `content_hash`: every extracted record dict has `content_hash == content_hash(fhir_resource)`.
- **Integration (mocked extraction):** upload → records carry content_hash; re-upload identical
  bytes → `duplicate_file`, 0 new rows; re-extract → record count converges (no doubling).
- **Fidelity / slow (real `test_data/note_361370_*.pdf`, needs `GEMINI_API_KEY`,
  `@pytest.mark.slow`):** upload the real note PDF → N records each with content_hash; re-upload
  identical bytes → `duplicate_file`, 0 new records.

## Out of Scope (Phase 2a)

- **Phase 2b:** extraction quality on non-standard documents (transcripts, phone notes) —
  coverage, precision, hallucinated/negated-finding suppression.
- Synthetic per-record identity for AI-extracted records (rejected: fragile under
  non-deterministic extraction).
- Changes to the fuzzy/LLM dedup heuristics.

## Validation Plan (post-build, manual)

1. Upload `test_data/note_361370_*.pdf` → records created (with content_hash).
2. Re-upload the identical file → `duplicate_file`, no new records, UI points at the original.
3. Re-trigger extraction on the file → record count for that file stays stable (replace, not add).
4. Confirm a clinical fact in the note that also exists in the structured extract is flagged by
   the existing dedup pipeline (cross-source overlap).
