# OCR Provider Refusal/Fallback Notices — Design Spec

**Date:** 2026-06-21
**Status:** Approved (brainstorming complete)
**Builds on:** PR #45 (OCR cross-provider fallback). Makes the fallback/refusal — currently server-log-only — visible to the user.

## Goal

When OCR routes through a vision provider that refuses a document (e.g. Gemini `RECITATION`/`content_filter`) and we silently fall back to another provider, the user should see what happened: which provider read their document, and that a refusal occurred. When **no** configured provider can read a document (currently a silent empty OCR → 0 records), surface that clearly as a warning. Decision: notices are **durable, per-file**, shown in **upload history + Admin → Extractions + file detail** (no toast). Scope: OCR only (the only operation that falls back today).

## Data model

Add a `notices` JSONB column to `uploaded_files` (list, server_default `[]`; migration chains from head `f2a3b4c5d6e7`). Each notice:

```json
{"type": "ocr_fallback" | "ocr_unreadable",
 "level": "info" | "warning",
 "message": "Read by Anthropic — Gemini declined this document (content policy).",
 "detail": {"used": "anthropic", "refused": ["gemini"], "attempts": [
     {"provider": "gemini", "status": "refused"},
     {"provider": "anthropic", "status": "ok"}]}}
```

- `ocr_fallback` (info): a provider refused/failed but a later one produced text. `used` ≠ the first attempt.
- `ocr_unreadable` (warning): every candidate returned empty/blocked → no text extracted.

`status` per attempt: `ok` (non-empty text), `refused` (empty/`content_filter`), `error` (exception).

Test DB note: `conftest.py` `create_all` adds new columns to NEW tables but does NOT ALTER existing ones — so `uploaded_files.notices` must be added to `medtimeline_test` via the migration / `ALTER TABLE` as well as dev (existing gotcha).

## Capture & threading (minimal-churn: an opt-in out-param)

Avoid changing the `(text, file_type)` return shapes. Thread an optional mutable `trace` list down; OCR appends attempt records to it:

- `_ocr_via_provider(parts, config, api_key, instruction, *, trace: list | None = None)` — for each candidate, append `{"provider": name, "status": "ok"|"refused"|"error"}` to `trace`; `refused` when `.text` is empty/blank, `error` on exception, `ok` on non-empty (and return immediately).
- `extract_text_from_pdf` / `extract_text_from_tiff` / `extract_text` gain `*, trace: list | None = None`, passed straight through. RTF and text-layer PDFs never touch the vision path, so `trace` stays empty (no notice).
- `_process_unstructured` creates `ocr_trace: list = []`, passes it into `extract_text(...)`, then builds a notice from it:
  - any `refused`/`error` attempt **and** a final `ok` → one `ocr_fallback` info notice.
  - attempts exist but none `ok` → one `ocr_unreadable` warning notice.
  - no attempts (local text / RTF) → no notice.
  - Append to `upload.notices` (read-modify-write the JSONB list) and persist.

A small pure helper `build_ocr_notice(trace: list) -> dict | None` (in `text_extractor.py` or a tiny `ocr_notices.py`) does the trace→notice mapping so it's unit-testable without the DB.

## API

The upload status responses already surface `progress_stage`/`progress_detail`/`ingestion_errors` (`api/upload.py` ~470, ~704). Add `notices: list` to:
- the per-file status object the list/status endpoints return, and
- the `UploadStatus`/file schema (whatever the status route serializes).

No new endpoint — `notices` rides along with the existing file status the frontend already polls.

## Frontend

Render notices wherever a file's status shows (theme-native, restraint — invoke frontend-design before building):
- **Upload history** (`app/(dashboard)/upload/page.tsx`): a small line/badge under a completed file — info notices muted ("Read by Anthropic — Gemini declined"), `warning` notices in the existing warning/error style.
- **Admin → Extractions tab** (`app/(dashboard)/admin/page.tsx`): same per-file notice line.
- Use the existing notice/error rendering patterns if present; otherwise a compact `<p>`/badge matching the card style. A short icon + message; the `detail` is available but not shown by default.
- `lib/api.ts`: include `notices` in the file-status type.

## Error handling

- Building/storing a notice is best-effort: a failure to write notices must never fail the extraction (wrap in try/except, log). The OCR text itself is already returned regardless.
- `ocr_unreadable` does NOT mark the file `failed` (the pipeline ran; the document was just unreadable) — it stays `completed` with a `warning` notice, so the user sees "processed, but this document couldn't be read" rather than a hard failure.

## Testing

- `build_ocr_notice`: fallback trace → info notice with correct `used`/`refused`; all-refused trace → warning; empty trace → None; single-ok trace (no refusal) → None (don't notify when the first provider just worked).
- `_ocr_via_provider` populates `trace` correctly across the fallback path (mocked providers: first empty, second ok → attempts `[refused, ok]`).
- `_process_unstructured` (or an integration-ish test): a mocked OCR fallback results in a `notices` entry on the file row.
- API: the file-status response includes `notices`.
- E2E (mocked): upload history renders an info notice for a fallback and a warning for unreadable.

## Out of scope (YAGNI)

- Toast/push notifications (decided against).
- Surfacing fallbacks for non-OCR operations (none fall back today).
- Per-record provenance of the OCR provider (the notice is per-file; provenance stays as-is).
