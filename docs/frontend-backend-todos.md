# Frontend → Backend Integration TODOs

Source of truth: the "MedTimeline Reimagined" design handoff (`fresh-api-map.js`).
These were the backend paths/features the new frontend needed but that did not exist.

**Status: 8 of 9 built (TDD-first). The 9th (`/notifications`) is intentionally not built.**
All paths are prefixed with `/api/v1` and require a Bearer JWT.

## Built

| # | Endpoint / change | Powers (UI) | Tests |
|---|---|---|---|
| 1 | `GET /dashboard/sources` → `{items:[{source,count}], total}` (distinct sources; excludes deleted/duplicates) | Overview "Where records come from" + Data-sources stat | `tests/test_todo_dashboard_sources.py` |
| 2 | `GET /records?status=<clinicalStatus>` | Overview active conditions / current meds | `tests/test_todo_record_query.py` |
| 3 | `GET /records/series?code_value=` → `{code_value, items:[{effective_date,value,unit}], total}` (numeric points, ascending) | Detail-sheet trend sparkline | `tests/test_todo_record_series.py` |
| 4 | `GET /records?sort=<date\|type\|display_text\|created>&order=<asc\|desc>` | Admin → Records column sort | `tests/test_todo_record_query.py` |
| 5 | `DELETE /upload/:id` — soft-deletes the upload **and cascade soft-deletes its health_records** (via `source_file_id`) + audit entry; user-scoped. Added `uploaded_files.deleted_at` (model + migration `c3d4e5f6a7b8`) and filtered it out of `/upload/history` | Admin → Uploads trash button | `tests/test_todo_upload_delete.py` |
| 6 | `GET /records/export?format=fhir-bundle` → a FHIR R4 collection `Bundle` of all non-deleted/non-duplicate records (downloads via Content-Disposition); 400 on unsupported format | Admin → System "Export all (FHIR)" | `tests/test_todo_record_export.py` |
| 7 | `GET /audit-log?page=&limit=` — paginated, user-scoped read of `audit_log` (the read itself is **not** audited, to stay deterministic) | Admin → System audit log table | `tests/test_todo_audit_log.py` |
| 8 | Patient identity: `GET /dashboard/patients` now returns decrypted `name` + `birth_date` (fail-open to `null`); `GET /auth/me` now returns `created_at` | Overview masthead + System account card ("Member since") | `tests/test_todo_patient_identity.py` |

## Not built (intentional)

- `GET /notifications` — the backend has no notifications concept. Recommendation stands:
  drop the affordance rather than fake it. (The top-bar bell currently links to Settings.)

## Frontend wiring — DONE

All UI surfaces are now wired to the live endpoints:

- Overview masthead → `/dashboard/patients` decrypted name/DOB; subline born-year.
- Overview "Where records come from" card + "Data sources" stat → `/dashboard/sources`.
- Overview active conditions / current meds → `/records?...&status=active`.
- Detail sheet → `/records/series?code_value=` neutral trend sparkline (observations).
- Admin → System audit table → `/audit-log`; "Export all (FHIR)" → `/records/export` (downloads a Bundle).
- Admin → Records + Records table → server-side `?sort=&order=` (sortable headers).
- Upload history → `DELETE /upload/:id` (real cascade delete + refetch, with confirm).

Verified: `tsc --noEmit` clean, `next build` green (19 routes).
