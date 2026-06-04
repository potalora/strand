# Snapshot Overview — Design Spec

_Date: 2026-06-03 · Source: Claude Design handoff bundle (`MedTimeline Overview Options.html`, committed "Snapshot" direction) + `BACKEND-TODOS.md`._

## Goal

Replace the home/Overview page with the committed **Snapshot** direction and build the
backend endpoints it needs to run on real data instead of the prototype's in-memory array.

**Guiding principle (must hold everywhere):** MedTimeline is **descriptive, not normative.**
It surfaces values, ranges, and flags *exactly as the source reported them* and never derives
targets, judgments, or recommendations. This matches CLAUDE.md Absolute Rule #1. No "at goal /
above target" chips, no invented reference ranges, no recommended follow-ups.

## Frontend — Full Snapshot replace (`app/(dashboard)/page.tsx`)

Layout, top to bottom:
1. **Masthead** — patient name (serif display), `{sex} · born {year} · {N} records · {span}`, secure chip.
2. **"Your most recent results"** — section head with the non-interpretation stance line +
   "All labs & vitals" link → markers grid. One `MarkerCard` per distinct observation code,
   sorted by recency of the latest reading. Each card: name, "as of {date}", big value+unit,
   sparkline (≥2 numeric readings) or gauge (single reading + source ref range), and a footer
   ("Previously {v} · {date}" or "{n} reading on file"; ref range "per source"; or "Per {source}").
3. **Bento** — "Conditions & medications on file" (two status-active columns) + "Recently added"
   (ingestion-ordered feed, fixed-width type badge column so titles align).
4. **DetailSheet** — reuse existing `RecordDetailSheet`; wire "Add to summary" + "Export FHIR".

**Reuse:** `Gauge`/`Sparkline` (`DataViz.tsx`), `RecordDetailSheet`, `RetroNav`, `FloatingDock`,
token CSS in `globals.css`, `record-icons.ts`, `constants.ts`, `api` client.
**Drop (prototype scaffolding only):** concept switcher, Tweaks panel, palette/type toggles.
**Add CSS** (from `overview-extra.css`, not yet in production): `.markers`, `.marker*`,
`.ov-sec-head`, `.ov-sec-note`, `.ov-link`, `.marker-asof`, feed-row badge column rule.

## Backend — new endpoints (TDD, user-scoped, audited)

All scope by `user_id`, exclude `deleted_at`/`is_duplicate`, log via `log_audit_event`. New literal
`/records/...` subpaths declared **before** `/{record_id}`.

### 1. `GET /api/v1/observations/by-code`  (new router `api/observations.py`)
One entry per distinct observation code, recency-sorted (latest.date desc, tie → higher count).
```
{ "items": [ {
    "code": "4548-4", "display": "Hemoglobin A1c", "category": "Diabetes"|null, "count": 4,
    "latest": { "id","value","unit","date","source","ref_low","ref_high","interpretation" },
    "prior":  { ...same... } | null,
    "series": [ {"date","value"} ... ]   // numeric only, ascending, unit-normalized
} ], "total": N }
```
ref range / interpretation / unit / value pulled from `fhir_resource` JSONB (mirror `/dashboard/labs`).
Non-numeric values (e.g. BP "128/78") kept verbatim in latest/prior; excluded from series.
Covers TODOs #1, #2, #6.

### 2. `GET /api/v1/records/{record_id}/fhir`
Returns the record's raw FHIR resource (`fhir_resource`) as JSON for single-record export. 404 if
not owned. Covers TODO #7.

### 3. `GET /api/v1/records/recent?limit=5`
Ordered by `created_at desc` (ingestion time, distinct from clinical date). Lightweight shape:
`{id, record_type, display_text, effective_date, created_at, source, value, unit}`. Covers TODO #3.

### 4. `GET /api/v1/records/stats`
`{ total, first_date, last_date, source_count }` for the masthead. Covers TODO #8.

### 5. `POST /api/v1/summary/items` (+ `GET`, + `DELETE /{item_id}`)
New table `summary_items(id, user_id, record_id, created_at)`, unique `(user_id, record_id)`.
Backs the detail sheet "Add to summary". Validates record ownership. Alembic migration + model.
Covers TODO #7.

### 6. Unit normalization — `services/utils/unit_normalization.py`  (TODO #5)
Pure, exhaustively-tested `normalize(code, value, unit) -> (value, canonical_unit)` with a curated
per-LOINC table + conversions (A1c %↔mmol/mol, cholesterol/trig/glucose mg/dL↔mmol/L,
creatinine mg/dL↔µmol/L, …). No-op when unit already canonical or unknown. Applied at read time in
the by-code endpoint so cross-source series are consistent. Original units preserved in fhir_resource.

## Deferred / flagged
- Ingest-time persistence of normalized units (read-time normalization used instead).
- Notifications / activity feed (TODO #9) — search already exists (`/records/search`).

## Verification (per CLAUDE.md)
1. API-level: pytest (TDD) for every new endpoint + normalizer; httpx/curl smoke against running server.
2. Frontend: user verifies the Overview through the UI.
