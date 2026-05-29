# Idempotent Incremental Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make re-uploading a cumulative EHR extract converge instead of duplicate — recognize records already ingested by their stable source id, skip identical ones, update changed ones (keeping prior versions), and only run the existing content-dedup pipeline on genuinely new records.

**Architecture:** A pre-insert identity gate (`idempotent_insert_records`) replaces direct `bulk_insert_records` calls in the structured ingest paths. For each record it derives a stable `(source_system, external_id)` and a `content_hash`, looks up existing rows in one batched query, and partitions into insert / update+snapshot / skip. The existing dedup pipeline is untouched and runs afterward on inserted records only.

**Tech Stack:** Python 3.12, SQLAlchemy 2.x async, PostgreSQL 16 (JSONB, partial unique index), Alembic, pytest + pytest-asyncio. Design spec: `docs/superpowers/specs/2026-05-29-idempotent-incremental-ingestion-design.md`.

**Conventions:** `from __future__ import annotations` at top of every new module. Type hints on all functions. No `print()` — use `logging`. DB head revision is `f1a2b3c4d5e6`.

**Test runner (IMPORTANT):** run pytest with the project venv interpreter, not the global pyenv (which has a broken `langsmith` plugin that fails collection): `cd backend && .venv/bin/python -m pytest …`. All `Run:` commands below assume this.

**Marker registration:** `pyproject.toml` registers only `slow` and `integration`. Before adding `@pytest.mark.fidelity` tests (Tasks 1, 12), register it in `[tool.pytest.ini_options] markers` to avoid `PytestUnknownMarkWarning`: add `"fidelity: marks tests requiring real user-provided data"`.

**Task 1 findings (already verified against the real extract):** `CcdaRenderer` preserves the CDA act `<id>` in `resource.identifier` as `[{"system": "urn:oid:<root-OID>", "value": "<extension>"}]` — NOT in `resource.id` (which is a renderer-generated UUID; never match on it). Some resource types (AllergyIntolerance, Medication, Practitioner) have `identifier=None` and will fall through to content-dedup (acceptable for Phase 1). Some ids appear as `{"nullFlavor": "UNK"}` dicts — guard against non-string id values. The probe PASSED, so **Task 4 uses the `resource.identifier` path (no raw-XML fallback needed)**.

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `backend/app/services/ingestion/content_hash.py` | Canonical content hash of a FHIR resource | Create |
| `backend/app/services/ingestion/identity.py` | Derive `(source_system, external_id)` from a record dict | Create |
| `backend/app/services/ingestion/idempotent_inserter.py` | Identity gate: partition + insert/update/skip + version snapshot | Create |
| `backend/app/models/record_version.py` | `RecordVersion` ORM model | Create |
| `backend/app/models/record.py` | Add `external_id`, `source_system`, `content_hash`, `version` columns | Modify |
| `backend/alembic/versions/b2c3d4e5f6a7_idempotency_fields.py` | Migration: columns + partial unique index + `record_versions` table | Create |
| `backend/app/services/ingestion/epic_mappers/base.py` | Add `source_table` + `primary_key_columns` class attrs | Modify |
| `backend/app/services/ingestion/epic_mappers/*.py` | Set `source_table`/`primary_key_columns` per mapper | Modify (14) |
| `backend/app/services/ingestion/epic_parser.py` | Populate `external_id`/`source_system` on record dict; swap insert call | Modify |
| `backend/app/services/ingestion/coordinator.py` | Swap 2 insert calls; thread `inserted` records to dedup | Modify |
| `backend/app/services/ingestion/fhir_parser.py` | Swap 4 insert calls | Modify |
| `backend/tests/test_content_hash.py` | Unit tests for hashing | Create |
| `backend/tests/test_identity.py` | Unit tests for identity extraction (all formats + negatives) | Create |
| `backend/tests/test_idempotent_inserter.py` | Unit + DB tests for the gate | Create |
| `backend/tests/test_incremental_ingestion.py` | Integration: double-ingest convergence | Create |
| `backend/tests/test_bulk_inserter.py` | Direct unit tests for `bulk_insert_records` (TDD-gap fix) | Create |
| `backend/tests/fidelity/test_incremental_fidelity.py` | Real-extract re-ingest delta = 0 | Create |
| `backend/scripts/backfill_idempotency_fields.py` | Backfill existing rows | Create |

---

## Task 1: CDA `<id>` verification spike (gating unknown)

Determine whether `CcdaRenderer` preserves the CDA act `<id>` into FHIR `resource.id`/`resource.identifier`. The CDA branch of `identity.py` (Task 4) depends on the answer. We prove it against the real `DOC0001.XML` before building.

**Files:**
- Test: `backend/tests/test_identity.py` (create with this one test first)

- [ ] **Step 1: Write the probe test**

```python
# backend/tests/test_identity.py
from __future__ import annotations

from pathlib import Path

import pytest

# Real extract location (gitignored). Skip when absent.
_XDM_DOC = Path(__file__).resolve().parents[2] / (
    "HealthSummary_May_29_2026/IHE_XDM/Pedro1/DOC0001.XML"
)


@pytest.mark.fidelity
@pytest.mark.skipif(not _XDM_DOC.exists(), reason="real XDM extract not present")
def test_cda_renderer_preserves_source_id():
    """Probe: does CcdaRenderer carry the CDA <id> into resource.id/identifier?

    This is a discovery test. We assert that AT LEAST ONE clinical resource
    produced from the real CDA carries either a non-UUID `id` or a populated
    `identifier`. If this fails, identity.py CDA branch must parse <id> directly.
    """
    from fhir_converter.renderers import CcdaRenderer

    renderer = CcdaRenderer()
    bundle = renderer.render_to_fhir("ccda", _XDM_DOC.read_text(encoding="utf-8"))

    has_identifier = False
    has_meaningful_id = False
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") in {"Bundle", "Composition", "Patient"}:
            continue
        if res.get("identifier"):
            has_identifier = True
        rid = res.get("id", "")
        # A bare UUID id is renderer-generated, not source-stable.
        if rid and "-" not in rid:
            has_meaningful_id = True

    # Record what we found for the implementer (printed on failure).
    assert has_identifier or has_meaningful_id, (
        "CcdaRenderer dropped source <id>; identity.py CDA branch needs a "
        "direct-XML fallback (parse act <id root extension>)."
    )
```

- [ ] **Step 2: Run the probe**

Run: `cd backend && python -m pytest tests/test_identity.py::test_cda_renderer_preserves_source_id -v -m fidelity -rs`
Expected: PASS (renderer preserves ids) or FAIL (need direct-XML fallback). **Record the outcome** — it selects Task 4's branch.

- [ ] **Step 3: Inspect the actual shape (no commit; informational)**

Run:
```bash
cd backend && python - <<'PY'
from pathlib import Path
from fhir_converter.renderers import CcdaRenderer
p = Path("../HealthSummary_May_29_2026/IHE_XDM/Pedro1/DOC0001.XML")
b = CcdaRenderer().render_to_fhir("ccda", p.read_text(encoding="utf-8"))
for e in b.get("entry", [])[:40]:
    r = e.get("resource", {})
    rt = r.get("resourceType")
    if rt in {"Bundle","Composition","Patient"}: continue
    print(rt, "| id=", r.get("id"), "| identifier=", r.get("identifier"))
PY
```
Expected: a printout showing whether `identifier`/`id` carry the Epic OID+extension. **This dictates Task 4.**

- [ ] **Step 4: Commit the probe**

```bash
cd backend && git add tests/test_identity.py
git commit -m "test: CDA source-id preservation probe against real extract"
```

> **Decision gate for Task 4:**
> - If PASS → Task 4 reads `resource.identifier[0]` / `resource.id`.
> - If FAIL → Task 4 adds `_extract_cda_id_from_xml()` that parses the act `<id root extension>` in `cda_parser.py` and stashes it on the resource as `identifier` **before** record-dict construction. (Excluding NPI root `2.16.840.1.113883.4.6` and person root `…4.2`.)

---

## Task 2: Content hashing module

A canonical hash that ignores volatile noise but captures clinical changes. Pure functions, fully unit-tested with pinned expected outputs.

**Files:**
- Create: `backend/app/services/ingestion/content_hash.py`
- Test: `backend/tests/test_content_hash.py`

- [ ] **Step 1: Write the failing tests (expected outputs first)**

```python
# backend/tests/test_content_hash.py
from __future__ import annotations

from app.services.ingestion.content_hash import canonicalize, content_hash


def _base_resource() -> dict:
    return {
        "resourceType": "Condition",
        "id": "abc",
        "code": {"text": "Type 2 Diabetes"},
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "meta": {"lastUpdated": "2024-01-01T00:00:00Z"},
    }


def test_hash_is_deterministic():
    assert content_hash(_base_resource()) == content_hash(_base_resource())


def test_meta_lastupdated_does_not_change_hash():
    a = _base_resource()
    b = _base_resource()
    b["meta"]["lastUpdated"] = "2025-12-31T23:59:59Z"
    assert content_hash(a) == content_hash(b)


def test_extraction_metadata_does_not_change_hash():
    a = _base_resource()
    b = _base_resource()
    b["_extraction_metadata"] = {"source_format": "cda_r2", "source_document": "DOC0001.XML"}
    assert content_hash(a) == content_hash(b)


def test_narrative_text_div_does_not_change_hash():
    a = _base_resource()
    b = _base_resource()
    b["text"] = {"status": "generated", "div": "<div>rendered html</div>"}
    assert content_hash(a) == content_hash(b)


def test_clinical_status_change_changes_hash():
    a = _base_resource()
    b = _base_resource()
    b["clinicalStatus"]["coding"][0]["code"] = "resolved"
    assert content_hash(a) != content_hash(b)


def test_key_order_does_not_change_hash():
    a = {"resourceType": "Observation", "code": {"text": "BP"}, "valueString": "120/80"}
    b = {"valueString": "120/80", "code": {"text": "BP"}, "resourceType": "Observation"}
    assert content_hash(a) == content_hash(b)


def test_canonicalize_strips_noise_keys():
    res = _base_resource()
    res["_extraction_metadata"] = {"x": 1}
    res["text"] = {"div": "<div/>"}
    out = canonicalize(res)
    assert "_extraction_metadata" not in out
    assert "text" not in out
    assert "lastUpdated" not in out.get("meta", {})


def test_hash_is_hex_sha256():
    h = content_hash(_base_resource())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_content_hash.py -v`
Expected: FAIL with `ModuleNotFoundError: app.services.ingestion.content_hash`

- [ ] **Step 3: Implement**

```python
# backend/app/services/ingestion/content_hash.py
"""Canonical content hashing for health records.

Produces a stable sha256 over the clinically meaningful content of a FHIR
resource, ignoring volatile fields (server timestamps, ingestion metadata,
rendered narrative) so that re-ingesting the same record yields the same hash
while a genuine clinical change yields a different one.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Top-level keys removed entirely before hashing.
_NOISE_KEYS = frozenset({"_extraction_metadata", "text"})
# Keys removed from the `meta` object before hashing.
_META_NOISE_KEYS = frozenset({"lastUpdated", "versionId", "source"})


def canonicalize(resource: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `resource` with volatile/noise fields removed."""
    out: dict[str, Any] = {}
    for key, value in resource.items():
        if key in _NOISE_KEYS:
            continue
        if key == "meta" and isinstance(value, dict):
            cleaned = {k: v for k, v in value.items() if k not in _META_NOISE_KEYS}
            if cleaned:
                out[key] = cleaned
            continue
        out[key] = value
    return out


def content_hash(resource: dict[str, Any]) -> str:
    """Stable hex sha256 of a resource's canonical clinical content."""
    canonical = canonicalize(resource)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest tests/test_content_hash.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/ingestion/content_hash.py tests/test_content_hash.py
git commit -m "feat: canonical content hashing for idempotent ingestion"
```

---

## Task 3: Identity extraction — dataclass + FHIR + Epic-passthrough

`extract_identity(record)` returns an `Identity` or `None`. It honors an `external_id`/`source_system` already placed on the record dict (Epic path, Task 5), else derives from `fhir_resource` for FHIR. CDA is added in Task 4.

**Files:**
- Modify: `backend/app/services/ingestion/identity.py` (currently only the probe test exists; create the module)
- Test: `backend/tests/test_identity.py` (append)

- [ ] **Step 1: Write failing tests (append to test_identity.py)**

```python
from app.services.ingestion.identity import Identity, extract_identity


def test_explicit_fields_take_precedence():
    rec = {
        "source_format": "epic_ehi",
        "external_id": "ORDER_MED_123",
        "source_system": "epic:ORDER_MED",
        "fhir_resource": {"resourceType": "MedicationRequest", "id": "ignored"},
    }
    ident = extract_identity(rec)
    assert ident == Identity(source_system="epic:ORDER_MED", external_id="ORDER_MED_123")


def test_fhir_resource_id():
    rec = {
        "source_format": "fhir_r4",
        "fhir_resource": {"resourceType": "Condition", "id": "cond-1"},
    }
    ident = extract_identity(rec)
    assert ident == Identity(source_system="fhir", external_id="Condition/cond-1")


def test_fhir_identifier_preferred_over_id():
    rec = {
        "source_format": "fhir_r4",
        "fhir_resource": {
            "resourceType": "Condition",
            "id": "gen-uuid",
            "identifier": [{"system": "urn:epic", "value": "PROB-9"}],
        },
    }
    ident = extract_identity(rec)
    assert ident == Identity(source_system="urn:epic", external_id="Condition/PROB-9")


def test_fhir_no_id_returns_none():
    rec = {"source_format": "fhir_r4", "fhir_resource": {"resourceType": "Condition"}}
    assert extract_identity(rec) is None


def test_unknown_format_returns_none():
    rec = {"source_format": "mystery", "fhir_resource": {"resourceType": "X", "id": "1"}}
    assert extract_identity(rec) is None


def test_extraction_never_raises_on_bad_input():
    assert extract_identity({"source_format": "fhir_r4"}) is None
    assert extract_identity({"source_format": "fhir_r4", "fhir_resource": None}) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_identity.py -v -k "not renderer"`
Expected: FAIL with `ImportError: cannot import name 'Identity'`

- [ ] **Step 3: Implement (FHIR + passthrough; CDA stub returns None for now)**

```python
# backend/app/services/ingestion/identity.py
"""Stable source-identity extraction for incremental (idempotent) ingestion.

Given a parsed record dict, derive a `(source_system, external_id)` pair that
uniquely and stably identifies the upstream record, so that a re-uploaded
cumulative extract can be matched against records already in the database.

Returns None when no stable identity can be derived; such records fall through
to the existing content/fuzzy dedup pipeline unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# CDA <id> roots that identify people/providers, not clinical acts.
CDA_NON_ACT_ROOTS = frozenset({
    "2.16.840.1.113883.4.6",   # NPI (provider)
    "2.16.840.1.113883.4.2",   # SSN-ish / person
})


@dataclass(frozen=True)
class Identity:
    """A stable upstream identity for a health record."""
    source_system: str
    external_id: str


def extract_identity(record: dict[str, Any]) -> Identity | None:
    """Derive a stable identity from a parsed record dict, or None."""
    try:
        # 1. Explicit fields win (Epic parser sets these from table PKs).
        ext = record.get("external_id")
        sys = record.get("source_system")
        if ext and sys:
            return Identity(source_system=str(sys), external_id=str(ext))

        source_format = record.get("source_format")
        resource = record.get("fhir_resource")
        if not isinstance(resource, dict):
            return None

        if source_format == "fhir_r4":
            return _from_fhir(resource)
        if source_format == "cda_r2":
            return _from_cda(resource)
        return None
    except Exception:  # never break ingestion on identity extraction
        logger.exception("identity extraction failed; treating as no-identity")
        return None


def _from_fhir(resource: dict[str, Any]) -> Identity | None:
    rtype = resource.get("resourceType")
    if not rtype:
        return None
    identifiers = resource.get("identifier")
    if isinstance(identifiers, list) and identifiers:
        first = identifiers[0]
        value = first.get("value")
        system = first.get("system") or "fhir"
        if value:
            return Identity(source_system=str(system), external_id=f"{rtype}/{value}")
    rid = resource.get("id")
    if rid:
        return Identity(source_system="fhir", external_id=f"{rtype}/{rid}")
    return None


def _from_cda(resource: dict[str, Any]) -> Identity | None:
    """CDA-derived FHIR identity. Filled in Task 4 per probe outcome."""
    return _from_fhir(resource)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest tests/test_identity.py -v -k "not renderer"`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/ingestion/identity.py tests/test_identity.py
git commit -m "feat: identity extraction for FHIR + explicit-field passthrough"
```

---

## Task 4: Identity extraction — CDA branch (per Task 1 outcome)

Implement `_from_cda` based on the probe. **If probe PASSED**, this task only adds the non-act-root filter + tests. **If probe FAILED**, also add `_extract_cda_id_from_xml` wired into `cda_parser.py`.

**Files:**
- Modify: `backend/app/services/ingestion/identity.py`
- Modify (only if probe FAILED): `backend/app/services/ingestion/cda_parser.py`
- Test: `backend/tests/test_identity.py` (append)

- [ ] **Step 1: Write failing tests (append)**

Real `CcdaRenderer` output (verified in Task 1): `system` is `urn:oid:<root>` (note the `urn:oid:` prefix), `value` is the extension; `resource.id` is a renderer UUID and must be ignored; some ids are `{"nullFlavor": "UNK"}` dicts. The filter must therefore strip `urn:oid:` before comparing roots, and guard non-string id values.

```python
def test_cda_identifier_with_urn_oid():
    rec = {
        "source_format": "cda_r2",
        "fhir_resource": {
            "resourceType": "Condition",
            "id": "881e0b55-1111-2222-3333-444455556666",  # renderer UUID, ignored
            "identifier": [{"system": "urn:oid:1.2.840.114350.1.13.516.2.7.2.768076", "value": "26510156"}],
        },
    }
    ident = extract_identity(rec)
    assert ident == Identity(
        source_system="urn:oid:1.2.840.114350.1.13.516.2.7.2.768076",
        external_id="Condition/26510156",
    )


def test_cda_npi_root_is_not_used_as_identity():
    """Provider NPI (urn:oid:2.16.840.1.113883.4.6) must not become the record identity."""
    rec = {
        "source_format": "cda_r2",
        "fhir_resource": {
            "resourceType": "Practitioner",
            "identifier": [{"system": "urn:oid:2.16.840.1.113883.4.6", "value": "1234567890"}],
        },
    }
    assert extract_identity(rec) is None


def test_cda_person_root_is_not_used_as_identity():
    rec = {
        "source_format": "cda_r2",
        "fhir_resource": {
            "resourceType": "Condition",
            "identifier": [{"system": "urn:oid:2.16.840.1.113883.4.2", "value": "999"}],
        },
    }
    # Only non-act identifier present, and id absent -> None
    assert extract_identity(rec) is None


def test_cda_nullflavor_id_does_not_crash():
    rec = {
        "source_format": "cda_r2",
        "fhir_resource": {"resourceType": "Practitioner", "id": {"nullFlavor": "UNK"}},
    }
    assert extract_identity(rec) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_identity.py -v -k "cda and not renderer"`
Expected: FAIL (`test_cda_npi_root_is_not_used_as_identity` returns an Identity instead of None)

- [ ] **Step 3: Implement the non-act-root filter (prefix-aware, id-shape-safe)**

```python
# Replace _from_cda in identity.py. Probe PASSED -> identifier path only, no raw-XML fallback.
def _strip_oid(system: str | None) -> str:
    """Normalize a FHIR system to a bare OID for root comparison."""
    s = system or ""
    return s[len("urn:oid:"):] if s.startswith("urn:oid:") else s


def _from_cda(resource: dict[str, Any]) -> Identity | None:
    rtype = resource.get("resourceType")
    if not rtype:
        return None
    identifiers = resource.get("identifier")
    if isinstance(identifiers, list):
        for ident in identifiers:
            if not isinstance(ident, dict):
                continue
            system = ident.get("system")
            value = ident.get("value")
            if not value or _strip_oid(system) in CDA_NON_ACT_ROOTS:
                continue
            return Identity(source_system=str(system or "cda"), external_id=f"{rtype}/{value}")
    rid = resource.get("id")
    # Ignore renderer UUIDs and non-string ids (e.g. {"nullFlavor": "UNK"}).
    if isinstance(rid, str) and rid and "-" not in rid:
        return Identity(source_system="cda", external_id=f"{rtype}/{rid}")
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_identity.py -v -k "cda and not renderer"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/ingestion/identity.py tests/test_identity.py
git commit -m "feat: CDA identity extraction with non-act-root filtering"
```

---

## Task 5: Epic mapper primary keys + parser wiring

Give each Epic mapper a `source_table` and `primary_key_columns`, and have `epic_parser` set `external_id`/`source_system` on each record dict from the row's PK values.

**Files:**
- Modify: `backend/app/services/ingestion/epic_mappers/base.py`
- Modify: all 14 mappers in `backend/app/services/ingestion/epic_mappers/`
- Modify: `backend/app/services/ingestion/epic_parser.py:120-148`
- Test: `backend/tests/test_identity.py` (append an Epic-PK builder test)

- [ ] **Step 1: Write failing test for the PK→identity helper**

```python
def test_epic_identity_from_row():
    from app.services.ingestion.identity import epic_identity

    row = {"ORDER_MED_ID": "555", "ORDER_ID": "9", "LINE": "2"}
    ident = epic_identity("ORDER_MED", ["ORDER_MED_ID", "LINE"], row)
    assert ident == Identity(source_system="epic:ORDER_MED", external_id="555|2")


def test_epic_identity_missing_pk_returns_none():
    from app.services.ingestion.identity import epic_identity

    assert epic_identity("ORDER_MED", ["ORDER_MED_ID"], {"OTHER": "x"}) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_identity.py -v -k "epic_identity"`
Expected: FAIL (`cannot import name 'epic_identity'`)

- [ ] **Step 3: Implement `epic_identity` in `identity.py`**

```python
def epic_identity(table: str, pk_columns: list[str], row: dict[str, str]) -> Identity | None:
    """Build an Identity from an Epic TSV row's primary-key column(s)."""
    parts: list[str] = []
    for col in pk_columns:
        val = (row.get(col) or "").strip()
        if not val:
            return None
        parts.append(val)
    if not parts:
        return None
    return Identity(source_system=f"epic:{table}", external_id="|".join(parts))
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest tests/test_identity.py -v -k "epic_identity"`
Expected: PASS (2 tests)

- [ ] **Step 5: Add class attrs to base mapper**

```python
# In EpicMapper (base.py), add as class-level attributes above to_fhir:
    source_table: str = ""
    primary_key_columns: list[str] = []
```

- [ ] **Step 6: Set attrs on each of the 14 mappers**

Add `source_table` and `primary_key_columns` to each mapper class. Use the Epic table PK column(s). Reference values (verify against the real export's column headers; adjust if the TSV uses different names):

| Mapper | `source_table` | `primary_key_columns` |
|--------|----------------|----------------------|
| `ProblemListMapper` | `PROBLEM_LIST` | `["PROBLEM_LIST_ID"]` |
| `MedicalHxMapper` | `MEDICAL_HX` | `["PAT_ID", "CONTACT_DATE", "LINE"]` |
| `OrderMedMapper` | `ORDER_MED` | `["ORDER_MED_ID"]` |
| `OrderResultsMapper` | `ORDER_RESULTS` | `["ORDER_PROC_ID", "ORD_VALUE_LINE", "RESULT_LINE"]` |
| `PatEncMapper` | `PAT_ENC` | `["PAT_ENC_CSN_ID"]` |
| `DocInformationMapper` | `DOC_INFORMATION` | `["DOCUMENT_ID"]` |
| `AllergyMapper` | `ALLERGY` | `["ALLERGY_ID"]` |
| `ImmuneMapper` | `IMMUNE` | `["IMMUNE_ID"]` |
| `OrderProcMapper` | `ORDER_PROC` | `["ORDER_PROC_ID"]` |
| `VitalsMapper` | `IP_FLWSHT_MEAS` | `["FSD_ID", "LINE", "FLO_MEAS_ID"]` |
| `ReferralMapper` | `REFERRAL` | `["REFERRAL_ID"]` |
| `EncounterDxMapper` | `PAT_ENC_DX` | `["PAT_ENC_CSN_ID", "DX_ID", "LINE"]` |
| `SocialHxMapper` | `SOCIAL_HX` | `["PAT_ID", "CONTACT_DATE", "LINE"]` |
| `FamilyHxMapper` | `FAMILY_HX` | `["PAT_ID", "LINE"]` |

Example edit (`medications.py`, inside `OrderMedMapper`):

```python
class OrderMedMapper(EpicMapper):
    source_table = "ORDER_MED"
    primary_key_columns = ["ORDER_MED_ID"]

    def to_fhir(self, row: dict[str, str]) -> dict | None:
        ...
```

- [ ] **Step 7: Wire `epic_parser` to set identity on the record dict**

In `backend/app/services/ingestion/epic_parser.py`, inside the row loop where `mapped = {...}` is built (around line 130-145), after constructing `mapped`, add:

```python
                        from app.services.ingestion.identity import epic_identity

                        ident = epic_identity(
                            mapper.source_table or table_name,
                            mapper.primary_key_columns,
                            row,
                        )
                        if ident is not None:
                            mapped["external_id"] = ident.external_id
                            mapped["source_system"] = ident.source_system
```

(Move the import to module top per ruff; shown inline for locality.)

- [ ] **Step 8: Run mapper + identity tests**

Run: `cd backend && python -m pytest tests/test_identity.py tests/test_ingestion.py -v`
Expected: PASS (identity tests pass; existing ingestion tests still green)

- [ ] **Step 9: Commit**

```bash
cd backend && git add app/services/ingestion/identity.py app/services/ingestion/epic_mappers/ app/services/ingestion/epic_parser.py tests/test_identity.py
git commit -m "feat: Epic table primary keys -> record identity"
```

---

## Task 6: Database migration — columns, index, record_versions

**Files:**
- Modify: `backend/app/models/record.py`
- Create: `backend/app/models/record_version.py`
- Create: `backend/alembic/versions/b2c3d4e5f6a7_idempotency_fields.py`

- [ ] **Step 1: Add columns to the `HealthRecord` model**

In `backend/app/models/record.py`, after `merge_metadata` (line 59), add:

```python
    external_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_system: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
```

Ensure `Integer` is imported from `sqlalchemy` at the top of the file (add to the existing import if missing).

- [ ] **Step 2: Create the `RecordVersion` model**

```python
# backend/app/models/record_version.py
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class RecordVersion(Base):
    __tablename__ = "record_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("health_records.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    fhir_resource: Mapped[dict] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    changed_fields: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    source_file_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
```

Confirm the `Base` import path matches `record.py` (use whatever `record.py` imports).

- [ ] **Step 3: Write the migration**

```python
# backend/alembic/versions/b2c3d4e5f6a7_idempotency_fields.py
"""idempotency fields: external_id, source_system, content_hash, version, record_versions

Revision ID: b2c3d4e5f6a7
Revises: f1a2b3c4d5e6
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b2c3d4e5f6a7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("health_records", sa.Column("external_id", sa.Text(), nullable=True))
    op.add_column("health_records", sa.Column("source_system", sa.Text(), nullable=True))
    op.add_column("health_records", sa.Column("content_hash", sa.Text(), nullable=True))
    op.add_column(
        "health_records",
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
    )
    op.create_index(
        "uq_health_records_identity",
        "health_records",
        ["user_id", "source_system", "external_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL AND external_id IS NOT NULL"),
    )
    op.create_table(
        "record_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("record_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("health_records.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("fhir_resource", postgresql.JSONB(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("changed_fields", postgresql.JSONB(), nullable=True),
        sa.Column("source_file_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_record_versions_record_id", "record_versions", ["record_id"])


def downgrade() -> None:
    op.drop_index("ix_record_versions_record_id", table_name="record_versions")
    op.drop_table("record_versions")
    op.drop_index("uq_health_records_identity", table_name="health_records")
    op.drop_column("health_records", "version")
    op.drop_column("health_records", "content_hash")
    op.drop_column("health_records", "source_system")
    op.drop_column("health_records", "external_id")
```

- [ ] **Step 4: Apply to the test DB and verify**

Run:
```bash
cd backend && DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_test alembic upgrade head
DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_test alembic current
```
Expected: head shows `b2c3d4e5f6a7`. Then verify column + index exist:
```bash
psql medtimeline_test -c "\d health_records" | grep -E "external_id|source_system|content_hash|version|uq_health_records_identity"
psql medtimeline_test -c "\dt record_versions"
```
Expected: all columns, the partial unique index, and the table are listed.

- [ ] **Step 5: Apply to dev DB too**

Run: `cd backend && alembic upgrade head && alembic current`
Expected: head `b2c3d4e5f6a7`.

- [ ] **Step 6: Commit**

```bash
cd backend && git add app/models/record.py app/models/record_version.py alembic/versions/b2c3d4e5f6a7_idempotency_fields.py
git commit -m "feat: migration for idempotency fields + record_versions table"
```

---

## Task 7: Idempotent inserter — pure partition logic

The decision core is a pure function: given existing rows (as a lookup map) and incoming records, classify each as insert / update / skip. Test exhaustively with no DB.

**Files:**
- Create: `backend/app/services/ingestion/idempotent_inserter.py`
- Test: `backend/tests/test_idempotent_inserter.py`

- [ ] **Step 1: Write failing tests (expected outputs first)**

```python
# backend/tests/test_idempotent_inserter.py
from __future__ import annotations

from app.services.ingestion.idempotent_inserter import Plan, plan_batch


def _rec(fmt="fhir_r4", rid="c1", code="active"):
    return {
        "source_format": fmt,
        "fhir_resource": {
            "resourceType": "Condition", "id": rid,
            "clinicalStatus": {"coding": [{"code": code}]},
        },
    }


def test_new_record_is_insert():
    plan = plan_batch([_rec(rid="c1")], existing={})
    assert [p.action for p in plan] == ["insert"]
    assert plan[0].identity is not None
    assert plan[0].content_hash is not None


def test_identical_existing_is_skip():
    rec = _rec(rid="c1", code="active")
    key = ("fhir", "Condition/c1")
    from app.services.ingestion.content_hash import content_hash
    existing = {key: ("row-uuid", content_hash(rec["fhir_resource"]), 1)}
    plan = plan_batch([rec], existing=existing)
    assert [p.action for p in plan] == ["skip"]


def test_changed_existing_is_update():
    rec = _rec(rid="c1", code="resolved")
    key = ("fhir", "Condition/c1")
    from app.services.ingestion.content_hash import content_hash
    old_hash = content_hash(_rec(rid="c1", code="active")["fhir_resource"])
    existing = {key: ("row-uuid", old_hash, 1)}
    plan = plan_batch([rec], existing=existing)
    assert plan[0].action == "update"
    assert plan[0].existing_id == "row-uuid"
    assert plan[0].new_version == 2


def test_no_identity_is_insert_fallthrough():
    rec = {"source_format": "fhir_r4", "fhir_resource": {"resourceType": "Condition"}}
    plan = plan_batch([rec], existing={})
    assert plan[0].action == "insert"
    assert plan[0].identity is None


def test_within_batch_duplicate_identity_second_is_skip():
    r1 = _rec(rid="dup", code="active")
    r2 = _rec(rid="dup", code="active")
    plan = plan_batch([r1, r2], existing={})
    assert [p.action for p in plan] == ["insert", "skip"]


def test_within_batch_duplicate_changed_second_is_update_of_first():
    r1 = _rec(rid="dup", code="active")
    r2 = _rec(rid="dup", code="resolved")
    plan = plan_batch([r1, r2], existing={})
    assert plan[0].action == "insert"
    assert plan[1].action == "update_pending"  # updates the just-planned insert
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_idempotent_inserter.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement the pure planner**

```python
# backend/app/services/ingestion/idempotent_inserter.py (part 1: planner)
"""Identity gate for incremental ingestion.

`plan_batch` is a pure classifier: given a map of existing identities and a
batch of incoming records, decide insert / update / skip for each. The DB-bound
`idempotent_insert_records` (added below) executes the plan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services.ingestion.content_hash import content_hash
from app.services.ingestion.identity import Identity, extract_identity

logger = logging.getLogger(__name__)

# existing identity -> (row_id, content_hash, version)
ExistingMap = dict[tuple[str, str], tuple[Any, str | None, int]]


@dataclass
class Plan:
    record: dict[str, Any]
    action: str  # insert | update | skip | update_pending
    identity: Identity | None
    content_hash: str | None
    existing_id: Any = None
    new_version: int = 1


def plan_batch(records: list[dict[str, Any]], existing: ExistingMap) -> list[Plan]:
    """Classify each record. `existing` maps identity-key -> (id, hash, version)."""
    plans: list[Plan] = []
    seen_in_batch: dict[tuple[str, str], int] = {}  # key -> index into plans

    for rec in records:
        ident = extract_identity(rec)
        resource = rec.get("fhir_resource") or {}
        chash = content_hash(resource) if isinstance(resource, dict) and resource else None

        if ident is None:
            plans.append(Plan(rec, "insert", None, chash))
            continue

        key = (ident.source_system, ident.external_id)

        # Within-batch duplicate identity.
        if key in seen_in_batch:
            prior_idx = seen_in_batch[key]
            if plans[prior_idx].content_hash == chash:
                plans.append(Plan(rec, "skip", ident, chash))
            else:
                plans.append(Plan(rec, "update_pending", ident, chash, existing_id=prior_idx))
            continue

        seen_in_batch[key] = len(plans)

        if key in existing:
            row_id, old_hash, old_version = existing[key]
            if old_hash == chash:
                plans.append(Plan(rec, "skip", ident, chash, existing_id=row_id, new_version=old_version))
            else:
                plans.append(Plan(rec, "update", ident, chash, existing_id=row_id, new_version=old_version + 1))
        else:
            plans.append(Plan(rec, "insert", ident, chash))

    return plans
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest tests/test_idempotent_inserter.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/ingestion/idempotent_inserter.py tests/test_idempotent_inserter.py
git commit -m "feat: pure partition planner for idempotent ingestion"
```

---

## Task 8: Idempotent inserter — DB execution + version snapshots

Add `idempotent_insert_records(db, records)` that loads existing identities in one query, runs `plan_batch`, then inserts/updates/snapshots, returning `{inserted, updated, unchanged, inserted_records}`.

**Files:**
- Modify: `backend/app/services/ingestion/idempotent_inserter.py`
- Test: `backend/tests/test_idempotent_inserter.py` (append DB tests)

- [ ] **Step 1: Write failing DB tests**

```python
import pytest
from app.models.record import HealthRecord
from app.models.record_version import RecordVersion
from sqlalchemy import select


@pytest.mark.asyncio
async def test_insert_then_reingest_identical_is_unchanged(db_session, create_test_patient):
    from app.services.ingestion.idempotent_inserter import idempotent_insert_records
    patient = await create_test_patient(db_session, "00000000-0000-0000-0000-000000000001")

    def batch():
        return [{
            "user_id": patient.user_id, "patient_id": patient.id, "source_file_id": None,
            "record_type": "condition", "fhir_resource_type": "Condition",
            "fhir_resource": {"resourceType": "Condition", "id": "c1",
                              "clinicalStatus": {"coding": [{"code": "active"}]}},
            "source_format": "fhir_r4", "display_text": "Cond",
        }]

    s1 = await idempotent_insert_records(db_session, batch()); await db_session.commit()
    assert s1["inserted"] == 1 and s1["unchanged"] == 0

    s2 = await idempotent_insert_records(db_session, batch()); await db_session.commit()
    assert s2["inserted"] == 0 and s2["unchanged"] == 1

    rows = (await db_session.execute(select(HealthRecord))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_changed_reingest_updates_and_snapshots(db_session, create_test_patient):
    from app.services.ingestion.idempotent_inserter import idempotent_insert_records
    patient = await create_test_patient(db_session, "00000000-0000-0000-0000-000000000002")

    def batch(code):
        return [{
            "user_id": patient.user_id, "patient_id": patient.id, "source_file_id": None,
            "record_type": "condition", "fhir_resource_type": "Condition",
            "fhir_resource": {"resourceType": "Condition", "id": "c2",
                              "clinicalStatus": {"coding": [{"code": code}]}},
            "source_format": "fhir_r4", "display_text": "Cond", "status": code,
        }]

    await idempotent_insert_records(db_session, batch("active")); await db_session.commit()
    s2 = await idempotent_insert_records(db_session, batch("resolved")); await db_session.commit()
    assert s2["updated"] == 1

    row = (await db_session.execute(select(HealthRecord))).scalars().one()
    assert row.version == 2
    assert row.fhir_resource["clinicalStatus"]["coding"][0]["code"] == "resolved"

    versions = (await db_session.execute(select(RecordVersion))).scalars().all()
    assert len(versions) == 1  # the prior (active) snapshot
    assert versions[0].fhir_resource["clinicalStatus"]["coding"][0]["code"] == "active"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_idempotent_inserter.py -v -k "reingest"`
Expected: FAIL (`cannot import name 'idempotent_insert_records'`)

- [ ] **Step 3: Implement the DB executor**

```python
# Append to idempotent_inserter.py
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from app.models.record_version import RecordVersion


async def _load_existing(db: AsyncSession, user_id, identities: list[Identity]) -> ExistingMap:
    if not identities:
        return {}
    keys = list({(i.source_system, i.external_id) for i in identities})
    result = await db.execute(
        select(HealthRecord.id, HealthRecord.source_system,
               HealthRecord.external_id, HealthRecord.content_hash, HealthRecord.version)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            tuple_(HealthRecord.source_system, HealthRecord.external_id).in_(keys),
        )
    )
    return {(r.source_system, r.external_id): (r.id, r.content_hash, r.version) for r in result.all()}


def _build_row(rec: dict[str, Any], ident: Identity | None, chash: str | None) -> HealthRecord:
    return HealthRecord(
        id=uuid.uuid4(),
        patient_id=rec["patient_id"], user_id=rec["user_id"],
        record_type=rec["record_type"], fhir_resource_type=rec["fhir_resource_type"],
        fhir_resource=rec["fhir_resource"], source_format=rec["source_format"],
        source_file_id=rec.get("source_file_id"),
        effective_date=rec.get("effective_date"), effective_date_end=rec.get("effective_date_end"),
        status=rec.get("status"), category=rec.get("category"),
        code_system=rec.get("code_system"), code_value=rec.get("code_value"),
        code_display=rec.get("code_display"), display_text=rec["display_text"],
        confidence_score=rec.get("confidence_score"), ai_extracted=rec.get("ai_extracted", False),
        external_id=ident.external_id if ident else None,
        source_system=ident.source_system if ident else None,
        content_hash=chash, version=1,
    )


async def idempotent_insert_records(db: AsyncSession, records: list[dict[str, Any]]) -> dict:
    """Insert new records, update changed ones (snapshotting prior versions),
    skip identical ones. Returns counts + the list of newly inserted record dicts."""
    if not records:
        return {"inserted": 0, "updated": 0, "unchanged": 0, "inserted_records": []}

    user_id = records[0]["user_id"]
    idents = [i for i in (extract_identity(r) for r in records) if i is not None]
    existing = await _load_existing(db, user_id, idents)
    plans = plan_batch(records, existing)

    inserted = updated = unchanged = 0
    inserted_records: list[dict] = []
    pending_rows: dict[int, HealthRecord] = {}  # plan index -> ORM row (for within-batch updates)

    for idx, p in enumerate(plans):
        if p.action == "insert":
            row = _build_row(p.record, p.identity, p.content_hash)
            db.add(row)
            pending_rows[idx] = row
            inserted += 1
            inserted_records.append(p.record)
        elif p.action == "skip":
            unchanged += 1
        elif p.action == "update_pending":
            target = pending_rows.get(p.existing_id)
            if target is not None:
                _snapshot(db, target)
                _apply_update(target, p)
                updated += 1
                inserted -= 1  # the insert it supersedes no longer counts as a fresh insert
                unchanged += 0
        elif p.action == "update":
            row = await db.get(HealthRecord, p.existing_id)
            if row is not None:
                _snapshot(db, row)
                _apply_update(row, p)
                updated += 1

    return {"inserted": inserted, "updated": updated, "unchanged": unchanged,
            "inserted_records": inserted_records}


def _snapshot(db: AsyncSession, row: HealthRecord) -> None:
    db.add(RecordVersion(
        id=uuid.uuid4(), record_id=row.id, version=row.version,
        fhir_resource=row.fhir_resource, content_hash=row.content_hash,
        changed_fields=None, source_file_id=row.source_file_id,
    ))


def _apply_update(row: HealthRecord, p: Plan) -> None:
    rec = p.record
    row.fhir_resource = rec["fhir_resource"]
    row.content_hash = p.content_hash
    row.version = p.new_version
    row.status = rec.get("status", row.status)
    row.effective_date = rec.get("effective_date", row.effective_date)
    row.display_text = rec.get("display_text", row.display_text)
    row.source_file_id = rec.get("source_file_id", row.source_file_id)
```

> Note: for `update_pending` (within-batch correction of a row not yet flushed), `version` stays 1 and we snapshot the pre-update in-memory state. This is an edge case; the integration test in Task 11 covers the common cross-upload path.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest tests/test_idempotent_inserter.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/ingestion/idempotent_inserter.py tests/test_idempotent_inserter.py
git commit -m "feat: DB executor for identity gate with version snapshots"
```

---

## Task 9: Wire ingest paths to the identity gate

Swap the 8 `bulk_insert_records` call sites in the structured paths to `idempotent_insert_records`, accumulate the new `updated`/`unchanged` stats, and run dedup on inserted records only.

**Files:**
- Modify: `backend/app/services/ingestion/coordinator.py` (lines 307, 393; stats around 144-167)
- Modify: `backend/app/services/ingestion/fhir_parser.py` (lines 405, 413, 464, 476)
- Modify: `backend/app/services/ingestion/epic_parser.py` (lines 150, 163)

- [ ] **Step 1: Add a helper that returns full stats**

In each call site, replace:
```python
count = await bulk_insert_records(db, batch)
stats["records_inserted"] += count
```
with:
```python
from app.services.ingestion.idempotent_inserter import idempotent_insert_records
result = await idempotent_insert_records(db, batch)
stats["records_inserted"] += result["inserted"]
stats.setdefault("records_updated", 0)
stats["records_updated"] += result["updated"]
stats.setdefault("records_unchanged", 0)
stats["records_unchanged"] += result["unchanged"]
```
(Hoist the import to module top.) Apply at all 8 sites.

- [ ] **Step 2: Surface new stats in coordinator response**

In `coordinator.py`, extend `ingestion_progress` (around line 144) to include `records_updated` and `records_unchanged` from `stats`.

- [ ] **Step 3: Run the full ingestion + pipeline suites**

Run: `cd backend && python -m pytest tests/test_ingestion.py tests/test_pipeline_integration.py tests/test_upload.py tests/test_xdm_ingestion.py tests/test_cda_parser.py -v`
Expected: PASS (existing behavior preserved; first-time inserts unchanged since DB starts empty per test).

- [ ] **Step 4: Commit**

```bash
cd backend && git add app/services/ingestion/coordinator.py app/services/ingestion/fhir_parser.py app/services/ingestion/epic_parser.py
git commit -m "feat: route structured ingestion through the identity gate"
```

---

## Task 10: Backfill existing rows

Populate `external_id`/`source_system`/`content_hash` for rows already in the DB so prior uploads participate in idempotency.

**Files:**
- Create: `backend/scripts/backfill_idempotency_fields.py`
- Test: covered by Task 11 (re-ingest after backfill)

- [ ] **Step 1: Implement the backfill script**

```python
# backend/scripts/backfill_idempotency_fields.py
"""Backfill external_id/source_system/content_hash for existing health_records.

Idempotent: safe to re-run. Processes in batches to bound memory.
Run: cd backend && python -m scripts.backfill_idempotency_fields
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from app.database import async_session_factory
from app.models.record import HealthRecord
from app.services.ingestion.content_hash import content_hash
from app.services.ingestion.identity import extract_identity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH = 500


async def main() -> None:
    async with async_session_factory() as db:
        offset = 0
        total = 0
        while True:
            rows = (await db.execute(
                select(HealthRecord)
                .where(HealthRecord.deleted_at.is_(None))
                .order_by(HealthRecord.created_at)
                .offset(offset).limit(BATCH)
            )).scalars().all()
            if not rows:
                break
            for row in rows:
                rec = {"source_format": row.source_format, "fhir_resource": row.fhir_resource,
                       "external_id": row.external_id, "source_system": row.source_system}
                ident = extract_identity(rec)
                if ident and not row.external_id:
                    row.external_id = ident.external_id
                    row.source_system = ident.source_system
                if row.fhir_resource and not row.content_hash:
                    row.content_hash = content_hash(row.fhir_resource)
                total += 1
            await db.commit()
            offset += BATCH
            logger.info("backfilled %d rows", total)
        logger.info("done; %d rows processed", total)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Dry-run against dev DB**

Run: `cd backend && python -m scripts.backfill_idempotency_fields`
Expected: logs `done; N rows processed` with no errors.

- [ ] **Step 3: Commit**

```bash
cd backend && git add scripts/backfill_idempotency_fields.py
git commit -m "feat: backfill script for idempotency fields"
```

---

## Task 11: Integration test — double-ingest convergence

**Files:**
- Create: `backend/tests/test_incremental_ingestion.py`

- [ ] **Step 1: Write the integration test**

```python
# backend/tests/test_incremental_ingestion.py
from __future__ import annotations

import pytest
from sqlalchemy import select, func

from app.models.record import HealthRecord
from app.services.ingestion.idempotent_inserter import idempotent_insert_records


def _fhir_batch(patient, n=5, status="active"):
    return [{
        "user_id": patient.user_id, "patient_id": patient.id, "source_file_id": None,
        "record_type": "condition", "fhir_resource_type": "Condition",
        "fhir_resource": {"resourceType": "Condition", "id": f"c{i}",
                          "clinicalStatus": {"coding": [{"code": status}]}},
        "source_format": "fhir_r4", "display_text": f"Cond {i}", "status": status,
    } for i in range(n)]


@pytest.mark.asyncio
async def test_reingesting_same_extract_converges(db_session, create_test_patient):
    patient = await create_test_patient(db_session, "00000000-0000-0000-0000-0000000000aa")

    s1 = await idempotent_insert_records(db_session, _fhir_batch(patient)); await db_session.commit()
    assert s1["inserted"] == 5

    # Re-upload the identical cumulative extract.
    s2 = await idempotent_insert_records(db_session, _fhir_batch(patient)); await db_session.commit()
    assert s2["inserted"] == 0
    assert s2["unchanged"] == 5

    count = (await db_session.execute(
        select(func.count()).select_from(HealthRecord)
        .where(HealthRecord.patient_id == patient.id)
    )).scalar()
    assert count == 5  # no duplication


@pytest.mark.asyncio
async def test_reingest_with_one_correction(db_session, create_test_patient):
    patient = await create_test_patient(db_session, "00000000-0000-0000-0000-0000000000bb")
    await idempotent_insert_records(db_session, _fhir_batch(patient)); await db_session.commit()

    # Newer extract: same 5 records, but c0 flipped to resolved.
    batch = _fhir_batch(patient)
    batch[0]["fhir_resource"]["clinicalStatus"]["coding"][0]["code"] = "resolved"
    s = await idempotent_insert_records(db_session, batch); await db_session.commit()
    assert s["inserted"] == 0 and s["updated"] == 1 and s["unchanged"] == 4
```

- [ ] **Step 2: Run**

Run: `cd backend && python -m pytest tests/test_incremental_ingestion.py -v`
Expected: PASS (2 tests)

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/test_incremental_ingestion.py
git commit -m "test: incremental re-ingest convergence (insert/update/skip)"
```

---

## Task 12: Fidelity test — real extract re-ingest delta = 0

**Files:**
- Create: `backend/tests/fidelity/test_incremental_fidelity.py`

- [ ] **Step 1: Write the fidelity test**

```python
# backend/tests/fidelity/test_incremental_fidelity.py
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select, func

from app.models.record import HealthRecord

_XDM_DIR = Path(__file__).resolve().parents[3] / "HealthSummary_May_29_2026"


@pytest.mark.fidelity
@pytest.mark.skipif(not _XDM_DIR.exists(), reason="real XDM extract not present")
@pytest.mark.asyncio
async def test_real_extract_reingest_is_idempotent(db_session, create_test_patient, tmp_path):
    """Ingest the real HealthSummary XDM twice; second pass adds 0 records."""
    from app.services.ingestion.coordinator import _ingest_xdm
    from app.models.uploaded_file import UploadedFile
    import uuid

    patient = await create_test_patient(db_session, "00000000-0000-0000-0000-0000000000cc")
    xdm_dir = _XDM_DIR / "IHE_XDM" / "Pedro1"
    metadata = xdm_dir / "METADATA.XML"

    up1 = UploadedFile(id=uuid.uuid4(), user_id=patient.user_id, filename="x1",
                       file_category="structured", ingestion_status="processing",
                       storage_path=str(metadata))
    db_session.add(up1); await db_session.commit()
    await _ingest_xdm(db_session, patient.user_id, patient.id, up1.id, xdm_dir, metadata)
    await db_session.commit()

    n1 = (await db_session.execute(
        select(func.count()).select_from(HealthRecord).where(HealthRecord.patient_id == patient.id)
    )).scalar()
    assert n1 > 0

    up2 = UploadedFile(id=uuid.uuid4(), user_id=patient.user_id, filename="x2",
                       file_category="structured", ingestion_status="processing",
                       storage_path=str(metadata))
    db_session.add(up2); await db_session.commit()
    await _ingest_xdm(db_session, patient.user_id, patient.id, up2.id, xdm_dir, metadata)
    await db_session.commit()

    n2 = (await db_session.execute(
        select(func.count()).select_from(HealthRecord).where(HealthRecord.patient_id == patient.id)
    )).scalar()
    assert n2 == n1, f"re-ingest changed record count: {n1} -> {n2}"
```

- [ ] **Step 2: Run (requires real extract present)**

Run: `cd backend && python -m pytest tests/fidelity/test_incremental_fidelity.py -v -m fidelity -rs`
Expected: PASS (or SKIP if extract absent). If the count grows, the CDA identity path is not matching — revisit Task 4 (likely the probe-failed branch is needed).

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/fidelity/test_incremental_fidelity.py
git commit -m "test: real-extract re-ingest idempotency fidelity"
```

---

## Task 13: TDD-gap audit + bulk_inserter direct tests

Deliver the audit promised in the spec and close the most glaring gap (`bulk_inserter` has no direct tests).

**Files:**
- Create: `backend/tests/test_bulk_inserter.py`
- Create: `docs/superpowers/specs/2026-05-29-tdd-gap-audit.md`

- [ ] **Step 1: Write direct bulk_inserter tests**

```python
# backend/tests/test_bulk_inserter.py
from __future__ import annotations

import pytest
from sqlalchemy import select, func

from app.models.record import HealthRecord
from app.services.ingestion.bulk_inserter import bulk_insert_records


@pytest.mark.asyncio
async def test_bulk_insert_empty_returns_zero(db_session):
    assert await bulk_insert_records(db_session, []) == 0


@pytest.mark.asyncio
async def test_bulk_insert_count_and_persistence(db_session, create_test_patient):
    patient = await create_test_patient(db_session, "00000000-0000-0000-0000-0000000000dd")
    recs = [{
        "user_id": patient.user_id, "patient_id": patient.id, "source_file_id": None,
        "record_type": "condition", "fhir_resource_type": "Condition",
        "fhir_resource": {"resourceType": "Condition", "id": f"b{i}"},
        "source_format": "fhir_r4", "display_text": f"C{i}",
    } for i in range(3)]
    n = await bulk_insert_records(db_session, recs)
    await db_session.commit()
    assert n == 3
    count = (await db_session.execute(select(func.count()).select_from(HealthRecord))).scalar()
    assert count == 3
```

- [ ] **Step 2: Run**

Run: `cd backend && python -m pytest tests/test_bulk_inserter.py -v`
Expected: PASS (2 tests)

- [ ] **Step 3: Write the audit report**

Create `docs/superpowers/specs/2026-05-29-tdd-gap-audit.md` documenting: (1) `bulk_inserter` previously untested (now fixed), (2) content-dedup reliance on real-data fixtures vs. pinned expected outputs, (3) absence of any pre-existing idempotency/re-upload tests (now added), and any further gaps found while implementing Tasks 1-12. List each gap, its risk, and whether this plan closed it or deferred it.

- [ ] **Step 4: Commit**

```bash
cd backend && git add tests/test_bulk_inserter.py ../docs/superpowers/specs/2026-05-29-tdd-gap-audit.md
git commit -m "test: bulk_inserter direct tests + TDD-gap audit report"
```

---

## Task 14: Full regression + lint

- [ ] **Step 1: Run the fast suite**

Run: `cd backend && python -m pytest -x -q`
Expected: all fast tests PASS. Investigate and fix any regression before proceeding.

- [ ] **Step 2: Lint**

Run: `cd backend && ruff check app/services/ingestion/ tests/test_identity.py tests/test_content_hash.py tests/test_idempotent_inserter.py`
Expected: no errors (fix import placement, line length ≤100).

- [ ] **Step 3: Final commit**

```bash
cd backend && git add -A
git commit -m "chore: lint + regression pass for idempotent ingestion"
```

---

## Self-Review Notes (author)

- **Spec coverage:** behavior contract → Tasks 7/8/11; data model → Task 6; identity per format → Tasks 3/4/5; content hash → Task 2; upsert flow + wiring → Tasks 8/9; backfill → Task 10; error handling (never-raise, user-scoped, atomic) → Tasks 3/8; test strategy → Tasks 2-13; TDD-gap audit → Task 13; CDA risk spike → Task 1. All covered.
- **Phase 2 (unstructured content-hash idempotency)** is intentionally out of scope; `content_hash.py` (Task 2) is built reusable for it.
- **Open implementation-time check:** Epic PK column names in Task 5 must be verified against the real export headers; adjust `primary_key_columns` if they differ.
