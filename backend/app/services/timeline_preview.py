from __future__ import annotations

from typing import Any

from app.schemas.timeline import TimelineGauge, TimelinePreview

# FHIR ObservationInterpretation codes → display label + abnormal-ness.
_ABNORMAL = {
    "H": "HIGH", "L": "LOW", "HH": "CRIT HIGH", "LL": "CRIT LOW",
    "HU": "HIGH", "LU": "LOW", "A": "ABNORMAL", "AA": "CRIT ABNORMAL",
    "POS": "POSITIVE", "DET": "DETECTED",
}
_NORMAL = {"N", "NR", "WNL", "NEG", "ND"}

# Encounter HL7 ActEncounterCode class → readable label.
_ENC_CLASS = {
    "AMB": "Ambulatory", "IMP": "Inpatient", "EMER": "Emergency",
    "VR": "Virtual", "HH": "Home Health", "OBSENC": "Observation",
    "ACUTE": "Acute", "SS": "Short Stay", "PRENC": "Pre-admission",
    "NONAC": "Non-acute", "FLD": "Field",
}

# Statuses that read as "no longer active" → muted (neutral) emphasis.
_MUTED_STATUS = {
    "stopped", "completed", "cancelled", "inactive", "resolved",
    "remission", "not-done", "entered-in-error", "on-hold",
}


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v: Any) -> str | None:
    """Render a scalar for display: ints without a trailing .0, else stripped str."""
    n = _num(v)
    if n is None:
        s = str(v).strip() if v not in (None, "") else None
        return s or None
    return str(int(n)) if n == int(n) else str(n)


def _coding_code(node: Any) -> str | None:
    """First ``coding[].code`` (trimmed) from a CodeableConcept-ish dict."""
    if not isinstance(node, dict):
        return None
    for c in node.get("coding", []) or []:
        if isinstance(c, dict) and c.get("code"):
            return str(c["code"]).strip()
    return None


def _cc_text(node: Any) -> str | None:
    """text → first coding.display → first coding.code from a CodeableConcept."""
    if not isinstance(node, dict):
        return None
    if node.get("text"):
        return str(node["text"]).strip()
    for c in node.get("coding", []) or []:
        if isinstance(c, dict):
            if c.get("display"):
                return str(c["display"]).strip()
            if c.get("code"):
                return str(c["code"]).strip()
    return None


def _status_emphasis(status: str | None) -> str:
    return "muted" if status and status.lower() in _MUTED_STATUS else "normal"


def _obs_category(r: dict) -> str:
    for cat in r.get("category", []) or []:
        if not isinstance(cat, dict):
            continue
        for c in cat.get("coding") or []:
            code = (c.get("code") or "").lower() if isinstance(c, dict) else ""
            if code in ("laboratory", "vital-signs", "social-history"):
                return code
    return "laboratory"


def _lab(r: dict) -> TimelinePreview:
    vq = r.get("valueQuantity") or {}
    value = _fmt(vq.get("value")) if vq else _fmt(r.get("valueString"))
    unit = (str(vq.get("unit")).strip() if vq.get("unit") else None) or None
    flag, emphasis = None, "normal"
    for ic in r.get("interpretation") or []:
        code = (_coding_code(ic) or "").upper()
        if code in _ABNORMAL:
            flag, emphasis = _ABNORMAL[code], "notable"
        elif code in _NORMAL:
            emphasis = "normal"
    gauge = None
    rr = r.get("referenceRange") or []
    if rr and isinstance(rr[0], dict):
        lo = _num((rr[0].get("low") or {}).get("value"))
        hi = _num((rr[0].get("high") or {}).get("value"))
        val = _num(vq.get("value"))
        if lo is not None and hi is not None and val is not None and hi != lo:
            gauge = TimelineGauge(value=val, low=lo, high=hi)
    return TimelinePreview(value=value, unit=unit, flag=flag, emphasis=emphasis, gauge=gauge)


def _vital(r: dict) -> TimelinePreview:
    # Blood pressure → "systolic/diastolic" from the standard LOINC components.
    sys = dia = None
    unit = None
    for comp in r.get("component") or []:
        if not isinstance(comp, dict):
            continue
        code = _coding_code(comp.get("code"))
        cvq = comp.get("valueQuantity") or {}
        if code == "8480-6":
            sys = _fmt(cvq.get("value"))
            unit = unit or cvq.get("unit")
        elif code == "8462-4":
            dia = _fmt(cvq.get("value"))
            unit = unit or cvq.get("unit")
    if sys and dia:
        return TimelinePreview(value=f"{sys}/{dia}", unit=(str(unit).strip() if unit else None))
    vq = r.get("valueQuantity") or {}
    value = _fmt(vq.get("value")) if vq else None
    unit = (str(vq.get("unit")).strip() if vq.get("unit") else None) or None
    return TimelinePreview(value=value, unit=unit)


def _social(r: dict) -> TimelinePreview:
    value = (
        _cc_text(r.get("valueCodeableConcept"))
        or _fmt(r.get("valueString"))
        or _fmt((r.get("valueQuantity") or {}).get("value"))
    )
    return TimelinePreview(value=value)


def _observation(r: dict) -> TimelinePreview:
    cat = _obs_category(r)
    if cat == "vital-signs":
        return _vital(r)
    if cat == "social-history":
        return _social(r)
    return _lab(r)


def _onset_facet(r: dict) -> str | None:
    onset = r.get("onsetDateTime") or r.get("onsetString")
    if not onset and isinstance(r.get("onsetPeriod"), dict):
        onset = r["onsetPeriod"].get("start")
    if not onset:
        return None
    year = str(onset)[:4]
    return f"onset {year}" if year.isdigit() else f"onset {onset}"


def _condition(r: dict) -> TimelinePreview:
    vcode = (_coding_code(r.get("verificationStatus")) or "").lower()
    if vcode in ("refuted", "entered-in-error"):
        flag, emphasis = "NEGATED", "muted"
    else:
        ccode = (_coding_code(r.get("clinicalStatus")) or "").lower()
        flag = ccode.upper().replace("-", " ") if ccode else None
        emphasis = _status_emphasis(ccode)
    facets = []
    onset = _onset_facet(r)
    if onset:
        facets.append(onset)
    return TimelinePreview(flag=flag, emphasis=emphasis, facets=facets)


def _medication(r: dict) -> TimelinePreview:
    di_list = r.get("dosageInstruction") or []
    di0 = di_list[0] if di_list and isinstance(di_list[0], dict) else {}
    value = None
    for dr in di0.get("doseAndRate") or []:
        dq = (dr or {}).get("doseQuantity") or {}
        v = _fmt(dq.get("value"))
        if v:
            unit = (dq.get("unit") or "").strip()
            value = f"{v} {unit}".strip()
            break
    facets = []
    route = _cc_text(di0.get("route"))
    if route:
        facets.append(route)
    freq = _cc_text((di0.get("timing") or {}).get("code"))
    if freq:
        facets.append(freq)
    status = (r.get("status") or "").lower()
    flag = status.upper().replace("-", " ") if status else None
    return TimelinePreview(value=value, flag=flag, emphasis=_status_emphasis(status), facets=facets)


def _allergy(r: dict) -> TimelinePreview:
    crit = (r.get("criticality") or "").lower()
    if crit in ("high", "low"):
        flag = crit.upper()
        emphasis = "notable" if crit == "high" else "normal"
    else:
        ccode = (_coding_code(r.get("clinicalStatus")) or "").lower()
        flag = ccode.upper() if ccode else None
        emphasis = _status_emphasis(ccode)
    facets = []
    for rxn in r.get("reaction") or []:
        for m in (rxn or {}).get("manifestation") or []:
            t = _cc_text(m)
            if t and t not in facets:
                facets.append(t)
    return TimelinePreview(flag=flag, emphasis=emphasis, facets=facets)


def _procedure(r: dict) -> TimelinePreview:
    status = (r.get("status") or "").lower()
    flag = status.upper().replace("-", " ") if status else None
    facets = []
    outcome = _cc_text(r.get("outcome"))
    if outcome:
        facets.append(outcome)
    for bs in r.get("bodySite") or []:
        t = _cc_text(bs)
        if t and t not in facets:
            facets.append(t)
    return TimelinePreview(flag=flag, emphasis=_status_emphasis(status), facets=facets)


def _immunization(r: dict) -> TimelinePreview:
    status = (r.get("status") or "").lower()
    flag = status.upper().replace("-", " ") if status else None
    facets = []
    for pa in r.get("protocolApplied") or []:
        dn = (pa or {}).get("doseNumberPositiveInt")
        if dn is None:
            dn = (pa or {}).get("doseNumberString")
        if dn is not None:
            facets.append(f"dose {dn}")
            break
    dq = r.get("doseQuantity") or {}
    value = None
    if dq.get("value") is not None:
        value = f"{_fmt(dq.get('value'))} {(dq.get('unit') or '').strip()}".strip()
    return TimelinePreview(value=value, flag=flag, emphasis=_status_emphasis(status), facets=facets)


def _encounter(r: dict) -> TimelinePreview:
    klass = r.get("class") or {}
    code = (klass.get("code") or "").upper() if isinstance(klass, dict) else ""
    flag = _ENC_CLASS.get(code) or (klass.get("display") if isinstance(klass, dict) else None)
    facets = []
    for rc in r.get("reasonCode") or []:
        t = _cc_text(rc)
        if t and t not in facets:
            facets.append(t)
    return TimelinePreview(flag=flag, emphasis="normal", facets=facets)


def _imaging(r: dict) -> TimelinePreview:
    facets = []
    concl = r.get("conclusion")
    if concl:
        snippet = str(concl).strip()
        facets.append(snippet[:80] + ("…" if len(snippet) > 80 else ""))
    for cat in r.get("category") or []:
        t = _cc_text(cat)
        if t and t not in facets:
            facets.append(t)
    for s in r.get("series") or []:
        mod = _cc_text((s or {}).get("modality"))
        if mod and mod not in facets:
            facets.append(mod)
    status = (r.get("status") or "").lower()
    flag = status.upper().replace("-", " ") if status else None
    return TimelinePreview(flag=flag, emphasis="normal", facets=facets[:3])


_DISPATCH = {
    "observation": _observation,
    "condition": _condition,
    "medication": _medication,
    "allergy": _allergy,
    "procedure": _procedure,
    "immunization": _immunization,
    "encounter": _encounter,
    "imaging": _imaging,
    "diagnostic_report": _imaging,
}


def build_timeline_preview(fhir_resource: dict | None, record_type: str) -> TimelinePreview | None:
    """Build a compact scalar preview for a timeline row from stored FHIR JSONB.

    Pure dict traversal (no ``fhir.resources``); mirrors the read-path pattern of
    ``timeline_service.extract_provider_display``. Returns ``None`` when the
    resource carries nothing worth surfacing inline, so the row falls back to its
    title-only rendering. ``emphasis`` is a NEUTRAL visual token only
    ("normal" | "notable" | "muted") — never a good/bad clinical judgment.
    """
    if not isinstance(fhir_resource, dict):
        return None
    builder = _DISPATCH.get(record_type)
    if builder is None:
        return None
    p = builder(fhir_resource)
    if p and (p.value or p.flag or p.facets or p.gauge):
        return p
    return None
