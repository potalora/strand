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
    bundle = renderer.render_to_fhir("CCD", _XDM_DOC.read_text(encoding="utf-8"))

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

    assert has_identifier or has_meaningful_id, (
        "CcdaRenderer dropped source <id>; identity.py CDA branch needs a "
        "direct-XML fallback (parse act <id root extension>)."
    )
