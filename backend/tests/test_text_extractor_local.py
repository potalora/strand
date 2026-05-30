from __future__ import annotations

from app.services.extraction.text_extractor import _render_tables


def test_render_tables_pipe_delimited():
    tables = [[["Test", "Value", "Units"], ["Glucose", "95", "mg/dL"], ["A1c", "5.4", "%"]]]
    out = _render_tables(tables)
    assert "Test | Value | Units" in out
    assert "Glucose | 95 | mg/dL" in out
    assert "A1c | 5.4 | %" in out


def test_render_tables_handles_none_cells():
    tables = [[["A", None, "C"], [None, "2", None]]]
    out = _render_tables(tables)
    assert "A |  | C" in out
    assert " | 2 | " in out


def test_render_tables_empty_returns_empty():
    assert _render_tables([]) == ""
    assert _render_tables(None) == ""
