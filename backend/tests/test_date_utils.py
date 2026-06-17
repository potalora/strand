from __future__ import annotations

from datetime import datetime, timezone

from app.utils.date_utils import parse_datetime


def test_parse_iso_date_gets_utc():
    """A bare ISO date parses to a UTC-aware midnight datetime."""
    result = parse_datetime("2024-09-24")
    assert result == datetime(2024, 9, 24, tzinfo=timezone.utc)


def test_parse_naive_datetime_gets_utc():
    """A naive datetime string is assumed UTC."""
    result = parse_datetime("2024-09-24 10:00:00")
    assert result == datetime(2024, 9, 24, 10, 0, 0, tzinfo=timezone.utc)


def test_parse_date_range_returns_none():
    """A date-RANGE string mis-parses into a degenerate -24:00 offset that
    Postgres cannot store; we must return None rather than that bad datetime."""
    result = parse_datetime("2024-09-24 - 2024-12-24")
    assert result is None


def test_parse_explicit_out_of_range_offset_returns_none():
    """An offset at or beyond ±24h is untrustworthy → None (never fabricate)."""
    result = parse_datetime("2024-09-24T20:24-24:00")
    assert result is None


def test_parse_valid_offset_is_preserved():
    """A valid in-range offset (e.g. +05:30) is preserved as-is."""
    result = parse_datetime("2024-09-24T10:00:00+05:30")
    assert result is not None
    assert result.utcoffset() is not None
    # +05:30 == 19800 seconds
    assert result.utcoffset().total_seconds() == 19800


def test_parse_empty_returns_none():
    assert parse_datetime("") is None


def test_parse_none_returns_none():
    assert parse_datetime(None) is None
