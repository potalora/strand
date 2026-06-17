from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from dateutil import parser as dateutil_parser

# Postgres ``timestamptz`` only accepts UTC offsets strictly within ±24h. dateutil
# will happily fabricate a degenerate offset (e.g. a date RANGE string like
# "2024-09-24 - 2024-12-24" parses to a -24:00 offset) that Python tolerates but
# asyncpg rejects on INSERT with a DataError. Anything at or beyond this bound is
# untrustworthy, so we discard it rather than store a wrong date.
_MAX_OFFSET = timedelta(hours=24)


def parse_datetime(
    value: str | None, default: datetime | None = None
) -> Optional[datetime]:
    """Parse a datetime string into a timezone-aware datetime, or None.

    Naive results are assumed UTC. A parsed offset is validated against the
    Postgres ``timestamptz`` bound (strictly within ±24h); an out-of-range
    offset signals a mis-parse and yields ``None`` (never a fabricated date).

    ``default`` fills fields absent from ``value`` (passed through to dateutil).
    Supply a fixed naive datetime (e.g. ``datetime(2000, 1, 1)``) when parsing
    generalized month/year values so the missing day resolves deterministically
    to the 1st instead of dateutil's implicit "today".
    """
    if not value:
        return None
    try:
        if default is not None:
            dt = dateutil_parser.parse(value, default=default)
        else:
            dt = dateutil_parser.parse(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        # ``utcoffset()`` itself raises ValueError for a degenerate ±24h offset,
        # so the bound check lives inside the try.
        offset = dt.utcoffset()
    except (ValueError, TypeError, OverflowError):
        return None

    if offset is None:
        return dt.replace(tzinfo=timezone.utc)
    if not (-_MAX_OFFSET < offset < _MAX_OFFSET):
        return None
    return dt
