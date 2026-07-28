from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class EpicMapper(ABC):
    """Abstract base class for Epic table → FHIR resource mappers."""

    source_table: str = ""
    primary_key_columns: list[str] = []

    @abstractmethod
    def to_fhir(self, row: dict[str, str]) -> dict | None:
        """Map a single TSV row to a FHIR-like resource dict.

        Returns None if the row cannot be mapped.
        """
        ...

    @staticmethod
    def parse_epic_date(value: str | None) -> datetime | None:
        """Parse Epic date formats like '5/21/2024 12:00:00 AM'."""
        if not value or not value.strip():
            return None
        formats = [
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %H:%M:%S %p",
            "%m/%d/%Y",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def safe_get(row: dict, key: str) -> str:
        """Safely get a value from a row dict, returning empty string if missing."""
        val = row.get(key, "")
        return val.strip() if val else ""
