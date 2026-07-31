"""Provider-neutral entity data transfer objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractedEntity:
    """A single entity extracted from clinical text."""

    entity_class: str
    text: str
    attributes: dict = field(default_factory=dict)
    start_pos: int | None = None
    end_pos: int | None = None
    confidence: float = 0.8
