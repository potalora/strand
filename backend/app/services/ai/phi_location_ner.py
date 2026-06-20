"""Clinical LOCATION de-identification (closes the city/geography Safe Harbor gap).

The legacy scrubber redacts street addresses with a regex but lets bare city /
town / county / facility names pass through, because the *general* spaCy model
mislabels drugs as places (``Rifaximin`` -> GPE) — blanket GPE NER would corrupt
clinical content. This module closes that gap with a **clinical** de-identifier
model (``StanfordAIMI/stanford-deidentifier-base``, MIT) that labels geographic
PHI without tagging medications as locations.

The model is loaded through a HuggingFace ``transformers`` token-classification
pipeline — the same backend Presidio's ``TransformersNlpEngine`` would use — but
invoked directly so the LOCATION pass has a small, testable, fail-open surface.

Safety properties (mirroring ``phi_ner``):

* **Fail-open, NON-latching.** If the model is missing / cannot download / errors
  at inference, the pass returns the text unchanged and de-identification of
  structured PHI + known identifiers + person names still happened upstream. A
  failure is NEVER latched into a process-global disable flag — every call retries
  the load until it succeeds (a single transient OOM must not silently disable
  geographic redaction for the life of the worker).
* **Behind a flag.** Only runs when ``settings.phi_location_ner_enabled`` is set;
  default off until validated on real data.
* **Drug names survive.** The clinical model does not tag medications as places;
  regression tests assert ``Rifaximin`` / ``Rituximab`` / ``Crohn's`` are kept.
"""

from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)

_LOCATION = "[LOCATION]"

# Decided clinical de-identifier model (oss-adoption-design §7.3 / WS-B kickoff):
# StanfordAIMI/stanford-deidentifier-base — MIT, best i2b2 location recall. Kept
# as a module constant (not a config field) so this branch does not touch
# config.py; `settings.phi_location_ner_model` overrides it if added later.
_DEFAULT_MODEL = "StanfordAIMI/stanford-deidentifier-base"


def _model_name() -> str:
    return getattr(settings, "phi_location_ner_model", None) or _DEFAULT_MODEL

# Token-classification labels (entity groups) that denote geographic PHI. The
# Stanford de-identifier emits coarse i2b2-style groups; we redact anything that
# names a place, hospital, or facility, and ONLY those — person/age/id/date/drug
# labels are handled elsewhere or intentionally left to other layers.
_LOCATION_LABEL_KEYS: tuple[str, ...] = (
    "LOC",          # LOCATION / LOC
    "CITY",
    "STATE",
    "COUNTRY",
    "ZIP",
    "STREET",
    "HOSPITAL",
    "FACILITY",
    "ORG",          # organizations are frequently clinics/hospitals (geographic)
    "GPE",
)

# Minimum model confidence to act on. Recall is the priority for de-id, but the
# clinical model is confident on true locations; a moderate floor trims noise
# without dropping real cities.
_SCORE_FLOOR = 0.35

_pipeline = None
_warned = False


def _is_location_label(entity_group: str | None) -> bool:
    if not entity_group:
        return False
    upper = entity_group.upper()
    return any(key in upper for key in _LOCATION_LABEL_KEYS)


def _get_pipeline():
    """Lazily build the transformers token-classification pipeline (singleton).

    A load failure is NOT latched: each call retries until the model is ready,
    matching the non-latching fail-open contract of ``phi_ner._get_nlp``. The
    warning is logged once to avoid spam. The model downloads from HuggingFace on
    first use (large); if that is blocked/offline the pass degrades gracefully.
    """
    global _pipeline, _warned
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        model_name = _model_name()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        _pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        _warned = False
        return _pipeline
    except Exception:  # noqa: BLE001 - missing/blocked model must not break scrubbing
        if not _warned:
            logger.warning(
                "Clinical location model %r unavailable; skipping LOCATION "
                "redaction this call (will retry; structured + known-identifier + "
                "person-name scrubbing still applied)",
                _model_name(),
                exc_info=True,
            )
            _warned = True
        return None


def warm_load_location_ner() -> bool:
    """Eagerly load the clinical location model (call at app startup).

    Returns True if the model is ready. Loading at boot avoids first-call latency
    and download races during concurrent extraction. Safe to call when the flag
    is off; callers gate on ``settings.phi_location_ner_enabled``.
    """
    return _get_pipeline() is not None


def redact_locations(text: str) -> tuple[str, dict[str, int]]:
    """Redact geographic PHI (cities, towns, counties, facilities) to ``[LOCATION]``.

    Drug names are NOT redacted — the clinical model does not label medications as
    places (unlike the general spaCy GPE model). Returns ``(redacted, {"locations": n})``;
    the input unchanged with an empty report when nothing matches or the model is
    unavailable (fail-open).
    """
    if not text or not text.strip():
        return text, {}

    pipe = _get_pipeline()
    if pipe is None:
        return text, {}

    try:
        entities = pipe(text)
    except Exception:  # noqa: BLE001 - inference error must not break scrubbing
        logger.warning("Location NER inference failed; skipping this call", exc_info=True)
        return text, {}

    spans: list[tuple[int, int]] = []
    for ent in entities:
        group = ent.get("entity_group") or ent.get("entity")
        score = float(ent.get("score", 0.0))
        start = ent.get("start")
        end = ent.get("end")
        if start is None or end is None:
            continue
        if score < _SCORE_FLOOR:
            continue
        if _is_location_label(group):
            spans.append((int(start), int(end)))

    if not spans:
        return text, {}

    # Merge whitespace-separated adjacent spans, then redact right-to-left so
    # earlier offsets stay valid.
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and text[merged[-1][1] : start].strip() == "":
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    redacted = text
    for start, end in reversed(merged):
        redacted = redacted[:start] + _LOCATION + redacted[end:]

    return redacted, {"locations": len(merged)}
