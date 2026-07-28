"""Deterministic, PHI-safe checkpoint key tests."""

from __future__ import annotations

import re

import pytest

from app.services.local_ai.checkpoints import (
    extraction_checkpoint_key,
    ocr_checkpoint_key,
    summary_checkpoint_key,
)
from app.services.local_ai.errors import LocalValidationError

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def test_checkpoint_keys_are_deterministic_and_kind_separated() -> None:
    ocr = ocr_checkpoint_key(_A, 1, "raster-v1", _D)
    assert ocr == ocr_checkpoint_key(_A, 1, "raster-v1", _D)
    assert _SHA256.fullmatch(ocr)

    extraction = extraction_checkpoint_key(_A, "raster-v1", "1", _D)
    summary = summary_checkpoint_key(_A, "raster-v1", _D)

    assert len({ocr, extraction, summary}) == 3


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("upload_hash", lambda: ocr_checkpoint_key(_B, 1, "raster-v1", _D)),
        ("page_number", lambda: ocr_checkpoint_key(_A, 2, "raster-v1", _D)),
        ("raster_version", lambda: ocr_checkpoint_key(_A, 1, "raster-v2", _D)),
        ("manifest_digest", lambda: ocr_checkpoint_key(_A, 1, "raster-v1", _C)),
    ],
)
def test_ocr_key_changes_for_every_dependency(field: str, changed) -> None:
    del field
    base = ocr_checkpoint_key(_A, 1, "raster-v1", _D)
    assert changed() != base


def test_extraction_key_changes_for_every_dependency() -> None:
    base = extraction_checkpoint_key(_A, "schema-v1", "prompt-v1", _D)
    assert base != extraction_checkpoint_key(_B, "schema-v1", "prompt-v1", _D)
    assert base != extraction_checkpoint_key(_A, "schema-v2", "prompt-v1", _D)
    assert base != extraction_checkpoint_key(_A, "schema-v1", "prompt-v2", _D)
    assert base != extraction_checkpoint_key(_A, "schema-v1", "prompt-v1", _C)


def test_summary_key_changes_for_every_dependency() -> None:
    base = summary_checkpoint_key(_A, "summary-v1", _D)
    assert base != summary_checkpoint_key(_B, "summary-v1", _D)
    assert base != summary_checkpoint_key(_A, "summary-v2", _D)
    assert base != summary_checkpoint_key(_A, "summary-v1", _C)


def test_unicode_versions_use_stable_canonical_utf8_json() -> None:
    first = extraction_checkpoint_key(_A, "schéma-一", "prompt-ß", _D)
    second = extraction_checkpoint_key(_A, "schéma-一", "prompt-ß", _D)
    assert first == second


@pytest.mark.parametrize("bad_digest", ["", "a" * 63, "A" * 64, "g" * 64, 42, None])
def test_digest_inputs_require_bounded_lowercase_sha256(bad_digest: object) -> None:
    with pytest.raises(LocalValidationError, match="digest"):
        ocr_checkpoint_key(bad_digest, 1, "raster-v1", _D)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_page", [True, False, 0, -1, 1.0, float("inf"), "1"])
def test_page_number_rejects_bool_non_integer_and_non_positive_values(
    bad_page: object,
) -> None:
    with pytest.raises(LocalValidationError, match="page number"):
        ocr_checkpoint_key(_A, bad_page, "raster-v1", _D)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_version", ["", " ", "x" * 129, True, 1, float("nan"), None]
)
def test_version_inputs_reject_empty_unbounded_and_non_scalar_values(
    bad_version: object,
) -> None:
    with pytest.raises(LocalValidationError, match="version"):
        summary_checkpoint_key(_A, bad_version, _D)  # type: ignore[arg-type]


def test_validation_errors_do_not_echo_raw_values() -> None:
    phi_canary = "Jane-Q-Public-MRN-000123"
    with pytest.raises(LocalValidationError) as exc:
        summary_checkpoint_key(_A, phi_canary * 20, _D)
    assert phi_canary not in str(exc.value)


@pytest.mark.parametrize("bad_version", ["\ud800", "一" * 86])
def test_version_rejects_unencodable_or_oversized_utf8(
    bad_version: str,
) -> None:
    with pytest.raises(LocalValidationError, match="version"):
        summary_checkpoint_key(_A, bad_version, _D)
