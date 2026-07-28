"""OvisOCR2 page-level inference with fixed greedy options."""

from __future__ import annotations

from collections.abc import Mapping

from .common import (
    Generate,
    LoadedRole,
    WorkerInputError,
    generate_content,
    load_role_from_payload,
    requested_output_tokens,
    validate_scratch_image,
    validate_token_budget,
)

OCR_INSTRUCTION = (
    "Extract this page faithfully as Markdown. Preserve reading order, tables, and formulas."
)
OCR_OUTPUT_CAP = 8192
_OCR_INPUT_KEYS = frozenset({"page_number", "scratch_dir", "image_path", "image_sha256"})
_OCR_TRANSPORT_KEYS = frozenset(
    {
        "job_id",
        "manifest_path",
        "model_dir",
        "manifest_identity",
        "max_output_tokens",
    }
)


def run_ocr(
    payload: Mapping[str, object],
    *,
    loaded: LoadedRole | None = None,
    generate_fn: Generate = generate_content,
) -> dict[str, object]:
    """Generate Markdown for exactly one explicitly selected page image."""

    if (
        not _OCR_INPUT_KEYS.issubset(payload)
        or set(payload) - _OCR_INPUT_KEYS - _OCR_TRANSPORT_KEYS
    ):
        raise WorkerInputError("OCR request is invalid.")
    page_number = payload.get("page_number")
    image_path = payload.get("image_path")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number <= 0:
        raise WorkerInputError("OCR request is invalid.")
    image = validate_scratch_image(
        image_path,
        payload.get("scratch_dir"),
        expected_sha256=payload.get("image_sha256"),
    )
    selected = loaded or load_role_from_payload("ocr", payload)
    max_tokens = requested_output_tokens(payload, selected, role_cap=OCR_OUTPUT_CAP)
    validate_token_budget(selected, [OCR_INSTRUCTION], max_output_tokens=max_tokens)
    markdown = generate_fn(
        model=selected.model,
        processor=selected.processor,
        prompt=OCR_INSTRUCTION,
        images=[image],
        max_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        input_token_limit=selected.decode_limits["max_input_tokens"],
    )
    if not isinstance(markdown, str):
        raise WorkerInputError("OCR result is invalid.")
    return {"markdown": markdown, "page_number": page_number}
