"""Bounded page-at-a-time rasterization for encrypted local documents."""

from __future__ import annotations

import warnings
import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError
from pypdfium2 import raw
from pypdfium2._helpers.misc import PdfiumError

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.scratch import ScratchJob

_PDF_MAGIC = b"%PDF-"
_TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


@dataclass(frozen=True)
class RasterizedPage:
    """One disposable, securely stored rasterized page."""

    page_number: int
    path: Path
    width: int
    height: int
    sha256: str


def _positive_limit(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LocalValidationError(f"{name} must be a positive integer.")
    return value


def _dimensions(width: object, height: object, max_pixels: int) -> tuple[int, int]:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise LocalValidationError("Document has invalid page dimensions.")
    if width * height > max_pixels:
        raise LocalValidationError("Rasterized page exceeds the pixel limit.")
    return width, height


def _validate_magic(source: BinaryIO, suffix: str) -> None:
    try:
        source.seek(0)
        magic = source.read(8)
        source.seek(0)
    except OSError:
        raise LocalValidationError("Decrypted document is unavailable.") from None
    if suffix == ".pdf" and not magic.startswith(_PDF_MAGIC):
        raise LocalValidationError("Document content does not match its suffix.")
    if suffix in {".tif", ".tiff"} and not magic.startswith(_TIFF_MAGICS):
        raise LocalValidationError("Document content does not match its suffix.")


def _hash_open_file(handle: BinaryIO) -> str:
    handle.flush()
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8192), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _preflight_pdf_page(page: object, scale: float, max_pixels: int) -> None:
    try:
        width, height = page.get_size()
        scaled_width = math.ceil(float(width) * scale)
        scaled_height = math.ceil(float(height) * scale)
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise LocalValidationError("Document has invalid page dimensions.") from None
    _dimensions(scaled_width, scaled_height, max_pixels)


def _iter_pdf(
    source: BinaryIO,
    scratch: ScratchJob,
    *,
    max_pages: int,
    max_pixels: int,
) -> Iterator[RasterizedPage]:
    try:
        document = pdfium.PdfDocument(source)
    except PdfiumError as exc:
        if exc.err_code in {raw.FPDF_ERR_PASSWORD, raw.FPDF_ERR_SECURITY}:
            raise LocalValidationError(
                "PDF is encrypted or password-protected."
            ) from None
        raise LocalValidationError("PDF document is malformed.") from None
    except Exception:
        raise LocalValidationError("PDF document is malformed.") from None

    try:
        try:
            page_count = len(document)
        except Exception:
            raise LocalValidationError("PDF document is malformed.") from None
        if page_count > max_pages:
            raise LocalValidationError("Document exceeds the page limit.")
        if page_count <= 0:
            raise LocalValidationError("PDF document is malformed.")

        for index in range(page_count):
            page = None
            bitmap = None
            image = None
            try:
                page = document[index]
                _preflight_pdf_page(page, 2.0, max_pixels)
                bitmap = page.render(scale=2.0)
                image = bitmap.to_pil()
                width, height = _dimensions(image.width, image.height, max_pixels)
                filename = f"page-{index + 1:04d}.png"
                output = scratch.reserve_file(filename)
                try:
                    with scratch.open_file(filename) as output_handle:
                        image.save(output_handle, format="PNG")
                        sha256 = _hash_open_file(output_handle)
                    output = scratch.verify_file(filename)
                    rasterized = RasterizedPage(
                        page_number=index + 1,
                        path=output,
                        width=width,
                        height=height,
                        sha256=sha256,
                    )
                except BaseException:
                    scratch.remove_file(filename)
                    raise
            except LocalValidationError:
                raise
            except Exception:
                raise LocalValidationError("PDF document is malformed.") from None
            finally:
                if image is not None:
                    image.close()
                if bitmap is not None:
                    bitmap.close()
                if page is not None:
                    page.close()

            try:
                yield rasterized
            finally:
                scratch.remove_file(filename)
    finally:
        document.close()


def _iter_tiff(
    source: BinaryIO,
    scratch: ScratchJob,
    *,
    max_pages: int,
    max_pixels: int,
) -> Iterator[RasterizedPage]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as document:
                page_number = 1
                while True:
                    try:
                        document.seek(page_number - 1)
                    except EOFError:
                        break
                    if page_number > max_pages:
                        raise LocalValidationError("Document exceeds the page limit.")
                    width, height = _dimensions(
                        document.width,
                        document.height,
                        max_pixels,
                    )
                    rendered = None
                    try:
                        rendered = document.convert("RGB")
                        filename = f"page-{page_number:04d}.png"
                        output = scratch.reserve_file(filename)
                        try:
                            with scratch.open_file(filename) as output_handle:
                                rendered.save(output_handle, format="PNG")
                                sha256 = _hash_open_file(output_handle)
                            output = scratch.verify_file(filename)
                            rasterized = RasterizedPage(
                                page_number=page_number,
                                path=output,
                                width=width,
                                height=height,
                                sha256=sha256,
                            )
                        except BaseException:
                            scratch.remove_file(filename)
                            raise
                    finally:
                        if rendered is not None:
                            rendered.close()
                    try:
                        yield rasterized
                    finally:
                        scratch.remove_file(filename)
                    page_number += 1
    except LocalValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise LocalValidationError("TIFF exceeds the decompression limit.") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise LocalValidationError("TIFF document is malformed.") from None


def iter_rasterized_pages(
    encrypted_path: Path | str,
    scratch: ScratchJob,
    *,
    max_pages: int = 500,
    max_pixels: int,
) -> Iterator[RasterizedPage]:
    """Yield at most one secure PNG at a time from an encrypted PDF or TIFF."""
    page_limit = _positive_limit(max_pages, "Page limit")
    pixel_limit = _positive_limit(max_pixels, "Pixel limit")
    source_path = Path(encrypted_path)
    suffix = source_path.suffix.lower()
    if suffix == ".rtf":
        raise LocalValidationError("RTF is text-only and cannot be rasterized.")
    if suffix not in {".pdf", ".tif", ".tiff"}:
        raise LocalValidationError("Unsupported document type.")

    source = scratch.decrypt_to_file(source_path)
    with scratch.open_file(source.name) as source_handle:
        _validate_magic(source_handle, suffix)
        if suffix == ".pdf":
            yield from _iter_pdf(
                source_handle,
                scratch,
                max_pages=page_limit,
                max_pixels=pixel_limit,
            )
        else:
            yield from _iter_tiff(
                source_handle,
                scratch,
                max_pages=page_limit,
                max_pixels=pixel_limit,
            )
