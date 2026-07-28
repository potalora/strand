"""Bounded, one-page-at-a-time local rasterization tests."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pytest
from PIL import Image
from pypdfium2 import raw
from pypdfium2._helpers.misc import PdfiumError

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.rasterizer import iter_rasterized_pages
from app.services.local_ai.scratch import ScratchJob
from app.utils.file_utils import EncryptedFileWriter


def _encrypt_bytes(path: Path, plaintext: bytes) -> Path:
    with path.open("wb") as destination:
        writer = EncryptedFileWriter(destination)
        for offset in range(0, len(plaintext), 1024):
            writer.write_chunk(plaintext[offset : offset + 1024])
        writer.finalize()
    return path


def _pdf_bytes(tmp_path: Path, pages: int = 2) -> bytes:
    path = tmp_path / "plain.pdf"
    images = [
        Image.new("RGB", (64, 48), (index * 40, 20, 100)) for index in range(pages)
    ]
    try:
        images[0].save(path, format="PDF", save_all=True, append_images=images[1:])
        return path.read_bytes()
    finally:
        for image in images:
            image.close()


def _tiff_bytes(tmp_path: Path, frames: int) -> bytes:
    path = tmp_path / "plain.tiff"
    images = [Image.new("RGB", (16, 12), (index, 10, 20)) for index in range(frames)]
    try:
        images[0].save(path, format="TIFF", save_all=True, append_images=images[1:])
        return path.read_bytes()
    finally:
        for image in images:
            image.close()


def _pngs(scratch: ScratchJob) -> list[Path]:
    return sorted(scratch.path.glob("page-*.png"))


def test_pdf_rasterizer_yields_pages_one_at_a_time_with_secure_hashes(
    tmp_path: Path,
) -> None:
    encrypted = _encrypt_bytes(tmp_path / "source.PDF", _pdf_bytes(tmp_path))

    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        pages = iter_rasterized_pages(encrypted, scratch, max_pixels=1_000_000)

        first = next(pages)
        assert first.page_number == 1
        assert first.width * first.height <= 1_000_000
        assert _pngs(scratch) == [first.path]
        assert stat.S_IMODE(first.path.stat().st_mode) == 0o600
        assert hashlib.sha256(first.path.read_bytes()).hexdigest() == first.sha256

        second = next(pages)
        assert second.page_number == 2
        assert not first.path.exists()
        assert _pngs(scratch) == [second.path]
        assert hashlib.sha256(second.path.read_bytes()).hexdigest() == second.sha256

        with pytest.raises(StopIteration):
            next(pages)
        assert not second.path.exists()
        assert _pngs(scratch) == []


def test_early_generator_close_removes_current_page(tmp_path: Path) -> None:
    encrypted = _encrypt_bytes(tmp_path / "source.pdf", _pdf_bytes(tmp_path))
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        pages = iter_rasterized_pages(encrypted, scratch, max_pixels=1_000_000)
        page = next(pages)
        assert page.path.exists()
        pages.close()
        assert not page.path.exists()


def test_tiff_over_limit_raises_instead_of_truncating(tmp_path: Path) -> None:
    encrypted = _encrypt_bytes(tmp_path / "source.tiff", _tiff_bytes(tmp_path, 26))
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError, match="page limit"):
            list(
                iter_rasterized_pages(
                    encrypted,
                    scratch,
                    max_pages=25,
                    max_pixels=1_000_000,
                )
            )
        assert _pngs(scratch) == []


@pytest.mark.parametrize(
    ("max_pages", "max_pixels"),
    [(0, 100), (True, 100), (1, 0), (1, False)],
)
def test_rasterizer_rejects_zero_and_bool_limits(
    tmp_path: Path,
    max_pages: object,
    max_pixels: object,
) -> None:
    encrypted = _encrypt_bytes(tmp_path / "source.pdf", _pdf_bytes(tmp_path, 1))
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError, match="positive integer"):
            list(
                iter_rasterized_pages(
                    encrypted,
                    scratch,
                    max_pages=max_pages,  # type: ignore[arg-type]
                    max_pixels=max_pixels,  # type: ignore[arg-type]
                )
            )


@pytest.mark.parametrize("suffix", [".pdf", ".tiff"])
def test_rasterizer_rejects_oversized_page_or_frame(
    tmp_path: Path,
    suffix: str,
) -> None:
    plaintext = (
        _pdf_bytes(tmp_path, 1) if suffix == ".pdf" else _tiff_bytes(tmp_path, 1)
    )
    encrypted = _encrypt_bytes(tmp_path / f"source{suffix}", plaintext)
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError, match="pixel limit"):
            list(iter_rasterized_pages(encrypted, scratch, max_pixels=10))


@pytest.mark.parametrize(
    ("filename", "plaintext", "message"),
    [
        ("bad.pdf", b"%PDF-not-valid", "malformed"),
        ("bad.tiff", b"II*\x00not-valid", "malformed"),
        ("mismatch.pdf", b"II*\x00not-a-pdf", "does not match"),
        ("mismatch.tiff", b"%PDF-not-a-tiff", "does not match"),
        ("note.rtf", b"{\\\\rtf1 text}", "text-only"),
        ("photo.png", b"\\x89PNG\\r\\n\\x1a\\n", "Unsupported"),
    ],
)
def test_rasterizer_fails_safely_for_malformed_mismatched_and_unsupported_inputs(
    tmp_path: Path,
    filename: str,
    plaintext: bytes,
    message: str,
) -> None:
    encrypted = _encrypt_bytes(tmp_path / filename, plaintext)
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError, match=message):
            list(iter_rasterized_pages(encrypted, scratch, max_pixels=1_000_000))


def test_password_protected_pdf_has_stable_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.rasterizer as rasterizer

    encrypted = _encrypt_bytes(tmp_path / "locked.pdf", b"%PDF-1.7\n")

    def password_error(*_args: object, **_kwargs: object) -> object:
        raise PdfiumError("sensitive library detail", raw.FPDF_ERR_PASSWORD)

    monkeypatch.setattr(rasterizer.pdfium, "PdfDocument", password_error)
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError, match="password-protected") as exc:
            list(iter_rasterized_pages(encrypted, scratch, max_pixels=1_000_000))
    assert "sensitive library detail" not in str(exc.value)


def test_tiff_decompression_bomb_has_stable_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.rasterizer as rasterizer

    encrypted = _encrypt_bytes(tmp_path / "bomb.tiff", b"II*\x00")
    monkeypatch.setattr(
        rasterizer.Image,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            Image.DecompressionBombError("sensitive detail")
        ),
    )
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError, match="decompression limit") as exc:
            list(iter_rasterized_pages(encrypted, scratch, max_pixels=1_000_000))
    assert "sensitive detail" not in str(exc.value)


def test_pdf_document_page_bitmap_and_image_close_on_early_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.rasterizer as rasterizer

    encrypted = _encrypt_bytes(tmp_path / "source.pdf", b"%PDF-1.7\n")
    closed: list[str] = []

    class FakeImage:
        width = 10
        height = 10

        def save(self, target, *, format: str) -> None:
            assert format == "PNG"
            target.write(b"png")

        def close(self) -> None:
            closed.append("image")

    class FakeBitmap:
        def to_pil(self) -> FakeImage:
            return FakeImage()

        def close(self) -> None:
            closed.append("bitmap")

    class FakePage:
        def get_size(self) -> tuple[float, float]:
            return 5.0, 5.0

        def render(self, *, scale: float) -> FakeBitmap:
            assert scale == 2.0
            return FakeBitmap()

        def close(self) -> None:
            closed.append("page")

    class FakeDocument:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> FakePage:
            assert index == 0
            return FakePage()

        def close(self) -> None:
            closed.append("document")

    monkeypatch.setattr(rasterizer.pdfium, "PdfDocument", lambda *_args: FakeDocument())
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        pages = iter_rasterized_pages(encrypted, scratch, max_pixels=1_000)
        page = next(pages)
        assert page.path.exists()
        pages.close()

    assert closed == ["image", "bitmap", "page", "document"]


def test_pdf_preflights_pixel_limit_before_render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.rasterizer as rasterizer

    encrypted = _encrypt_bytes(tmp_path / "source.pdf", b"%PDF-1.7\n")
    closed: list[str] = []

    class HugePage:
        def get_size(self) -> tuple[float, float]:
            return 10_000.0, 10_000.0

        def render(self, *, scale: float) -> object:
            raise AssertionError(f"render called at scale {scale}")

        def close(self) -> None:
            closed.append("page")

    class FakeDocument:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> HugePage:
            assert index == 0
            return HugePage()

        def close(self) -> None:
            closed.append("document")

    monkeypatch.setattr(rasterizer.pdfium, "PdfDocument", lambda *_args: FakeDocument())
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        with pytest.raises(LocalValidationError, match="pixel limit"):
            list(iter_rasterized_pages(encrypted, scratch, max_pixels=1_000))

    assert closed == ["page", "document"]


def test_raster_output_swap_never_writes_through_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.rasterizer as rasterizer

    encrypted = _encrypt_bytes(tmp_path / "source.pdf", b"%PDF-1.7\n")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"KEEP")
    scratch_ref: ScratchJob | None = None

    class SwappingImage:
        width = 10
        height = 10

        def save(self, target, *, format: str) -> None:
            assert format == "PNG"
            assert scratch_ref is not None
            path = scratch_ref.path / "page-0001.png"
            path.rename(scratch_ref.path / "moved-page.png")
            path.symlink_to(outside)
            if hasattr(target, "write"):
                target.write(b"png")
            else:
                Path(target).write_bytes(b"png")

        def close(self) -> None:
            pass

    class FakeBitmap:
        def to_pil(self) -> SwappingImage:
            return SwappingImage()

        def close(self) -> None:
            pass

    class FakePage:
        def get_size(self) -> tuple[float, float]:
            return 5.0, 5.0

        def render(self, *, scale: float) -> FakeBitmap:
            return FakeBitmap()

        def close(self) -> None:
            pass

    class FakeDocument:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            pass

    monkeypatch.setattr(rasterizer.pdfium, "PdfDocument", lambda *_args: FakeDocument())
    with ScratchJob(tmp_path / "scratch", "job-1") as scratch:
        scratch_ref = scratch
        with pytest.raises(LocalValidationError, match="changed unexpectedly"):
            list(iter_rasterized_pages(encrypted, scratch, max_pixels=1_000))

    assert outside.read_bytes() == b"KEEP"
