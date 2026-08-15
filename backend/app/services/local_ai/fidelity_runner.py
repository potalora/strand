"""Real-model fidelity orchestration for the strict-local release corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageFont

from app.services.local_ai.artifact_store import (
    manifest_sha256 as hash_manifest,
    resolve_retained_candidate_pack,
)
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.extraction_schema import (
    FACT_CATEGORY_NAMES,
    NUEXTRACT_TEMPLATE_V1,
    ClinicalDocumentExtraction,
    EvidenceFact,
)
from app.services.local_ai.extraction_validator import validate_clinical_extraction
from app.services.local_ai.fidelity_metrics import (
    FidelityFactObservation,
    FidelityMetrics,
    score_fidelity,
)
from app.services.local_ai.grounded_summary import (
    build_grounded_summary_input,
    normalize_observation_summary_value,
    validate_and_render_summary,
)
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
)
from app.services.local_ai.model_manager import LocalModelManager
from app.services.local_ai.rasterizer import iter_rasterized_pages
from app.services.local_ai.scratch import ScratchJob
from app.services.local_ai.types import ModelRole

_PAGE_SIZE = (1240, 1754)
_SUPPORTED_RENDER_FORMATS = frozenset({"pdf", "tiff"})
_SUPPORTED_RENDER_STYLES = frozenset(
    {
        "clean",
        "table",
        "skew-low-contrast",
        "handwriting",
        "poor-illumination",
        "poor-scan",
    }
)
_DOCUMENT_KEYS = frozenset(
    {
        "id",
        "render",
        "critical_numeric_tokens",
        "expected_facts",
        "forbidden_facts",
    }
)
_RENDER_KEYS = frozenset({"format", "style", "lines"})
_PRIVATE_DOCUMENT_KEYS = frozenset(
    {
        "id",
        "source_file",
        "critical_numeric_tokens",
        "expected_facts",
        "forbidden_facts",
    }
)
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "content_free",
        "fixture_suite_version",
        "fixture_suite_sha256",
        "manifest_sha256",
        "synthetic_documents",
        "private_documents",
        "metrics",
        "private_metrics",
    }
)
_REPORT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_REPORT_BYTES = 16 * 1024
FIDELITY_SUITE_VERSION = "local-ai-fidelity-v1"
FIDELITY_CORPUS_SHA256 = (
    "5654f214d4fb0a404d3abf3572812f209e05b20be0deb379dddd7db663d1a826"
)
RELEASE_SYNTHETIC_DOCUMENT_COUNT = 6


class FidelityManager(Protocol):
    """Contained-worker surface used by the real runner and fake CI gate."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def run_attested(
        self,
        manifest: LocalAIManifest,
        role: ModelRole,
        payload: dict[str, Any],
    ) -> Any: ...


@dataclass(frozen=True)
class FidelityRender:
    """One deterministic synthetic raster-document recipe."""

    format: str
    style: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class FidelityDocument:
    """Ground truth and source recipe for one fidelity document."""

    document_id: str
    render: FidelityRender | None
    source_path: Path | None
    critical_numeric_tokens: tuple[str, ...]
    expected_facts: tuple[dict[str, object], ...]
    forbidden_facts: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class FidelityCorpus:
    """A validated, versioned fidelity corpus."""

    suite_version: str
    corpus_sha256: str
    documents: tuple[FidelityDocument, ...]


@dataclass(frozen=True)
class MaterializedFidelityDocument:
    """One generated image-only PDF or TIFF and its ground truth."""

    document_id: str
    path: Path
    critical_numeric_tokens: tuple[str, ...]
    expected_facts: tuple[dict[str, object], ...]
    forbidden_facts: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class FidelityRunReport:
    """Content-free identity, counts, and hard-gate metrics for one real run."""

    schema_version: int
    content_free: bool
    fixture_suite_version: str
    fixture_suite_sha256: str
    manifest_sha256: str
    synthetic_documents: int
    private_documents: int
    metrics: FidelityMetrics
    private_metrics: FidelityMetrics | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "content_free": self.content_free,
            "fixture_suite_version": self.fixture_suite_version,
            "fixture_suite_sha256": self.fixture_suite_sha256,
            "manifest_sha256": self.manifest_sha256,
            "synthetic_documents": self.synthetic_documents,
            "private_documents": self.private_documents,
            "metrics": self.metrics.as_report(),
            "private_metrics": (
                self.private_metrics.as_report()
                if self.private_metrics is not None
                else None
            ),
        }

    def assert_release_thresholds(self) -> None:
        """Require the committed suite and optional private suite to pass alone."""

        self.metrics.assert_release_thresholds()
        if self.private_metrics is not None:
            self.private_metrics.assert_release_thresholds()


def _document_count(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 10_000
    ):
        raise LocalValidationError("Local fidelity report is invalid.")
    return value


def build_fidelity_report(
    *,
    metrics: FidelityMetrics,
    fixture_suite_version: str,
    fixture_suite_sha256: str,
    manifest_sha256: str,
    synthetic_documents: int,
    private_documents: int,
    private_metrics: FidelityMetrics | None = None,
) -> FidelityRunReport:
    """Build a strictly bounded report that cannot contain document content."""

    if (
        not isinstance(metrics, FidelityMetrics)
        or fixture_suite_version != FIDELITY_SUITE_VERSION
        or fixture_suite_sha256 != FIDELITY_CORPUS_SHA256
        or not isinstance(manifest_sha256, str)
        or _REPORT_SHA256.fullmatch(manifest_sha256) is None
        or (
            private_metrics is not None
            and not isinstance(private_metrics, FidelityMetrics)
        )
    ):
        raise LocalValidationError("Local fidelity report is invalid.")
    synthetic_count = _document_count(synthetic_documents)
    private_count = _document_count(private_documents)
    if synthetic_count <= 0:
        raise LocalValidationError("Local fidelity report is invalid.")
    if (private_count == 0) != (private_metrics is None):
        raise LocalValidationError("Local fidelity report is invalid.")
    return FidelityRunReport(
        schema_version=1,
        content_free=True,
        fixture_suite_version=fixture_suite_version,
        fixture_suite_sha256=fixture_suite_sha256,
        manifest_sha256=manifest_sha256,
        synthetic_documents=synthetic_count,
        private_documents=private_count,
        metrics=metrics,
        private_metrics=private_metrics,
    )


def _parse_report(value: object) -> FidelityRunReport:
    if type(value) is not dict or set(value) != _REPORT_KEYS:
        raise LocalValidationError("Local fidelity report is invalid.")
    if value.get("schema_version") != 1 or value.get("content_free") is not True:
        raise LocalValidationError("Local fidelity report is invalid.")
    metrics = FidelityMetrics.from_report(value.get("metrics"))
    raw_private_metrics = value.get("private_metrics")
    private_metrics = (
        FidelityMetrics.from_report(raw_private_metrics)
        if raw_private_metrics is not None
        else None
    )
    return build_fidelity_report(
        metrics=metrics,
        fixture_suite_version=value.get("fixture_suite_version"),  # type: ignore[arg-type]
        fixture_suite_sha256=value.get("fixture_suite_sha256"),  # type: ignore[arg-type]
        manifest_sha256=value.get("manifest_sha256"),  # type: ignore[arg-type]
        synthetic_documents=value.get("synthetic_documents"),  # type: ignore[arg-type]
        private_documents=value.get("private_documents"),  # type: ignore[arg-type]
        private_metrics=private_metrics,
    )


def parse_fidelity_report(value: object) -> FidelityRunReport:
    """Validate one already-decoded, content-free fidelity report."""

    return _parse_report(value)


def parse_fidelity_report_bytes(encoded: bytes) -> FidelityRunReport:
    """Parse the exact bounded bytes later bound into release evidence."""

    if (
        not isinstance(encoded, bytes)
        or not encoded
        or len(encoded) > _MAX_REPORT_BYTES
    ):
        raise LocalValidationError("Local fidelity report is unavailable.")
    try:
        raw = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        raise LocalValidationError("Local fidelity report is unavailable.") from None
    return _parse_report(raw)


def _report_parent(path: Path) -> Path:
    parent = path.parent
    try:
        if parent.is_symlink():
            raise LocalValidationError("Local fidelity report path is unsafe.")
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = parent.resolve(strict=True)
        if resolved != parent.absolute() or not parent.is_dir():
            raise LocalValidationError("Local fidelity report path is unsafe.")
    except LocalValidationError:
        raise
    except OSError:
        raise LocalValidationError(
            "Local fidelity report path is unavailable."
        ) from None
    return parent


def write_fidelity_report(path: Path | str, report: FidelityRunReport) -> None:
    """Atomically persist only the allowlisted content-free report fields."""

    if not isinstance(report, FidelityRunReport):
        raise LocalValidationError("Local fidelity report is invalid.")
    verified = _parse_report(report.as_dict())
    try:
        encoded = json.dumps(
            verified.as_dict(),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError):
        raise LocalValidationError("Local fidelity report is invalid.") from None
    if not encoded or len(encoded) > _MAX_REPORT_BYTES:
        raise LocalValidationError("Local fidelity report is invalid.")

    target = Path(path)
    parent = _report_parent(target)
    temporary = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        raise LocalValidationError(
            "Local fidelity report could not be written."
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def load_fidelity_report(path: Path | str) -> FidelityRunReport:
    """Load a bounded report with no tolerance for fields that could carry PHI."""

    target = Path(path)
    try:
        metadata = target.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_REPORT_BYTES
        ):
            raise LocalValidationError("Local fidelity report is unavailable.")
        encoded = target.read_bytes()
    except LocalValidationError:
        raise
    except OSError:
        raise LocalValidationError("Local fidelity report is unavailable.") from None
    return parse_fidelity_report_bytes(encoded)


def _safe_line(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2_000
        or any(ord(character) < 32 for character in value)
    ):
        raise LocalValidationError("Local fidelity corpus is invalid.")
    return value


def _safe_fact(value: object) -> dict[str, object]:
    if type(value) is not dict or not value:
        raise LocalValidationError("Local fidelity corpus is invalid.")
    fact: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise LocalValidationError("Local fidelity corpus is invalid.")
        if isinstance(item, str):
            fact[key] = _safe_line(item)
        elif type(item) in {int, float, bool} or item is None:
            fact[key] = item
        else:
            raise LocalValidationError("Local fidelity corpus is invalid.")
    category = fact.get("category")
    if not isinstance(category, str) or not category:
        raise LocalValidationError("Local fidelity corpus is invalid.")
    return fact


def load_fidelity_corpus(path: Path | str) -> FidelityCorpus:
    """Load the committed synthetic corpus without accepting loose structure."""

    raw, corpus_sha256 = _read_corpus_json(Path(path))
    if (
        type(raw) is not dict
        or set(raw) != {"schema_version", "suite_version", "documents"}
        or raw.get("schema_version") != 1
        or raw.get("suite_version") != FIDELITY_SUITE_VERSION
        or not isinstance(raw.get("documents"), list)
        or not raw["documents"]
    ):
        raise LocalValidationError("Local fidelity corpus is invalid.")

    documents: list[FidelityDocument] = []
    seen_ids: set[str] = set()
    for raw_document in raw["documents"]:
        if type(raw_document) is not dict or set(raw_document) != _DOCUMENT_KEYS:
            raise LocalValidationError("Local fidelity corpus is invalid.")
        document_id = raw_document.get("id")
        if (
            not isinstance(document_id, str)
            or not document_id
            or len(document_id) > 128
            or not document_id.replace("-", "").replace("_", "").isalnum()
            or document_id in seen_ids
        ):
            raise LocalValidationError("Local fidelity corpus is invalid.")
        seen_ids.add(document_id)

        raw_render = raw_document.get("render")
        if type(raw_render) is not dict or set(raw_render) != _RENDER_KEYS:
            raise LocalValidationError("Local fidelity corpus is invalid.")
        render_format = raw_render.get("format")
        style = raw_render.get("style")
        lines = raw_render.get("lines")
        if (
            render_format not in _SUPPORTED_RENDER_FORMATS
            or style not in _SUPPORTED_RENDER_STYLES
            or not isinstance(lines, list)
            or not lines
            or len(lines) > 100
        ):
            raise LocalValidationError("Local fidelity corpus is invalid.")

        tokens = raw_document.get("critical_numeric_tokens")
        expected = raw_document.get("expected_facts")
        forbidden = raw_document.get("forbidden_facts")
        if (
            not isinstance(tokens, list)
            or not tokens
            or not isinstance(expected, list)
            or not expected
            or not isinstance(forbidden, list)
        ):
            raise LocalValidationError("Local fidelity corpus is invalid.")
        documents.append(
            FidelityDocument(
                document_id=document_id,
                render=FidelityRender(
                    format=str(render_format),
                    style=str(style),
                    lines=tuple(_safe_line(line) for line in lines),
                ),
                source_path=None,
                critical_numeric_tokens=tuple(_safe_line(token) for token in tokens),
                expected_facts=tuple(_safe_fact(fact) for fact in expected),
                forbidden_facts=tuple(_safe_fact(fact) for fact in forbidden),
            )
        )
    return FidelityCorpus(
        suite_version=FIDELITY_SUITE_VERSION,
        corpus_sha256=corpus_sha256,
        documents=tuple(documents),
    )


def _read_corpus_json(path: Path) -> tuple[dict[str, object], str]:
    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > 4 * 1024 * 1024
        ):
            raise LocalValidationError("Local fidelity corpus is unavailable.")
        encoded = path.read_bytes()
        if len(encoded) != metadata.st_size:
            raise LocalValidationError("Local fidelity corpus is unavailable.")
        raw = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except LocalValidationError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        raise LocalValidationError("Local fidelity corpus is unavailable.") from None
    if type(raw) is not dict:
        raise LocalValidationError("Local fidelity corpus is invalid.")
    return raw, hashlib.sha256(encoded).hexdigest()


def _private_source(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise LocalValidationError("Local fidelity corpus is invalid.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or str(relative) != value
        or relative.suffix.lower() not in {".pdf", ".tif", ".tiff"}
    ):
        raise LocalValidationError("Local fidelity corpus is invalid.")
    current = root
    try:
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise LocalValidationError("Local fidelity corpus is invalid.")
        source = current.resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            raise LocalValidationError("Local fidelity corpus is invalid.")
        metadata = source.stat()
    except LocalValidationError:
        raise
    except OSError:
        raise LocalValidationError("Local fidelity corpus is invalid.") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise LocalValidationError("Local fidelity corpus is invalid.")
    return source


def load_private_fidelity_corpus(directory: Path | str) -> FidelityCorpus:
    """Load owner-supplied sources only when an explicit golden sidecar exists."""

    root = Path(directory).absolute()
    try:
        metadata = root.lstat()
    except OSError:
        raise LocalValidationError("Local fidelity corpus is unavailable.") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise LocalValidationError("Local fidelity corpus is unavailable.")
    raw, corpus_sha256 = _read_corpus_json(root / "local-ai-corpus-v1.json")
    if (
        set(raw) != {"schema_version", "suite_version", "documents"}
        or raw.get("schema_version") != 1
        or raw.get("suite_version") != FIDELITY_SUITE_VERSION
        or not isinstance(raw.get("documents"), list)
        or not raw["documents"]
    ):
        raise LocalValidationError("Local fidelity corpus is invalid.")

    seen_ids: set[str] = set()
    documents: list[FidelityDocument] = []
    for raw_document in raw["documents"]:
        if (
            type(raw_document) is not dict
            or set(raw_document) != _PRIVATE_DOCUMENT_KEYS
        ):
            raise LocalValidationError("Local fidelity corpus is invalid.")
        document_id = raw_document.get("id")
        tokens = raw_document.get("critical_numeric_tokens")
        expected = raw_document.get("expected_facts")
        forbidden = raw_document.get("forbidden_facts")
        if (
            not isinstance(document_id, str)
            or not document_id
            or len(document_id) > 128
            or not document_id.replace("-", "").replace("_", "").isalnum()
            or document_id in seen_ids
            or not isinstance(tokens, list)
            or not tokens
            or not isinstance(expected, list)
            or not expected
            or not isinstance(forbidden, list)
        ):
            raise LocalValidationError("Local fidelity corpus is invalid.")
        seen_ids.add(document_id)
        documents.append(
            FidelityDocument(
                document_id=document_id,
                render=None,
                source_path=_private_source(root, raw_document.get("source_file")),
                critical_numeric_tokens=tuple(_safe_line(token) for token in tokens),
                expected_facts=tuple(_safe_fact(fact) for fact in expected),
                forbidden_facts=tuple(_safe_fact(fact) for fact in forbidden),
            )
        )
    return FidelityCorpus(
        suite_version=FIDELITY_SUITE_VERSION,
        corpus_sha256=corpus_sha256,
        documents=tuple(documents),
    )


def validate_release_fidelity_corpus(corpus: FidelityCorpus) -> None:
    """Bind the release gate to the committed synthetic corpus bytes."""

    if (
        not isinstance(corpus, FidelityCorpus)
        or corpus.suite_version != FIDELITY_SUITE_VERSION
        or corpus.corpus_sha256 != FIDELITY_CORPUS_SHA256
    ):
        raise LocalValidationError("Local release fidelity corpus is invalid.")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_lines(image: Image.Image, render: FidelityRender) -> None:
    draw = ImageDraw.Draw(image)
    if render.style == "table":
        font = _font(32)
        left = 90
        top = 130
        row_height = 76
        widths = (430, 310, 310)
        for row_index, line in enumerate(render.lines):
            cells = tuple(cell.strip() for cell in line.split("|"))
            if len(cells) == 1:
                draw.text(
                    (left, top + row_index * row_height), cells[0], fill=20, font=font
                )
                continue
            if len(cells) != 3:
                raise LocalValidationError("Local fidelity corpus is invalid.")
            row_top = top + row_index * row_height
            x = left
            for cell, width in zip(cells, widths, strict=True):
                draw.rectangle(
                    (x, row_top, x + width, row_top + row_height),
                    outline=40,
                    width=2,
                )
                draw.text((x + 14, row_top + 17), cell, fill=20, font=font)
                x += width
        return

    font = _font(34)
    fill = 78 if render.style == "skew-low-contrast" else 18
    if render.style == "handwriting":
        for line_index, line in enumerate(render.lines):
            x = 100
            y = 130 + line_index * 82
            for character_index, character in enumerate(line):
                draw.text(
                    (x, y + (character_index % 3) - 1),
                    character,
                    fill=fill,
                    font=font,
                    stroke_width=1,
                )
                x += int(draw.textlength(character, font=font))
        return
    for index, line in enumerate(render.lines):
        draw.text((100, 130 + index * 82), line, fill=fill, font=font)


def _render_image(render: FidelityRender) -> Image.Image:
    background = 238 if render.style == "skew-low-contrast" else 255
    image = Image.new("L", _PAGE_SIZE, background)
    _draw_lines(image, render)
    if render.style == "poor-illumination":
        shading = Image.new("L", _PAGE_SIZE, 0)
        shading_draw = ImageDraw.Draw(shading)
        for y in range(0, _PAGE_SIZE[1], 16):
            shade = 15 + (y * 65 // _PAGE_SIZE[1])
            shading_draw.rectangle((0, y, _PAGE_SIZE[0], y + 15), fill=shade)
        shaded = Image.blend(image, shading, 0.22)
        image.close()
        return shaded
    if render.style == "poor-scan":
        reduced = image.resize((620, 877), Image.Resampling.BILINEAR)
        image.close()
        scanned = reduced.resize(_PAGE_SIZE, Image.Resampling.NEAREST)
        scan_draw = ImageDraw.Draw(scanned)
        for y in range(0, _PAGE_SIZE[1], 29):
            scan_draw.line((0, y, _PAGE_SIZE[0], y), fill=220, width=1)
        return scanned
    if render.style != "skew-low-contrast":
        return image
    rotated = image.rotate(
        1.35,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=background,
    )
    image.close()
    return rotated


def _save_rendered(image: Image.Image, target: Path, render_format: str) -> None:
    fixed_timestamp = time.gmtime(0)
    if render_format == "pdf":
        converted = image.convert("RGB")
        try:
            converted.save(
                target,
                format="PDF",
                resolution=150.0,
                title="MedTimeline local fidelity fixture",
                author="MedTimeline",
                creator="MedTimeline",
                producer="Pillow",
                creationDate=fixed_timestamp,
                modDate=fixed_timestamp,
            )
        finally:
            converted.close()
    else:
        image.save(
            target,
            format="TIFF",
            compression="tiff_deflate",
            dpi=(150.0, 150.0),
        )
    os.chmod(target, 0o600)
    metadata = target.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise LocalValidationError("Local fidelity fixture could not be secured.")


def materialize_fidelity_documents(
    corpus: FidelityCorpus,
    output_dir: Path | str,
) -> tuple[MaterializedFidelityDocument, ...]:
    """Render each synthetic source as an image-only PDF or TIFF."""

    root = Path(output_dir)
    try:
        if root.is_symlink():
            raise LocalValidationError("Local fidelity fixture directory is unsafe.")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(root, 0o700)
    except FileExistsError:
        raise LocalValidationError(
            "Local fidelity fixture directory already exists."
        ) from None
    except OSError:
        raise LocalValidationError(
            "Local fidelity fixture directory is unavailable."
        ) from None

    materialized: list[MaterializedFidelityDocument] = []
    for document in corpus.documents:
        if document.render is None or document.source_path is not None:
            raise LocalValidationError("Local synthetic fidelity corpus is invalid.")
        suffix = ".pdf" if document.render.format == "pdf" else ".tiff"
        target = root / f"{document.document_id}{suffix}"
        image = _render_image(document.render)
        try:
            _save_rendered(image, target, document.render.format)
        except (OSError, ValueError):
            raise LocalValidationError(
                "Local fidelity fixture could not be rendered."
            ) from None
        finally:
            image.close()
        materialized.append(
            MaterializedFidelityDocument(
                document_id=document.document_id,
                path=target,
                critical_numeric_tokens=document.critical_numeric_tokens,
                expected_facts=document.expected_facts,
                forbidden_facts=document.forbidden_facts,
            )
        )
    return tuple(materialized)


def _private_documents(
    corpus: FidelityCorpus | None,
) -> tuple[MaterializedFidelityDocument, ...]:
    if corpus is None:
        return ()
    documents: list[MaterializedFidelityDocument] = []
    for document in corpus.documents:
        if document.render is not None or document.source_path is None:
            raise LocalValidationError("Local private fidelity corpus is invalid.")
        documents.append(
            MaterializedFidelityDocument(
                document_id=document.document_id,
                path=document.source_path,
                critical_numeric_tokens=document.critical_numeric_tokens,
                expected_facts=document.expected_facts,
                forbidden_facts=document.forbidden_facts,
            )
        )
    return tuple(documents)


def _artifact(manifest: LocalAIManifest, role: ModelRole) -> ManifestArtifact:
    try:
        return next(item for item in manifest.artifacts if item.role is role)
    except StopIteration:
        raise LocalValidationError("Local fidelity manifest is invalid.") from None


def _manifest_identity(
    manifest: LocalAIManifest,
    role: ModelRole,
) -> dict[str, object]:
    artifact = _artifact(manifest, role)
    return {
        "schema_version": manifest.schema_version,
        "pack_revision": manifest.pack_revision,
        "platform": manifest.platform,
        "runtime": dict(manifest.runtime),
        "validation_suite_version": manifest.validation_suite_version,
        "roles": sorted(item.value for item in ModelRole),
        "role": role.value,
        "repository": artifact.repository,
        "revision": artifact.revision,
        "quantization": artifact.quantization,
        "license": artifact.license,
        "attribution": artifact.attribution,
        "manifest_sha256": hash_manifest(manifest),
    }


def _transport(
    *,
    manifest: LocalAIManifest,
    role: ModelRole,
    manifest_path: Path,
    model_dir: Path,
    scratch: ScratchJob,
    job_id: str,
) -> dict[str, object]:
    artifact = _artifact(manifest, role)
    payload: dict[str, object] = {
        "job_id": job_id,
        "manifest_path": str(manifest_path),
        "manifest_identity": _manifest_identity(manifest, role),
        "model_dir": str(model_dir),
        "max_output_tokens": artifact.decode_limits["max_output_tokens"],
    }
    if role is not ModelRole.SUMMARY:
        payload["scratch_dir"] = str(scratch.path.absolute())
    return payload


def _validated_ocr_result(value: object, *, page_number: int) -> str:
    if (
        type(value) is not dict
        or set(value) != {"markdown", "page_number"}
        or value.get("page_number") != page_number
        or not isinstance(value.get("markdown"), str)
        or not value["markdown"].strip()
    ):
        raise LocalValidationError("Local fidelity OCR output is invalid.")
    return value["markdown"]


def _enum_value(value: object) -> object:
    return value.value if hasattr(value, "value") else value


def _summary_content(category: str, fact: EvidenceFact) -> dict[str, object]:
    fields_by_category: dict[str, tuple[str, tuple[str, ...]]] = {
        "medications": (
            "medication",
            (
                "name",
                "dose_value",
                "dose_unit",
                "route",
                "frequency",
                "status",
                "date",
            ),
        ),
        "conditions": (
            "condition",
            ("name", "assertion", "relationship", "date"),
        ),
        "procedures": (
            "procedure",
            ("name", "assertion", "date", "provider"),
        ),
        "labs": (
            "observation",
            (
                "name",
                "value",
                "unit",
                "reference_range",
                "interpretation",
                "date",
            ),
        ),
        "allergies": (
            "allergy",
            (
                "substance",
                "reaction",
                "severity",
                "status",
                "date",
                "assertion",
            ),
        ),
        "encounters": (
            "encounter",
            ("name", "visit_type", "date", "provider", "facility", "status"),
        ),
        "immunizations": (
            "immunization",
            (
                "name",
                "date",
                "status",
                "route",
                "site",
                "dose",
                "manufacturer",
                "lot",
            ),
        ),
        "vital_signs": (
            "observation",
            ("name", "value", "unit", "date"),
        ),
        "diagnostic_reports": (
            "diagnostic_report",
            (
                "name",
                "findings",
                "interpretation",
                "date",
                "category",
                "performer",
                "status",
                "assertion",
            ),
        ),
        "care_plans": (
            "care_plan",
            ("title", "plan_items", "status", "date"),
        ),
    }
    try:
        record_type, names = fields_by_category[category]
    except KeyError:
        raise LocalValidationError("Local fidelity extraction is invalid.") from None
    content: dict[str, object] = {"record_type": record_type}
    for name in names:
        value = getattr(fact, name, None)
        if value is not None and value != []:
            content[name] = _enum_value(value)
    if (
        category == "conditions"
        and content.get("assertion") == "family_history"
        and "relationship" not in content
    ):
        content["relationship"] = "family"
    if category in {"labs", "vital_signs"}:
        content["category"] = "lab" if category == "labs" else "vital_signs"
        if "value" in content:
            try:
                summary_value, summary_unit = normalize_observation_summary_value(
                    content["value"],
                    content.get("unit"),
                )
            except ValueError:
                raise LocalValidationError(
                    "Local fidelity extraction has an invalid observation value."
                ) from None
            content["value"] = summary_value
            if summary_unit is None:
                content.pop("unit", None)
            else:
                content["unit"] = summary_unit
    return content


def _scored_fact(category: str, fact: EvidenceFact) -> dict[str, object]:
    value = fact.model_dump(
        mode="json",
        exclude={
            "fact_id",
            "verbatim",
            "page_number",
            "evidence_excerpt",
            "confidence",
            "normalized_value",
            "normalization_method",
            "normalization_version",
        },
        exclude_none=True,
    )
    if category == "medications":
        value["value"] = value.get("dose_value")
        value["unit"] = value.get("dose_unit")
    return {"category": category, **value}


def _leaf_paths(value: Mapping[str, object]) -> tuple[str, ...]:
    paths: list[str] = []

    def visit(item: object, path: str) -> None:
        if type(item) is dict and item:
            for key in sorted(item):
                visit(item[key], f"{path}/{key}")
            return
        if type(item) is list and item:
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
            return
        paths.append(path)

    for key in sorted(value):
        visit(value[key], f"/{key}")
    return tuple(paths)


def _summary_candidates(
    extractions: list[ClinicalDocumentExtraction],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[list[dict[str, object]]],
]:
    facts: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    scored: list[dict[str, object]] = []
    scored_groups: list[list[dict[str, object]]] = []
    record_number = 0
    for extraction in extractions:
        scored_group: list[dict[str, object]] = []
        for category in FACT_CATEGORY_NAMES:
            for fact in getattr(extraction, category):
                if fact.evidence_id is None:
                    raise LocalValidationError(
                        "Local fidelity extraction lacks evidence."
                    )
                record_number += 1
                content = _summary_content(category, fact)
                paths = _leaf_paths(content)
                facts.append(
                    {
                        "record_id": f"fidelity-record-{record_number}",
                        "content": content,
                        "evidence_ids": [fact.evidence_id],
                    }
                )
                evidence.append(
                    {
                        "id": fact.evidence_id,
                        "excerpt": fact.evidence_excerpt,
                        "page_number": fact.page_number,
                        "section": category,
                        "field_paths": list(paths),
                    }
                )
                scored_fact = _scored_fact(category, fact)
                scored_fact["_summary_required_field_paths"] = [
                    path for path in paths if path.startswith("/value/")
                ]
                scored.append(scored_fact)
                scored_group.append(scored_fact)
        scored_groups.append(scored_group)
    return facts, evidence, scored, scored_groups


def _summary_claims(
    value: object,
) -> list[dict[str, object]]:
    claims: list[dict[str, object]] = []
    sections = getattr(value, "sections", None)
    if not isinstance(sections, tuple):
        raise LocalValidationError("Local fidelity summary output is invalid.")
    for section in sections:
        for claim in section.claims:
            claims.append(
                {
                    "fact_id": claim.fact_id,
                    "evidence_ids": list(claim.evidence_ids),
                    "field_paths": list(claim.field_paths),
                }
            )
    return claims


async def run_fidelity_suite(
    *,
    manifest: LocalAIManifest,
    manifest_path: Path | str,
    model_dir: Path | str,
    corpus: FidelityCorpus,
    scratch_root: Path | str,
    manager: FidelityManager | None = None,
    private_corpus: FidelityCorpus | None = None,
) -> FidelityRunReport:
    """Run generated raster documents through locked OCR, extraction, and summary."""

    if (
        not isinstance(manifest, LocalAIManifest)
        or not isinstance(corpus, FidelityCorpus)
        or (
            private_corpus is not None
            and (
                not isinstance(private_corpus, FidelityCorpus)
                or private_corpus.suite_version != corpus.suite_version
            )
        )
    ):
        raise LocalValidationError("Local fidelity configuration is invalid.")
    validate_release_fidelity_corpus(corpus)
    locked_manifest = Path(manifest_path).absolute()
    pack_path = Path(model_dir).absolute()
    try:
        manifest_info = locked_manifest.lstat()
        pack_info = pack_path.lstat()
    except OSError:
        raise LocalValidationError(
            "Local fidelity model pack is unavailable."
        ) from None
    if (
        stat.S_ISLNK(manifest_info.st_mode)
        or not stat.S_ISREG(manifest_info.st_mode)
        or manifest_info.st_nlink != 1
        or stat.S_ISLNK(pack_info.st_mode)
        or not stat.S_ISDIR(pack_info.st_mode)
    ):
        raise LocalValidationError("Local fidelity model pack is unavailable.")

    root = Path(scratch_root)
    run_id = uuid.uuid4().hex
    fixture_root = root / f"fixtures-{run_id}"
    synthetic_documents = materialize_fidelity_documents(corpus, fixture_root)
    private_documents = _private_documents(private_corpus)
    documents = (*synthetic_documents, *private_documents)
    selected_manager: FidelityManager = manager or LocalModelManager()
    started = False
    numeric_groups: list[list[tuple[str, str]]] = []
    extractions: list[ClinicalDocumentExtraction] = []
    schema_groups: list[list[bool]] = []
    try:
        await selected_manager.start()
        started = True
        for document_number, document in enumerate(documents, start=1):
            pages: dict[int, str] = {}
            document_schema_results: list[bool] = []
            job_name = f"fidelity-{run_id}-{document_number}"
            with ScratchJob(root / "jobs", job_name) as scratch:
                rasterized = iter_rasterized_pages(
                    document.path,
                    scratch,
                    max_pixels=40_000_000,
                )
                try:
                    for page in rasterized:
                        role = ModelRole.OCR
                        value = await selected_manager.run_attested(
                            manifest,
                            role,
                            {
                                **_transport(
                                    manifest=manifest,
                                    role=role,
                                    manifest_path=locked_manifest,
                                    model_dir=pack_path,
                                    scratch=scratch,
                                    job_id=(f"{job_name}-ocr-{page.page_number}"),
                                ),
                                "page_number": page.page_number,
                                "image_path": str(page.path.absolute()),
                                "image_sha256": page.sha256,
                            },
                        )
                        markdown = _validated_ocr_result(
                            value,
                            page_number=page.page_number,
                        )
                        pages[page.page_number] = markdown
                        document_schema_results.append(True)
                finally:
                    rasterized.close()

                role = ModelRole.EXTRACTION
                value = await selected_manager.run_attested(
                    manifest,
                    role,
                    {
                        **_transport(
                            manifest=manifest,
                            role=role,
                            manifest_path=locked_manifest,
                            model_dir=pack_path,
                            scratch=scratch,
                            job_id=f"{job_name}-extraction",
                        ),
                        "page_markdown": [
                            {"page_number": page, "markdown": pages[page]}
                            for page in sorted(pages)
                        ],
                        "image_paths": {},
                        "schema": NUEXTRACT_TEMPLATE_V1,
                    },
                )
                extraction = validate_clinical_extraction(
                    value,
                    pages=pages,
                    upload_id=f"fidelity-{document_number}",
                    strict_local=True,
                )
                extractions.append(extraction)
                document_schema_results.append(True)
            document_ocr = "\n".join(pages[page] for page in sorted(pages))
            numeric_groups.append(
                [(token, document_ocr) for token in document.critical_numeric_tokens]
            )
            schema_groups.append(document_schema_results)
        summary_facts, summary_evidence, scored_facts, scored_groups = (
            _summary_candidates(extractions)
        )
        summary_input = build_grounded_summary_input(
            facts=summary_facts,
            evidence=summary_evidence,
            requested_scope={"summary_type": "full_health"},
        )
        summary_job_name = f"fidelity-{run_id}-summary"
        with ScratchJob(root / "jobs", summary_job_name) as scratch:
            role = ModelRole.SUMMARY
            summary_value = await selected_manager.run_attested(
                manifest,
                role,
                {
                    **summary_input.model_dump(mode="json"),
                    **_transport(
                        manifest=manifest,
                        role=role,
                        manifest_path=locked_manifest,
                        model_dir=pack_path,
                        scratch=scratch,
                        job_id=summary_job_name,
                    ),
                },
            )
        rendered = validate_and_render_summary(
            summary_value,
            facts={item.fact_id: item for item in summary_input.facts},
            evidence={item.evidence_id: item for item in summary_input.evidence},
            uncertainties={
                item.uncertainty_id: item for item in summary_input.uncertainty_labels
            },
        )
        for index, fact in enumerate(summary_input.facts):
            scored_facts[index]["fact_id"] = fact.fact_id
            scored_facts[index]["evidence_ids"] = list(fact.evidence_ids)
        fact_observations = [
            FidelityFactObservation(
                expected=document.expected_facts,
                forbidden=document.forbidden_facts,
                accepted=tuple(accepted),
            )
            for document, accepted in zip(
                documents,
                scored_groups,
                strict=True,
            )
        ]
        summary_claims = _summary_claims(rendered.selection_document)

        def score_partition(start: int, end: int) -> FidelityMetrics:
            observations = fact_observations[start:end]
            accepted_ids = {
                fact_id
                for observation in observations
                for accepted in observation.accepted
                if isinstance((fact_id := accepted.get("fact_id")), str)
            }
            return score_fidelity(
                numeric_observations=[
                    item for group in numeric_groups[start:end] for item in group
                ],
                fact_observations=observations,
                summary_claims=[
                    claim
                    for claim in summary_claims
                    if claim.get("fact_id") in accepted_ids
                ],
                schema_results=[
                    item for group in schema_groups[start:end] for item in group
                ]
                + [True],
            )

        synthetic_count = len(synthetic_documents)
        metrics = score_partition(0, synthetic_count)
        private_metrics = (
            score_partition(synthetic_count, len(documents))
            if private_documents
            else None
        )
        return build_fidelity_report(
            metrics=metrics,
            fixture_suite_version=corpus.suite_version,
            fixture_suite_sha256=corpus.corpus_sha256,
            manifest_sha256=hash_manifest(manifest),
            synthetic_documents=len(synthetic_documents),
            private_documents=len(private_documents),
            private_metrics=private_metrics,
        )
    finally:
        try:
            if started:
                await selected_manager.stop()
        finally:
            try:
                shutil.rmtree(fixture_root)
            except OSError:
                pass


def resolve_installed_fidelity_pack(
    *,
    manifest_path: Path | str,
    model_root: Path | str,
) -> tuple[LocalAIManifest, Path, Path]:
    """Resolve a v2 lock over one hash-verified retained artifact tree."""

    try:
        candidate = resolve_retained_candidate_pack(
            manifest_path=manifest_path,
            model_root=model_root,
        )
    except (OSError, LocalValidationError):
        raise LocalValidationError(
            "Validated local fidelity model pack is unavailable."
        ) from None
    return candidate.manifest, candidate.manifest_path, candidate.pack_path


async def run_installed_fidelity_suite(
    *,
    corpus_path: Path | str,
    manifest_path: Path | str,
    model_root: Path | str,
    scratch_root: Path | str,
    private_fixtures_dir: Path | str | None = None,
) -> FidelityRunReport:
    """Run the v2 candidate over exact retained bytes with optional private goldens."""

    try:
        candidate = resolve_retained_candidate_pack(
            manifest_path=manifest_path,
            model_root=model_root,
        )
    except (OSError, LocalValidationError):
        raise LocalValidationError(
            "Validated local fidelity model pack is unavailable."
        ) from None
    private_corpus = (
        load_private_fidelity_corpus(private_fixtures_dir)
        if private_fixtures_dir is not None
        else None
    )
    report = await run_fidelity_suite(
        manifest=candidate.manifest,
        manifest_path=candidate.manifest_path,
        model_dir=candidate.pack_path,
        corpus=load_fidelity_corpus(corpus_path),
        private_corpus=private_corpus,
        scratch_root=scratch_root,
    )
    try:
        candidate.revalidate()
    except LocalValidationError:
        raise LocalValidationError(
            "Validated local fidelity model pack is unavailable."
        ) from None
    return report
