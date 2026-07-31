"""Offline runtime and synthetic fixture validation for a staged Apple pack."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw

from app.config import settings
from app.services.local_ai.artifact_store import manifest_sha256
from app.services.local_ai.errors import LocalAIError, LocalValidationError
from app.services.local_ai.extraction_schema import NUEXTRACT_TEMPLATE_V1
from app.services.local_ai.extraction_validator import validate_clinical_extraction
from app.services.local_ai.grounded_summary import (
    SERVER_MEDICAL_DISCLAIMER,
    build_grounded_summary_input,
    validate_and_render_summary,
)
from app.services.local_ai.manifest import LocalAIManifest, ManifestArtifact
from app.services.local_ai.model_manager import LocalModelManager
from app.services.local_ai.pack_operations import platform_profile
from app.services.local_ai.types import ModelRole
from app.services.local_ai.validation_receipt import (
    RuntimeValidationReceipt,
    _issue_runtime_validation_receipt,
)

_SUPPORTED_RUNTIME = {"name": "mlx-vlm", "version": "0.5.0"}
_SUPPORTED_FIXTURE_SUITE = "local-ai-fixtures-v1"
_FIXTURE_TEXT = "Hemoglobin A1c 6.8 %"


class _Manager(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def run(self, role: ModelRole, payload: dict[str, Any]) -> Any: ...


def _artifact(manifest: LocalAIManifest, role: ModelRole) -> ManifestArtifact:
    try:
        return next(item for item in manifest.artifacts if item.role is role)
    except StopIteration:
        raise LocalValidationError("Local model validation fixture failed") from None


def _identity(
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
        "manifest_sha256": manifest_sha256(manifest),
    }


def _assert_candidate_path(pack_path: Path) -> Path:
    try:
        absolute = pack_path.resolve(strict=True)
        metadata = pack_path.lstat()
    except OSError as exc:
        raise LocalValidationError(
            "Local model validation candidate is unavailable"
        ) from exc
    if (
        pack_path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or absolute != pack_path.absolute()
    ):
        raise LocalValidationError("Local model validation candidate is unavailable")
    return absolute


def _make_scratch(root: Path) -> Path:
    if root.is_symlink():
        raise LocalValidationError("Local model validation scratch is unavailable")
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        resolved_root = root.resolve(strict=True)
        path = resolved_root / f"pack-validation-{uuid.uuid4().hex}"
        path.mkdir(mode=0o700)
        path.chmod(0o700)
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise LocalValidationError(
            "Local model validation scratch is unavailable"
        ) from exc
    return resolved_path


def _write_manifest(path: Path, manifest: LocalAIManifest) -> Path:
    target = path / "manifest.lock.json"
    descriptor = -1
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            descriptor = -1
            json.dump(
                asdict(manifest),
                stream,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError) as exc:
        raise LocalValidationError("Local model validation fixture failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return target


def _write_fixture_image(path: Path) -> tuple[Path, str]:
    target = path / "fixture.png"
    try:
        image = Image.new("RGB", (640, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.text((32, 64), _FIXTURE_TEXT, fill="black")
        image.save(target, format="PNG")
        target.chmod(0o600)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise LocalValidationError("Local model validation fixture failed") from exc
    return target, digest


def _transport(
    *,
    manifest: LocalAIManifest,
    role: ModelRole,
    manifest_path: Path,
    pack_path: Path,
    scratch: Path,
) -> dict[str, object]:
    artifact = _artifact(manifest, role)
    payload: dict[str, object] = {
        "job_id": f"pack-validation-{role.value}",
        "manifest_path": str(manifest_path),
        "manifest_identity": _identity(manifest, role),
        "model_dir": str(pack_path),
        "max_output_tokens": artifact.decode_limits["max_output_tokens"],
    }
    if role is not ModelRole.SUMMARY:
        payload["scratch_dir"] = str(scratch)
    return payload


def _validate_ocr(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {"markdown", "page_number"}:
        raise LocalValidationError("Local model validation fixture failed")
    markdown = value.get("markdown")
    if (
        value.get("page_number") != 1
        or not isinstance(markdown, str)
        or "6.8" not in markdown
        or "%" not in markdown
    ):
        raise LocalValidationError("Local model validation fixture failed")
    return markdown


async def verify_pack_candidate(
    manifest: LocalAIManifest,
    pack_path: Path,
    *,
    manager: _Manager | None = None,
    scratch_root: Path | None = None,
) -> RuntimeValidationReceipt:
    """Load all roles offline and pass the bounded synthetic fixture chain."""

    platform_name, compatible = platform_profile()
    if (
        platform_name != "apple_silicon"
        or not compatible
        or manifest.platform != "apple_silicon"
    ):
        raise LocalValidationError("Local model validation platform is incompatible")
    if (
        manifest.runtime != _SUPPORTED_RUNTIME
        or manifest.validation_suite_version != _SUPPORTED_FIXTURE_SUITE
    ):
        raise LocalValidationError("Local model validation runtime is incompatible")
    candidate = _assert_candidate_path(Path(pack_path))
    scratch = _make_scratch(
        Path(scratch_root)
        if scratch_root is not None
        else Path(settings.local_ai_scratch_dir) / "pack-validation"
    )
    selected_manager: _Manager = manager or LocalModelManager()
    started = False
    try:
        manifest_path = _write_manifest(scratch, manifest)
        image_path, image_sha256 = _write_fixture_image(scratch)
        await selected_manager.start()
        started = True

        ocr_artifact = _artifact(manifest, ModelRole.OCR)
        ocr_value = await selected_manager.run(
            ModelRole.OCR,
            {
                **_transport(
                    manifest=manifest,
                    role=ModelRole.OCR,
                    manifest_path=manifest_path,
                    pack_path=candidate,
                    scratch=scratch,
                ),
                "page_number": 1,
                "image_path": str(image_path),
                "image_sha256": image_sha256,
                "max_output_tokens": ocr_artifact.decode_limits["max_output_tokens"],
            },
        )
        markdown = _validate_ocr(ocr_value)

        extraction_artifact = _artifact(manifest, ModelRole.EXTRACTION)
        extraction_value = await selected_manager.run(
            ModelRole.EXTRACTION,
            {
                **_transport(
                    manifest=manifest,
                    role=ModelRole.EXTRACTION,
                    manifest_path=manifest_path,
                    pack_path=candidate,
                    scratch=scratch,
                ),
                "page_markdown": [{"page_number": 1, "markdown": markdown}],
                "image_paths": {},
                "schema": NUEXTRACT_TEMPLATE_V1,
                "max_output_tokens": extraction_artifact.decode_limits[
                    "max_output_tokens"
                ],
            },
        )
        extraction = validate_clinical_extraction(
            extraction_value,
            pages={1: markdown},
            upload_id="pack-validation",
            strict_local=True,
        )
        if (
            len(extraction.labs) != 1
            or extraction.labs[0].name != "Hemoglobin A1c"
            or extraction.labs[0].value != "6.8"
            or extraction.labs[0].unit != "%"
        ):
            raise LocalValidationError("Local model validation fixture failed")
        evidence = extraction.evidence[0]

        summary_input = build_grounded_summary_input(
            facts=[
                {
                    "record_id": "pack-validation-record",
                    "content": {
                        "record_type": "observation",
                        "name": extraction.labs[0].name,
                        "value": extraction.labs[0].value,
                        "unit": extraction.labs[0].unit,
                    },
                    "evidence_ids": [evidence.id],
                }
            ],
            evidence=[
                {
                    "id": evidence.id,
                    "excerpt": extraction.labs[0].evidence_excerpt,
                    "page_number": evidence.page_number,
                    "section": "Laboratory",
                    "field_paths": ["/name", "/value", "/unit"],
                }
            ],
            requested_scope={"summary_type": "full_health"},
        )
        summary_artifact = _artifact(manifest, ModelRole.SUMMARY)
        summary_value = await selected_manager.run(
            ModelRole.SUMMARY,
            {
                **summary_input.model_dump(mode="json"),
                **_transport(
                    manifest=manifest,
                    role=ModelRole.SUMMARY,
                    manifest_path=manifest_path,
                    pack_path=candidate,
                    scratch=scratch,
                ),
                "max_output_tokens": summary_artifact.decode_limits[
                    "max_output_tokens"
                ],
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
        if not rendered.markdown.endswith(SERVER_MEDICAL_DISCLAIMER):
            raise LocalValidationError("Local model validation fixture failed")
        return _issue_runtime_validation_receipt(manifest)
    except LocalValidationError:
        raise
    except LocalAIError as exc:
        raise LocalValidationError("Local model validation fixture failed") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalValidationError("Local model validation fixture failed") from exc
    finally:
        if started:
            try:
                await selected_manager.stop()
            except LocalAIError:
                pass
        try:
            shutil.rmtree(scratch)
        except OSError:
            pass
