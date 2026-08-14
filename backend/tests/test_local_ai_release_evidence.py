"""Strict release evidence binds one manifest to benchmark and fidelity reports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _stable_release_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.runtime_identity import WorkerRuntimeIdentity

    monkeypatch.setattr(
        promotion,
        "normalize_worker_runtime_binding",
        lambda command, project: ((str(command),), Path(project)),
        raising=False,
    )
    monkeypatch.setattr(
        promotion,
        "resolve_worker_runtime_identity",
        lambda _command, _project: WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256="c" * 64,
        ),
        raising=False,
    )


def _manifest() -> object:
    from app.services.local_ai.manifest import (
        LocalAIManifest,
        ManifestArtifact,
        ManifestFile,
    )
    from app.services.local_ai.types import ModelRole

    return LocalAIManifest(
        schema_version=2,
        pack_revision="apple-m4-16gb-v2",
        platform="apple_silicon",
        runtime={
            "name": "mlx-vlm",
            "version": "0.5.0",
            "worker_identity_scheme": "local-ai-worker-bundle.v1",
            "worker_bundle_sha256": "c" * 64,
        },
        validation_suite_version="v1",
        artifacts=tuple(
            ManifestArtifact(
                role=role,
                repository="owner/model",
                revision="a" * 40,
                quantization="4bit",
                license="mit",
                attribution="source",
                decode_limits={"max_input_tokens": 1, "max_output_tokens": 1},
                files=(
                    ManifestFile(path="model.safetensors", sha256="b" * 64, size=1),
                ),
            )
            for role in ModelRole
        ),
    )


def _benchmark(manifest: object) -> dict[str, object]:
    from app.services.local_ai.artifact_store import manifest_sha256

    run = {
        "cold_start_seconds": 1.0,
        "duration_seconds": 1.0,
        "peak_rss_bytes": 8 * 1024**3,
        "mlx_peak_memory_bytes": 6 * 1024**3,
        "mlx_active_memory_after_bytes": 1,
        "throughput_per_second": 1.0,
    }
    return {
        "schema_version": 1,
        "content_free": True,
        "machine": {
            "platform_profile": "apple_silicon",
            "model": "Mac",
            "physical_memory_bytes": 16 * 1024**3,
            "os_version": "1",
        },
        "manifest": {
            "pack_revision": manifest.pack_revision,
            "sha256": manifest_sha256(manifest),
            "runtime_name": "mlx-vlm",
            "runtime_version": "0.5.0",
            "worker_identity_scheme": manifest.runtime["worker_identity_scheme"],
            "worker_bundle_sha256": manifest.runtime["worker_bundle_sha256"],
        },
        "processes": {"max_live_models": 1, "roles_started": 9},
        "roles": {
            role: [{**run, "run": index} for index in range(1, 4)]
            for role in ("ocr", "extraction", "summary")
        },
        "system": {
            "swap_before_bytes": 0,
            "swap_after_bytes": 0,
            "swap_delta_bytes": 0,
            "memory_pressure_termination": False,
            "sustained_swap_thrashing": False,
        },
        "reclamation": {
            "baseline_active_memory_bytes": 1,
            "final_active_memory_bytes": 1,
            "peak_active_memory_bytes": 1,
            "final_active_memory_ratio": 0.1,
        },
    }


def _fidelity(manifest: object) -> dict[str, object]:
    from app.services.local_ai.artifact_store import manifest_sha256
    from app.services.local_ai.fidelity_metrics import FidelityMetrics
    from app.services.local_ai.fidelity_runner import (
        FIDELITY_CORPUS_SHA256,
        FIDELITY_SUITE_VERSION,
        build_fidelity_report,
    )

    return build_fidelity_report(
        metrics=FidelityMetrics(
            critical_numeric_exact=1.0,
            critical_precision=1.0,
            critical_recall=1.0,
            accepted_output_schema_validity=1.0,
            forbidden_extraction_facts=0,
            unsupported_summary_facts=0,
            accepted_facts_without_evidence=0,
            summary_fact_recall=1.0,
            summary_typed_field_recall=1.0,
        ),
        fixture_suite_version=FIDELITY_SUITE_VERSION,
        fixture_suite_sha256=FIDELITY_CORPUS_SHA256,
        manifest_sha256=manifest_sha256(manifest),
        synthetic_documents=6,
        private_documents=0,
    ).as_dict()


def _write_fidelity(path: Path, manifest: object) -> None:
    path.write_text(
        json.dumps(_fidelity(manifest), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_release_evidence_requires_exact_manifest_benchmark_and_fidelity(
    tmp_path: Path,
) -> None:
    from app.services.local_ai.release_evidence import (
        build_release_evidence,
        load_release_evidence,
    )

    manifest = _manifest()
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(_benchmark(manifest)), encoding="utf-8")
    fidelity = tmp_path / "fidelity.json"
    _write_fidelity(fidelity, manifest)
    payload = build_release_evidence(
        manifest,
        _benchmark(manifest),
        _fidelity(manifest),
    )
    payload["benchmark_sha256"] = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    payload["fidelity_sha256"] = hashlib.sha256(fidelity.read_bytes()).hexdigest()
    evidence = tmp_path / "release.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_release_evidence(
        evidence,
        manifest=manifest,
        benchmark_path=benchmark,
        fidelity_path=fidelity,
    )
    assert loaded.expected_memory_bytes("ocr") > 8 * 1024**3
    assert loaded.fidelity_suite_version == "local-ai-fidelity-v1"
    assert loaded.fidelity_corpus_sha256 == (
        "5654f214d4fb0a404d3abf3572812f209e05b20be0deb379dddd7db663d1a826"
    )

    payload["roles"]["ocr"] = 17 * 1024**3
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="release evidence"):
        load_release_evidence(
            evidence,
            manifest=manifest,
            benchmark_path=benchmark,
            fidelity_path=fidelity,
        )


def test_release_loader_rejects_hashed_fidelity_report_below_threshold(
    tmp_path: Path,
) -> None:
    from app.services.local_ai.release_evidence import (
        build_release_evidence,
        load_release_evidence,
    )

    manifest = _manifest()
    benchmark_value = _benchmark(manifest)
    fidelity_value = _fidelity(manifest)
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(benchmark_value), encoding="utf-8")
    fidelity = tmp_path / "fidelity.json"
    fidelity_value["metrics"]["critical_recall"] = 0.94
    fidelity.write_text(json.dumps(fidelity_value), encoding="utf-8")
    payload = build_release_evidence(manifest, benchmark_value, _fidelity(manifest))
    payload["benchmark_sha256"] = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    payload["fidelity_sha256"] = hashlib.sha256(fidelity.read_bytes()).hexdigest()
    evidence = tmp_path / "release.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception, match="release evidence"):
        load_release_evidence(
            evidence,
            manifest=manifest,
            benchmark_path=benchmark,
            fidelity_path=fidelity,
        )


def test_release_loader_rejects_incomplete_canonical_fidelity_count(
    tmp_path: Path,
) -> None:
    from app.services.local_ai.errors import LocalValidationError
    from app.services.local_ai.release_evidence import (
        build_release_evidence,
        load_release_evidence,
    )

    manifest = _manifest()
    benchmark_value = _benchmark(manifest)
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(benchmark_value), encoding="utf-8")
    fidelity_value = _fidelity(manifest)
    fidelity_value["synthetic_documents"] = 1
    fidelity = tmp_path / "fidelity.json"
    fidelity.write_text(json.dumps(fidelity_value), encoding="utf-8")
    payload = build_release_evidence(manifest, benchmark_value, _fidelity(manifest))
    payload["benchmark_sha256"] = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    payload["fidelity_sha256"] = hashlib.sha256(fidelity.read_bytes()).hexdigest()
    evidence = tmp_path / "release.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LocalValidationError, match="release evidence"):
        load_release_evidence(
            evidence,
            manifest=manifest,
            benchmark_path=benchmark,
            fidelity_path=fidelity,
        )


def test_promotion_rejects_incomplete_canonical_fidelity_count(tmp_path: Path) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.errors import LocalValidationError

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(_benchmark(manifest)), encoding="utf-8")
    fidelity_value = _fidelity(manifest)
    fidelity_value["synthetic_documents"] = 1
    fidelity_path = tmp_path / "fidelity.json"
    fidelity_path.write_text(json.dumps(fidelity_value), encoding="utf-8")
    output = tmp_path / "release.json"

    with pytest.raises(LocalValidationError, match="release"):
        promotion.promote(
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
            fidelity_path=fidelity_path,
            output=output,
        )

    assert not output.exists()


def test_promotion_rejects_private_fidelity_results(tmp_path: Path) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.errors import LocalValidationError

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(_benchmark(manifest)), encoding="utf-8")
    fidelity_value = _fidelity(manifest)
    fidelity_value["private_documents"] = 1
    fidelity_value["private_metrics"] = dict(fidelity_value["metrics"])
    fidelity_path = tmp_path / "fidelity.json"
    fidelity_path.write_text(json.dumps(fidelity_value), encoding="utf-8")
    output = tmp_path / "release.json"

    with pytest.raises(LocalValidationError, match="release"):
        promotion.promote(
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
            fidelity_path=fidelity_path,
            output=output,
        )

    assert not output.exists()


def test_release_loader_rejects_private_fidelity_results(tmp_path: Path) -> None:
    from app.services.local_ai.errors import LocalValidationError
    from app.services.local_ai.release_evidence import (
        build_release_evidence,
        load_release_evidence,
    )

    manifest = _manifest()
    benchmark_value = _benchmark(manifest)
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(benchmark_value), encoding="utf-8")
    fidelity_value = _fidelity(manifest)
    fidelity_value["private_documents"] = 1
    fidelity_value["private_metrics"] = dict(fidelity_value["metrics"])
    fidelity = tmp_path / "fidelity.json"
    fidelity.write_text(json.dumps(fidelity_value), encoding="utf-8")
    payload = build_release_evidence(manifest, benchmark_value, _fidelity(manifest))
    payload["benchmark_sha256"] = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    payload["fidelity_sha256"] = hashlib.sha256(fidelity.read_bytes()).hexdigest()
    evidence = tmp_path / "release.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LocalValidationError, match="release evidence"):
        load_release_evidence(
            evidence,
            manifest=manifest,
            benchmark_path=benchmark,
            fidelity_path=fidelity,
        )


def test_promotion_rejects_expected_role_memory_above_sixteen_gib(
    tmp_path: Path,
) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.errors import LocalValidationError

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    benchmark_value = _benchmark(manifest)
    for run in benchmark_value["roles"]["summary"]:
        run["peak_rss_bytes"] = 16 * 1024**3
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_value), encoding="utf-8")
    fidelity_path = tmp_path / "fidelity.json"
    _write_fidelity(fidelity_path, manifest)
    output = tmp_path / "release.json"

    with pytest.raises(LocalValidationError, match="release"):
        promotion.promote(
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
            fidelity_path=fidelity_path,
            output=output,
        )

    assert not output.exists()


def test_expected_memory_is_ceiled_to_64mib_with_ten_percent_headroom() -> None:
    from app.services.local_ai.release_evidence import expected_memory_bytes

    assert (
        expected_memory_bytes([{"peak_rss_bytes": 100, "mlx_peak_memory_bytes": 200}])
        == 64 * 1024**2
    )
    assert (
        expected_memory_bytes(
            [{"peak_rss_bytes": 64 * 1024**2, "mlx_peak_memory_bytes": 0}]
        )
        == 128 * 1024**2
    )


def test_promotion_hashes_the_same_bounded_benchmark_bytes_it_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.release_evidence import _read_small_regular

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(_benchmark(manifest)), encoding="utf-8")
    fidelity_path = tmp_path / "fidelity.json"
    _write_fidelity(fidelity_path, manifest)
    output = tmp_path / "release.json"
    benchmark_reads = 0
    fidelity_reads = 0

    def counted_read(path: Path) -> bytes:
        nonlocal benchmark_reads, fidelity_reads
        if path == benchmark_path:
            benchmark_reads += 1
        if path == fidelity_path:
            fidelity_reads += 1
        return _read_small_regular(path)

    monkeypatch.setattr(promotion, "_read_small_regular", counted_read)

    promotion.promote(
        manifest_path=manifest_path,
        benchmark_path=benchmark_path,
        fidelity_path=fidelity_path,
        output=output,
    )

    assert benchmark_reads == 1
    assert fidelity_reads == 1
    promoted = json.loads(output.read_text(encoding="utf-8"))
    assert (
        promoted["benchmark_sha256"]
        == hashlib.sha256(_read_small_regular(benchmark_path)).hexdigest()
    )
    assert (
        promoted["fidelity_sha256"]
        == hashlib.sha256(_read_small_regular(fidelity_path)).hexdigest()
    )


def test_promotion_rejects_fidelity_report_below_threshold(tmp_path: Path) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.errors import LocalValidationError

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(_benchmark(manifest)), encoding="utf-8")
    fidelity_value = _fidelity(manifest)
    fidelity_value["metrics"]["critical_precision"] = 0.97
    fidelity_path = tmp_path / "fidelity.json"
    fidelity_path.write_text(json.dumps(fidelity_value), encoding="utf-8")

    with pytest.raises(LocalValidationError, match="promotion failed"):
        promotion.promote(
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
            fidelity_path=fidelity_path,
            output=tmp_path / "release.json",
        )


def test_bounded_reader_rejects_replaced_non_regular_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.release_evidence as release_evidence
    from app.services.local_ai.errors import LocalValidationError

    source = tmp_path / "benchmark.json"
    source.write_text("{}", encoding="utf-8")
    real_open = release_evidence.os.open

    def replaced_open(_path: Path, flags: int) -> int:
        return real_open("/dev/null", flags)

    monkeypatch.setattr(release_evidence.os, "open", replaced_open)

    with pytest.raises(LocalValidationError, match="unavailable"):
        release_evidence._read_small_regular(source)


@pytest.mark.parametrize(
    "decode_error", [RecursionError(), ValueError("integer limit")]
)
def test_release_loader_normalizes_bounded_json_decode_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decode_error: Exception,
) -> None:
    import app.services.local_ai.release_evidence as release_evidence
    from app.services.local_ai.errors import LocalValidationError

    evidence = tmp_path / "release.json"
    benchmark = tmp_path / "benchmark.json"
    fidelity = tmp_path / "fidelity.json"
    evidence.write_text("{}", encoding="utf-8")
    benchmark.write_text("{}", encoding="utf-8")
    fidelity.write_text("{}", encoding="utf-8")

    def fail_decode(_value: object) -> object:
        raise decode_error

    monkeypatch.setattr(release_evidence.json, "loads", fail_decode)

    with pytest.raises(LocalValidationError, match="unavailable"):
        release_evidence.load_release_evidence(
            evidence,
            manifest=_manifest(),
            benchmark_path=benchmark,
            fidelity_path=fidelity,
        )


@pytest.mark.parametrize(
    "decode_error", [RecursionError(), ValueError("integer limit")]
)
def test_release_promotion_normalizes_bounded_json_decode_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decode_error: Exception,
) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.errors import LocalValidationError

    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text("{}", encoding="utf-8")
    fidelity = tmp_path / "fidelity.json"
    fidelity.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(promotion, "load_manifest", lambda _path: _manifest())

    def fail_decode(_value: object) -> object:
        raise decode_error

    monkeypatch.setattr(promotion.json, "loads", fail_decode)

    with pytest.raises(LocalValidationError, match="promotion failed"):
        promotion.promote(
            manifest_path=tmp_path / "manifest.json",
            benchmark_path=benchmark,
            fidelity_path=fidelity,
            output=tmp_path / "release.json",
        )


def test_promotion_rejects_benchmark_worker_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.errors import LocalValidationError
    from app.services.local_ai.runtime_identity import WorkerRuntimeIdentity

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    benchmark_value = _benchmark(manifest)
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_value), encoding="utf-8")
    fidelity_path = tmp_path / "fidelity.json"
    _write_fidelity(fidelity_path, manifest)
    monkeypatch.setattr(
        promotion,
        "normalize_worker_runtime_binding",
        lambda command, project: ((str(command),), Path(project)),
        raising=False,
    )
    monkeypatch.setattr(
        promotion,
        "resolve_worker_runtime_identity",
        lambda _command, _project: WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256="b" * 64,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        promotion, "validate_acceptance", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        promotion, "load_release_evidence", lambda *_args, **_kwargs: None
    )

    with pytest.raises(LocalValidationError, match="promotion failed"):
        promotion.promote(
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
            fidelity_path=fidelity_path,
            output=tmp_path / "release.json",
        )

    assert not (tmp_path / "release.json").exists()


def test_promotion_rejects_benchmark_worker_identity_scheme_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.promote_local_ai_release as promotion
    from app.services.local_ai.errors import LocalValidationError

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    benchmark_value = _benchmark(manifest)
    benchmark_value["manifest"]["worker_identity_scheme"] = "local-ai-worker-bundle.v2"
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_value), encoding="utf-8")
    fidelity_path = tmp_path / "fidelity.json"
    _write_fidelity(fidelity_path, manifest)
    output = tmp_path / "release.json"
    monkeypatch.setattr(
        promotion, "validate_acceptance", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        promotion, "load_release_evidence", lambda *_args, **_kwargs: None
    )

    with pytest.raises(LocalValidationError, match="promotion failed"):
        promotion.promote(
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
            fidelity_path=fidelity_path,
            output=output,
        )

    assert not output.exists()


def test_release_loader_rejects_benchmark_worker_identity_scheme_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_local_ai as benchmark_cli
    from app.services.local_ai.errors import LocalValidationError
    from app.services.local_ai.release_evidence import (
        build_release_evidence,
        load_release_evidence,
    )

    manifest = _manifest()
    benchmark_value = _benchmark(manifest)
    benchmark_value["manifest"]["worker_identity_scheme"] = "local-ai-worker-bundle.v2"
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps(benchmark_value), encoding="utf-8")
    fidelity_value = _fidelity(manifest)
    fidelity = tmp_path / "fidelity.json"
    fidelity.write_text(json.dumps(fidelity_value), encoding="utf-8")
    payload = build_release_evidence(manifest, benchmark_value, fidelity_value)
    payload["benchmark_sha256"] = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    payload["fidelity_sha256"] = hashlib.sha256(fidelity.read_bytes()).hexdigest()
    evidence = tmp_path / "release.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        benchmark_cli,
        "validate_acceptance",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(LocalValidationError, match="release evidence"):
        load_release_evidence(
            evidence,
            manifest=manifest,
            benchmark_path=benchmark,
            fidelity_path=fidelity,
        )


@pytest.mark.asyncio
async def test_fidelity_wrapper_rejects_worker_identity_drift_before_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_local_ai_fidelity as fidelity_cli
    from app.services.local_ai.runtime_identity import WorkerRuntimeIdentity

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    called = False

    async def forbidden_suite(**_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("fidelity suite must not run")

    monkeypatch.setattr(fidelity_cli, "run_installed_fidelity_suite", forbidden_suite)
    monkeypatch.setattr(
        fidelity_cli,
        "normalize_worker_runtime_binding",
        lambda command, project: ((str(command),), Path(project)),
        raising=False,
    )
    monkeypatch.setattr(
        fidelity_cli,
        "resolve_worker_runtime_identity",
        lambda _command, _project: WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256="b" * 64,
        ),
        raising=False,
    )
    args = fidelity_cli._parser().parse_args(
        [
            "--manifest",
            str(manifest_path),
            "--model-root",
            str(tmp_path / "models"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--output",
            str(tmp_path / "fidelity.json"),
        ]
    )

    assert await fidelity_cli._execute(args) == 1
    assert called is False


def test_fidelity_wrapper_ignores_inherited_private_fixture_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_local_ai_fidelity as fidelity_cli

    monkeypatch.setenv("REAL_MEDICAL_FIXTURES_DIR", "/private/medical/fixtures")

    args = fidelity_cli._parser().parse_args([])

    assert args.private_fixtures_dir is None


@pytest.mark.asyncio
async def test_fidelity_wrapper_serializes_no_private_run_from_hostile_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_local_ai_fidelity as fidelity_cli
    from app.services.local_ai.fidelity_runner import (
        load_fidelity_report,
        parse_fidelity_report,
    )
    from app.services.local_ai.runtime_identity import WorkerRuntimeIdentity

    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    output = tmp_path / "fidelity.json"
    monkeypatch.setenv("REAL_MEDICAL_FIXTURES_DIR", "/private/medical/fixtures")
    monkeypatch.setattr(
        fidelity_cli,
        "normalize_worker_runtime_binding",
        lambda command, project: ((str(command),), Path(project)),
    )
    monkeypatch.setattr(
        fidelity_cli,
        "resolve_worker_runtime_identity",
        lambda _command, _project: WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256=manifest.runtime["worker_bundle_sha256"],
        ),
    )

    async def synthetic_only_suite(**kwargs: object) -> object:
        assert kwargs["private_fixtures_dir"] is None
        return parse_fidelity_report(_fidelity(manifest))

    monkeypatch.setattr(
        fidelity_cli,
        "run_installed_fidelity_suite",
        synthetic_only_suite,
    )
    args = fidelity_cli._parser().parse_args(
        [
            "--manifest",
            str(manifest_path),
            "--model-root",
            str(tmp_path / "models"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--output",
            str(output),
        ]
    )

    assert await fidelity_cli._execute(args) == 0
    report = load_fidelity_report(output)
    assert report.private_documents == 0
    assert report.private_metrics is None
