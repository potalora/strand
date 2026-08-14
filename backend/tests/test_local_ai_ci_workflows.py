"""Static contracts for strict-local CI workflow configuration."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings

WORKFLOWS = (
    ".github/workflows/backend-ci.yml",
    ".github/workflows/local-ai-contract-ci.yml",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_ci_uses_the_application_encryption_key_name_and_format() -> None:
    """Clean runners must configure the AES-256-GCM key the app actually reads."""

    for relative in WORKFLOWS:
        content = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert "\n      DATABASE_ENCRYPTION_KEY: " in content
        assert "\n      ENCRYPTION_KEY: " not in content
        value = next(
            line.split(":", 1)[1].strip()
            for line in content.splitlines()
            if line.strip().startswith("DATABASE_ENCRYPTION_KEY:")
        )
        assert len(value) == 64
        int(value, 16)


def test_strict_local_contract_ci_separates_ubuntu_and_apple_mlx_jobs() -> None:
    """Keep Apple-only MLX installation off the Ubuntu fake-worker job."""

    content = (
        REPOSITORY_ROOT / ".github/workflows/local-ai-contract-ci.yml"
    ).read_text(encoding="utf-8")

    assert "runs-on: ubuntu-latest" in content
    assert "runs-on: macos-15" in content
    assert "Install isolated MLX worker dependencies" in content
    assert "Run offline worker contracts without model downloads" in content

    ubuntu_job = content.split("  contracts:", 1)[1].split("  apple-mlx-worker:", 1)[0]
    assert "workers/local_ai/apple_mlx" not in ubuntu_job


def test_strict_local_contract_ci_watches_runtime_and_migration_wiring() -> None:
    """Changes to strict-local install/profile wiring must run its contracts."""

    content = (
        REPOSITORY_ROOT / ".github/workflows/local-ai-contract-ci.yml"
    ).read_text(encoding="utf-8")

    for path in (
        '      - "backend/app/config.py"',
        '      - "backend/app/services/local_ai/pack_operations.py"',
        '      - "backend/scripts/local_ai_pack.py"',
        '      - "backend/scripts/benchmark_local_ai.py"',
        '      - "backend/alembic/versions/**"',
        '      - "justfile"',
        '      - "scripts/setup-local-ai-macos.sh"',
    ):
        assert path in content


def test_strict_local_contract_ci_upgrades_only_an_isolated_migration_database() -> (
    None
):
    """Migration smoke checks must not downgrade or reuse the ordinary test DB."""

    content = (
        REPOSITORY_ROOT / ".github/workflows/local-ai-contract-ci.yml"
    ).read_text(encoding="utf-8")

    assert "createdb -h 127.0.0.1 -U postgres medtimeline_migrations_ci" in content
    assert "medtimeline_migrations_ci uv run alembic upgrade head" in content
    assert (
        "dropdb -h 127.0.0.1 -U postgres --if-exists medtimeline_migrations_ci"
        in content
    )
    assert "alembic downgrade" not in content


def test_backend_ci_keeps_full_lint_but_scopes_the_formatter_gate() -> None:
    """Do not make strict-local CI depend on unrelated legacy formatting debt."""

    content = (REPOSITORY_ROOT / ".github/workflows/backend-ci.yml").read_text(
        encoding="utf-8"
    )

    assert "uv run ruff check app scripts tests" in content
    assert "uv run ruff format --check app scripts tests" not in content
    for path in (
        "app/services/ai/grounded_routing.py",
        "app/services/ai/prompt_grounding.py",
        "app/services/local_ai",
        "tests/test_grounded_*.py",
        "tests/test_local_ai_*.py",
        "tests/test_strict_local_*.py",
    ):
        assert path in content


def test_v2_pack_paths_are_the_only_active_defaults_and_release_recipes() -> None:
    configured = Settings(_env_file=None)
    assert configured.local_ai_manifest_path.endswith("/apple-m4-16gb-v2.lock.json")
    assert configured.local_ai_release_evidence_path.endswith(
        "/apple-m4-16gb-v2.release.json"
    )

    justfile = (REPOSITORY_ROOT / "justfile").read_text(encoding="utf-8")
    promotion = justfile.split("local-ai-release-promote:", 1)[1].split("\n\n", 1)[0]
    assert "apple-m4-16gb-v2.lock.json" in promotion
    assert "apple-m4-16gb-v2.release.json" in promotion
    assert "apple-m4-16gb-v1" not in promotion

    environment = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    assert (
        "LOCAL_AI_MANIFEST_PATH=./app/model_manifests/apple-m4-16gb-v2.lock.json"
    ) in environment.splitlines()
    assert (
        "LOCAL_AI_RELEASE_EVIDENCE_PATH="
        "./app/model_manifests/apple-m4-16gb-v2.release.json"
    ) in environment.splitlines()
    assert "apple-m4-16gb-v1" not in "\n".join(
        line for line in environment.splitlines() if line.startswith("LOCAL_AI_")
    )


def test_static_v2_catalog_and_lock_preserve_model_revisions() -> None:
    import json

    manifests = REPOSITORY_ROOT / "backend" / "app" / "model_manifests"
    catalog_v1 = json.loads((manifests / "catalog-v1.json").read_text())
    catalog_v2 = json.loads((manifests / "catalog-v2.json").read_text())
    lock_v1 = json.loads((manifests / "apple-m4-16gb-v1.lock.json").read_text())
    lock_v2 = json.loads((manifests / "apple-m4-16gb-v2.lock.json").read_text())

    expected_digest = "f847c4b69848029c0e6de7edfedbbfa2ea775910d424adc1e8b714974cd4db65"
    for value in (catalog_v2, lock_v2):
        assert value["schema_version"] == 2
        assert value["pack_revision"] == "apple-m4-16gb-v2"
        assert value["runtime"]["worker_identity_scheme"] == (
            "local-ai-worker-bundle.v1"
        )
        assert value["runtime"]["worker_bundle_sha256"] == expected_digest

    assert catalog_v2["candidates"] == catalog_v1["candidates"]
    assert lock_v2["artifacts"] == lock_v1["artifacts"]


def test_strict_local_contract_ci_covers_setup_only_change_contracts() -> None:
    """Setup and justfile-only PRs need their deterministic contract nodes here."""

    local_ai_workflow = (
        REPOSITORY_ROOT / ".github/workflows/local-ai-contract-ci.yml"
    ).read_text(encoding="utf-8")
    selected_contract_tests = local_ai_workflow.split("uv run pytest \\", 1)[1].split(
        "\n            -q", 1
    )[0]
    backend_workflow = (REPOSITORY_ROOT / ".github/workflows/backend-ci.yml").read_text(
        encoding="utf-8"
    )

    for test_file in (
        "tests/test_local_ai_setup_script.py",
        "tests/test_local_ai_benchmark.py",
        "tests/test_local_ai_candidate_pack_cli.py",
        "tests/test_local_ai_ci_workflows.py",
        "tests/test_local_ai_pack_cli.py",
        "tests/test_local_ai_runtime_identity.py",
        "tests/test_local_ai_release_evidence.py",
    ):
        assert test_file in selected_contract_tests

    workflow_triggers = local_ai_workflow.split("permissions:", 1)[0]
    for watched_path in (
        ".env.example",
        "backend/scripts/local_ai_candidate_pack.py",
        "backend/scripts/lock_local_ai_manifest.py",
        "backend/scripts/run_local_ai_fidelity.py",
        "scripts/setup-local-ai-macos.sh",
        "justfile",
    ):
        assert workflow_triggers.count(f'- "{watched_path}"') == 2

    assert '"scripts/setup-local-ai-macos.sh"' not in backend_workflow
    assert '"justfile"' not in backend_workflow
