"""Static contracts for strict-local CI workflow configuration."""

from __future__ import annotations

from pathlib import Path

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
