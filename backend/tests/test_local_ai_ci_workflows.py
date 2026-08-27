"""Static contracts for strict-local CI workflow configuration."""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.config import Settings

WORKFLOWS = (
    ".github/workflows/backend-ci.yml",
    ".github/workflows/local-ai-contract-ci.yml",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_local_only_playwright_profile_is_network_denial_not_attested_worker() -> None:
    content = (REPOSITORY_ROOT / "frontend/playwright.config.ts").read_text(
        encoding="utf-8"
    )
    local_profile = content.split("if (localOnly) {", 1)[1].split("} else {", 1)[0]

    assert 'APP_ENV: "test"' in local_profile
    assert 'DATABASE_ENCRYPTION_KEY: "00".repeat(32)' in local_profile
    assert 'REAL_MEDICAL_FIXTURES_DIR: ""' in local_profile
    assert 'MEDTIMELINE_LEGACY_DEV_FIXTURES_DIR: ""' in local_profile
    assert 'E2E_ATTESTED_STRICT_PACK: ""' in local_profile
    assert 'LOCAL_AI_WORKER_COMMAND: "/usr/bin/false"' in local_profile
    assert "LOCAL_AI_WORKER_PROJECT_DIR: runtimePaths.nonWorkerProject" in local_profile
    assert "UPLOAD_DIR: runtimePaths.uploads" in local_profile
    assert "TEMP_EXTRACT_DIR: runtimePaths.tempExtract" in local_profile
    assert "LOCAL_AI_SCRATCH_DIR: runtimePaths.scratch" in local_profile
    assert "LOCAL_AI_MODEL_DIR: runtimePaths.models" in local_profile
    assert "E2E_RUNTIME_ROOT" in content
    assert "E2E_OUTPUT_ROOT" in content
    assert "fs.lstatSync" in content
    assert "fs.realpathSync.native" in content
    assert "stats.dev" in content
    assert "stats.ino" in content
    assert "stats.uid" in content
    assert "stats.mode & 0o777" in content
    assert (
        "path.dirname(rootIdentity.realPath) !== runtimeParentIdentity.realPath"
        in content
    )
    assert 'path.join(outputIdentity.realPath, "artifacts")' in content
    assert "outputArtifactsIdentity.realPath" in content
    assert "outputDir: localOnlyOutputDir" in content
    assert "e2e_local_ai_worker.py" not in local_profile
    assert "apple-m4-16gb-v1.lock.json" in local_profile
    assert "apple-m4-16gb-v1.release.json" in local_profile
    assert "backend/artifacts/local-ai-benchmark.json" in local_profile
    assert "backend/artifacts/local-ai-fidelity.json" in local_profile
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "VERTEX_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        assert f'{name}: ""' in local_profile
    assert 'LLM_PROVIDER: "gemini"' in local_profile
    for name in (
        "LLM_SUMMARY_PROVIDER",
        "LLM_SECTION_PROVIDER",
        "LLM_DEDUP_PROVIDER",
        "LLM_EXTRACTION_PROVIDER",
    ):
        assert f'{name}: ""' in local_profile


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


def test_local_only_structured_fixture_helpers_are_constrained() -> None:
    content = (REPOSITORY_ROOT / "frontend/e2e/helpers/api-client.ts").read_text(
        encoding="utf-8"
    )
    assert "async uploadTrackedSyntheticFhirCloudAssisted():" in content
    assert "async attemptTrackedSyntheticFhirUsingStoredPreference():" in content
    assert (
        "uploadGeneratedPaginationFhirCloudAssisted(\n    bundleJson: string" in content
    )
    assert "uploadDeterministicStructuredFixtureCloudAssisted" not in content
    assert "sample_fhir_bundle.json" in content


def test_local_only_browser_docs_separate_network_denial_from_pack_attestation() -> (
    None
):
    content = (REPOSITORY_ROOT / "docs/operations-strict-local-ai.md").read_text(
        encoding="utf-8"
    )
    heading = "## Run browser tests with local-only enforcement"
    assert content.splitlines().count(heading) == 1
    section_start = content.index(f"{heading}\n") + len(heading) + 1
    section_tail = content[section_start:]
    section_end = section_tail.index("\n## ")
    section = section_tail[:section_end]
    command_fence = section.index("```bash\n")
    assert "```" not in section[:command_fence]
    assert "\n### " not in section[:command_fence]
    command_start = command_fence + len("```bash\n")
    command_end = section.index("\n```", command_start)
    command = section[command_start:command_end]

    subprocess.run(
        ["/bin/bash", "-n"],
        input=command,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "proves network confinement" in section
    assert (
        "does not prove that an attested strict-local model pack is installed"
        in section
    )
    assert "`/usr/bin/false`" in section
    assert "startup sentinel" in section
    assert "explicit `cloud_assisted`" in section
    assert "background dedup" in section
    assert "provider construction" in section
    assert "three summary model-execution cases" in section
    assert "24-character random token" in section
    assert "database OID and owner" in section
    assert "not absolute protection" in section
    assert "malicious PostgreSQL cluster administrator" in section

    assert 'task_evidence_parent="$task_repo_root/frontend/test-results"' in command
    evidence_creation = command.index('if [ ! -e "$task_evidence_parent" ]')
    evidence_link_guard = command.index('[ ! -L "$task_evidence_parent" ]')
    evidence_mkdir = command.index('mkdir -m 700 -- "$task_evidence_parent"')
    evidence_creation_end = command.index("\nfi", evidence_mkdir)
    assert (
        evidence_creation < evidence_link_guard < evidence_mkdir < evidence_creation_end
    )
    parent_validation = command.split("validate_owned_parent() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert 'test -d "$task_parent" && test ! -L "$task_parent"' in parent_validation
    assert (
        'test "$(cd -P -- "$task_parent" && pwd -P)" = "$task_parent"'
        in parent_validation
    )
    assert 'test "$(stat -f \'%u\' "$task_parent")" -eq "$(id -u)"' in parent_validation
    assert 'task_parent_mode="$(stat -f \'%Lp\' "$task_parent")"' in parent_validation
    assert 'test "$((8#$task_parent_mode & 022))" -eq 0' in parent_validation

    evidence_validation = command.index('validate_owned_parent "$task_evidence_parent"')
    runtime_parent_creation = command.index(
        'if [ ! -e "$task_runtime_parent" ] && [ ! -L "$task_runtime_parent" ]'
    )
    runtime_parent_validation = command.index(
        'validate_owned_parent "$task_runtime_parent"'
    )
    runtime_parent_identity = command.index(
        "task_runtime_parent_identity=\"$(stat -f '%d:%i:%u:%HT:%Lp' "
        '"$task_runtime_parent")"'
    )
    output_parent_creation = command.index(
        'if [ ! -e "$task_output_parent" ] && [ ! -L "$task_output_parent" ]'
    )
    output_parent_validation = command.index(
        'validate_owned_parent "$task_output_parent"'
    )
    output_parent_identity = command.index(
        "task_output_parent_identity=\"$(stat -f '%d:%i:%u:%HT:%Lp' "
        '"$task_output_parent")"'
    )
    assert (
        evidence_validation
        < runtime_parent_creation
        < runtime_parent_validation
        < runtime_parent_identity
        < output_parent_creation
        < output_parent_validation
        < output_parent_identity
    )

    runtime_creation = command.index(
        'task_runtime_root="$(mktemp -d '
        '"$task_runtime_parent/docs.XXXXXXXXXXXXXXXXXXXXXXXX")"'
    )
    runtime_validation = command.index(
        'validate_owned_root "$task_runtime_root" "$task_runtime_parent"'
    )
    runtime_identity = command.index(
        "task_runtime_root_identity=\"$(stat -f '%d:%i:%u:%HT:%Lp' "
        '"$task_runtime_root")"'
    )
    output_creation = command.index(
        'task_output_root="$(mktemp -d '
        '"$task_output_parent/docs.XXXXXXXXXXXXXXXXXXXXXXXX")"'
    )
    output_validation = command.index(
        'validate_owned_root "$task_output_root" "$task_output_parent"'
    )
    output_identity = command.index(
        "task_output_root_identity=\"$(stat -f '%d:%i:%u:%HT:%Lp' "
        '"$task_output_root")"'
    )
    assert (
        runtime_creation
        < runtime_validation
        < runtime_identity
        < output_creation
        < output_validation
        < output_identity
    )

    token_derivation = command.index('task_run_token="${task_runtime_root##*.}"')
    token_validation = command.index('test "${#task_run_token}" -eq 24')
    token_alnum_validation = command.index(
        "*[!abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789]*"
    )
    token_lowercase = command.index("tr '[:upper:]' '[:lower:]'")
    database_name_assignment = command.index(
        'task_e2e_database="medtimeline_e2e_$task_run_token"'
    )
    assert 'test "${#task_e2e_database}" -le 63' in command
    assert "medtimeline_e2e_*)" in command
    assert "*[!a-z0-9_]*" in command
    assert "medtimeline_e2e_local" not in section

    database_query = command.split("query_task_database_identity() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert "psql -X -qAt -v ON_ERROR_STOP=1" in database_query
    assert '-v task_database="$task_e2e_database"' in database_query
    assert " -c " not in database_query
    assert "<<'SQL'" in database_query
    assert "SELECT oid::text || ':' || datdba::text" in database_query
    assert "WHERE datname = :'task_database';" in database_query

    database_name_validation = command.index('test "${#task_e2e_database}" -le 63')
    database_creation = command.index(
        'createdb -h 127.0.0.1 -p 5432 "$task_e2e_database"'
    )
    database_charset_validation = command.rindex(
        'case "$task_e2e_database" in', 0, database_creation
    )
    database_created_flag = command.index("task_created_e2e_database=1")
    database_identity = command.index(
        'task_e2e_database_identity="$(query_task_database_identity)"'
    )
    assert (
        token_derivation
        < token_validation
        < token_alnum_validation
        < token_lowercase
        < database_name_assignment
        < database_name_validation
        < database_charset_validation
        < database_creation
        < database_created_flag
        < database_identity
    )
    command_lines = command.splitlines()
    database_creation_line = command_lines.index(
        'createdb -h 127.0.0.1 -p 5432 "$task_e2e_database"'
    )
    assert command_lines[database_creation_line + 1] == "task_created_e2e_database=1"
    assert command_lines[database_creation_line + 2] == (
        'task_e2e_database_identity="$(query_task_database_identity)"'
    )
    assert 'task_e2e_database_oid="${task_e2e_database_identity%%:*}"' in command
    assert 'task_e2e_database_owner="${task_e2e_database_identity#*:}"' in command

    cleanup = command.split("cleanup_local_e2e_docs() {", 1)[1].split(
        "\n}\ntrap cleanup_local_e2e_docs EXIT", 1
    )[0]
    database_cleanup = cleanup.split(
        '  if [ "$task_created_e2e_database" -eq 1 ]; then', 1
    )[1].split('  if [ -n "$task_runtime_root" ]; then', 1)[0]
    assert command.count('dropdb -h 127.0.0.1 -p 5432 "$task_e2e_database"') == 1
    assert database_cleanup.count("dropdb") == 1
    current_identity_query = database_cleanup.index(
        'task_current_e2e_database_identity="$(query_task_database_identity '
        '2>/dev/null)"'
    )
    absent_database = database_cleanup.index(
        'if [ -z "$task_current_e2e_database_identity" ]; then\n        :'
    )
    matching_identity = database_cleanup.index(
        '[ "$task_current_e2e_database_identity" = '
        '"$task_e2e_database_identity" ]; then'
    )
    database_drop = database_cleanup.index(
        'dropdb -h 127.0.0.1 -p 5432 "$task_e2e_database"'
    )
    database_drop_suppression = database_cleanup.index(">/dev/null 2>&1", database_drop)
    assert (
        current_identity_query
        < absent_database
        < matching_identity
        < database_drop
        < database_drop_suppression
    )
    assert database_cleanup.count("task_cleanup_status=1") >= 3

    safe_remove = command.split("safe_remove_owned_directory() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert 'test -n "$task_remove_root" && test -n "$task_remove_parent"' in safe_remove
    assert 'test -n "$task_expected_parent_identity"' in safe_remove
    assert 'test -n "$task_expected_root_identity"' in safe_remove
    assert "stat -f '%d:%i:%u:%HT:%Lp'" in safe_remove
    assert 'test "$task_remove_root_real" = "$task_remove_root"' in safe_remove
    assert 'test "$(dirname -- "$task_remove_root_real")" = \\' in safe_remove
    assert command.count('rm -rf -- "$task_remove_root"') == 1
    assert 'rm -rf -- "$task_runtime_parent"' not in command
    assert 'rm -rf -- "$task_output_parent"' not in command
    assert 'rm -rf -- "$task_evidence_parent"' not in command
    assert "chmod" not in command

    warning = "local-only browser cleanup could not remove all task-owned state"
    runtime_cleanup_start = cleanup.index('if [ -n "$task_runtime_root" ]; then')
    output_cleanup_start = cleanup.index('if [ -n "$task_output_root" ]; then')
    warning_start = cleanup.index('if [ "$task_cleanup_status" -ne 0 ]; then')
    runtime_cleanup = cleanup[runtime_cleanup_start:output_cleanup_start]
    output_cleanup = cleanup[output_cleanup_start:warning_start]
    for cleanup_branch in (runtime_cleanup, output_cleanup):
        safe_remove_call = cleanup_branch.index("safe_remove_owned_directory")
        stderr_suppression = cleanup_branch.index("2>/dev/null")
        status_failure = cleanup_branch.index("task_cleanup_status=1")
        assert safe_remove_call < stderr_suppression < status_failure
        assert cleanup_branch.count("2>/dev/null") == 1
    assert cleanup.count(warning) == 1
    assert cleanup.count(">&2") == 1
    assert cleanup.count("2>/dev/null") == 3
    assert cleanup.count(">/dev/null 2>&1") == 1
    original_status_branch = cleanup.index('if [ "$task_original_status" -ne 0 ]; then')
    original_status_exit = cleanup.index(
        'exit "$task_original_status"', original_status_branch
    )
    cleanup_status_exit = cleanup.index(
        'exit "$task_cleanup_status"', original_status_exit
    )
    assert warning_start < cleanup.index(warning) < original_status_branch
    assert original_status_branch < original_status_exit < cleanup_status_exit

    assert "createdb -h 127.0.0.1 -p 5432" in command
    assert "dropdb --if-exists" not in command
    assert "E2E_RUNTIME_ROOT" in command
    assert "E2E_OUTPUT_ROOT" in command
    assert "mktemp -d" in command
    assert "pwd -P" in command
    assert "test ! -L" in command
    assert "env -u DATABASE_ENCRYPTION_KEY -u UPLOAD_DIR -u TEMP_EXTRACT_DIR" in command
    assert "LLM_PROVIDER=gemini" in command
    assert "LLM_SUMMARY_PROVIDER=" in command
    assert "LLM_SECTION_PROVIDER=" in command
    assert "LLM_DEDUP_PROVIDER=" in command
    assert "LLM_EXTRACTION_PROVIDER=" in command
    assert "E2E worker returns" not in section
    assert "already sandboxed model worker" not in section
