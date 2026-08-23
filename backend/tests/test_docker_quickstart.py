"""Contracts for the loopback-only Docker quickstart."""

from __future__ import annotations

from pathlib import Path
import shlex

from app.config import Settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VALID_KEY = "ab" * 32
STRONG_SECRET = "x" * 48


def _environment_value(source: str, name: str) -> str:
    prefix = f"{name}="
    return next(
        line.removeprefix(prefix)
        for line in source.splitlines()
        if line.startswith(prefix)
    )


def _yaml_mapping_value(source: str, *path: str) -> str:
    parents: list[tuple[int, str]] = []
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip())
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        while parents and parents[-1][0] >= indentation:
            parents.pop()
        current_path = (*[parent[1] for parent in parents], key)
        if current_path == path:
            return value.strip()
        if not value.strip():
            parents.append((indentation, key))
    raise KeyError(".".join(path))


def _dockerfile_instructions(source: str) -> list[str]:
    instructions: list[str] = []
    current = ""
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            current += f"{stripped[:-1].rstrip()} "
            continue
        current += stripped
        instructions.append(current)
        current = ""
    return instructions


def _final_dockerfile_environment(source: str) -> dict[str, str]:
    instructions = _dockerfile_instructions(source)
    final_stage_start = max(
        index
        for index, instruction in enumerate(instructions)
        if instruction.upper().startswith("FROM ")
    )
    environment: dict[str, str] = {}
    for instruction in instructions[final_stage_start + 1 :]:
        if not instruction.startswith("ENV "):
            continue
        for assignment in shlex.split(instruction.removeprefix("ENV ")):
            name, separator, value = assignment.partition("=")
            if separator:
                environment[name] = value
    return environment


def test_docker_env_example_uses_nonproduction_mode_for_loopback_http() -> None:
    example = (REPOSITORY_ROOT / ".env.docker.example").read_text(encoding="utf-8")

    assert _environment_value(example, "CORS_ORIGINS") == "http://localhost:3000"
    assert (
        _environment_value(example, "NEXT_PUBLIC_API_URL")
        == "http://localhost:8000/api/v1"
    )
    configured = Settings(
        app_env=_environment_value(example, "APP_ENV"),
        jwt_secret_key=STRONG_SECRET,
        database_encryption_key=VALID_KEY,
    )
    assert configured.is_production is False


def test_compose_fallback_uses_nonproduction_mode_for_loopback_http() -> None:
    compose = (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert (
        _yaml_mapping_value(compose, "services", "backend", "environment", "APP_ENV")
        == "${APP_ENV:-development}"
    )


def test_bare_backend_image_still_defaults_to_production() -> None:
    dockerfile = (REPOSITORY_ROOT / "backend" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert _final_dockerfile_environment(dockerfile)["APP_ENV"] == "production"
