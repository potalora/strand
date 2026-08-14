#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: ./scripts/setup-local-ai-macos.sh [--download-pack]" \
        "" \
        "Install the isolated Apple MLX worker for the optional local-AI path." \
        "This command does not download model files by default." \
        "" \
        "  --download-pack  Install the locked model pack after the worker runtime." \
        "  --help           Show this help."
}

download_pack=false
for argument in "$@"; do
    case "$argument" in
        --download-pack)
            download_pack=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unsupported argument: %s\n' "$argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    printf '%s\n' "ERROR: the MLX runtime requires Apple Silicon macOS." >&2
    exit 1
fi

physical_memory_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
if [[ ! "$physical_memory_bytes" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "ERROR: physical memory could not be determined." >&2
    exit 1
fi
minimum_memory_bytes=$((16 * 1024 * 1024 * 1024))
if (( physical_memory_bytes < minimum_memory_bytes )); then
    printf '%s\n' "ERROR: the validated Apple profile requires at least 16 GB of unified memory." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "ERROR: uv is required. Install it with 'brew install uv'." >&2
    exit 1
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_directory/.." && pwd)"
worker_directory="$repository_root/workers/local_ai/apple_mlx"
pack_cli="$repository_root/backend/scripts/local_ai_pack.py"

if [[ "$download_pack" == true ]]; then
    if [[ ! -f "$pack_cli" ]]; then
        printf '%s\n' "ERROR: the model-pack command is not available in this checkout." >&2
        exit 1
    fi
    (
        cd "$repository_root/backend"
        uv run python scripts/local_ai_pack.py preflight
    )
fi

free_kib="$(df -Pk "$repository_root" | awk 'NR == 2 {print $4}')"
if [[ ! "$free_kib" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "ERROR: available disk space could not be determined." >&2
    exit 1
fi
required_free_kib=$((2 * 1024 * 1024))
if [[ "$download_pack" == true ]]; then
    # The current pack is about 10 GB. Keep 15 GiB free after installation so
    # OCR scratch, database work, and macOS swap cannot consume the last space.
    required_free_kib=$((25 * 1024 * 1024))
fi
if (( free_kib < required_free_kib )); then
    required_gib=$((required_free_kib / 1024 / 1024))
    printf 'ERROR: at least %s GiB of free disk is required for this operation.\n' \
        "$required_gib" >&2
    exit 1
fi

umask 077
printf '%s\n' "Installing the isolated Apple MLX worker runtime..."
uv sync --frozen --project "$worker_directory"

worker_venv="$worker_directory/.venv"
worker_python="$worker_venv/bin/python"
worker_launcher="$worker_venv/bin/local-ai-mlx-worker"
shopt -s nullglob
site_packages_candidates=("$worker_venv"/lib/python*/site-packages)
shopt -u nullglob
if (( ${#site_packages_candidates[@]} != 1 )); then
    printf '%s\n' \
        "ERROR: expected exactly one worker site-packages directory after sync." >&2
    exit 1
fi
site_packages="${site_packages_candidates[0]}"
worker_lib="$worker_venv/lib"
worker_python_directory="${site_packages%/site-packages}"
for runtime_directory in \
    "$worker_venv" \
    "$worker_lib" \
    "$worker_python_directory" \
    "$site_packages"; do
    if [[ -L "$runtime_directory" || ! -d "$runtime_directory" ]]; then
        printf '%s\n' \
            "ERROR: worker site-packages must be a regular in-venv directory." >&2
        exit 1
    fi
done
worker_venv_real="$(cd -- "$worker_venv" && pwd -P)"
site_packages_real="$(cd -- "$site_packages" && pwd -P)"
case "$site_packages_real" in
    "$worker_venv_real"/lib/python*/site-packages)
        ;;
    *)
        printf '%s\n' \
            "ERROR: worker site-packages must be a regular in-venv directory." >&2
        exit 1
        ;;
esac

virtualenv_pth="$site_packages/_virtualenv.pth"
virtualenv_py="$site_packages/_virtualenv.py"
pth_present=false
py_present=false
if [[ -e "$virtualenv_pth" || -L "$virtualenv_pth" ]]; then
    pth_present=true
fi
if [[ -e "$virtualenv_py" || -L "$virtualenv_py" ]]; then
    py_present=true
fi
if [[ "$pth_present" == true || "$py_present" == true ]]; then
    if [[ "$pth_present" != true \
        || "$py_present" != true \
        || -L "$virtualenv_pth" \
        || ! -f "$virtualenv_pth" \
        || -L "$virtualenv_py" \
        || ! -f "$virtualenv_py" ]]; then
        printf '%s\n' \
            "ERROR: uv bootstrap must be absent or an exact regular-file pair." >&2
        exit 1
    fi
    rm -- "$virtualenv_pth" "$virtualenv_py"
fi
if [[ -e "$virtualenv_pth" \
    || -L "$virtualenv_pth" \
    || -e "$virtualenv_py" \
    || -L "$virtualenv_py" ]]; then
    printf '%s\n' "ERROR: uv bootstrap removal did not complete." >&2
    exit 1
fi

if [[ ! -x "$worker_python" ]]; then
    printf '%s\n' "ERROR: the worker venv interpreter is unavailable." >&2
    exit 1
fi
if [[ -L "$worker_launcher" || ! -f "$worker_launcher" || ! -x "$worker_launcher" ]]; then
    printf '%s\n' "ERROR: the worker launcher is unavailable." >&2
    exit 1
fi

validation_scratch="$(mktemp -d "${TMPDIR:-/tmp}/local-ai-worker-validation.XXXXXX")"
chmod 700 "$validation_scratch"
cleanup_validation_scratch() {
    if [[ -n "${validation_scratch:-}" \
        && -d "$validation_scratch" \
        && ! -L "$validation_scratch" ]]; then
        rm -rf -- "$validation_scratch"
    fi
}
trap cleanup_validation_scratch EXIT
expected_body="$validation_scratch/expected-body"
cat <<'PYTHON_LAUNCHER' > "$expected_body"
# -*- coding: utf-8 -*-
import sys
from local_ai_mlx_worker.__main__ import main
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
PYTHON_LAUNCHER
expected_polyglot_launcher="$validation_scratch/expected-polyglot-launcher"
{
    printf '%s\n' '#!/bin/sh'
    printf "'''exec' '%s' \"\$0\" \"\$@\"\n" "$worker_python"
    printf '%s\n' "' '''"
    cat "$expected_body"
} > "$expected_polyglot_launcher"
expected_direct_launcher="$validation_scratch/expected-direct-launcher"
{
    printf '#!%s\n' "$worker_python"
    cat "$expected_body"
} > "$expected_direct_launcher"
if ! cmp -s "$expected_polyglot_launcher" "$worker_launcher" \
    && ! cmp -s "$expected_direct_launcher" "$worker_launcher"; then
    printf '%s\n' "ERROR: the worker launcher is invalid." >&2
    exit 1
fi

validation_pycache="$validation_scratch/pycache"
mkdir "$validation_pycache"
chmod 700 "$validation_pycache"
env -u PYTHONPATH -u PYTHONHOME \
    PYTHONNOUSERSITE=1 \
    PYTHONSAFEPATH=1 \
    PYTHONPYCACHEPREFIX="$validation_pycache" \
    "$worker_python" -c \
    "from local_ai_mlx_worker.__main__ import RUNTIME; assert RUNTIME == 'mlx-vlm-0.5.0'"
cleanup_validation_scratch
validation_scratch=""
trap - EXIT

if [[ "$download_pack" == true ]]; then
    printf '%s\n' "Downloading and validating the locked local model pack..."
    (
        cd "$repository_root/backend"
        uv run python scripts/local_ai_pack.py install
    )
else
    printf '%s\n' \
        "Runtime installed. Model files were not downloaded." \
        "Run 'just local-ai-pack-download' when you want to install the optional pack."
fi
