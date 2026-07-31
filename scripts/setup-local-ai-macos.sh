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
uv run --project "$worker_directory" python -c \
    "from local_ai_mlx_worker.__main__ import RUNTIME; assert RUNTIME == 'mlx-vlm-0.5.0'"

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
