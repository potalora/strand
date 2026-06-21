#!/usr/bin/env bash
# gen-secrets.sh — fill the blank secrets in the Docker .env, idempotently.
#
#   - Generates strong random values for DB_PASSWORD, JWT_SECRET_KEY,
#     DATABASE_ENCRYPTION_KEY.
#   - NEVER overwrites an existing non-empty value.
#   - NEVER touches any other setting in .env.
#   - Creates .env from .env.docker.example if it doesn't exist yet.
#
# Usage:  scripts/gen-secrets.sh [path-to-env-file]   (default: <repo>/.env)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${1:-$ROOT/.env}"
EXAMPLE_FILE="$ROOT/.env.docker.example"

if ! command -v openssl >/dev/null 2>&1; then
  echo "error: openssl is required to generate secrets" >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  if [ -f "$EXAMPLE_FILE" ]; then
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    echo "Created $ENV_FILE from $(basename "$EXAMPLE_FILE")"
  else
    : > "$ENV_FILE"
    echo "Created empty $ENV_FILE"
  fi
fi

# Set a key only if it is missing or empty. Hex values are URL-safe (no / + =),
# so DB_PASSWORD is safe to embed inside DATABASE_URL.
set_secret() {
  key="$1"
  val="$2"
  if grep -qE "^${key}=.+" "$ENV_FILE"; then
    echo "  ${key}: already set — left unchanged"
    return
  fi
  if grep -qE "^${key}=" "$ENV_FILE"; then
    # Key present but empty → replace in place (portable: no sed -i).
    tmp="$(mktemp)"
    sed "s|^${key}=.*|${key}=${val}|" "$ENV_FILE" > "$tmp"
    cat "$tmp" > "$ENV_FILE"
    rm -f "$tmp"
    echo "  ${key}: filled"
  else
    # Key absent → append.
    printf '%s=%s\n' "$key" "$val" >> "$ENV_FILE"
    echo "  ${key}: added"
  fi
}

echo "Generating secrets in $ENV_FILE ..."
set_secret DB_PASSWORD             "$(openssl rand -hex 24)"
set_secret JWT_SECRET_KEY          "$(openssl rand -hex 32)"
set_secret DATABASE_ENCRYPTION_KEY "$(openssl rand -hex 32)"
echo "Done."

if [ ! -x "$0" ]; then
  echo "Tip: make this script executable with:  chmod +x scripts/gen-secrets.sh"
fi
