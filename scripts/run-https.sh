#!/usr/bin/env bash
# Serve the local dev stack over HTTPS via portless (https://medtimeline.localhost).
#
# Why: the plain dev stack binds loopback HTTP only, so the FastAPI backend never
# observes an `https` scheme and never emits HSTS. portless fronts both apps with
# trusted local TLS and injects `X-Forwarded-Proto: https`, which the backend's
# security-headers middleware already honors in dev — so HSTS finally fires.
#
# Layout:
#   frontend  ->  https://medtimeline.localhost       (portless app "medtimeline")
#   backend   ->  https://api.medtimeline.localhost    (portless app "api.medtimeline")
#
# portless assigns each app an ephemeral port via $PORT, so this never binds
# 8000/3000 and won't collide with a plain `just dev` stack on those ports.
#
# One-time setup (operator, requires sudo — see docs/operations-local-https.md):
#   npm install -g portless && portless trust
#
# Run:
#   just up-https            # or: bash scripts/run-https.sh
# Stop: Ctrl-C (both apps are torn down; the shared portless proxy daemon stays up).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# App names map directly to the .localhost hostnames portless serves.
FRONTEND_NAME="${PORTLESS_FRONTEND_NAME:-medtimeline}"
BACKEND_NAME="${PORTLESS_BACKEND_NAME:-api.medtimeline}"
FRONTEND_HOST="https://${FRONTEND_NAME}.localhost"
BACKEND_HOST="https://${BACKEND_NAME}.localhost"

if ! command -v portless >/dev/null 2>&1; then
  cat >&2 <<EOF
error: 'portless' is not installed.

  npm install -g portless   # Apache-2.0, ~0.4 MB, no runtime deps
  portless trust            # one-time: generate + trust the local CA (uses sudo)

See docs/operations-local-https.md for the full runbook.
EOF
  exit 1
fi

# The browser loads the https frontend, so its API calls must also be https
# (mixed content is blocked) and cross-origin to the api subdomain. Tell the
# frontend where the API lives and let the backend accept that origin. Env vars
# override .env, and APP_ENV stays "development" so no prod-only HTTPS redirect /
# trusted-host middleware is installed (that would break the loopback flow).
export NEXT_PUBLIC_API_URL="${BACKEND_HOST}/api/v1"
export CORS_ORIGINS="${FRONTEND_HOST},http://localhost:3000"

echo "Starting HTTPS dev stack via portless"
echo "  frontend -> ${FRONTEND_HOST}"
echo "  backend  -> ${BACKEND_HOST}  (CORS_ORIGINS=${CORS_ORIGINS})"
echo "  NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}"
echo

pids=()
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "Stopping HTTPS dev stack..."
  for pid in "${pids[@]:-}"; do
    [ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Backend: portless injects $PORT; uvicorn must be told to bind it (uvicorn does
# not read $PORT on its own). --proxy-headers makes uvicorn trust portless's
# X-Forwarded-* from loopback so request.url.scheme reads as https too. HSTS does
# not depend on that flag (the middleware reads the X-Forwarded-Proto header
# directly), so a forwarded-ip mismatch still yields HSTS.
(
  cd "${ROOT}/backend"
  exec portless "${BACKEND_NAME}" sh -c \
    'exec uv run uvicorn app.main:app --host 127.0.0.1 --port "$PORT" --proxy-headers --forwarded-allow-ips="127.0.0.1,::1"'
) &
pids+=("$!")

# Frontend: Next.js reads $PORT from the environment, so portless wiring is enough.
(
  cd "${ROOT}/frontend"
  exec portless "${FRONTEND_NAME}" npm run dev
) &
pids+=("$!")

echo "Both apps launching. First boot fetches/loads models and compiles Next — give it a moment."
echo "Verify HSTS:  curl -sI ${BACKEND_HOST}/api/v1/health | grep -i strict-transport"
wait
