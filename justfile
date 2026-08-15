# MedTimeline task runner — run `just` (no args) to list recipes.
#   Full container stack:  just up
#   Native dev (app on host, db+redis in Docker):  just setup  →  just dev
set dotenv-load := true

# List available recipes
default:
    @just --list

# Generate strong secrets into .env (idempotent; creates .env from the example if missing)
gen-secrets:
    bash scripts/gen-secrets.sh

# Build + start the full containerized stack (localhost only)
up:
    @[ -f .env ] || bash scripts/gen-secrets.sh
    docker compose up -d --build
    @echo ""
    @echo "MedTimeline starting →  frontend http://localhost:3000   ·   API http://localhost:8000"

# Stop the stack (named volumes / data are preserved)
down:
    docker compose down

# Follow logs from all services
logs:
    docker compose logs -f

# Show service status
ps:
    docker compose ps

# One-time native dev setup: db+redis in Docker, app deps installed on the host (uv + npm)
setup:
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d db redis
    cd backend && uv sync && uv run python -m spacy download en_core_web_md && uv run alembic upgrade head
    cd frontend && npm install
    @echo "Setup complete.  Start the app with:  just dev"

# Add the optional on-device clinical-NLP stack (heavier: medspaCy + scispaCy, ~1.8GB RAM)
setup-clinical:
    cd backend && uv sync --extra clinical-nlp
    @echo "Now install the scispaCy NER model (not on PyPI):"
    @echo "  cd backend && uv run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"

# Install only the optional host-native Apple MLX worker. Model downloads stay explicit.
local-ai-runtime-install:
    ./scripts/setup-local-ai-macos.sh

# Download, hash-check, fixture-validate, and atomically activate the locked pack.
local-ai-pack-download:
    cd backend && uv run python scripts/local_ai_pack.py install

# Re-run the exact offline runtime and synthetic-fixture validation gate.
local-ai-candidate-verify:
    cd backend && uv run python scripts/local_ai_candidate_pack.py verify

# Re-run the promoted release-evidence-required lifecycle validation gate.
local-ai-pack-verify:
    cd backend && uv run python scripts/local_ai_pack.py verify

# Run the content-free three-cold-run Apple resource release gate.
local-ai-benchmark:
    cd backend && uv run python scripts/benchmark_local_ai.py --runs 3 --output artifacts/local-ai-benchmark.json

# Run only the source-controlled synthetic fidelity suite.
local-ai-fidelity:
    cd backend && env -u REAL_MEDICAL_FIXTURES_DIR uv run python scripts/run_local_ai_fidelity.py --output artifacts/local-ai-fidelity.json

# Bind passing benchmark and fidelity artifacts to the exact shipped lock.
local-ai-release-promote:
    cd backend && uv run python scripts/promote_local_ai_release.py --manifest app/model_manifests/apple-m4-16gb-v2.lock.json --benchmark artifacts/local-ai-benchmark.json --fidelity artifacts/local-ai-fidelity.json --output app/model_manifests/apple-m4-16gb-v2.release.json

# Remove model artifacts only. Clinical records, evidence, and checkpoints remain.
local-ai-pack-remove:
    cd backend && uv run python scripts/local_ai_pack.py remove

# Native dev: bring up db+redis, then print how to run backend + frontend (two processes)
dev:
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d db redis
    @echo "db + redis up on 127.0.0.1:5432 / 6379.  Now run, in two terminals:"
    @echo "  just backend     (cd backend && uv run uvicorn app.main:app --reload --port 8000)"
    @echo "  just frontend    (cd frontend && npm run dev)"

# Run the backend natively with hot-reload — needs db+redis (`just dev`) first
backend:
    cd backend && uv run uvicorn app.main:app --reload --port 8000

# Run the frontend dev server natively — needs the backend running
frontend:
    cd frontend && npm run dev

# Serve the dev stack over local HTTPS via portless (https://medtimeline.localhost)
# so the backend sees an https scheme and emits HSTS. One-time setup (operator):
#   npm install -g portless && portless trust   (see docs/operations-local-https.md)
up-https:
    bash scripts/run-https.sh

# Fast backend test suite (excludes slow / live-Gemini)
test:
    cd backend && uv run pytest -m "not slow"
