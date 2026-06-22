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
