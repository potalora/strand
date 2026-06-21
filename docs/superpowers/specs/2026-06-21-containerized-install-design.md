# Containerized install — design spec (2026-06-21)

**Goal:** Let someone go from a fresh clone to a running MedTimeline in one command, cross-platform, without the macOS-only Homebrew setup. Two layers, desktop deferred.

**Audience (decided):** self-hosters / developers. Docker is an acceptable prerequisite. A non-technical double-click desktop app (Tauri) is a separate, larger initiative and is **out of scope here**.

## Layer 0 — cross-platform dev ergonomics

Replace the macOS-only `scripts/setup-local.sh` as the *primary* path (keep it as a documented native alternative) with:

- **`uv`** for the backend (env + Python pin + deps + the opt-in `clinical-nlp` extra). `uv.lock` already exists.
- **`corepack` + `pnpm`** for the frontend.
- **`docker compose`** for Postgres + Redis (the only cross-platform option; Homebrew is macOS-only).
- A **`justfile`** as the single entrypoint: `just setup`, `just dev`, `just test`, `just up`, `just down`, `just gen-secrets`, `just setup-clinical`.

Dev flow: `git clone → just setup → just dev`.

## Layer A — one-command self-host

`docker compose up -d` brings up the whole stack. Build-from-clone in v1; prebuilt GHCR images (multi-arch) added so tagged releases also support a no-clone "download the compose file + up" flow.

### Services (`docker-compose.yml`)
| Service | Image / build | Notes |
|---|---|---|
| `db` | `postgres:16` | mounts `scripts/init-db.sql` → `/docker-entrypoint-initdb.d/` (pgcrypto, uuid-ossp, pg_trgm); `pg_isready` healthcheck; named volume `pgdata` |
| `redis` | `redis:7` | `redis-cli ping` healthcheck |
| `migrate` | backend image | one-shot `alembic upgrade head`, `depends_on: db: service_healthy`, `restart: "no"` |
| `backend` | `./backend` | bind **`127.0.0.1:8000`**; `depends_on` db+redis healthy, migrate completed; named volume for uploads |
| `frontend` | `./frontend` | bind **`127.0.0.1:3000`**; `depends_on: backend` |

### Key decisions (settled by research)
- **Bind `127.0.0.1` only** — local-first/privacy; remote/TLS is an optional Caddy overlay later, not the default.
- **One-shot migrate service** (not entrypoint migrations) — runs exactly once, fails loudly, decoupled from app start.
- **Bake `en_core_web_md` at image build** — first boot must work offline (the PHI-NER warm-load needs it).
- **`clinical-nlp` OFF by default** in the image (lighter; Gemini extraction via the hybrid engine's fail-open). On-device extraction via `--build-arg CLINICAL_NLP=true`.
- **No auto-update** — wrong for health data; Watchtower conflicts with pinned tags. Upgrade = bump `APP_VERSION` → `docker compose pull && up -d`.
- **Pinned image tags** via `metadata-action` semver; multi-arch `linux/amd64,linux/arm64`.
- **Secrets generated, never defaulted** — `gen-secrets` fills `DB_PASSWORD`, `JWT_SECRET_KEY`, `DATABASE_ENCRYPTION_KEY` (config hard-fails on empty/default).
- **`NEXT_PUBLIC_API_URL` is build-time** — baked to `http://localhost:8000/api/v1` for the default localhost self-host; remote access needs a rebuild or the proxy overlay (documented).

### File manifest
- `backend/Dockerfile`, `backend/.dockerignore`
- `frontend/Dockerfile`, `frontend/.dockerignore`, `frontend/next.config.*` (+ `output: 'standalone'`)
- `docker-compose.yml`, `.env.docker.example`
- `justfile`, `scripts/gen-secrets.sh`
- `.github/workflows/docker-publish.yml`
- `README.md` (Quick start: Docker one-command + native `just`)

## Verification
Docker is available locally (28.x). Verify by actually running:
1. `docker compose build` (both images build; spaCy model baked).
2. `docker compose up -d`; wait for healthchecks; `migrate` exits 0.
3. `curl 127.0.0.1:8000/api/v1/health` → 200; register/login round-trip.
4. Load `http://localhost:3000`, confirm the app talks to the API.
5. `docker compose down && up -d` → data persists (named volume).

## Out of scope
Desktop app (Tauri/SQLite/sidecar), reverse-proxy/TLS default, Kubernetes/Helm.
