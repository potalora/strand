# syntax=docker/dockerfile:1

# =============================================================================
# MedTimeline backend image.
#
# Build context is ./backend, so all COPY paths are relative to backend/.
# The app RUNS FROM SOURCE (no [build-system] in pyproject.toml) — we install
# only the locked dependencies into a venv, then copy that venv + the source.
#
# Both stages use the SAME python:3.11-slim base so the venv's hard-coded
# interpreter path (/app/.venv/bin/python) stays valid when copied between
# stages. Do NOT swap the base for the astral uv image.
# =============================================================================

# -----------------------------------------------------------------------------
# Builder: resolve + install dependencies into /app/.venv via uv.
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# uv binary (pinned-by-tag latest) — used only at build time.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# build-essential: needed if CLINICAL_NLP=true drags in sdists that compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# false (default): base deps only. true: + .[clinical-nlp] (scispaCy/medspaCy).
ARG CLINICAL_NLP=false

# Lockfile + manifest first so the dependency layer caches independently of source.
COPY pyproject.toml uv.lock ./

# Install dependencies only (never the project itself — it runs from source).
# Cache mount keeps uv's wheel cache across builds without baking it into a layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$CLINICAL_NLP" = "true" ]; then \
        uv sync --frozen --no-install-project --no-dev --extra clinical-nlp; \
    else \
        uv sync --frozen --no-install-project --no-dev; \
    fi

# uv-created venvs have no pip, but spaCy's `download` shells out to pip — add pip
# to the venv first, then bake the PERSON-NER model so first boot is offline-safe
# (PHI scrubber). spaCy picks the model version matching the resolved spaCy.
RUN uv pip install --python /app/.venv/bin/python pip \
    && /app/.venv/bin/python -m spacy download en_core_web_md

# Optional clinical-NLP scispaCy NER model. Only when CLINICAL_NLP=true.
# (Untested path — default build is false. Included for completeness.)
RUN if [ "$CLINICAL_NLP" = "true" ]; then \
        /app/.venv/bin/pip install \
            https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz; \
    fi

# -----------------------------------------------------------------------------
# Runtime: copy the venv + source, run as a non-root user.
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# libgomp1: runtime OpenMP lib for numpy/spaCy. curl: container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user (fixed high uid for predictable volume ownership).
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Dependencies first (venv), then application source.
COPY --from=builder /app/.venv /app/.venv
COPY . /app/

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UPLOAD_DIR=/data/uploads \
    TEMP_EXTRACT_DIR=/data/tmp

# Persistent data dirs (uploads + temp extraction), owned by the runtime user.
RUN mkdir -p /data/uploads /data/tmp \
    && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
