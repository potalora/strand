# syntax=docker/dockerfile:1

# =============================================================================
# MedTimeline frontend image (Next.js 16, App Router).
#
# Build context is ./frontend. Lockfile present is package-lock.json (no
# pnpm-lock.yaml), so we install with `npm ci`.
#
# Produces a Next.js "standalone" build (next.config.ts sets
# output: 'standalone'): a self-contained server.js + trimmed node_modules,
# which keeps the runtime image small. NEXT_PUBLIC_API_URL is a build-time
# inlined value (baked into the client bundle), so it must be set BEFORE
# `npm run build`.
# =============================================================================

# -----------------------------------------------------------------------------
# deps: install node_modules from the lockfile (cached on package*.json).
# -----------------------------------------------------------------------------
FROM node:22-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# -----------------------------------------------------------------------------
# build: compile the Next.js standalone output.
# -----------------------------------------------------------------------------
FROM node:22-bookworm-slim AS build
WORKDIR /app

# Public API base URL — inlined into the client bundle at build time.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# -----------------------------------------------------------------------------
# runtime: serve the standalone server as the non-root `node` user.
# -----------------------------------------------------------------------------
FROM node:22-bookworm-slim AS runtime
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Standalone bundle: server.js + minimal node_modules + traced app files.
# .next/static and public/ are NOT included in standalone and must be added.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public

# `node` user/group ship with the base image (uid/gid 1000).
USER node

EXPOSE 3000

CMD ["node", "server.js"]
