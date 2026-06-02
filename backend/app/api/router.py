from __future__ import annotations

from fastapi import APIRouter

from app.api import audit, auth, dashboard, dedup, records, summary, timeline, upload

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(records.router)
api_router.include_router(timeline.router)
api_router.include_router(upload.router)
api_router.include_router(summary.router)
api_router.include_router(dedup.router)
api_router.include_router(dashboard.router)
api_router.include_router(audit.router)
