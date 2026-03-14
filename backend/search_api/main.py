"""
backend/search_api/main.py

Search API — FastAPI application (Phase 1 placeholder).
Full hybrid-search logic (BM25 + vector + RRF) will be implemented in Phase 2.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kyoto Lab Matching — Search API",
    description="Hybrid search over Kyoto University research labs.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    """Liveness probe."""
    return {"status": "ok", "phase": 1}


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    return {"message": "Search API is running. Full implementation coming in Phase 2."}
