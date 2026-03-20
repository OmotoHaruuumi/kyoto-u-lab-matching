"""
backend/embedding_api/main.py

Embedding API — FastAPI application.
Endpoint: POST /api/v1/embed
  - Accepts a list of text strings
  - Calls Google Gemini text-embedding-004 to generate embeddings (via google-genai SDK)
  - Returns a list of float vectors

In development / mock mode (MOCK_EMBEDDING=true) a deterministic random
vector is returned instead of calling the Gemini API, so the service can
be tested without a valid API key.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from google import genai as google_genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "models/text-embedding-004")
EMBEDDING_DIM: int = 768
MOCK_MODE: bool = os.environ.get("MOCK_EMBEDDING", "false").lower() == "true"
DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")

# Initialise the google-genai client (new SDK)
_genai_client: google_genai.Client | None = None
if GEMINI_API_KEY:
    _genai_client = google_genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kyoto Lab Matching — Embedding API",
    description="Generates text embeddings via Google Gemini text-embedding-004.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="List of text strings to embed (max 100 per request).",
        examples=[["機械学習を用いた医療診断の研究", "量子コンピューティングの基礎研究"]],
    )
    task_type: str = Field(
        default="RETRIEVAL_DOCUMENT",
        description=(
            "Gemini embedding task type. "
            "Use RETRIEVAL_QUERY for query texts, RETRIEVAL_DOCUMENT for documents."
        ),
    )


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int
    mock: bool


# ---------------------------------------------------------------------------
# Helper: mock embedding (deterministic per input text)
# ---------------------------------------------------------------------------
def _mock_embedding(text: str) -> list[float]:
    """
    Returns a deterministic pseudo-random unit vector derived from the
    SHA-256 hash of the input text.  Useful for testing without an API key.
    """
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    norm = (sum(v**2 for v in vec) ** 0.5) or 1.0
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# Helper: call Gemini API via google-genai SDK
# ---------------------------------------------------------------------------
async def _gemini_embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Call Gemini text-embedding-004 and return vectors using the google-genai SDK."""
    if _genai_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini client not initialised. Set GEMINI_API_KEY.",
        )
    try:
        result = _genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=genai_types.EmbedContentConfig(task_type=task_type),
        )
        return [list(e.values) for e in result.embeddings]
    except Exception as exc:
        logger.exception("Gemini embedding call failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error: {exc}",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    """Readiness probe: checks DB and Redis connectivity."""
    # --- DB check ---
    db_status = "ok"
    if DATABASE_URL:
        try:
            engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
        except Exception as exc:
            db_status = f"error: {exc}"
    else:
        db_status = "not configured"

    # --- Redis check ---
    redis_status = "ok"
    try:
        r = aioredis.from_url(REDIS_URL, socket_connect_timeout=3)
        await r.ping()
        await r.aclose()
    except Exception as exc:
        redis_status = f"error: {exc}"

    overall = "ok" if db_status == "ok" and redis_status == "ok" else "degraded"
    return {
        "status": overall,
        "mock": MOCK_MODE,
        "model": EMBEDDING_MODEL,
        "db": db_status,
        "redis": redis_status,
    }


@app.post(
    "/api/v1/embed",
    response_model=EmbedResponse,
    summary="Generate text embeddings",
    tags=["embedding"],
)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """
    Generate vector embeddings for a list of text strings.

    - **texts**: 1–100 input strings
    - **task_type**: Gemini task type (default: RETRIEVAL_DOCUMENT)

    Returns a list of float vectors of dimension 768.
    """
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 100 texts per request.",
        )

    if MOCK_MODE or not GEMINI_API_KEY:
        logger.info("Mock mode: returning deterministic embeddings for %d texts.", len(request.texts))
        embeddings = [_mock_embedding(t) for t in request.texts]
        return EmbedResponse(
            embeddings=embeddings,
            model=f"mock:{EMBEDDING_MODEL}",
            dim=EMBEDDING_DIM,
            mock=True,
        )

    logger.info("Calling Gemini API for %d texts (task_type=%s).", len(request.texts), request.task_type)
    embeddings = await _gemini_embed(request.texts, request.task_type)
    return EmbedResponse(
        embeddings=embeddings,
        model=EMBEDDING_MODEL,
        dim=EMBEDDING_DIM,
        mock=False,
    )
