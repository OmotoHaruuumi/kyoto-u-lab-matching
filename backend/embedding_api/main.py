"""
backend/embedding_api/main.py

Embedding API — FastAPI application.
Endpoint: POST /api/v1/embed
  - Accepts a list of text strings
  - Calls Google Gemini text-embedding-004 to generate embeddings
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

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kyoto Lab Matching — Embedding API",
    description="Generates text embeddings via Google Gemini text-embedding-004.",
    version="0.1.0",
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
# Helper: call Gemini API
# ---------------------------------------------------------------------------
async def _gemini_embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Call Gemini text-embedding-004 and return vectors."""
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type=task_type,
        )
        return result["embedding"] if isinstance(result["embedding"][0], list) else [result["embedding"]]
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
    return {"status": "ok", "mock": MOCK_MODE, "model": EMBEDDING_MODEL}


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
