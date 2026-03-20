"""
backend/search_api/main.py

Search API — FastAPI application.
Implements Hybrid Search (Vector + Keyword) over Kyoto University research labs.
Uses Weighted RRF (Reciprocal Rank Fusion) for score integration.

Tuning parameters (all overridable via env vars):
  RRF_K              — RRF constant (default 30; lower = stronger rank-based signal)
  RRF_VECTOR_WEIGHT  — weight for vector search hits (default 0.7)
  RRF_KEYWORD_WEIGHT — weight for keyword search hits (default 0.3)
  SEARCH_CANDIDATE_LIMIT — candidate pool size per search mode (default 100)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
import httpx
import redis.asyncio as aioredis
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector

from backend.shared.database import async_session_maker, get_db
from backend.shared.models import EmbeddingChunk, Lab

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kyoto Lab Matching — Search API",
    description="Hybrid search over Kyoto University research labs using pgvector and weighted RRF.",
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
class ChunkMatch(BaseModel):
    chunk_text: str
    source_type: str
    combined_score: float

class LabResult(BaseModel):
    lab_id: int
    name: str
    name_en: str | None
    department: str | None
    faculty: str | None
    lab_url: str | None
    description: str | None
    keywords: list[str] | None
    matched_chunks: list[ChunkMatch]
    total_score: float

class SearchResponse(BaseModel):
    query: str
    results: list[LabResult]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")
EMBEDDING_API_URL: str = os.environ.get("EMBEDDING_API_URL", "http://embedding_api:8001")

# RRF tuning — all overridable at runtime via environment variables
RRF_K: int = int(os.environ.get("RRF_K", "30"))
VECTOR_WEIGHT: float = float(os.environ.get("RRF_VECTOR_WEIGHT", "0.7"))
KEYWORD_WEIGHT: float = float(os.environ.get("RRF_KEYWORD_WEIGHT", "0.3"))
CANDIDATE_LIMIT: int = int(os.environ.get("SEARCH_CANDIDATE_LIMIT", "100"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def get_query_embedding(query: str) -> list[float]:
    """Call the embedding API to get the vector for the search query."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{EMBEDDING_API_URL}/api/v1/embed",
                json={"texts": [query], "task_type": "RETRIEVAL_QUERY"},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
    except Exception as exc:
        logger.error(f"Failed to get query embedding: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed.",
        )


def compute_rrf(
    rank_vector: int | None,
    rank_keyword: int | None,
    k: int = RRF_K,
    w_v: float = VECTOR_WEIGHT,
    w_k: float = KEYWORD_WEIGHT,
) -> float:
    """
    Weighted Reciprocal Rank Fusion score.
    Vector search is weighted higher (semantic relevance > literal match).
    If a rank is None, that component contributes 0.
    """
    score = 0.0
    if rank_vector is not None:
        score += w_v / (k + rank_vector)
    if rank_keyword is not None:
        score += w_k / (k + rank_keyword)
    return score


def _keyword_conditions(query: str):
    """
    Build OR conditions for keyword search.
    Splits the query into tokens and matches any token against chunk_text.
    Single-character tokens are skipped to reduce noise.
    """
    tokens = [t.strip() for t in query.split() if len(t.strip()) > 1]
    if not tokens:
        tokens = [query]
    return [EmbeddingChunk.chunk_text.ilike(f"%{token}%") for token in tokens]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    """Readiness probe: checks DB and Redis connectivity."""
    # --- DB check ---
    db_status = "ok"
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"

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
        "db": db_status,
        "redis": redis_status,
    }


@app.get(
    "/api/v1/search",
    response_model=SearchResponse,
    summary="Hybrid search for labs",
    tags=["search"],
)
async def search(
    q: str = Query(..., description="Search query string", min_length=1),
    limit: int = Query(10, description="Max lab results to return", ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Executes a hybrid search combining pgvector cosine distance and token-based
    keyword matching. Results are merged using Weighted RRF (vector weight 0.7,
    keyword weight 0.3) with k=30 for sharper rank differentiation.
    """
    logger.info(f"Search query='{q}' limit={limit}")

    # 1. Get query embedding
    query_vector = await get_query_embedding(q)

    # 2. Vector Search — top CANDIDATE_LIMIT chunks by cosine distance
    stmt_vector = (
        select(EmbeddingChunk.id, EmbeddingChunk.lab_id)
        .order_by(EmbeddingChunk.embedding.cosine_distance(query_vector))
        .limit(CANDIDATE_LIMIT)
    )
    res_vector = await db.execute(stmt_vector)
    vector_rows = res_vector.all()
    vector_ranks = {row.id: rank for rank, row in enumerate(vector_rows, start=1)}

    # 3. Keyword Search — token-based OR match, top CANDIDATE_LIMIT chunks
    kw_conditions = _keyword_conditions(q)
    stmt_keyword = (
        select(EmbeddingChunk.id, EmbeddingChunk.lab_id)
        .where(or_(*kw_conditions))
        .limit(CANDIDATE_LIMIT)
    )
    res_keyword = await db.execute(stmt_keyword)
    keyword_rows = res_keyword.all()
    keyword_ranks = {row.id: rank for rank, row in enumerate(keyword_rows, start=1)}

    # 4. Weighted RRF scoring at chunk level
    all_chunk_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())
    if not all_chunk_ids:
        return SearchResponse(query=q, results=[])

    chunk_scores: dict[int, float] = {
        cid: compute_rrf(vector_ranks.get(cid), keyword_ranks.get(cid))
        for cid in all_chunk_ids
    }

    # 5. Fetch full chunk data & aggregate by Lab
    stmt_chunks = select(EmbeddingChunk).where(EmbeddingChunk.id.in_(all_chunk_ids))
    res_chunks = await db.execute(stmt_chunks)
    chunks = res_chunks.scalars().all()

    lab_scores: dict[int, float] = {}
    lab_matched_chunks: dict[int, list[ChunkMatch]] = {}

    for chunk in chunks:
        score = chunk_scores[chunk.id]
        lid = chunk.lab_id
        lab_scores[lid] = lab_scores.get(lid, 0.0) + score
        lab_matched_chunks.setdefault(lid, []).append(
            ChunkMatch(
                chunk_text=chunk.chunk_text,
                source_type=chunk.source_type,
                combined_score=score,
            )
        )

    # Sort chunks within each lab by score DESC
    for lid in lab_matched_chunks:
        lab_matched_chunks[lid].sort(key=lambda x: x.combined_score, reverse=True)

    # 6. Fetch Lab master data for the top matched labs
    top_lab_ids = sorted(lab_scores, key=lab_scores.__getitem__, reverse=True)[:limit]
    if not top_lab_ids:
        return SearchResponse(query=q, results=[])

    stmt_labs = select(Lab).where(Lab.id.in_(top_lab_ids))
    res_labs = await db.execute(stmt_labs)
    labs_dict = {lab.id: lab for lab in res_labs.scalars().all()}

    # 7. Construct response (preserve score-sorted order)
    results = [
        LabResult(
            lab_id=lab.id,
            name=lab.name,
            name_en=lab.name_en,
            department=lab.department,
            faculty=lab.faculty,
            lab_url=lab.lab_url,
            description=lab.description,
            keywords=lab.keywords,
            matched_chunks=lab_matched_chunks[lab.id][:3],
            total_score=lab_scores[lab.id],
        )
        for lid in top_lab_ids
        if (lab := labs_dict.get(lid))
    ]

    return SearchResponse(query=q, results=results)
