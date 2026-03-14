"""
backend/search_api/main.py

Search API — FastAPI application.
Implements Hybrid Search (Vector + Keyword) over Kyoto University research labs.
Uses RRF (Reciprocal Rank Fusion) for score integration.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
import httpx
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector

from backend.shared.database import get_db
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
    description="Hybrid search over Kyoto University research labs using pgvector and RRF.",
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
# Helpers
# ---------------------------------------------------------------------------
async def get_query_embedding(query: str) -> list[float]:
    """Call the embedding API to get the vector for the search query."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://embedding_api:8001/api/v1/embed",
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


def compute_rrf(rank_vector: int | None, rank_keyword: int | None, k: int = 60) -> float:
    """
    Computes reciprocal rank fusion score.
    Higher is better. If a rank is None, it contributes 0.
    """
    score = 0.0
    if rank_vector is not None:
        score += 1.0 / (k + rank_vector)
    if rank_keyword is not None:
        score += 1.0 / (k + rank_keyword)
    return score


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    """Liveness probe."""
    return {"status": "ok", "phase": 2}


@app.get(
    "/api/v1/search",
    response_model=SearchResponse,
    summary="Hybrid search for labs",
    tags=["search"],
)
async def search(
    q: str = Query(..., description="Search query string"),
    limit: int = Query(10, description="Max lab results to return", ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Executes a hybrid search combining pgvector cosine distance and ILIKE keyword match.
    Results are merged using RRF (Reciprocal Rank Fusion).
    """
    logger.info(f"Received search query: '{q}'")

    # 1. Get query embedding
    query_vector = await get_query_embedding(q)

    # 2. Vector Search (Top 50 chunks)
    # Cosine distance: smaller is closer. We order by distance ASC.
    stmt_vector = (
        select(EmbeddingChunk.id, EmbeddingChunk.lab_id)
        .order_by(EmbeddingChunk.embedding.cosine_distance(query_vector))
        .limit(50)
    )
    res_vector = await db.execute(stmt_vector)
    vector_rows = res_vector.all()
    
    # rank is 1-indexed
    vector_ranks = {row.id: rank for rank, row in enumerate(vector_rows, start=1)}

    # 3. Keyword Search (Top 50 chunks)
    # Simple ILIKE match for Phase 2 prototype
    stmt_keyword = (
        select(EmbeddingChunk.id, EmbeddingChunk.lab_id)
        .where(EmbeddingChunk.chunk_text.ilike(f"%{q}%"))
        .limit(50)
    )
    res_keyword = await db.execute(stmt_keyword)
    keyword_rows = res_keyword.all()

    keyword_ranks = {row.id: rank for rank, row in enumerate(keyword_rows, start=1)}

    # 4. RRF Scoring (chunk level)
    all_chunk_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())
    if not all_chunk_ids:
        return SearchResponse(query=q, results=[])

    chunk_scores: dict[int, float] = {}
    for cid in all_chunk_ids:
        r_v = vector_ranks.get(cid)
        r_k = keyword_ranks.get(cid)
        chunk_scores[cid] = compute_rrf(r_v, r_k)

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
        
        if lid not in lab_matched_chunks:
            lab_matched_chunks[lid] = []
            
        lab_matched_chunks[lid].append(
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
    top_lab_ids = sorted(lab_scores.keys(), key=lambda lid: lab_scores[lid], reverse=True)[:limit]
    
    if not top_lab_ids:
        return SearchResponse(query=q, results=[])

    stmt_labs = select(Lab).where(Lab.id.in_(top_lab_ids))
    res_labs = await db.execute(stmt_labs)
    labs_dict = {lab.id: lab for lab in res_labs.scalars().all()}

    # 7. Construct response
    results = []
    for lid in top_lab_ids:
        lab = labs_dict[lid]
        results.append(
            LabResult(
                lab_id=lab.id,
                name=lab.name,
                name_en=lab.name_en,
                department=lab.department,
                faculty=lab.faculty,
                lab_url=lab.lab_url,
                description=lab.description,
                keywords=lab.keywords,
                matched_chunks=lab_matched_chunks[lid][:3],  # return top 3 chunks per lab
                total_score=lab_scores[lid],
            )
        )

    return SearchResponse(query=q, results=results)
