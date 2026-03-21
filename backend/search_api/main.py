"""
backend/search_api/main.py

Search API — FastAPI application.
Implements Hybrid Search (Vector + Keyword) over Kyoto University research labs.
Uses Weighted RRF (Reciprocal Rank Fusion) for score integration.

Tuning parameters (all overridable via env vars):
  RRF_K                        — RRF constant (default 30; lower = stronger rank-based signal)
  RRF_VECTOR_WEIGHT            — weight for vector search hits (default 0.7)
  RRF_KEYWORD_WEIGHT           — weight for keyword search hits (default 0.3)
  SEARCH_CANDIDATE_LIMIT       — candidate pool size per search mode (default 100)
  CHUNK_WEIGHT_LAB_DESCRIPTION — score multiplier for lab_description chunks (default 1.5)
  CHUNK_WEIGHT_RESEARCH_THEME  — score multiplier for research_theme chunks (default 1.0)
  QUERY_REWRITE_ENABLED        — enable LLM-based bilingual query expansion (default true)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
import httpx
import redis.asyncio as aioredis
from fastapi.middleware.cors import CORSMiddleware
from google import genai as google_genai
from google.genai import types as genai_types
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
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Config (all overridable via environment variables)
# ---------------------------------------------------------------------------
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")
EMBEDDING_API_URL: str = os.environ.get("EMBEDDING_API_URL", "http://embedding_api:8001")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# RRF tuning
RRF_K: int = int(os.environ.get("RRF_K", "30"))
VECTOR_WEIGHT: float = float(os.environ.get("RRF_VECTOR_WEIGHT", "0.7"))
KEYWORD_WEIGHT: float = float(os.environ.get("RRF_KEYWORD_WEIGHT", "0.3"))
CANDIDATE_LIMIT: int = int(os.environ.get("SEARCH_CANDIDATE_LIMIT", "100"))

# Chunk-type score multipliers
CHUNK_WEIGHTS: dict[str, float] = {
    "lab_description": float(os.environ.get("CHUNK_WEIGHT_LAB_DESCRIPTION", "1.5")),
    "research_theme": float(os.environ.get("CHUNK_WEIGHT_RESEARCH_THEME", "1.0")),
}

# Query rewriting
QUERY_REWRITE_ENABLED: bool = os.environ.get("QUERY_REWRITE_ENABLED", "true").lower() == "true"

# Gemini client for query rewriting
_genai_client: google_genai.Client | None = None
if GEMINI_API_KEY:
    _genai_client = google_genai.Client(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set — query rewriting will be disabled.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRewrite(BaseModel):
    keywords_ja: list[str] = Field(description="関連する日本語専門用語・研究分野キーワード")
    keywords_en: list[str] = Field(description="Equivalent English technical keywords")


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
    rewritten_query: str | None
    results: list[LabResult]


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------
def _build_expanded_query(original: str, rewrite: QueryRewrite) -> str:
    """Combine original query with bilingual keyword expansion for embedding."""
    ja_text = " ".join(rewrite.keywords_ja)
    en_text = " ".join(rewrite.keywords_en)
    return f"{original}\n{ja_text}\n{en_text}"


async def rewrite_query(query: str) -> tuple[str, str | None]:
    """
    Expand the search query into bilingual academic keywords using Gemini.
    Returns (text_to_embed, rewritten_query_for_logging).
    Falls back to the original query on any error.
    """
    if not QUERY_REWRITE_ENABLED or _genai_client is None:
        return query, None

    prompt = f"""ユーザーの研究室検索クエリを、日本語と英語の学術専門用語に展開してください。

クエリ: "{query}"

【keywords_ja】: このクエリが意図する研究に関連する日本語の専門用語を5〜10個。汎用語（「研究」「AI」「大学」）ではなく研究分野・手法・応用領域の具体的な術語を選んでください。
【keywords_en】: 同じ内容の英語専門用語を5〜10個。"""

    try:
        def _call() -> QueryRewrite:
            response = _genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QueryRewrite,
                    temperature=0.1,
                ),
            )
            data = json.loads(response.text)
            return QueryRewrite.model_validate(data)

        rewrite = await asyncio.to_thread(_call)
        expanded = _build_expanded_query(query, rewrite)
        logger.info(f"Query rewrite — ja: {rewrite.keywords_ja} / en: {rewrite.keywords_en}")
        return expanded, expanded
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return query, None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
async def get_query_embedding(query_text: str) -> list[float]:
    """Call the embedding API to get the vector for the (possibly rewritten) search query."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{EMBEDDING_API_URL}/api/v1/embed",
                json={"texts": [query_text], "task_type": "RETRIEVAL_QUERY"},
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
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
    db_status = "ok"
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"

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
        "query_rewrite_enabled": QUERY_REWRITE_ENABLED and _genai_client is not None,
        "chunk_weights": CHUNK_WEIGHTS,
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
    keyword matching. Results are merged using Weighted RRF.

    Query rewriting expands the input into bilingual academic keywords so that
    Japanese queries surface English-language lab pages and vice versa.
    Chunk-type weights (configurable via env vars) give lab_description chunks
    higher influence than individual research_theme chunks.
    """
    logger.info(f"Search query='{q}' limit={limit}")

    # 1. Rewrite query into bilingual academic keywords
    embed_text, rewritten = await rewrite_query(q)

    # 2. Get query embedding (uses expanded text if rewriting succeeded)
    query_vector = await get_query_embedding(embed_text)

    # 3. Vector Search — top CANDIDATE_LIMIT chunks by cosine distance
    stmt_vector = (
        select(EmbeddingChunk.id, EmbeddingChunk.lab_id)
        .order_by(EmbeddingChunk.embedding.cosine_distance(query_vector))
        .limit(CANDIDATE_LIMIT)
    )
    res_vector = await db.execute(stmt_vector)
    vector_rows = res_vector.all()
    vector_ranks = {row.id: rank for rank, row in enumerate(vector_rows, start=1)}

    # 4. Keyword Search — use original query tokens (not expanded) to avoid noise
    kw_conditions = _keyword_conditions(q)
    stmt_keyword = (
        select(EmbeddingChunk.id, EmbeddingChunk.lab_id)
        .where(or_(*kw_conditions))
        .limit(CANDIDATE_LIMIT)
    )
    res_keyword = await db.execute(stmt_keyword)
    keyword_rows = res_keyword.all()
    keyword_ranks = {row.id: rank for rank, row in enumerate(keyword_rows, start=1)}

    # 5. Weighted RRF scoring at chunk level
    all_chunk_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())
    if not all_chunk_ids:
        return SearchResponse(query=q, rewritten_query=rewritten, results=[])

    chunk_scores: dict[int, float] = {
        cid: compute_rrf(vector_ranks.get(cid), keyword_ranks.get(cid))
        for cid in all_chunk_ids
    }

    # 6. Fetch full chunk data
    stmt_chunks = select(EmbeddingChunk).where(EmbeddingChunk.id.in_(all_chunk_ids))
    res_chunks = await db.execute(stmt_chunks)
    chunks = res_chunks.scalars().all()

    # 7. Aggregate by lab, applying chunk-type weights
    lab_scores: dict[int, float] = {}
    lab_matched_chunks: dict[int, list[ChunkMatch]] = {}

    for chunk in chunks:
        base_score = chunk_scores[chunk.id]
        weight = CHUNK_WEIGHTS.get(chunk.source_type, 1.0)
        weighted_score = base_score * weight

        lid = chunk.lab_id
        lab_scores[lid] = lab_scores.get(lid, 0.0) + weighted_score
        lab_matched_chunks.setdefault(lid, []).append(
            ChunkMatch(
                chunk_text=chunk.chunk_text,
                source_type=chunk.source_type,
                combined_score=weighted_score,
            )
        )

    # Sort chunks within each lab by score DESC
    for lid in lab_matched_chunks:
        lab_matched_chunks[lid].sort(key=lambda x: x.combined_score, reverse=True)

    # 8. Fetch Lab master data for the top matched labs
    top_lab_ids = sorted(lab_scores, key=lab_scores.__getitem__, reverse=True)[:limit]
    if not top_lab_ids:
        return SearchResponse(query=q, rewritten_query=rewritten, results=[])

    stmt_labs = select(Lab).where(Lab.id.in_(top_lab_ids))
    res_labs = await db.execute(stmt_labs)
    labs_dict = {lab.id: lab for lab in res_labs.scalars().all()}

    # 9. Construct response (preserve score-sorted order)
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

    return SearchResponse(query=q, rewritten_query=rewritten, results=results)
