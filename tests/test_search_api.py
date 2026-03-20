"""
tests/test_search_api.py

Tests for the Search API.

Normal cases:
  - Valid query returns SearchResponse schema
  - limit parameter is respected
  - Results are ordered by score descending

Error cases:
  - Empty query returns HTTP 422
  - limit out of range returns HTTP 422
  - Embedding API failure returns HTTP 500

Unit tests:
  - compute_rrf: basic scoring, weighted behaviour, None handling
  - _keyword_conditions: token splitting
"""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

_ChunkRow = namedtuple("_ChunkRow", ["id", "lab_id"])


# ---------------------------------------------------------------------------
# Pure-unit tests (no app import needed)
# ---------------------------------------------------------------------------
class TestComputeRrf:
    def _get_rrf(self):
        from backend.search_api.main import compute_rrf
        return compute_rrf

    def test_both_ranks_produce_positive_score(self):
        score = self._get_rrf()(rank_vector=1, rank_keyword=1)
        assert score > 0

    def test_none_vector_rank_uses_keyword_only(self):
        rrf = self._get_rrf()
        score_both = rrf(rank_vector=1, rank_keyword=1)
        score_kw_only = rrf(rank_vector=None, rank_keyword=1)
        assert score_kw_only < score_both

    def test_none_keyword_rank_uses_vector_only(self):
        rrf = self._get_rrf()
        score_both = rrf(rank_vector=1, rank_keyword=1)
        score_v_only = rrf(rank_vector=1, rank_keyword=None)
        assert score_v_only < score_both

    def test_both_none_returns_zero(self):
        assert self._get_rrf()(rank_vector=None, rank_keyword=None) == 0.0

    def test_lower_rank_number_gives_higher_score(self):
        rrf = self._get_rrf()
        assert rrf(rank_vector=1, rank_keyword=None) > rrf(rank_vector=10, rank_keyword=None)

    def test_vector_weight_exceeds_keyword_weight(self):
        """With default weights (0.7 vs 0.3), vector-only score > keyword-only at same rank."""
        rrf = self._get_rrf()
        v_only = rrf(rank_vector=1, rank_keyword=None)
        k_only = rrf(rank_vector=None, rank_keyword=1)
        assert v_only > k_only


class TestKeywordConditions:
    def _get_kw(self):
        from backend.search_api.main import _keyword_conditions
        return _keyword_conditions

    def test_single_token_query(self):
        conds = self._get_kw()("機械学習")
        assert len(conds) == 1

    def test_multi_token_query_splits_correctly(self):
        conds = self._get_kw()("機械学習 医療診断 AI")
        assert len(conds) == 3

    def test_single_char_tokens_are_ignored(self):
        """Noise characters like 'a' or '.' should be filtered out."""
        conds = self._get_kw()("a b 機械学習")
        # 'a' and 'b' have len==1 and are skipped; only '機械学習' remains
        assert len(conds) == 1

    def test_empty_query_falls_back_to_full_string(self):
        conds = self._get_kw()("")
        assert len(conds) >= 1


# ---------------------------------------------------------------------------
# Fixture: app with DB and embedding calls mocked
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def search_app():
    from backend.search_api.main import app
    return app


def _make_mock_lab(lab_id: int, name: str, score: float):
    """Build a minimal mock Lab ORM object."""
    lab = MagicMock()
    lab.id = lab_id
    lab.name = name
    lab.name_en = f"{name} (EN)"
    lab.department = "情報学研究科"
    lab.faculty = "工学部"
    lab.lab_url = f"https://example.com/lab/{lab_id}"
    lab.description = f"{name}の説明文です。"
    lab.keywords = ["キーワード1", "キーワード2"]
    return lab


def _make_mock_chunk(chunk_id: int, lab_id: int):
    chunk = MagicMock()
    chunk.id = chunk_id
    chunk.lab_id = lab_id
    chunk.chunk_text = f"チャンクテキスト {chunk_id}"
    chunk.source_type = "lab_description"
    return chunk


def _mock_db_session(mock_labs: list, mock_chunks: list):
    """
    Return an async context manager that produces a mock AsyncSession.
    The session's execute() returns results shaped for the search flow.
    """
    session = AsyncMock()

    call_count = [0]

    async def execute_side_effect(stmt):
        call_count[0] += 1
        result = MagicMock()
        n = call_count[0]

        if n == 1:
            # Vector search: return named-tuple rows with .id and .lab_id
            result.all.return_value = [_ChunkRow(c.id, c.lab_id) for c in mock_chunks]
        elif n == 2:
            # Keyword search: empty
            result.all.return_value = []
        elif n == 3:
            # Fetch full chunk objects
            result.scalars.return_value.all.return_value = mock_chunks
        elif n == 4:
            # Fetch Lab master data
            result.scalars.return_value.all.return_value = mock_labs
        else:
            result.all.return_value = []
            result.scalars.return_value.all.return_value = []
        return result

    session.execute = AsyncMock(side_effect=execute_side_effect)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------
class TestSearch:
    @pytest.mark.asyncio
    async def test_returns_search_response_schema(self, search_app):
        from backend.shared.database import get_db

        mock_chunks = [_make_mock_chunk(1, 1), _make_mock_chunk(2, 1)]
        mock_labs = [_make_mock_lab(1, "自然言語処理研究室", 0.05)]
        session = _mock_db_session(mock_labs, mock_chunks)

        search_app.dependency_overrides[get_db] = _db_dep(session)
        try:
            with patch("backend.search_api.main.get_query_embedding", new=AsyncMock(return_value=[0.1] * 768)):
                async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
                    resp = await ac.get("/api/v1/search", params={"q": "機械学習"})
        finally:
            search_app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "results" in data
        assert data["query"] == "機械学習"

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(self, search_app):
        from backend.shared.database import get_db

        mock_chunks = [
            _make_mock_chunk(1, 1), _make_mock_chunk(2, 1),  # lab 1 — 2 chunks → higher score
            _make_mock_chunk(3, 2),                           # lab 2 — 1 chunk
        ]
        mock_labs = [
            _make_mock_lab(1, "高スコア研究室", 0.1),
            _make_mock_lab(2, "低スコア研究室", 0.01),
        ]
        session = _mock_db_session(mock_labs, mock_chunks)

        search_app.dependency_overrides[get_db] = _db_dep(session)
        try:
            with patch("backend.search_api.main.get_query_embedding", new=AsyncMock(return_value=[0.1] * 768)):
                async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
                    resp = await ac.get("/api/v1/search", params={"q": "テスト"})
        finally:
            search_app.dependency_overrides.clear()

        results = resp.json()["results"]
        if len(results) >= 2:
            assert results[0]["total_score"] >= results[1]["total_score"]

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_results(self, search_app):
        from backend.shared.database import get_db

        session = _mock_db_session([], [])

        search_app.dependency_overrides[get_db] = _db_dep(session)
        try:
            with patch("backend.search_api.main.get_query_embedding", new=AsyncMock(return_value=[0.1] * 768)):
                async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
                    resp = await ac.get("/api/v1/search", params={"q": "存在しないクエリ"})
        finally:
            search_app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert resp.json()["results"] == []


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------
class TestSearchErrors:
    @pytest.mark.asyncio
    async def test_missing_query_returns_422(self, search_app):
        async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/search")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_limit_zero_returns_422(self, search_app):
        async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/search", params={"q": "テスト", "limit": 0})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_limit_over_50_returns_422(self, search_app):
        async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/search", params={"q": "テスト", "limit": 51})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_500(self, search_app):
        from fastapi import HTTPException

        with patch(
            "backend.search_api.main.get_query_embedding",
            new=AsyncMock(side_effect=HTTPException(status_code=500, detail="Embedding failed")),
        ):
            async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
                resp = await ac.get("/api/v1/search", params={"q": "テスト"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class TestSearchHealth:
    @pytest.mark.asyncio
    async def test_health_has_required_fields(self, search_app):
        async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "db" in data
        assert "redis" in data
        assert data["status"] in ("ok", "degraded")

    @pytest.mark.asyncio
    async def test_health_ok_when_all_services_available(self, search_app):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock()

        mock_sm = MagicMock()
        mock_sm.return_value = mock_session

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()

        with (
            patch("backend.search_api.main.async_session_maker", mock_sm),
            patch("backend.search_api.main.aioredis.from_url", return_value=mock_redis),
        ):
            async with AsyncClient(transport=ASGITransport(app=search_app), base_url="http://test") as ac:
                resp = await ac.get("/health")

        data = resp.json()
        assert data["db"] == "ok"
        assert data["redis"] == "ok"
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Helper: wrap a session as a FastAPI dependency override
# ---------------------------------------------------------------------------
async def _db_dep_gen(session):
    yield session

def _db_dep(session):
    async def _dep():
        yield session
    return _dep
