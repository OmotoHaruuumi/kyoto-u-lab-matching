"""
tests/test_embedding_api.py

Tests for the Embedding API.

Normal cases:
  - Mock mode returns deterministic 768-dim vectors
  - Multiple texts return one vector per text
  - Vector values are floats and unit-normalised

Error cases:
  - Exceeding 100 texts returns HTTP 422
  - Empty texts list returns HTTP 422 (pydantic min_length=1)
  - Health endpoint reports DB/Redis status (mocked)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# Fixture: import app after env vars are set by conftest.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def embedding_app():
    from backend.embedding_api.main import app
    return app


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
async def _post_embed(app, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        return await ac.post("/api/v1/embed", json={"texts": texts, "task_type": task_type})


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------
class TestEmbed:
    @pytest.mark.asyncio
    async def test_single_text_returns_768_dim_vector(self, embedding_app):
        resp = await _post_embed(embedding_app, ["機械学習を用いた医療診断"])
        assert resp.status_code == 200
        data = resp.json()
        assert data["mock"] is True
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 768

    @pytest.mark.asyncio
    async def test_multiple_texts_return_matching_count(self, embedding_app):
        texts = ["自然言語処理", "ロボット工学", "量子コンピューティング"]
        resp = await _post_embed(embedding_app, texts)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["embeddings"]) == len(texts)

    @pytest.mark.asyncio
    async def test_embedding_is_unit_normalised(self, embedding_app):
        resp = await _post_embed(embedding_app, ["テスト入力"])
        vec = resp.json()["embeddings"][0]
        norm = sum(v ** 2 for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5, f"Vector not unit-normalised: norm={norm}"

    @pytest.mark.asyncio
    async def test_same_text_returns_same_vector(self, embedding_app):
        """Mock embeddings are deterministic."""
        r1 = await _post_embed(embedding_app, ["再現性テスト"])
        r2 = await _post_embed(embedding_app, ["再現性テスト"])
        assert r1.json()["embeddings"] == r2.json()["embeddings"]

    @pytest.mark.asyncio
    async def test_different_texts_return_different_vectors(self, embedding_app):
        r = await _post_embed(embedding_app, ["テキストA", "テキストB"])
        assert r.json()["embeddings"][0] != r.json()["embeddings"][1]

    @pytest.mark.asyncio
    async def test_retrieval_query_task_type_accepted(self, embedding_app):
        resp = await _post_embed(embedding_app, ["クエリ"], task_type="RETRIEVAL_QUERY")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------
class TestEmbedErrors:
    @pytest.mark.asyncio
    async def test_too_many_texts_returns_422(self, embedding_app):
        texts = [f"text {i}" for i in range(101)]
        resp = await _post_embed(embedding_app, texts)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_texts_list_returns_422(self, embedding_app):
        async with AsyncClient(transport=ASGITransport(app=embedding_app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/embed", json={"texts": []})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_texts_field_returns_422(self, embedding_app):
        async with AsyncClient(transport=ASGITransport(app=embedding_app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/embed", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_status_field(self, embedding_app):
        """Health endpoint should always return a JSON with 'status' key,
        even if DB/Redis are unavailable (degraded is acceptable in tests)."""
        async with AsyncClient(transport=ASGITransport(app=embedding_app), base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")
        assert "db" in data
        assert "redis" in data

    @pytest.mark.asyncio
    async def test_health_ok_when_db_and_redis_available(self, embedding_app):
        """Simulate both DB and Redis being reachable."""
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)
        mock_engine.dispose = AsyncMock()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()

        with (
            patch("backend.embedding_api.main.create_async_engine", return_value=mock_engine),
            patch("backend.embedding_api.main.aioredis.from_url", return_value=mock_redis),
        ):
            async with AsyncClient(transport=ASGITransport(app=embedding_app), base_url="http://test") as ac:
                resp = await ac.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["db"] == "ok"
        assert data["redis"] == "ok"
        assert data["status"] == "ok"
