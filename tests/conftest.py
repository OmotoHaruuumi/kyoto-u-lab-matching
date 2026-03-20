"""
tests/conftest.py

Shared fixtures for all tests.
DATABASE_URL must be set before importing any backend module that uses database.py.
"""
import os

import pytest

# Set required env vars BEFORE any backend module is imported.
# These point to non-existent services; individual tests mock DB/Redis as needed.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test_db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("MOCK_EMBEDDING", "true")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("EMBEDDING_API_URL", "http://localhost:8001")
