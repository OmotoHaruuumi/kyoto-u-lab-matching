"""
backend/shared/models.py

SQLAlchemy ORM models.
PostgreSQL-specific types (ARRAY, pgvector Vector) are used throughout.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.shared.database import Base

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 768  # text-embedding-004 output dimension


# ---------------------------------------------------------------------------
# labs — Research Laboratory master
# ---------------------------------------------------------------------------
class Lab(Base):
    __tablename__ = "labs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    name_en: Mapped[Optional[str]] = mapped_column(String(512))
    department: Mapped[Optional[str]] = mapped_column(String(256))
    faculty: Mapped[Optional[str]] = mapped_column(String(256))
    lab_url: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    keywords: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    professors: Mapped[list[Professor]] = relationship(
        "Professor", back_populates="lab", cascade="all, delete-orphan"
    )
    research_themes: Mapped[list[ResearchTheme]] = relationship(
        "ResearchTheme", back_populates="lab", cascade="all, delete-orphan"
    )
    embedding_chunks: Mapped[list[EmbeddingChunk]] = relationship(
        "EmbeddingChunk", back_populates="lab", cascade="all, delete-orphan"
    )
    data_sources: Mapped[list[DataSource]] = relationship(
        "DataSource", back_populates="lab", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# professors — Faculty member information
# ---------------------------------------------------------------------------
class Professor(Base):
    __tablename__ = "professors"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    lab_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("labs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    name_en: Mapped[Optional[str]] = mapped_column(String(256))
    title: Mapped[Optional[str]] = mapped_column(String(128))  # e.g. 教授, 准教授
    email: Mapped[Optional[str]] = mapped_column(String(256))
    profile_url: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    lab: Mapped[Lab] = relationship("Lab", back_populates="professors")

    __table_args__ = (UniqueConstraint("lab_id", "name", name="uq_professor_lab_name"),)


# ---------------------------------------------------------------------------
# research_themes — Individual research theme entries
# ---------------------------------------------------------------------------
class ResearchTheme(Base):
    __tablename__ = "research_themes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    lab_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("labs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    keywords: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    lab: Mapped[Lab] = relationship("Lab", back_populates="research_themes")


# ---------------------------------------------------------------------------
# embedding_chunks — Text chunks and their vector embeddings
# ---------------------------------------------------------------------------
class EmbeddingChunk(Base):
    __tablename__ = "embedding_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    lab_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("labs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Source type: "lab_description" | "research_theme" | "professor_profile"
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[Optional[int]] = mapped_column(BigInteger)  # FK to source row
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    lab: Mapped[Lab] = relationship("Lab", back_populates="embedding_chunks")

    __table_args__ = (
        # IVFFlat index for approximate nearest-neighbour search (cosine)
        Index(
            "ix_embedding_chunks_embedding_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# ---------------------------------------------------------------------------
# data_sources — Crawl / scrape management
# ---------------------------------------------------------------------------
class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    lab_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("labs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    # Status: "pending" | "crawling" | "done" | "failed"
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    last_crawled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))  # SHA-256
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    lab: Mapped[Lab] = relationship("Lab", back_populates="data_sources")

    __table_args__ = (UniqueConstraint("lab_id", "url", name="uq_datasource_lab_url"),)
