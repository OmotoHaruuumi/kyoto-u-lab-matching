"""replace_ivfflat_with_hnsw

Revision ID: c4f2a1b8e3d5
Revises: b3a1c9e2f7d4
Create Date: 2026-03-21

"""
from typing import Sequence, Union
from alembic import op

revision: str = 'c4f2a1b8e3d5'
down_revision: Union[str, None] = 'b3a1c9e2f7d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index('ix_embedding_chunks_embedding_cosine', table_name='embedding_chunks')
    op.execute("""
        CREATE INDEX ix_embedding_chunks_embedding_hnsw
        ON embedding_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.drop_index('ix_embedding_chunks_embedding_hnsw', table_name='embedding_chunks')
    op.execute("""
        CREATE INDEX ix_embedding_chunks_embedding_cosine
        ON embedding_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
