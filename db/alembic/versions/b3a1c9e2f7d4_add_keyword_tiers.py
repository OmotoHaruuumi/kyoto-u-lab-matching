"""add keyword tiers to labs

Revision ID: b3a1c9e2f7d4
Revises: ee63f0cec60d
Create Date: 2026-03-21

Replace flat keywords array with two-tier keyword structure:
  keywords_primary   — up to 3 identity-defining keywords
  keywords_secondary — up to 7 important but non-core keywords
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "b3a1c9e2f7d4"
down_revision: Union[str, None] = "ee63f0cec60d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("labs", sa.Column("keywords_primary", postgresql.ARRAY(sa.Text()), nullable=True))
    op.add_column("labs", sa.Column("keywords_secondary", postgresql.ARRAY(sa.Text()), nullable=True))
    op.drop_column("labs", "keywords")


def downgrade() -> None:
    op.add_column("labs", sa.Column("keywords", postgresql.ARRAY(sa.Text()), nullable=True))
    op.drop_column("labs", "keywords_primary")
    op.drop_column("labs", "keywords_secondary")
