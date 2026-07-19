"""baseline schema

Revision ID: 0001_baseline
Revises:
Create Date: 2026-07-18
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, index=True),
        sa.Column("merchant", sa.String(200)),
        sa.Column("amount", sa.Float),
        sa.Column("category", sa.String(80), index=True),
        sa.Column("description", sa.Text),
        sa.Column("is_anomaly", sa.Boolean, server_default=sa.false()),
        sa.Column("embedding", sa.JSON(), nullable=True),
    )
    op.create_table(
        "insights",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("period", sa.String(20), index=True),
        sa.Column("summary_text", sa.Text),
        sa.Column("generated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("session_id", sa.String(64), index=True),
        sa.Column("role", sa.String(16)),
        sa.Column("content", sa.Text),
        sa.Column("tool_calls", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("insights")
    op.drop_table("transactions")
