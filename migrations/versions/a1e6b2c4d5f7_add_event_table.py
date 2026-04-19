"""Add event table for analytics

Revision ID: a1e6b2c4d5f7
Revises: 4d12dd92e576
Create Date: 2026-04-19 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'a1e6b2c4d5f7'
down_revision = '4d12dd92e576'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'event',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('session_id', sa.String(length=64), nullable=True),
        sa.Column('upload_id', sa.Integer(), nullable=True),
        sa.Column('data', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.ForeignKeyConstraint(['upload_id'], ['upload.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('event', schema=None) as batch_op:
        batch_op.create_index('ix_event_created_at', ['created_at'], unique=False)
        batch_op.create_index('ix_event_event_type', ['event_type'], unique=False)
        batch_op.create_index('ix_event_user_id', ['user_id'], unique=False)
        batch_op.create_index('ix_event_session_id', ['session_id'], unique=False)
        batch_op.create_index('ix_event_upload_id', ['upload_id'], unique=False)


def downgrade():
    with op.batch_alter_table('event', schema=None) as batch_op:
        batch_op.drop_index('ix_event_upload_id')
        batch_op.drop_index('ix_event_session_id')
        batch_op.drop_index('ix_event_user_id')
        batch_op.drop_index('ix_event_event_type')
        batch_op.drop_index('ix_event_created_at')
    op.drop_table('event')
