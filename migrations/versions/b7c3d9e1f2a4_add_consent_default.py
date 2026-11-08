"""Record whether the training checkbox was ticked by default

Revision ID: b7c3d9e1f2a4
Revises: a1e6b2c4d5f7
Create Date: 2026-11-08 01:00:00.000000

The checkbox became opt-out (ticked by default) when the 2026-11-08 deploy
went live at 00:47:47 UTC. Uploads before then were made under opt-in, and
the privacy policy promises those without consent are never used for training.

"""
from alembic import op
import sqlalchemy as sa


revision = 'b7c3d9e1f2a4'
down_revision = 'a1e6b2c4d5f7'
branch_labels = None
depends_on = None

OPT_OUT_LIVE = '2026-11-08 00:47:47'


def upgrade():
    with op.batch_alter_table('upload') as batch_op:
        batch_op.add_column(sa.Column('consent_default', sa.Boolean(), nullable=True))
    op.execute(
        "UPDATE upload SET consent_default = "
        f"CASE WHEN created_at < '{OPT_OUT_LIVE}' THEN false ELSE true END"
    )


def downgrade():
    with op.batch_alter_table('upload') as batch_op:
        batch_op.drop_column('consent_default')
