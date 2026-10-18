"""Create the asset registry, measurement, feature and rejected message tables."""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.create_table(
        "machines",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.String(10), nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("kind", sa.String(50)),
        sa.Column("rpm", sa.Float()),
        sa.Column("load_n", sa.Float()),
        sa.Column("fixed_speed", sa.Boolean()),
        sa.UniqueConstraint("source", "name"),
        sa.CheckConstraint("source in ('femto', 'ims', 'sca')", name="source"),
    )
    op.create_table(
        "sensors",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("machine_id", sa.Integer(), sa.ForeignKey("machines.id"), nullable=False),
        sa.Column("placement", sa.String(20)),
        sa.Column("channels", sa.ARRAY(sa.String(20)), nullable=False),
        sa.Column("unit", sa.String(10)),
        sa.Column("bearing", sa.String(50)),
        sa.Column("fault_ftf", sa.Float()),
        sa.Column("fault_bpf", sa.Float()),
        sa.Column("fault_bpfo", sa.Float()),
        sa.Column("fault_bpfi", sa.Float()),
        sa.UniqueConstraint("machine_id", "placement", postgresql_nulls_not_distinct=True),
    )
    op.create_table(
        "measurements",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("sensor_id", sa.Integer(), sa.ForeignKey("sensors.id"), nullable=False),
        sa.Column("recorded_at", sa.DateTime()),
        sa.Column("elapsed_s", sa.Float()),
        sa.Column("rpm", sa.Float()),
        sa.Column("sampling_hz", sa.Float(), nullable=False),
        sa.Column("injected", sa.String(10)),
        sa.Column("sequence", sa.Integer()),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("received_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("sensor_id", "sequence", "sent_at"),
    )
    op.create_table(
        "channel_features",
        sa.Column("measurement_id", sa.BigInteger(), sa.ForeignKey("measurements.id"), primary_key=True),
        sa.Column("channel", sa.String(20), primary_key=True),
        *[
            sa.Column(name, sa.Float(), nullable=False)
            for name in (
                "mean", "std", "rms", "peak", "crest", "kurtosis", "skewness", "flat_top_fraction",
                "centroid_hz", "dominant_hz", "band1_share", "band2_share", "band3_share", "band4_share",
            )
        ],
        *[sa.Column(name, sa.Float()) for name in ("fault_ftf", "fault_bpf", "fault_bpfo", "fault_bpfi")],
    )
    op.create_table(
        "rejected_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("topic", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("reasons", sa.ARRAY(sa.String()), nullable=False),
    )


def downgrade() -> None:
    for table in ("rejected_messages", "channel_features", "measurements", "sensors", "machines"):
        op.drop_table(table)