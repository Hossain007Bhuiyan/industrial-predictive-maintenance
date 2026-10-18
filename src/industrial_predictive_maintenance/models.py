from __future__ import annotations
from datetime import datetime
from sqlalchemy import ARRAY, BigInteger, CheckConstraint, DateTime, ForeignKey, MetaData, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    # Fixed constraint names, so later migrations can find and change them.
    metadata = MetaData(
        naming_convention={
            "pk": "pk_%(table_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
        }
    )


class Machine(Base):
    __tablename__ = "machines"
    __table_args__ = (
        UniqueConstraint("source", "name"),
        CheckConstraint("source in ('femto', 'ims', 'sca')", name="source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(10))
    # Bearing1_1, test1 or case8. FEMTO and SCA use the asset names from the replay messages.
    name: Mapped[str] = mapped_column(String(50))
    # Roller, Pump, Engine and so on, as the dataset describes the machine.
    kind: Mapped[str | None] = mapped_column(String(50))
    # Shaft speed and radial load where the dataset gives one fixed value.
    rpm: Mapped[float | None]
    load_n: Mapped[float | None]
    fixed_speed: Mapped[bool | None]


class Sensor(Base):
    __tablename__ = "sensors"
    # FEMTO messages have no placement, so an empty placement must also count only once per machine.
    __table_args__ = (UniqueConstraint("machine_id", "placement", postgresql_nulls_not_distinct=True),)

    id: Mapped[int] = mapped_column(primary_key=True)
    machine_id: Mapped[int] = mapped_column(ForeignKey("machines.id"))
    placement: Mapped[str | None] = mapped_column(String(20))
    channels: Mapped[list[str]] = mapped_column(ARRAY(String(20)))
    # Empty for IMS, which doesn't document its unit.
    unit: Mapped[str | None] = mapped_column(String(10))
    bearing: Mapped[str | None] = mapped_column(String(50))
    # Fault frequencies as multiples of the shaft speed. Empty unless verified values exist.
    fault_ftf: Mapped[float | None]
    fault_bpf: Mapped[float | None]
    fault_bpfo: Mapped[float | None]
    fault_bpfi: Mapped[float | None]


class Measurement(Base):
    __tablename__ = "measurements"
    # MQTT can deliver a message twice. The copy has the same sequence and sent_at and is refused.
    __table_args__ = (UniqueConstraint("sensor_id", "sequence", "sent_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sensor_id: Mapped[int] = mapped_column(ForeignKey("sensors.id"))
    # SCA and IMS give local times without a time zone, so none is added.
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime)
    elapsed_s: Mapped[float | None]
    rpm: Mapped[float | None]
    sampling_hz: Mapped[float]
    # Kept only for evaluating drift detection, never as an input to it.
    injected: Mapped[str | None] = mapped_column(String(10))
    # Only streamed messages have these. Rows loaded from the feature tables don't.
    sequence: Mapped[int | None]
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChannelFeatures(Base):
    __tablename__ = "channel_features"

    measurement_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("measurements.id"), primary_key=True)
    channel: Mapped[str] = mapped_column(String(20), primary_key=True)
    mean: Mapped[float]
    std: Mapped[float]
    rms: Mapped[float]
    peak: Mapped[float]
    crest: Mapped[float]
    kurtosis: Mapped[float]
    skewness: Mapped[float]
    flat_top_fraction: Mapped[float]
    centroid_hz: Mapped[float]
    dominant_hz: Mapped[float]
    band1_share: Mapped[float]
    band2_share: Mapped[float]
    band3_share: Mapped[float]
    band4_share: Mapped[float]
    # Empty when the sensor has no fault frequencies.
    fault_ftf: Mapped[float | None]
    fault_bpf: Mapped[float | None]
    fault_bpfo: Mapped[float | None]
    fault_bpfi: Mapped[float | None]


class RejectedMessage(Base):
    __tablename__ = "rejected_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    topic: Mapped[str] = mapped_column(String)
    size_bytes: Mapped[int]
    reasons: Mapped[list[str]] = mapped_column(ARRAY(String))