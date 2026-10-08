"""Ingestion service: receive vibration messages over MQTT, validate them,
compute their features and store the result.

Every valid message becomes one row in data/stream/features.jsonl. Rejected
messages are written to data/stream/rejected.jsonl with the reasons. MQTT can
deliver a message twice with QoS 1, so duplicates are recognised and skipped.

The "injected" field of a message is stored for evaluating drift detection
later. It must never be used as an input for detection itself.

The fault frequency multiples of the SCA machines are looked up in the dataset
for now. In Phase 3 they move to the asset registry in the database.

Run:
    uv run python -m industrial_predictive_maintenance.ingest
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import paho.mqtt.client as mqtt
from pydantic import ValidationError
from industrial_predictive_maintenance import datasets, features, messages
from industrial_predictive_maintenance.data_download import DATA_DIR

MQTT_HOST = os.environ.get("IPM_MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.environ.get("IPM_MQTT_PORT", "1883"))
TOPICS = [("ipm/+/+/vibration", 1), ("ipm/+/+/+/vibration", 1)]
STREAM_DIR = DATA_DIR / "stream"
DUPLICATE_WINDOW = 10_000


@lru_cache(maxsize=64)
def fault_orders(source: str, asset: str, placement: str | None) -> dict[str, float] | None:
    if source != "sca" or placement is None:
        return None
    return datasets.sca_fault_orders(int(asset.removeprefix("case")), "test", placement)


def to_row(topic: str, message: messages.VibrationMessage, received_at: datetime) -> dict:
    orders = fault_orders(message.source, message.asset, message.placement)
    values = features.extract(message.signal, message.sampling_hz, message.channels, message.rpm, orders)
    row = {
        "received_at": received_at.isoformat(),
        "topic": topic,
        "source": message.source,
        "asset": message.asset,
        "placement": message.placement,
        "sequence": message.sequence,
        "sent_at": message.sent_at.isoformat(),
        "latency_ms": round((received_at - message.sent_at).total_seconds() * 1000, 1),
        "elapsed_s": message.elapsed_s,
        "recorded_at": message.recorded_at.isoformat() if message.recorded_at else None,
        "rpm": message.rpm,
        "sampling_hz": message.sampling_hz,
        "injected": message.injected.kind if message.injected else None,
        **values,
    }
    # JSON has no NaN, so missing feature values are stored as null.
    return {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}

def reasons(error: ValidationError) -> list[str]:
    """One readable line per problem, with the field name when there is one."""
    return [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" if e["loc"] else e["msg"] for e in error.errors()]

class Ingestor:
    def __init__(self, out_dir: Path = STREAM_DIR) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.features_file = out_dir / "features.jsonl"
        self.rejected_file = out_dir / "rejected.jsonl"
        self.seen: OrderedDict[tuple, None] = OrderedDict()
        self.accepted = 0
        self.rejected = 0
        self.duplicates = 0

    def handle(self, topic: str, payload: bytes, received_at: datetime | None = None) -> str:
        """Process one message. Returns "accepted", "rejected" or "duplicate"."""
        received_at = received_at or datetime.now(timezone.utc)
        try:
            message = messages.parse(payload)
        except ValidationError as error:
            self._reject(topic, received_at, len(payload), reasons(error))
            return "rejected"

        key = (topic, message.sequence, message.sent_at)
        if key in self.seen:
            self.duplicates += 1
            return "duplicate"
        self._remember(key)

        try:
            row = to_row(topic, message, received_at)
        except Exception as error:  # a single bad message must never stop the service
            self._reject(topic, received_at, len(payload), [f"feature error: {error}"])
            return "rejected"
        self._append(self.features_file, row)
        self.accepted += 1
        return "accepted"

    def _remember(self, key: tuple) -> None:
        self.seen[key] = None
        if len(self.seen) > DUPLICATE_WINDOW:
            self.seen.popitem(last=False)

    def _reject(self, topic: str, received_at: datetime, size: int, reasons: list[str]) -> None:
        row = {"received_at": received_at.isoformat(), "topic": topic, "bytes": size, "reasons": reasons}
        self._append(self.rejected_file, row)
        self.rejected += 1

    @staticmethod
    def _append(path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    ingestor = Ingestor()

    def on_connect(client, userdata, flags, reason_code, properties):
        # Subscribing here means the subscription is renewed after a reconnect.
        client.subscribe(TOPICS)
        print(f"connected to {MQTT_HOST}:{MQTT_PORT}, writing to {ingestor.features_file.parent}")

    def on_message(client, userdata, msg):
        result = ingestor.handle(msg.topic, msg.payload)
        print(
            f"{result:<9}  {msg.topic}  "
            f"accepted {ingestor.accepted}  rejected {ingestor.rejected}  duplicates {ingestor.duplicates}"
        )

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ipm-ingest")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        print(f"stopped: accepted {ingestor.accepted}  rejected {ingestor.rejected}  duplicates {ingestor.duplicates}")


if __name__ == "__main__":
    main()