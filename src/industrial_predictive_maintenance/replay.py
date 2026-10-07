"""Replay real bearing recordings over MQTT as if the sensors were live.

Each message carries one recording of one sensor: the vibration samples as
little-endian float32 bytes in base64 plus the metadata a real sensor would
send. Labels and remaining life are never sent, since a real sensor does not
know them.

Every message also has an "injected" field. It is null for real recordings.
When the drift injector changes a recording, it describes the change, so
synthetic data is always marked.

Examples:
    uv run python -m industrial_predictive_maintenance.replay femto Bearing1_1 --speed 10
    uv run python -m industrial_predictive_maintenance.replay sca 8 DS --part test --speed 86400
    uv run python -m industrial_predictive_maintenance.replay femto Bearing1_1 --speed 10 --drift gain --drift-start 100
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone

import numpy as np
import paho.mqtt.client as mqtt

from industrial_predictive_maintenance import datasets
from industrial_predictive_maintenance.drift import KINDS, DriftInjector

MQTT_HOST = os.environ.get("IPM_MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.environ.get("IPM_MQTT_PORT", "1883"))
TOPIC_PREFIX = "ipm"

# Seconds since the first recording, topic, metadata and samples.
Message = tuple[float, str, dict, np.ndarray]


def encode_samples(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii")


def to_payload(meta: dict, samples: np.ndarray, sequence: int, injected: dict | None) -> dict:
    return {
        **meta,
        "shape": list(samples.shape),
        "samples": encode_samples(samples),
        "injected": injected,
        "sequence": sequence,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }


def femto_messages(bearing: str) -> Iterator[Message]:
    catalog = datasets.femto_catalog()
    rows = catalog[catalog["bearing"] == bearing]
    if rows.empty:
        raise SystemExit(f"Unknown FEMTO bearing: {bearing}")
    topic = f"{TOPIC_PREFIX}/femto/{bearing}/vibration"
    for row in rows.itertuples():
        meta = {
            "source": "femto",
            "asset": bearing,
            "sensor": "accelerometer",
            "channels": ["horizontal", "vertical"],
            "unit": "g",
            "sampling_hz": datasets.FEMTO_SAMPLING_HZ,
            "rpm": int(row.rpm),
            "elapsed_s": float(row.seconds),
        }
        yield float(row.seconds), topic, meta, datasets.read_femto(row.path)


def sca_messages(case: int, placement: str, part: str) -> Iterator[Message]:
    catalog = datasets.sca_catalog()
    rows = catalog[(catalog["case"] == case) & (catalog["placement"] == placement) & (catalog["part"] == part)]
    if rows.empty:
        raise SystemExit(f"No SCA data for case {case}, placement {placement}, part {part}")
    start = rows["time"].iat[0]
    topic = f"{TOPIC_PREFIX}/sca/case{case}/{placement}/vibration"
    for row in rows.itertuples():
        meta = {
            "source": "sca",
            "asset": f"case{case}",
            "placement": placement,
            "sensor": "accelerometer",
            "channels": ["vibration"],
            "unit": "m/s^2",
            "sampling_hz": float(row.sampling_hz),
            "rpm": float(row.rpm),
            "recorded_at": row.time.isoformat(),
        }
        samples = datasets.read_sca(case, part, placement, int(row.measurement))
        yield (row.time - start).total_seconds(), topic, meta, samples


def replay(messages: Iterator[Message], speed: float, limit: int | None, injector: DriftInjector | None) -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ipm-replay")
    client.connect(MQTT_HOST, MQTT_PORT)
    client.loop_start()
    started = time.monotonic()
    try:
        for sequence, (elapsed, topic, meta, samples) in enumerate(messages, start=1):
            # Keep the recorded time gaps, divided by the replay speed.
            time.sleep(max(0.0, started + elapsed / speed - time.monotonic()))
            injected = None
            if injector is not None:
                samples, injected = injector.apply(sequence, samples)
            payload = to_payload(meta, samples, sequence, injected)
            client.publish(topic, json.dumps(payload), qos=1).wait_for_publish()
            status = "real" if injected is None else f"{injected['kind']} drift, progress {injected['progress']}"
            print(f"{sequence:>6}  {topic}  shape {payload['shape']}  {status}")
            if limit and sequence >= limit:
                break
    finally:
        client.loop_stop()
        client.disconnect()


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--speed", type=float, default=1.0, help="how many times faster than recorded")
    common.add_argument("--limit", type=int, help="stop after this many messages")
    common.add_argument("--drift", choices=KINDS, help="change the replayed signal on purpose")
    common.add_argument("--drift-start", type=int, default=1, help="first message number that is changed")
    common.add_argument("--drift-strength", type=float, default=0.5, help="size of the change at full strength")
    common.add_argument("--drift-ramp", type=int, default=1, help="messages until the change reaches full strength")

    parser = argparse.ArgumentParser(description="Replay real bearing recordings over MQTT.")
    sources = parser.add_subparsers(dest="source", required=True)
    femto = sources.add_parser("femto", parents=[common], help="replay one FEMTO bearing")
    femto.add_argument("bearing", help="for example Bearing1_1")
    sca = sources.add_parser("sca", parents=[common], help="replay one SCA sensor")
    sca.add_argument("case", type=int, help="case number from 1 to 11")
    sca.add_argument("placement", help="for example DS or FS")
    sca.add_argument("--part", choices=["train", "test"], default="test")
    args = parser.parse_args()

    injector = None
    if args.drift:
        injector = DriftInjector(args.drift, args.drift_start, args.drift_strength, args.drift_ramp)
    if args.source == "femto":
        messages = femto_messages(args.bearing)
    else:
        messages = sca_messages(args.case, args.placement, args.part)
    replay(messages, args.speed, args.limit, injector)


if __name__ == "__main__":
    main()