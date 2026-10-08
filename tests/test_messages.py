"""Tests for the MQTT message schema.

Valid messages are built from real recordings with the replay code. Every
invalid case is a real message with one deliberate defect, to check that
the schema rejects it.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from pydantic import ValidationError

from industrial_predictive_maintenance import datasets, messages, replay
from industrial_predictive_maintenance.drift import DriftInjector

needs_femto = pytest.mark.skipif(not datasets.FEMTO_DIR.is_dir(), reason="FEMTO data not downloaded")
needs_sca = pytest.mark.skipif(not datasets.SCA_DIR.is_dir(), reason="SCA data not downloaded")


@pytest.fixture(scope="module")
def femto_message() -> tuple[dict, np.ndarray]:
    _, _, meta, samples = next(replay.femto_messages("Bearing1_1"))
    return meta, samples


def with_nan(encoded: str) -> str:
    values = np.frombuffer(base64.b64decode(encoded), dtype="<f4").copy()
    values[0] = np.nan
    return base64.b64encode(values.tobytes()).decode("ascii")


DEFECTS = {
    "missing field": lambda p: p.pop("samples"),
    "unknown field": lambda p: p.update(label=3),
    "negative speed": lambda p: p.update(rpm=-1),
    "wrong channel names": lambda p: p.update(channels=["horizontal"]),
    "shape does not match samples": lambda p: p.update(shape=[2561, 2]),
    "invalid base64": lambda p: p.update(samples="not base64!"),
    "NaN sample": lambda p: p.update(samples=with_nan(p["samples"])),
}


@needs_femto
def test_valid_femto_message(femto_message):
    meta, samples = femto_message
    message = messages.parse(json.dumps(replay.to_payload(meta, samples, 1, None)))
    np.testing.assert_array_equal(message.signal, samples)
    assert message.injected is None


@needs_sca
def test_valid_sca_message():
    _, _, meta, samples = next(replay.sca_messages(8, "DS", "test"))
    message = messages.parse(json.dumps(replay.to_payload(meta, samples, 1, None)))
    np.testing.assert_array_equal(message.signal, samples)
    assert message.recorded_at is not None


@needs_femto
def test_message_with_injected_drift(femto_message):
    meta, samples = femto_message
    changed, injected = DriftInjector("gain", start=1).apply(1, samples)
    message = messages.parse(json.dumps(replay.to_payload(meta, changed, 1, injected)))
    assert message.injected.kind == "gain"


@needs_femto
@pytest.mark.parametrize("defect", DEFECTS)
def test_defective_message_is_rejected(femto_message, defect):
    meta, samples = femto_message
    payload = replay.to_payload(meta, samples, 1, None)
    DEFECTS[defect](payload)
    with pytest.raises(ValidationError):
        messages.parse(json.dumps(payload))