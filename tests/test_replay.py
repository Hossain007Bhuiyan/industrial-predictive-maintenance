"""Tests for the replay message format, run on the full raw data."""

from __future__ import annotations
import base64
import json
import numpy as np
import pytest
from industrial_predictive_maintenance import datasets, replay

needs_femto = pytest.mark.skipif(not datasets.FEMTO_DIR.is_dir(), reason="FEMTO data not downloaded")
needs_sca = pytest.mark.skipif(not datasets.SCA_DIR.is_dir(), reason="SCA data not downloaded")


def decode(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["samples"])
    return np.frombuffer(raw, dtype="<f4").reshape(payload["shape"])


@needs_femto
def test_femto_message_keeps_the_exact_samples():
    elapsed, topic, meta, samples = next(replay.femto_messages("Bearing1_1"))
    payload = replay.to_payload(meta, samples, sequence=1, injected=None)
    catalog = datasets.femto_catalog()
    original = datasets.read_femto(catalog.loc[catalog["bearing"] == "Bearing1_1", "path"].iat[0])
    np.testing.assert_array_equal(decode(payload), original)
    assert topic == "ipm/femto/Bearing1_1/vibration"
    assert elapsed == 0
    assert payload["injected"] is None
    json.dumps(payload)


@needs_sca
def test_sca_message_keeps_the_exact_samples():
    elapsed, topic, meta, samples = next(replay.sca_messages(8, "DS", "test"))
    payload = replay.to_payload(meta, samples, sequence=1, injected=None)
    np.testing.assert_array_equal(decode(payload), datasets.read_sca(8, "test", "DS", 0))
    assert topic == "ipm/sca/case8/DS/vibration"
    assert elapsed == 0
    json.dumps(payload)


@needs_sca
def test_messages_never_contain_ground_truth():
    _, _, meta, samples = next(replay.sca_messages(8, "DS", "test"))
    payload = replay.to_payload(meta, samples, sequence=1, injected=None)
    assert not {"label", "rul", "fault_origin"} & set(payload)