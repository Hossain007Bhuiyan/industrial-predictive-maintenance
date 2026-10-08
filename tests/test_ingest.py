"""Tests for the ingestion service, run on real messages without a broker."""

from __future__ import annotations

import json

import pytest

from industrial_predictive_maintenance import datasets, features, replay
from industrial_predictive_maintenance.drift import DriftInjector
from industrial_predictive_maintenance.ingest import Ingestor

needs_femto = pytest.mark.skipif(not datasets.FEMTO_DIR.is_dir(), reason="FEMTO data not downloaded")
needs_sca = pytest.mark.skipif(not datasets.SCA_DIR.is_dir(), reason="SCA data not downloaded")

FEMTO_TOPIC = "ipm/femto/Bearing1_1/vibration"
SCA_TOPIC = "ipm/sca/case8/DS/vibration"


def rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def femto_message():
    _, _, meta, samples = next(replay.femto_messages("Bearing1_1"))
    return meta, samples


@needs_femto
def test_valid_message_is_stored_with_the_same_features_as_the_batch_code(tmp_path, femto_message):
    meta, samples = femto_message
    ingestor = Ingestor(tmp_path)
    payload = json.dumps(replay.to_payload(meta, samples, 1, None)).encode()
    assert ingestor.handle(FEMTO_TOPIC, payload) == "accepted"
    [row] = rows(ingestor.features_file)
    expected = features.extract(samples, meta["sampling_hz"], meta["channels"])
    for name, value in expected.items():
        if "_fault_" in name:
            assert row[name] is None
        else:
            assert row[name] == pytest.approx(value)
    assert row["asset"] == "Bearing1_1"
    assert row["injected"] is None


@needs_sca
def test_sca_message_gets_fault_features(tmp_path):
    _, _, meta, samples = next(replay.sca_messages(8, "DS", "test"))
    ingestor = Ingestor(tmp_path)
    assert ingestor.handle(SCA_TOPIC, json.dumps(replay.to_payload(meta, samples, 1, None)).encode()) == "accepted"
    [row] = rows(ingestor.features_file)
    assert row["vibration_fault_BPFO"] > 0
    assert row["recorded_at"] is not None


@needs_femto
def test_defective_message_is_rejected_with_a_reason(tmp_path, femto_message):
    meta, samples = femto_message
    payload = replay.to_payload(meta, samples, 1, None)
    payload["label"] = 3
    ingestor = Ingestor(tmp_path)
    assert ingestor.handle(FEMTO_TOPIC, json.dumps(payload).encode()) == "rejected"
    assert not ingestor.features_file.exists()
    [row] = rows(ingestor.rejected_file)
    assert any("Extra inputs" in reason for reason in row["reasons"])


@needs_femto
def test_duplicate_message_is_stored_once(tmp_path, femto_message):
    meta, samples = femto_message
    ingestor = Ingestor(tmp_path)
    payload = json.dumps(replay.to_payload(meta, samples, 1, None)).encode()
    assert ingestor.handle(FEMTO_TOPIC, payload) == "accepted"
    assert ingestor.handle(FEMTO_TOPIC, payload) == "duplicate"
    assert len(rows(ingestor.features_file)) == 1


@needs_femto
def test_injected_drift_is_recorded(tmp_path, femto_message):
    meta, samples = femto_message
    changed, injected = DriftInjector("gain", start=1).apply(1, samples)
    ingestor = Ingestor(tmp_path)
    ingestor.handle(FEMTO_TOPIC, json.dumps(replay.to_payload(meta, changed, 1, injected)).encode())
    [row] = rows(ingestor.features_file)
    assert row["injected"] == "gain"