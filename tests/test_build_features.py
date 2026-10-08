"""Tests for the batch feature tables, run on real recordings."""

from __future__ import annotations
import json
import numpy as np
import pandera.pandas as pa
import pytest
from industrial_predictive_maintenance import build_features, datasets, replay
from industrial_predictive_maintenance.ingest import Ingestor

needs_femto = pytest.mark.skipif(not datasets.FEMTO_DIR.is_dir(), reason="FEMTO data not downloaded")
needs_ims = pytest.mark.skipif(not datasets.IMS_DIR.is_dir(), reason="IMS data not downloaded")
needs_sca = pytest.mark.skipif(not datasets.SCA_DIR.is_dir(), reason="SCA data not downloaded")


@pytest.fixture(scope="module")
def femto_rows():
    return build_features.femto_table(datasets.femto_catalog().head(3))


@needs_femto
def test_femto_rows_pass_the_schema(femto_rows):
    schema = build_features.feature_schema(build_features.FEMTO_CHANNELS, datasets.FEMTO_SAMPLING_HZ)
    schema.validate(femto_rows, lazy=True)
    assert len(femto_rows) == 3


@needs_femto
@pytest.mark.parametrize(
    "column, value",
    [("horizontal_rms", -1.0), ("vertical_kurtosis", np.nan), ("horizontal_band1_share", 1.5)],
)
def test_schema_rejects_impossible_values(femto_rows, column, value):
    # A real row with one deliberate defect must fail the schema.
    broken = femto_rows.copy()
    broken.loc[0, column] = value
    schema = build_features.feature_schema(build_features.FEMTO_CHANNELS, datasets.FEMTO_SAMPLING_HZ)
    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(broken, lazy=True)


@needs_ims
def test_ims_gives_one_row_per_bearing():
    catalog = datasets.ims_catalog()
    first_of_each_test = catalog.groupby("test").head(1)
    table = build_features.ims_table(first_of_each_test)
    assert len(table) == 12
    assert table.groupby("test")["bearing"].apply(list).to_dict() == {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4]}
    assert table.loc[table["failed"], ["test", "bearing"]].values.tolist() == [[1, 3], [1, 4], [2, 1], [3, 3]]
    test1, others = table[table["test"] == 1], table[table["test"] != 1]
    build_features.feature_schema(["x", "y"], datasets.IMS_SAMPLING_HZ).validate(test1, lazy=True)
    build_features.feature_schema(["vibration"], datasets.IMS_SAMPLING_HZ).validate(others, lazy=True)


@needs_sca
def test_batch_and_stream_give_the_same_features(tmp_path):
    catalog = datasets.sca_catalog()
    first = catalog[(catalog["case"] == 8) & (catalog["part"] == "test") & (catalog["placement"] == "DS")].head(1)
    batch = build_features.sca_table(first).iloc[0]

    _, topic, meta, samples = next(replay.sca_messages(8, "DS", "test"))
    ingestor = Ingestor(tmp_path)
    ingestor.handle(topic, json.dumps(replay.to_payload(meta, samples, 1, None)).encode())
    stream = json.loads(ingestor.features_file.read_text().splitlines()[0])

    names = [name for name in stream if name.startswith("vibration_")]
    assert len(names) == 18
    for name in names:
        assert stream[name] == pytest.approx(batch[name]), name