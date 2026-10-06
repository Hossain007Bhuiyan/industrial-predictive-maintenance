"""Tests for the dataset loaders, run on the full raw data.

They need the downloaded data in data/raw and are skipped when it is
missing, for example on a CI machine without the datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from industrial_predictive_maintenance import datasets

needs_femto = pytest.mark.skipif(not datasets.FEMTO_DIR.is_dir(), reason="FEMTO data not downloaded")
needs_ims = pytest.mark.skipif(not datasets.IMS_DIR.is_dir(), reason="IMS data not downloaded")
needs_sca = pytest.mark.skipif(not datasets.SCA_DIR.is_dir(), reason="SCA data not downloaded")


@pytest.fixture(scope="module")
def femto() -> pd.DataFrame:
    return datasets.femto_catalog()


@pytest.fixture(scope="module")
def ims() -> pd.DataFrame:
    return datasets.ims_catalog()


@pytest.fixture(scope="module")
def sca() -> pd.DataFrame:
    return datasets.sca_catalog()


# FEMTO


@needs_femto
def test_femto_has_all_bearings(femto):
    assert femto["bearing"].nunique() == 17
    assert len(femto) == 24_889
    assert femto.groupby("set")["bearing"].nunique().to_dict() == {"learning": 6, "test": 11}


@needs_femto
def test_femto_condition_matches_bearing_name(femto):
    for row in femto.drop_duplicates("bearing").itertuples():
        assert row.condition == int(row.bearing[7])
        assert (row.rpm, row.load_n) == datasets.FEMTO_CONDITIONS[row.condition]


@needs_femto
def test_femto_recordings_are_ten_seconds_apart(femto):
    for bearing, group in femto.groupby("bearing"):
        steps = np.diff(group["seconds"].to_numpy())
        assert group["seconds"].iat[0] == 0
        assert (steps > 0).all(), bearing
        assert abs(np.median(steps) - 10) < 0.1, bearing


@needs_femto
@pytest.mark.parametrize("bearing", ["Bearing1_1", "Bearing1_4"])
def test_read_femto_handles_both_separators(femto, bearing):
    # Bearing1_4 uses semicolons. The other bearings use commas.
    path = femto.loc[femto["bearing"] == bearing, "path"].iat[0]
    signal = datasets.read_femto(path)
    assert signal.shape == (2560, 2)
    assert signal.dtype == np.float32
    assert np.isfinite(signal).all()


@needs_femto
def test_femto_broken_clock_values_are_only_in_bearing1_1(femto):
    broken = femto.loc[~femto["clock_ok"], ["bearing", "recording"]]
    assert broken.values.tolist() == [["Bearing1_1", 2121], ["Bearing1_1", 2122]]


# IMS


@needs_ims
def test_ims_documented_recordings(ims):
    assert ims.groupby("test").size().to_dict() == {1: 2156, 2: 984, 3: 4448}
    assert ims.loc[ims["test"] == 3, "time"].max() == datasets.IMS_TEST3_DOCUMENTED_END


@needs_ims
def test_ims_full_archive_keeps_extra_test3_recordings():
    full = datasets.ims_catalog(documented_only=False)
    assert (full["test"] == 3).sum() == 6324


@needs_ims
@pytest.mark.parametrize("test, channels", [(1, 8), (2, 4), (3, 4)])
def test_read_ims_shape(ims, test, channels):
    path = ims.loc[ims["test"] == test, "path"].iat[0]
    signal = datasets.read_ims(path)
    assert signal.shape == (20480, channels)
    assert np.isfinite(signal).all()


# SCA


@needs_sca
def test_sca_has_all_measurements(sca):
    assert sca["case"].nunique() == 11
    assert len(sca) == 6644
    assert set(sca["label"].unique()) <= {-1, 0, 1, 2, 3}


@needs_sca
def test_sca_times_are_sorted(sca):
    for key, group in sca.groupby(["case", "part", "placement"]):
        assert group["time"].is_monotonic_increasing, key


@needs_sca
@pytest.mark.parametrize("rate, length", [(2560, 8192), (5120, 16384)])
def test_read_sca_case9_mixed_rates(sca, rate, length):
    # Case 9 test mixes two sampling rates. Both measurements last 3.2 s.
    row = sca[(sca["case"] == 9) & (sca["part"] == "test") & (sca["sampling_hz"] == rate)].iloc[0]
    signal = datasets.read_sca(9, "test", row["placement"], int(row["measurement"]))
    assert len(signal) == length
    assert np.isfinite(signal).all()


@needs_sca
def test_sca_fault_orders_have_four_frequencies():
    orders = datasets.sca_fault_orders(1, "test", "DS")
    assert set(orders) == {"FTF", "BPF", "BPFO", "BPFI"}
    assert all(value > 0 for value in orders.values())