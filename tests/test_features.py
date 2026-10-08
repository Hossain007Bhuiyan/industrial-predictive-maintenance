"""Tests for the feature extraction, run on real recordings."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kurtosis

from industrial_predictive_maintenance import datasets, features

needs_femto = pytest.mark.skipif(not datasets.FEMTO_DIR.is_dir(), reason="FEMTO data not downloaded")
needs_sca = pytest.mark.skipif(not datasets.SCA_DIR.is_dir(), reason="SCA data not downloaded")


@pytest.fixture(scope="module")
def recording() -> np.ndarray:
    catalog = datasets.femto_catalog()
    return datasets.read_femto(catalog.loc[catalog["bearing"] == "Bearing1_1", "path"].iat[0])


@needs_femto
def test_time_features_match_a_direct_calculation(recording):
    x = recording[:, 0].astype(np.float64)
    f = features.time_features(recording[:, 0])
    assert f["rms"] == pytest.approx(np.sqrt(np.mean(x**2)))
    assert f["peak"] == pytest.approx(np.abs(x).max())
    assert f["crest"] == pytest.approx(f["peak"] / f["rms"])
    assert f["kurtosis"] == pytest.approx(kurtosis(x, fisher=False))


@needs_femto
def test_band_shares_add_up_to_one(recording):
    f = features.spectrum_features(recording[:, 0], datasets.FEMTO_SAMPLING_HZ)
    assert sum(f[f"band{i}_share"] for i in range(1, features.BANDS + 1)) == pytest.approx(1.0)
    assert 0 < f["centroid_hz"] < datasets.FEMTO_SAMPLING_HZ / 2


@needs_femto
def test_extract_names_every_channel(recording):
    f = features.extract(recording, datasets.FEMTO_SAMPLING_HZ, ["horizontal", "vertical"])
    assert len(f) == 36
    assert all(name.startswith(("horizontal_", "vertical_")) for name in f)
    assert all(np.isnan(f[f"horizontal_fault_{name}"]) for name in features.FAULT_NAMES)
    assert all(np.isfinite(value) for name, value in f.items() if "_fault_" not in name)


@needs_femto
def test_extract_rejects_a_wrong_channel_count(recording):
    with pytest.raises(ValueError):
        features.extract(recording, datasets.FEMTO_SAMPLING_HZ, ["only_one"])


@needs_sca
@pytest.mark.parametrize("case, fault", [(1, "BPFI"), (8, "BPFO")])
def test_known_fault_shows_at_its_frequency(case, fault):
    # Case 1 has an inner ring fault and case 8 an outer ring fault.
    sca = datasets.sca_catalog()
    rows = sca[(sca["case"] == case) & (sca["placement"] == "DS") & (sca["rpm"] > 0)]
    orders = datasets.sca_fault_orders(case, "test", "DS")
    ratio = {}
    for name, row in (("normal", rows[rows["part"] == "train"].iloc[0]), ("faulty", rows[rows["part"] == "test"].iloc[-1])):
        x = datasets.read_sca(case, row["part"], "DS", int(row["measurement"]))
        ratio[name] = features.fault_features(x, row["sampling_hz"], row["rpm"], orders)[f"fault_{fault}"]
    assert ratio["faulty"] > 2 * ratio["normal"]