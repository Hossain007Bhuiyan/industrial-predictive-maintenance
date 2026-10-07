"""Tests for the drift injector, run on real FEMTO recordings."""

from __future__ import annotations

import numpy as np
import pytest

from industrial_predictive_maintenance import datasets
from industrial_predictive_maintenance.drift import DriftInjector

needs_femto = pytest.mark.skipif(not datasets.FEMTO_DIR.is_dir(), reason="FEMTO data not downloaded")


@pytest.fixture(scope="module")
def recordings() -> list[np.ndarray]:
    catalog = datasets.femto_catalog()
    paths = catalog.loc[catalog["bearing"] == "Bearing1_1", "path"].iloc[:3]
    return [datasets.read_femto(p) for p in paths]


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=0))


@needs_femto
def test_messages_before_the_start_are_untouched(recordings):
    injector = DriftInjector("gain", start=2)
    samples, info = injector.apply(1, recordings[0])
    assert info is None
    np.testing.assert_array_equal(samples, recordings[0])


@needs_femto
def test_gain_multiplies_the_signal(recordings):
    injector = DriftInjector("gain", start=1, strength=0.5)
    samples, info = injector.apply(1, recordings[0])
    np.testing.assert_allclose(samples, recordings[0] * 1.5, rtol=1e-6)
    assert info["kind"] == "gain"
    assert info["progress"] == 1.0


@needs_femto
def test_ramp_grows_the_change_step_by_step(recordings):
    injector = DriftInjector("gain", start=1, strength=0.5, ramp=2)
    first, info = injector.apply(1, recordings[0])
    np.testing.assert_allclose(first, recordings[0] * 1.25, rtol=1e-6)
    assert info["progress"] == 0.5


@needs_femto
def test_offset_shifts_the_mean_by_a_share_of_the_rms(recordings):
    injector = DriftInjector("offset", start=1, strength=0.5)
    samples, _ = injector.apply(1, recordings[0])
    shift = samples.mean(axis=0) - recordings[0].mean(axis=0)
    np.testing.assert_allclose(shift, 0.5 * rms(recordings[0]), rtol=1e-4)


@needs_femto
def test_clip_cuts_the_peak(recordings):
    injector = DriftInjector("clip", start=1, strength=0.5)
    samples, _ = injector.apply(1, recordings[0])
    np.testing.assert_allclose(np.abs(samples).max(axis=0), 0.5 * np.abs(recordings[0]).max(axis=0), rtol=1e-6)


@needs_femto
def test_stuck_repeats_the_last_real_recording(recordings):
    injector = DriftInjector("stuck", start=2)
    injector.apply(1, recordings[0])
    second, _ = injector.apply(2, recordings[1])
    third, _ = injector.apply(3, recordings[2])
    np.testing.assert_array_equal(second, recordings[0])
    np.testing.assert_array_equal(third, recordings[0])


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        DriftInjector("noise", start=1)
    with pytest.raises(ValueError):
        DriftInjector("clip", start=1, strength=1.0)
    with pytest.raises(ValueError):
        DriftInjector("gain", start=0)