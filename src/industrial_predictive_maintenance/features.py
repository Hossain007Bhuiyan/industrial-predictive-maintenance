"""Features computed from one vibration recording.

The same functions are used for streamed messages and for the batch
datasets, so training and live prediction always see identical features.

Per channel:
    time domain       mean, std, rms, peak, crest factor, kurtosis, skewness
                      and the share of samples at the recording's peak level
    frequency domain  spectral centroid, dominant frequency and the share of
                      energy in four equal bands up to the Nyquist frequency
    fault frequencies amplitude in the envelope spectrum at each bearing fault
                      frequency, relative to the median level of that spectrum.
                      Only computed when the shaft speed and the fault
                      frequency multiples are known (SCA). Otherwise NaN.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

BANDS = 4
FAULT_NAMES = ("FTF", "BPF", "BPFO", "BPFI")
HARMONICS = 3
# Half width of the window searched around a fault frequency: 1 % of the frequency, at least 2 bins.
FAULT_TOLERANCE = 0.01
FLAT_TOP_LEVEL = 0.999


def time_features(x: np.ndarray) -> dict[str, float]:
    x = x.astype(np.float64)
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.abs(x).max())
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "rms": rms,
        "peak": peak,
        "crest": peak / rms if rms > 0 else np.nan,
        "kurtosis": float(kurtosis(x, fisher=False)),
        "skewness": float(skew(x)),
        "flat_top_fraction": float(np.mean(np.abs(x) >= FLAT_TOP_LEVEL * peak)) if peak > 0 else np.nan,
    }


def spectrum_features(x: np.ndarray, sampling_hz: float) -> dict[str, float]:
    x = x.astype(np.float64) - x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sampling_hz)
    power, freqs = power[1:], freqs[1:]  # drop the zero frequency
    total = power.sum()
    features = {
        "centroid_hz": float((freqs * power).sum() / total) if total > 0 else np.nan,
        "dominant_hz": float(freqs[np.argmax(power)]),
    }
    edges = np.linspace(0, sampling_hz / 2, BANDS + 1)
    for i in range(BANDS):
        in_band = (freqs > edges[i]) & (freqs <= edges[i + 1])
        features[f"band{i + 1}_share"] = float(power[in_band].sum() / total) if total > 0 else np.nan
    return features


def envelope_spectrum(x: np.ndarray, sampling_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64) - x.mean()
    envelope = np.abs(hilbert(x))
    envelope -= envelope.mean()
    # A Hann window keeps a strong peak from leaking into the neighbouring frequencies.
    window = np.hanning(len(envelope))
    amplitude = np.abs(np.fft.rfft(envelope * window)) * 2 / window.sum()
    return np.fft.rfftfreq(len(envelope), 1 / sampling_hz), amplitude


def fault_features(
    x: np.ndarray, sampling_hz: float, rpm: float | None, orders: dict[str, float] | None
) -> dict[str, float]:
    empty = {f"fault_{name}": np.nan for name in FAULT_NAMES}
    if rpm is None or rpm <= 0 or not orders:
        return empty
    freqs, amplitude = envelope_spectrum(x, sampling_hz)
    level = np.median(amplitude[1:])
    if level <= 0:
        return empty
    shaft_hz = rpm / 60
    bin_width = freqs[1] - freqs[0]
    features = {}
    for name in FAULT_NAMES:
        peaks = []
        for harmonic in range(1, HARMONICS + 1):
            target = orders[name] * shaft_hz * harmonic
            if target >= freqs[-1]:
                break
            half_width = max(FAULT_TOLERANCE * target, 2 * bin_width)
            window = (freqs >= target - half_width) & (freqs <= target + half_width)
            peaks.append(amplitude[window].max())
        features[f"fault_{name}"] = float(np.mean(peaks) / level) if peaks else np.nan
    return features


def extract(
    samples: np.ndarray,
    sampling_hz: float,
    channels: list[str],
    rpm: float | None = None,
    orders: dict[str, float] | None = None,
) -> dict[str, float]:
    """All features of one recording. Names are prefixed with the channel name."""
    samples = samples.reshape(len(samples), -1)
    if samples.shape[1] != len(channels):
        raise ValueError(f"{samples.shape[1]} channels in the samples but {len(channels)} channel names")
    features = {}
    for i, channel in enumerate(channels):
        x = samples[:, i]
        values = {
            **time_features(x),
            **spectrum_features(x, sampling_hz),
            **fault_features(x, sampling_hz, rpm, orders),
        }
        features.update({f"{channel}_{name}": value for name, value in values.items()})
    return features


def main() -> None:
    """Check the features on real recordings."""
    from industrial_predictive_maintenance import datasets

    femto = datasets.femto_catalog()
    paths = femto.loc[femto["bearing"] == "Bearing1_1", "path"]
    channels = ["horizontal", "vertical"]
    first = extract(datasets.read_femto(paths.iat[0]), datasets.FEMTO_SAMPLING_HZ, channels)
    last = extract(datasets.read_femto(paths.iat[-1]), datasets.FEMTO_SAMPLING_HZ, channels)
    print(f"features per FEMTO recording: {len(first)}")
    print("\nFEMTO Bearing1_1                 first       last")
    for name in ("horizontal_rms", "horizontal_peak", "horizontal_kurtosis", "horizontal_centroid_hz", "vertical_rms"):
        print(f"  {name:<24} {first[name]:>10.4f} {last[name]:>10.4f}")
    rms_rises = last["horizontal_rms"] > first["horizontal_rms"] and last["vertical_rms"] > first["vertical_rms"]
    print("  RMS higher in the last recording:", rms_rises)

    sca = datasets.sca_catalog()
    for case, expected in ((1, "BPFI"), (8, "BPFO")):
        rows = sca[(sca["case"] == case) & (sca["placement"] == "DS") & (sca["rpm"] > 0)]
        normal = rows[rows["part"] == "train"].iloc[0]
        faulty = rows[rows["part"] == "test"].iloc[-1]
        orders = datasets.sca_fault_orders(case, "test", "DS")
        result = {}
        for name, row in (("normal", normal), ("faulty", faulty)):
            x = datasets.read_sca(case, row["part"], "DS", int(row["measurement"]))
            result[name] = fault_features(x, row["sampling_hz"], row["rpm"], orders)
        print(f"\nSCA case {case} DS, labels: normal {normal['label']}, faulty {faulty['label']}")
        print("  envelope amplitude relative to the median level")
        for fault in FAULT_NAMES:
            print(
                f"  {fault:<5} normal {result['normal'][f'fault_{fault}']:>8.2f}"
                f"   faulty {result['faulty'][f'fault_{fault}']:>8.2f}"
            )
        higher = result["faulty"][f"fault_{expected}"] > result["normal"][f"fault_{expected}"]
        print(f"  {expected} higher in the faulty measurement:", higher)


if __name__ == "__main__":
    main()