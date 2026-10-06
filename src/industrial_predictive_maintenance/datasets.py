"""Load the raw FEMTO, IMS and SCA bearing data.
Each dataset has a catalog function that lists every recording with its
metadata and a read function that returns the vibration signal of one
recording. Signals are only read when needed, since all datasets at once
would not fit comfortably in memory.
"""

from __future__ import annotations
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from industrial_predictive_maintenance.data_download import RAW_DIR

FEMTO_DIR = RAW_DIR / "femto"
IMS_DIR = RAW_DIR / "ims"
SCA_DIR = RAW_DIR / "sca" / "SCA bearing dataset"

# PRONOSTIA operating conditions: shaft speed in rpm and radial load in N.
FEMTO_CONDITIONS = {1: (1800, 4000), 2: (1650, 4200), 3: (1500, 5000)}
FEMTO_SETS = {"Learning_set": "learning", "Full_Test_Set": "test"}
FEMTO_SAMPLING_HZ = 25_600

IMS_TESTS = {1: ("1st_test", 8), 2: ("2nd_test", 4), 3: ("4th_test/txt", 4)}
IMS_SAMPLING_HZ = 20_000
# The official documentation ends test 3 here. The archive holds 1,876 more
# recordings after this point that the documentation does not describe.
IMS_TEST3_DOCUMENTED_END = pd.Timestamp("2004-04-04 19:01:57")

SCA_TOP_LEVEL = {"id", "assetDescription", "faultOrigin", "faultType", "fromDate", "toDate", "fixedSpeed"}


# FEMTO

# Recordings are 10 s apart. A larger jump between two files means a broken clock value.
FEMTO_MAX_STEP_S = 600


def _femto_separator(path: Path) -> str:
    with path.open() as f:
        return ";" if ";" in f.readline() else ","


def _femto_clock(path: Path) -> float:
    """Time of day of a recording in seconds, read from its first row."""
    with path.open() as f:
        h, m, s, us = (float(v) for v in f.readline().replace(";", ",").split(",")[:4])
    return h * 3600 + m * 60 + s + us / 1e6


def _femto_elapsed_seconds(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the first recording plus a flag that tells whether each value comes from the file's clock.

    In Bearing1_1 the files acc_02121 and acc_02122 carry a clock value about
    six hours earlier than their neighbours, while the files around them are
    exactly 30 s apart. Clock values that do not move forward by a plausible
    step are replaced by linear interpolation between the valid neighbours.
    """
    clock = np.array([_femto_clock(path) for path in files])
    elapsed = np.zeros(len(clock))
    valid = np.zeros(len(clock), dtype=bool)
    valid[0] = True
    last = 0
    for i in range(1, len(clock)):
        step = clock[i] - clock[last]
        if step < -43_200:
            step += 86_400  # the recording passed midnight
        if 0 < step <= FEMTO_MAX_STEP_S:
            elapsed[i] = elapsed[last] + step
            valid[i] = True
            last = i
    index = np.arange(len(clock))
    elapsed[~valid] = np.interp(index[~valid], index[valid], elapsed[valid])
    return elapsed, valid


def femto_catalog() -> pd.DataFrame:
    rows = []
    for folder, set_name in FEMTO_SETS.items():
        for bearing_dir in sorted((FEMTO_DIR / folder).glob("Bearing*")):
            bearing = bearing_dir.name
            condition = int(bearing[7])
            rpm, load_n = FEMTO_CONDITIONS[condition]
            files = sorted(bearing_dir.glob("acc_*.csv"))
            elapsed, clock_ok = _femto_elapsed_seconds(files)
            for number, path in enumerate(files, start=1):
                rows.append(
                    {
                        "bearing": bearing,
                        "set": set_name,
                        "condition": condition,
                        "rpm": rpm,
                        "load_n": load_n,
                        "recording": number,
                        "seconds": float(elapsed[number - 1]),
                        "clock_ok": bool(clock_ok[number - 1]),
                        "path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def read_femto(path: Path | str) -> np.ndarray:
    """Horizontal and vertical vibration in g, shape (2560, 2)."""
    path = Path(path)
    values = pd.read_csv(path, header=None, sep=_femto_separator(path)).to_numpy()
    return values[:, 4:6].astype(np.float32)


# IMS


def ims_catalog(documented_only: bool = True) -> pd.DataFrame:
    rows = []
    for test, (folder, channels) in IMS_TESTS.items():
        for path in sorted((IMS_DIR / folder).glob("20*")):
            time = pd.to_datetime(path.name, format="%Y.%m.%d.%H.%M.%S")
            if documented_only and test == 3 and time > IMS_TEST3_DOCUMENTED_END:
                continue
            rows.append({"test": test, "time": time, "channels": channels, "path": str(path)})
    return pd.DataFrame(rows)


def read_ims(path: Path | str) -> np.ndarray:
    """Vibration of all channels, shape (20480, channels). The unit is not documented."""
    return pd.read_csv(path, header=None, sep="\t").to_numpy(dtype=np.float32)


# SCA


@lru_cache(maxsize=2)
def _load_sca_file(case: int, part: str) -> dict:
    return loadmat(SCA_DIR / str(case) / f"{part}.mat")


def _sca_placements(mat: dict) -> list[str]:
    return [k for k in mat if not k.startswith("__") and k not in SCA_TOP_LEVEL]


def sca_catalog() -> pd.DataFrame:
    rows = []
    cases = sorted(int(p.name) for p in SCA_DIR.iterdir() if p.is_dir())
    for case in cases:
        for part in ("train", "test"):
            mat = _load_sca_file(case, part)
            asset = str(np.ravel(mat["assetDescription"])[0])
            fault_origin = str(np.ravel(mat["faultOrigin"])[0])
            for placement in _sca_placements(mat):
                s = mat[placement][0, 0]
                times = pd.to_datetime([str(t).strip() for t in np.ravel(s["time"])], format="ISO8601")
                rates = np.ravel(s["samplingRate"])
                rpms = np.ravel(s["RPM"])
                labels = np.ravel(s["label"])
                bearing = str(np.ravel(s["assetName"])[0])
                for i, time in enumerate(times):
                    rows.append(
                        {
                            "case": case,
                            "part": part,
                            "placement": placement,
                            "measurement": i,
                            "time": time,
                            "sampling_hz": float(rates[i]),
                            "rpm": float(rpms[i]),
                            "label": int(labels[i]),
                            "bearing": bearing,
                            "asset": asset,
                            "fault_origin": fault_origin,
                        }
                    )
    return pd.DataFrame(rows)


def read_sca(case: int, part: str, placement: str, measurement: int) -> np.ndarray:
    """Vibration of one measurement in m/s^2. The length depends on the measurement."""
    raw = _load_sca_file(case, part)[placement][0, 0]["rawData"]
    row = raw[measurement] if raw.dtype != object else raw.ravel()[measurement]
    return np.asarray(row, dtype=np.float32).ravel()


def sca_fault_orders(case: int, part: str, placement: str) -> dict[str, float]:
    """Fault frequencies as multiples of the shaft speed in Hz (FTF, BPF, BPFO and BPFI)."""
    s = _load_sca_file(case, part)[placement][0, 0]
    frequencies = s["faultFrequencies"][0, 0]
    return {
        name.removesuffix("Multiple"): float(np.ravel(frequencies[name])[0])
        for name in frequencies.dtype.names
    }


# Check


def _femto_peaks(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bearing, group in catalog.groupby("bearing", sort=False):
        peaks = np.array([np.abs(read_femto(p)).max() for p in group["path"]])
        above = np.flatnonzero(peaks > 20)
        rows.append(
            {
                "bearing": bearing,
                "set": group["set"].iat[0],
                "recordings": len(peaks),
                "max_g": round(float(peaks.max()), 2),
                "max_at": int(peaks.argmax()) + 1,
                "first_above_20g": int(above[0]) + 1 if above.size else None,
                "last_g": round(float(peaks[-1]), 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    femto = femto_catalog()
    ims = ims_catalog()
    sca = sca_catalog()
    print(f"FEMTO  {femto['bearing'].nunique()} bearings, {len(femto)} recordings")
    print(f"IMS    recordings per test (documented part): {ims.groupby('test').size().to_dict()}")
    print(f"SCA    {sca['case'].nunique()} cases, {len(sca)} measurements")

    print("\nFEMTO peak vibration per recording over the whole life (g)")
    print(_femto_peaks(femto).to_string(index=False))

    print("\nSCA case 9 test: placement, sampling rate and signal length")
    nine = sca[(sca["case"] == 9) & (sca["part"] == "test")]
    lengths = Counter(
        (row.placement, row.sampling_hz, len(read_sca(9, "test", row.placement, row.measurement)))
        for row in nine.itertuples()
    )
    for (placement, rate, length), count in sorted(lengths.items()):
        print(f"       {placement}  {rate:g} Hz  {length} samples  {count} measurements")


if __name__ == "__main__":
    main()