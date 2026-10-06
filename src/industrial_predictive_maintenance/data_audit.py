"""Audit the raw bearing datasets before any processing.
Checks file structure, separators, sample counts and recording intervals
for FEMTO and IMS. Reports the vibration peak at the end of each FEMTO
bearing, which should be near the 20 g end-of-life limit. Summarises
every SCA case. Exits with code 1 if a structural problem is found.
"""

from __future__ import annotations
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from industrial_predictive_maintenance.data_download import RAW_DIR

FEMTO_DIR = RAW_DIR / "femto"
IMS_DIR = RAW_DIR / "ims"
SCA_DIR = RAW_DIR / "sca" / "SCA bearing dataset"

FEMTO_ROWS = 2560
FEMTO_COLUMNS = 6
IMS_ROWS = 20480
IMS_CHANNELS = {"1st_test": 8, "2nd_test": 4, "4th_test/txt": 4}
IMS_TEST3_DOCUMENTED_END = "2004.04.04.19.01.57"
SCA_TOP_LEVEL = {"id", "assetDescription", "faultOrigin", "faultType", "fromDate", "toDate", "fixedSpeed"}


def separator(line: str) -> str:
    if ";" in line:
        return ";"
    if "," in line:
        return ","
    if "\t" in line:
        return "tab"
    return "space"


def field_count(line: str) -> int:
    return len(re.split(r"[;,\s]+", line.strip()))


def first_line_and_rows(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    first = data.split(b"\n", 1)[0].decode().strip()
    rows = data.count(b"\n") + (0 if data.endswith(b"\n") else 1)
    return first, rows


def load_numeric(path: Path) -> np.ndarray:
    with path.open() as f:
        sep = separator(f.readline())
    delimiter = sep if sep in (";", ",") else None
    return np.loadtxt(path, delimiter=delimiter, ndmin=2)


def femto_seconds(line: str) -> float:
    h, m, s, us = (float(v) for v in re.split(r"[;,\s]+", line.strip())[:4])
    return h * 3600 + m * 60 + s + us / 1e6


def interval_summary(times: list[float], limit: float, wrap_midnight: bool) -> tuple[float, int]:
    steps = np.diff(np.asarray(times))
    if wrap_midnight:
        steps[steps < 0] += 86400
    return float(np.median(steps)), int((steps > limit).sum())


def fmt(counter: Counter) -> str:
    return "/".join(sorted(counter)) or "-"


def audit_femto(problems: list[str]) -> None:
    print("FEMTO  bearing      acc  temp  acc_sep  temp_sep  bad  step_s  gaps>15s  last_peak_g")
    for bearing in sorted(FEMTO_DIR.glob("*/Bearing*")):
        acc = sorted(bearing.glob("acc_*.csv"))
        temp = sorted(bearing.glob("temp_*.csv"))
        acc_seps, times, bad = Counter(), [], 0
        for f in acc:
            line, rows = first_line_and_rows(f)
            acc_seps[separator(line)] += 1
            times.append(femto_seconds(line))
            if rows != FEMTO_ROWS or field_count(line) != FEMTO_COLUMNS:
                bad += 1
        temp_seps = Counter(separator(first_line_and_rows(f)[0]) for f in temp)
        step, gaps = interval_summary(times, 15.0, wrap_midnight=True)
        peak = float(np.abs(load_numeric(acc[-1])[:, 4:6]).max())
        if bad:
            problems.append(f"FEMTO {bearing.name}: {bad} files with an unexpected shape")
        print(
            f"       {bearing.name:<11} {len(acc):>5} {len(temp):>5}  {fmt(acc_seps):<8} "
            f"{fmt(temp_seps):<9} {bad:>3}  {step:>6.1f}  {gaps:>8}  {peak:>11.2f}"
        )


def audit_ims(problems: list[str]) -> None:
    print("\nIMS    test          files  sep    bad  step_min  gaps>20min  first                last")
    for name, channels in IMS_CHANNELS.items():
        files = sorted(p for p in (IMS_DIR / name).iterdir() if p.name.startswith("20"))
        seps, bad = Counter(), 0
        for f in files:
            line, rows = first_line_and_rows(f)
            seps[separator(line)] += 1
            if rows != IMS_ROWS or field_count(line) != channels:
                bad += 1
        times = [
            datetime.strptime(f.name, "%Y.%m.%d.%H.%M.%S").replace(tzinfo=timezone.utc).timestamp()
            for f in files
        ]
        step, gaps = interval_summary(times, 20 * 60, wrap_midnight=False)
        if bad:
            problems.append(f"IMS {name}: {bad} files with an unexpected shape")
        print(
            f"       {name:<13} {len(files):>5}  {fmt(seps):<5} {bad:>4}  {step / 60:>8.1f}  "
            f"{gaps:>10}  {files[0].name}  {files[-1].name}"
        )
    documented = sum(
        1 for p in (IMS_DIR / "4th_test/txt").iterdir()
        if p.name.startswith("20") and p.name <= IMS_TEST3_DOCUMENTED_END
    )
    print(f"       test 3 files up to the documented end: {documented}")


def audit_sca(problems: list[str]) -> None:
    print("\nSCA    case part   place  n    samples  rate_hz             unit   rpm0  sorted  period                   labels")
    cases = sorted((p for p in SCA_DIR.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    for case in cases:
        for part in ("train", "test"):
            m = loadmat(case / f"{part}.mat")
            places = [k for k in m if not k.startswith("__") and k not in SCA_TOP_LEVEL]
            for place in places:
                s = m[place][0, 0]
                raw = s["rawData"]
                samples = raw.shape[-1] if raw.dtype != object else "ragged"
                rates = "/".join(f"{r:g}" for r in sorted({float(r) for r in np.ravel(s["samplingRate"])}))
                unit = str(np.ravel(s["unit"])[0])
                rpm0 = int((np.ravel(s["RPM"]) == 0).sum())
                times = [str(t).strip() for t in np.ravel(s["time"])]
                ordered = times == sorted(times)
                labels = Counter(int(v) for v in np.ravel(s["label"]))
                label_text = " ".join(f"{k}:{labels[k]}" for k in sorted(labels))
                if not ordered:
                    problems.append(f"SCA case {case.name} {part} {place}: timestamps not in order")
                print(
                    f"       {case.name:<4} {part:<6} {place:<6} {len(times):<4} {samples!s:<8} {rates:<19} "
                    f"{unit:<6} {rpm0:>4}  {ordered!s:<6}  {times[0][:10]}..{times[-1][:10]}  {label_text}"
                )


def main() -> int:
    problems: list[str] = []
    audit_femto(problems)
    audit_ims(problems)
    audit_sca(problems)
    if problems:
        print("\nProblems found:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nNo structural problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())