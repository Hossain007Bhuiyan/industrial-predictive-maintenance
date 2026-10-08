"""Build the feature tables of all three datasets from the raw recordings.

Uses the same feature code as the ingestion service, including the same
lookup of the SCA fault frequencies. Every table is checked with a Pandera
schema before it is saved as a Parquet file in data/processed:
    features_femto.parquet  one row per recording of each of the 17 bearings
    features_ims.parquet    one row per recording of each of the 4 bearings in a test
    features_sca.parquet    one row per measurement of each sensor in the 11 cases

Run:
    uv run python -m industrial_predictive_maintenance.build_features
    uv run python -m industrial_predictive_maintenance.build_features --only sca
"""

from __future__ import annotations

import argparse
import sys
import time
import pandas as pd
import pandera.pandas as pa
from industrial_predictive_maintenance import datasets, features
from industrial_predictive_maintenance.data_download import DATA_DIR
from industrial_predictive_maintenance.ingest import fault_orders

PROCESSED_DIR = DATA_DIR / "processed"
FEMTO_CHANNELS = ["horizontal", "vertical"]
# Bearings that failed at the end of each IMS test, from the official documentation.
IMS_FAILED = {1: {3, 4}, 2: {1}, 3: {3}}


def _progress(name: str, done: int, total: int, started: float) -> None:
    if done % 1000 == 0 or done == total:
        print(f"  {name}: {done}/{total} recordings ({time.monotonic() - started:.0f} s)", flush=True)


def femto_table(catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    catalog = datasets.femto_catalog() if catalog is None else catalog
    rows, started = [], time.monotonic()
    for done, row in enumerate(catalog.itertuples(), start=1):
        values = features.extract(datasets.read_femto(row.path), datasets.FEMTO_SAMPLING_HZ, FEMTO_CHANNELS)
        rows.append(
            {
                "bearing": row.bearing,
                "set": row.set,
                "condition": row.condition,
                "rpm": row.rpm,
                "load_n": row.load_n,
                "recording": row.recording,
                "seconds": row.seconds,
                "clock_ok": row.clock_ok,
                **values,
            }
        )
        _progress("FEMTO", done, len(catalog), started)
    return pd.DataFrame(rows)


def ims_table(catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    catalog = datasets.ims_catalog() if catalog is None else catalog
    rows, started = [], time.monotonic()
    for done, row in enumerate(catalog.itertuples(), start=1):
        signal = datasets.read_ims(row.path)
        # Test 1 has two channels (x and y) per bearing. Tests 2 and 3 have one.
        per_bearing = signal.shape[1] // 4
        names = ["x", "y"] if per_bearing == 2 else ["vibration"]
        for bearing in range(1, 5):
            columns = signal[:, (bearing - 1) * per_bearing : bearing * per_bearing]
            values = features.extract(columns, datasets.IMS_SAMPLING_HZ, names)
            rows.append(
                {
                    "test": row.test,
                    "bearing": bearing,
                    "failed": bearing in IMS_FAILED[row.test],
                    "time": row.time,
                    "after_documented_end": row.after_documented_end,
                    **values,
                }
            )
        _progress("IMS", done, len(catalog), started)
    return pd.DataFrame(rows)


def sca_table(catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    catalog = datasets.sca_catalog() if catalog is None else catalog
    rows, started = [], time.monotonic()
    for done, row in enumerate(catalog.itertuples(), start=1):
        x = datasets.read_sca(row.case, row.part, row.placement, row.measurement)
        orders = fault_orders("sca", f"case{row.case}", row.placement)
        values = features.extract(x, row.sampling_hz, ["vibration"], row.rpm, orders)
        rows.append(
            {
                "case": row.case,
                "part": row.part,
                "placement": row.placement,
                "measurement": row.measurement,
                "time": row.time,
                "sampling_hz": row.sampling_hz,
                "rpm": row.rpm,
                "label": row.label,
                "fault_origin": row.fault_origin,
                "asset": row.asset,
                "bearing": row.bearing,
                **values,
            }
        )
        _progress("SCA", done, len(catalog), started)
    return pd.DataFrame(rows)


def feature_schema(channels: list[str], sampling_hz: float | str) -> pa.DataFrameSchema:
    """Schema for the feature columns of the given channels.

    sampling_hz is either a fixed rate or the name of a column holding the rate of each row.
    """

    def nyquist(df: pd.DataFrame) -> pd.Series | float:
        return (df[sampling_hz] if isinstance(sampling_hz, str) else sampling_hz) / 2

    columns, checks = {}, []
    for c in channels:
        bands = [f"{c}_band{i}_share" for i in range(1, features.BANDS + 1)]
        columns.update(
            {
                f"{c}_mean": pa.Column(float),
                f"{c}_std": pa.Column(float, pa.Check.ge(0)),
                f"{c}_rms": pa.Column(float, pa.Check.gt(0)),
                f"{c}_peak": pa.Column(float, pa.Check.gt(0)),
                # Peak over RMS and the Pearson kurtosis can never be below 1.
                f"{c}_crest": pa.Column(float, pa.Check.ge(1)),
                f"{c}_kurtosis": pa.Column(float, pa.Check.ge(1)),
                f"{c}_skewness": pa.Column(float),
                f"{c}_flat_top_fraction": pa.Column(float, pa.Check.in_range(0, 1, include_min=False)),
                f"{c}_centroid_hz": pa.Column(float, pa.Check.gt(0)),
                f"{c}_dominant_hz": pa.Column(float, pa.Check.gt(0)),
                **{band: pa.Column(float, pa.Check.in_range(0, 1)) for band in bands},
                **{f"{c}_fault_{n}": pa.Column(float, pa.Check.ge(0), nullable=True) for n in features.FAULT_NAMES},
            }
        )
        checks += [
            pa.Check(lambda df, bands=bands: (df[bands].sum(axis=1) - 1).abs() < 1e-6, name=f"{c} band shares add up to 1"),
            pa.Check(lambda df, c=c: df[f"{c}_centroid_hz"] <= nyquist(df), name=f"{c} centroid below Nyquist"),
        ]
    return pa.DataFrameSchema(columns, checks=checks)


def save(name: str, table: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"features_{name}.parquet"
    table.to_parquet(path, index=False)
    print(f"saved {path}: {len(table)} rows, {table.shape[1]} columns")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the feature tables from the raw recordings.")
    parser.add_argument("--only", choices=["femto", "ims", "sca"], help="build one table only")
    args = parser.parse_args()
    try:
        if args.only in (None, "femto"):
            femto = femto_table()
            feature_schema(FEMTO_CHANNELS, datasets.FEMTO_SAMPLING_HZ).validate(femto, lazy=True)
            save("femto", femto)
        if args.only in (None, "ims"):
            ims = ims_table()
            feature_schema(["x", "y"], datasets.IMS_SAMPLING_HZ).validate(ims[ims["test"] == 1], lazy=True)
            feature_schema(["vibration"], datasets.IMS_SAMPLING_HZ).validate(ims[ims["test"] != 1], lazy=True)
            save("ims", ims)
        if args.only in (None, "sca"):
            sca = sca_table()
            feature_schema(["vibration"], "sampling_hz").validate(sca, lazy=True)
            save("sca", sca)
    except pa.errors.SchemaErrors as error:
        cases = error.failure_cases
        print(f"schema check failed: {len(cases)} problems")
        print(cases.groupby(["column", "check"], dropna=False).size().rename("count").to_string())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())