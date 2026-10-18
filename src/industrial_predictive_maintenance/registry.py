# Fills the asset registry: one row per machine and one per sensor position for FEMTO, IMS and SCA.
# Values come from the dataset files and the IMS readme. Running it again updates the same rows.
#     uv run --env-file .env python -m industrial_predictive_maintenance.registry

from __future__ import annotations

import math

import numpy as np
from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import Engine

from industrial_predictive_maintenance import datasets
from industrial_predictive_maintenance.db import make_engine
from industrial_predictive_maintenance.models import Machine, Sensor

# From the IMS readme: four Rexnord ZA-2115 bearings on one shaft, constant 2000 rpm
# and a radial load of 6000 lbs (1 lbf = 4.4482216152605 N).
IMS_RPM = 2000
IMS_LOAD_N = 6000 * 4.4482216152605
IMS_BEARING = "Rexnord ZA-2115"

FAULT_COLUMNS = {"FTF": "fault_ftf", "BPF": "fault_bpf", "BPFO": "fault_bpfo", "BPFI": "fault_bpfi"}

# One machine row and its sensor rows.
Entry = tuple[dict, list[dict]]


# One sensor row. The fault frequency columns stay empty when none are given.
def sensor(
    placement: str | None, channels: list[str], unit: str | None, bearing: str | None = None, faults: dict | None = None
) -> dict:
    row = {"placement": placement, "channels": channels, "unit": unit, "bearing": bearing}
    for column in FAULT_COLUMNS.values():
        row[column] = (faults or {}).get(column)
    return row


# One machine per FEMTO bearing run, with the speed and load of its operating condition.
def femto_entries() -> list[Entry]:
    entries = []
    for folder in datasets.FEMTO_SETS:
        for bearing_dir in sorted((datasets.FEMTO_DIR / folder).glob("Bearing*")):
            rpm, load_n = datasets.FEMTO_CONDITIONS[int(bearing_dir.name[7])]
            # Each condition has one nominal speed. Whether it was held constant during
            # the run is not verified, so fixed_speed stays empty.
            machine = {"source": "femto", "name": bearing_dir.name, "kind": "PRONOSTIA test rig",
                       "rpm": rpm, "load_n": load_n, "fixed_speed": None}
            entries.append((machine, [sensor(None, ["horizontal", "vertical"], "g")]))
    return entries


# One machine per IMS test, with a sensor for each of the four bearings on the shaft.
def ims_entries() -> list[Entry]:
    entries = []
    for test, (_, channels) in datasets.IMS_TESTS.items():
        names = ["x", "y"] if channels // 4 == 2 else ["vibration"]
        machine = {"source": "ims", "name": f"test{test}", "kind": "IMS test rig",
                   "rpm": IMS_RPM, "load_n": IMS_LOAD_N, "fixed_speed": True}
        # The readme gives no unit for the vibration values.
        sensors = [sensor(f"bearing{b}", names, None, IMS_BEARING) for b in range(1, 5)]
        entries.append((machine, sensors))
    return entries


# First value of a MATLAB text field, without surrounding spaces.
def _text(value) -> str:
    return str(np.ravel(value)[0]).strip()


# One SCA case as described by its train or test file.
def sca_entry(case: int, part: str) -> Entry:
    mat = datasets._load_sca_file(case, part)
    fixed = int(np.ravel(mat["fixedSpeed"])[0])
    if fixed not in (0, 1):
        raise SystemExit(f"SCA case {case} {part}: unexpected fixedSpeed {fixed}")
    # The speed changes between measurements, so it's stored with each measurement instead.
    machine = {"source": "sca", "name": f"case{case}", "kind": _text(mat["assetDescription"]),
               "rpm": None, "load_n": None, "fixed_speed": bool(fixed)}
    sensors = []
    for placement in sorted(datasets._sca_placements(mat)):
        s = mat[placement][0, 0]
        orders = datasets.sca_fault_orders(case, part, placement)
        faults = {FAULT_COLUMNS[name]: value for name, value in orders.items()}
        sensors.append(sensor(placement, ["vibration"], _text(s["unit"]), _text(s["assetName"]), faults))
    return machine, sensors


# A number can come back from the database with its last digit rounded, so floats get a tiny tolerance.
def same_value(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9)
    return a == b


# True when two machines have the same values and the same sensor rows in the same order.
def same_entry(a: Entry, b: Entry) -> bool:
    rows_a, rows_b = [a[0], *a[1]], [b[0], *b[1]]
    return len(rows_a) == len(rows_b) and all(
        x.keys() == y.keys() and all(same_value(x[k], y[k]) for k in x) for x, y in zip(rows_a, rows_b)
    )


# One machine per SCA case, taken from the test file once it's clear train describes the same machine.
def sca_entries() -> list[Entry]:
    entries = []
    for case in sorted(int(p.name) for p in datasets.SCA_DIR.iterdir() if p.is_dir()):
        test, train = sca_entry(case, "test"), sca_entry(case, "train")
        # One row per position is only right if both parts describe the same machine and bearings.
        if not same_entry(test, train):
            raise SystemExit(f"SCA case {case}: train and test describe different machines or bearings")
        entries.append(test)
    return entries


# Writes everything in one transaction, so either all rows are saved or none.
def write(engine: Engine, entries: list[Entry]) -> None:
    # Existing rows are updated in place, so running it again keeps the same ids.
    # An insert with ON CONFLICT would use up a new id from the sequence on every run.
    with engine.begin() as conn:
        for machine, sensors in entries:
            machine_id = conn.scalar(
                select(Machine.id).where(Machine.source == machine["source"], Machine.name == machine["name"])
            )
            if machine_id is None:
                machine_id = conn.scalar(insert(Machine).values(**machine).returning(Machine.id))
            else:
                conn.execute(update(Machine).where(Machine.id == machine_id).values(**machine))
            for row in sensors:
                # is_not_distinct_from also matches the empty FEMTO placement.
                sensor_id = conn.scalar(
                    select(Sensor.id).where(
                        Sensor.machine_id == machine_id, Sensor.placement.is_not_distinct_from(row["placement"])
                    )
                )
                if sensor_id is None:
                    conn.execute(insert(Sensor).values(machine_id=machine_id, **row))
                else:
                    conn.execute(update(Sensor).where(Sensor.id == sensor_id).values(**row))


# Sort key, so stored and expected machines are compared in the same order.
def _key(entry: Entry) -> tuple[str, str]:
    return entry[0]["source"], entry[0]["name"]


# Everything in the registry, in the same shape as the rows built above.
def read(engine: Engine) -> list[Entry]:
    machine_cols = ["source", "name", "kind", "rpm", "load_n", "fixed_speed"]
    sensor_cols = ["placement", "channels", "unit", "bearing", *FAULT_COLUMNS.values()]
    with engine.connect() as conn:
        machines = conn.execute(select(Machine).order_by(Machine.id)).mappings().all()
        sensors = conn.execute(select(Sensor).order_by(Sensor.id)).mappings().all()
    entries = []
    for m in machines:
        rows = [{c: s[c] for c in sensor_cols} for s in sensors if s["machine_id"] == m["id"]]
        entries.append(({c: m[c] for c in machine_cols}, rows))
    return entries


# Builds the rows, writes them and checks the database against them.
def main() -> None:
    entries = {"femto": femto_entries(), "ims": ims_entries(), "sca": sca_entries()}
    for source, group in entries.items():
        print(f"{source}: {len(group)} machines, {sum(len(s) for _, s in group)} sensors")

    engine = make_engine()
    expected = [e for group in entries.values() for e in group]
    write(engine, expected)
    stored = read(engine)
    matches = len(stored) == len(expected) and all(
        same_entry(a, b) for a, b in zip(sorted(stored, key=_key), sorted(expected, key=_key))
    )
    print("database matches the datasets:", matches)
    print(f"rows in database: {len(stored)} machines, {sum(len(s) for _, s in stored)} sensors")
    with engine.connect() as conn:
        top = conn.scalar(select(func.max(Machine.id))), conn.scalar(select(func.max(Sensor.id)))
    print(f"highest ids: machine {top[0]}, sensor {top[1]}")


if __name__ == "__main__":
    main()