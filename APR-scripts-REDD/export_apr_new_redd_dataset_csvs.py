"""
Export REDD Building 1 train/validation/test splits to CSV.

Reads the same date ranges used by every script in this folder (see
test_lnn_redd_dataset.py for the full rationale) directly from the raw
NILMTK HDF5 file (APR-new-REDD-dataset/redd.h5) and writes them out as CSVs
in APR-new-REDD-dataset/, mirroring the UKDALE_HF_*.csv convention used by
APR-new-House2-dataset/ (timestamp index column, 'aggregate' for the mains
column).

test is written as ONE CSV (REDD_test.csv) built from TWO non-adjacent
windows (2011-05-11->05-13 and 2011-05-23->05-25) concatenated by row, real
timestamps preserved -- there's a genuine ~10-day gap in the timestamp
column between them. Scripts reading this CSV back must detect that gap
and window each contiguous side independently, never slide a window across
it (see _split_into_blocks / create_sequences_concat in
test_lnn_redd_dataset.py).

Splits:
    train        : 2011-04-18 -> 2011-04-28  (10 days)
    validation   : 2011-04-30 -> 2011-05-03  ( 3 days)
    test         : 2011-05-11 -> 2011-05-13  ( 2 days)  +
                   2011-05-23 -> 2011-05-25  ( 2 days)  = 4 days total, 1 file

Output columns: timestamp, aggregate, dishwasher, fridge, microwave, washing_machine
"""

import os
import numpy as np
import pandas as pd

DATASET_DIR   = os.path.join(os.path.dirname(__file__), '..', 'APR-new-REDD-dataset')
H5_FILENAME   = 'redd.h5'
BUILDING      = 1
RESAMPLE_FREQ = '3s'
TIMEZONE      = 'US/Eastern'

APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

MAINS_METERS     = [1, 2]
APPLIANCE_METERS = {
    'dishwasher':      [6],
    'fridge':          [5],
    'microwave':       [11],
    'washing_machine': [10, 20],
}

SPLIT_RANGES = {
    'train':      ('2011-04-18', '2011-04-28'),
    'validation': ('2011-04-30', '2011-05-03'),
}
TEST_RANGES = [
    ('2011-05-11', '2011-05-13'),
    ('2011-05-23', '2011-05-25'),
]


def _read_meter(h5_path, building, meter):
    df = pd.read_hdf(h5_path, key=f'/building{building}/elec/meter{meter}/table')
    ts = pd.to_datetime(df['index'], unit='ns', utc=True).dt.tz_convert(TIMEZONE)
    return pd.Series(df['values_block_0'].values.astype(np.float32), index=pd.DatetimeIndex(ts))


def _load_channel(h5_path, building, meters, target_index):
    """Resample one or more meters onto target_index and sum them (handles
    REDD's split-phase mains and washer_dryer's motor + heating-element
    sub-meters)."""
    total = pd.Series(0.0, index=target_index)
    for m in meters:
        s = _read_meter(h5_path, building, m)
        r = s.resample(RESAMPLE_FREQ, origin=target_index[0]).mean().reindex(target_index)
        total = total.add(r.fillna(0), fill_value=0)
    return total


def build_df(h5_path, building, start, end):
    target_index = pd.date_range(
        start=pd.Timestamp(start, tz=TIMEZONE), end=pd.Timestamp(end, tz=TIMEZONE),
        freq=RESAMPLE_FREQ, inclusive='left')
    df = pd.DataFrame(index=target_index)
    df.index.name = 'timestamp'
    df['aggregate'] = _load_channel(h5_path, building, MAINS_METERS, target_index)
    for app, meters in APPLIANCE_METERS.items():
        df[app] = _load_channel(h5_path, building, meters, target_index)
    return df


def main():
    h5_path = os.path.join(DATASET_DIR, H5_FILENAME)
    if not os.path.exists(h5_path):
        raise SystemExit(f"Error: {h5_path} not found.")

    print(f"Reading REDD h5 data from '{h5_path}' (building {BUILDING}) ...")

    for name, (start, end) in SPLIT_RANGES.items():
        df = build_df(h5_path, BUILDING, start, end)
        out_path = os.path.join(DATASET_DIR, f'REDD_{name}.csv')
        df.to_csv(out_path)
        n_days = len(df) * 3 / 86400
        print(f"  {name:<12} {len(df):>7,} rows  (~{n_days:.2f} days)  "
              f"{df.index.min()} -> {df.index.max()}  -> {out_path}")

    test_blocks = [build_df(h5_path, BUILDING, s, e) for s, e in TEST_RANGES]
    test_df = pd.concat(test_blocks)
    out_path = os.path.join(DATASET_DIR, 'REDD_test.csv')
    test_df.to_csv(out_path)
    n_days = sum(len(b) for b in test_blocks) * 3 / 86400
    print(f"  {'test':<12} {len(test_df):>7,} rows  (~{n_days:.2f} days, "
          f"{len(test_blocks)} non-adjacent blocks)  "
          f"{test_df.index.min()} -> {test_df.index.max()}  -> {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
