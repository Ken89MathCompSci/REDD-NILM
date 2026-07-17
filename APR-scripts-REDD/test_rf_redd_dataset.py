"""
Random Forest baseline for NILM -- REDD h5 splits.

Companion to APR-scripts-UK-dale/test_rf_apr_new_house2_dataset.py, adapted
to REDD's raw NILMTK HDF5 file (APR-new-REDD-dataset/redd.h5) instead of
APR-new-House2-dataset/ CSVs or the old data/redd/*.pkl slices.
See test_lnn_redd_dataset.py for the full data-source / column-mapping /
threshold notes shared by every script in this folder.

Same design as the UK-dale companion script: one RandomForestRegressor per
appliance predicting continuous power (W) at the window midpoint, using 10
windowed statistics of the aggregate signal (mean, std, min, max, range,
median, first, last, mean|diff|, max|diff|) over the same WIN=100/STRIDE=5
midpoint-targeted windows used by the neural baselines in this folder. No
feature/target scaling (tree splits are scale-invariant) and no
sample_weight/class_weight imbalance correction -- the UK-dale companion
script found upweighting rare-appliance ON samples backfires (inflates a
small positive bias across OFF windows that clears the threshold), so this
stays unweighted here too.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Source Code'))
from utils import calculate_nilm_metrics

DATASET_DIR   = os.path.join(os.path.dirname(__file__), '..', 'APR-new-REDD-dataset')
H5_FILENAME   = 'redd.h5'
BUILDING      = 1
RESAMPLE_FREQ = '3s'
TIMEZONE      = 'US/Eastern'

APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD_W = 10.0

MAINS_METERS     = [1, 2]
APPLIANCE_METERS = {
    'dishwasher':      [6],
    'fridge':          [5],
    'microwave':       [11],
    'washing_machine': [10, 20],
}
SPLIT_RANGES = {
    'train': ('2011-04-18', '2011-04-28'),
    'val':   ('2011-04-30', '2011-05-03'),
    'test':  ('2011-05-23', '2011-05-25'),
}

WIN    = 100
STRIDE = 5

N_ESTIMATORS     = 300
MAX_DEPTH        = 20
MIN_SAMPLES_LEAF = 5
RANDOM_STATE     = 42

FEATURE_NAMES = ['mean', 'std', 'min', 'max', 'range', 'median',
                  'first', 'last', 'mean_abs_diff', 'max_abs_diff']


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


def load_data(dataset_dir=DATASET_DIR, building=BUILDING):
    """Load train / val / test from APR-new-REDD-dataset/redd.h5."""
    h5_path = os.path.join(dataset_dir, H5_FILENAME)
    print(f"Loading REDD h5 data from '{h5_path}' (building {building}) ...")

    splits = {}
    for name, (start, end) in SPLIT_RANGES.items():
        target_index = pd.date_range(
            start=pd.Timestamp(start, tz=TIMEZONE), end=pd.Timestamp(end, tz=TIMEZONE),
            freq=RESAMPLE_FREQ, inclusive='left')
        df = pd.DataFrame(index=target_index)
        df['main'] = _load_channel(h5_path, building, MAINS_METERS, target_index)
        for app, meters in APPLIANCE_METERS.items():
            df[app] = _load_channel(h5_path, building, meters, target_index)
        splits[name] = df
        print(f"  {name:6s}: {len(df):>7,} rows  {df.index.min()} -> {df.index.max()}")

    return {'train': splits['train'], 'val': splits['val'], 'test': splits['test']}


def build_window_features(mains: np.ndarray, appliance_vals: np.ndarray,
                          win: int = WIN, stride: int = STRIDE):
    """
    Windowed statistical features from the aggregate signal, midpoint-
    targeted (same alignment convention as create_sequences() elsewhere in
    this folder): X[i] summarizes mains[i*stride : i*stride+win], and
    y[i] is the appliance's continuous power (W) at that window's midpoint.
    """
    all_windows    = sliding_window_view(mains, win)          # (len(mains)-win+1, win)
    start_indices  = np.arange(0, len(mains) - win, stride)
    windows        = all_windows[start_indices]                # (n_windows, win)
    mid_indices    = start_indices + win // 2

    diffs = np.diff(windows, axis=1)
    feat = np.stack([
        windows.mean(axis=1),
        windows.std(axis=1),
        windows.min(axis=1),
        windows.max(axis=1),
        windows.max(axis=1) - windows.min(axis=1),
        np.median(windows, axis=1),
        windows[:, 0],
        windows[:, -1],
        np.abs(diffs).mean(axis=1),
        np.abs(diffs).max(axis=1),
    ], axis=1).astype(np.float32)

    y_power = appliance_vals[mid_indices].astype(np.float32)
    return feat, y_power


def train_rf_on_appliance(data_dict, appliance_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"\nBuilding windowed features for {appliance_name}...")
    X_tr, y_tr = build_window_features(train_data['main'].values, train_data[appliance_name].values)
    X_va, y_va = build_window_features(val_data['main'].values,   val_data[appliance_name].values)
    X_te, y_te = build_window_features(test_data['main'].values,  test_data[appliance_name].values)

    on_tr = (y_tr > THRESHOLD_W).mean() * 100
    on_va = (y_va > THRESHOLD_W).mean() * 100
    on_te = (y_te > THRESHOLD_W).mean() * 100
    print(f"  Train: {X_tr.shape}  ON={on_tr:.2f}%")
    print(f"  Val:   {X_va.shape}  ON={on_va:.2f}%")
    print(f"  Test:  {X_te.shape}  ON={on_te:.2f}%")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    print(f"Fitting RandomForestRegressor ({N_ESTIMATORS} trees) for {appliance_name}...")
    model.fit(X_tr, y_tr)

    val_pred  = model.predict(X_va)
    test_pred = model.predict(X_te)
    val_metrics  = calculate_nilm_metrics(y_va, val_pred,  threshold=THRESHOLD_W)
    test_metrics = calculate_nilm_metrics(y_te, test_pred, threshold=THRESHOLD_W)

    print(f"  Val  -- F1={val_metrics['f1']:.4f}  P={val_metrics['precision']:.4f}  "
          f"R={val_metrics['recall']:.4f}  MAE={val_metrics['mae']:.2f}  SAE={val_metrics['sae']:.4f}")
    print(f"  Test -- F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  SAE={test_metrics['sae']:.4f}  "
          f"TP={test_metrics['TP']:,}  TN={test_metrics['TN']:,}  "
          f"FP={test_metrics['FP']:,}  FN={test_metrics['FN']:,}")

    # -- Feature importances plot --
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(FEATURE_NAMES)), importances[order], color='steelblue')
    plt.xticks(range(len(FEATURE_NAMES)), [FEATURE_NAMES[i] for i in order], rotation=45, ha='right')
    plt.title(f'{appliance_name} -- RF feature importances')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rf_redd_{appliance_name}_feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'REDD (APR-new-REDD-dataset/redd.h5, Building 1)',
        'model': 'RandomForestRegressor',
        'threshold_w': THRESHOLD_W,
        'window_size': WIN,
        'stride': STRIDE,
        'model_params': {
            'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH,
            'min_samples_leaf': MIN_SAMPLES_LEAF, 'sample_weight': 'none (unweighted -- see docstring)',
            'random_state': RANDOM_STATE,
        },
        'feature_names': FEATURE_NAMES,
        'feature_importances': {FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))},
        'val_metrics': {k: float(v) for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
    }
    with open(os.path.join(save_dir, f'rf_redd_{appliance_name}_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics


def main():
    data_dict = load_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(
        os.path.dirname(__file__), '..', 'models', f"rf_redd_dataset_{timestamp}")

    all_results = {}
    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"Training RandomForestRegressor on {appliance_name}")
        print(f"{'='*60}")
        appliance_dir = os.path.join(base_save_dir, appliance_name)
        test_metrics = train_rf_on_appliance(data_dict, appliance_name, appliance_dir)
        all_results[appliance_name] = test_metrics

    summary = {
        'timestamp': timestamp,
        'dataset': 'REDD',
        'model': 'RandomForestRegressor',
        'dataset_splits': {
            'training':   'Building 1, 2011-04-18 -> 2011-04-28 (10 days)',
            'validation': 'Building 1, 2011-04-30 -> 2011-05-03 (3 days)',
            'testing':    'Building 1, 2011-05-23 -> 2011-05-25 (2 days)',
        },
        'window_size': WIN, 'stride': STRIDE, 'threshold_w': THRESHOLD_W,
        'model_params': {
            'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH,
            'min_samples_leaf': MIN_SAMPLES_LEAF, 'sample_weight': 'none (unweighted -- see docstring)',
        },
        'results': {app: {k: float(v) for k, v in m.items()} for app, m in all_results.items()},
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nRandom Forest REDD testing completed. Results saved to {base_save_dir}\n")
    print(f"{'Appliance':<18} {'F1':>7} {'Prec':>7} {'Rec':>7} {'MAE':>7} {'SAE':>7} "
          f"{'TP':>7} {'TN':>7} {'FP':>7} {'FN':>7}")
    print("-" * 92)
    for app in APPLIANCES:
        m = all_results[app]
        print(f"{app:<18} {m['f1']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['mae']:>7.2f} {m['sae']:>7.4f} "
              f"{m['TP']:>7,d} {m['TN']:>7,d} {m['FP']:>7,d} {m['FN']:>7,d}")


if __name__ == "__main__":
    h5_path = os.path.join(DATASET_DIR, H5_FILENAME)
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found!")
        sys.exit(1)

    main()
