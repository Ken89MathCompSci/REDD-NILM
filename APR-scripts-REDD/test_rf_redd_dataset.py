"""
Random Forest baseline for NILM -- REDD CSV splits.

Companion to APR-scripts-UK-dale/test_rf_apr_new_house2_dataset.py, adapted
to REDD's exported CSVs (APR-new-REDD-dataset/REDD_{train,validation,test}.csv
-- see export_apr_new_redd_dataset_csvs.py) instead of the raw redd.h5 file.
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

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-REDD-dataset')
TRAIN_CSV   = 'REDD_train.csv'
VAL_CSV     = 'REDD_validation.csv'
TEST_CSV    = 'REDD_test.csv'
TEST_GAP_TOLERANCE = '10s'   # any bigger gap than this in REDD_test.csv marks a block boundary

APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD_W = 10.0

WIN    = 100
STRIDE = 5

N_ESTIMATORS     = 300
MAX_DEPTH        = 20
MIN_SAMPLES_LEAF = 5
RANDOM_STATE     = 42

FEATURE_NAMES = ['mean', 'std', 'min', 'max', 'range', 'median',
                  'first', 'last', 'mean_abs_diff', 'max_abs_diff']


def _read_csv(path):
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    return df.rename(columns={'aggregate': 'main'})


def _split_into_blocks(df, gap_tolerance=TEST_GAP_TOLERANCE):
    """Split a DataFrame into contiguous blocks wherever the timestamp index
    jumps by more than gap_tolerance -- see test_lnn_redd_dataset.py."""
    deltas = df.index.to_series().diff()
    gap_td = pd.Timedelta(gap_tolerance)
    split_points = np.where(deltas > gap_td)[0]
    if len(split_points) == 0:
        return [df]
    blocks, start = [], 0
    for sp in split_points:
        blocks.append(df.iloc[start:sp])
        start = sp
    blocks.append(df.iloc[start:])
    return blocks


def load_data(dataset_dir=DATASET_DIR):
    """Load train / val / test from APR-new-REDD-dataset/REDD_*.csv.

    train/val are single DataFrames; test is a LIST of DataFrames (two
    separate clean windows recovered from REDD_test.csv -- see
    _split_into_blocks) that must be windowed independently and concatenated
    only after windowing (build_window_features_concat), never joined as raw
    timestamps.
    """
    print(f"Loading REDD CSV data from '{dataset_dir}' ...")

    train_df = _read_csv(os.path.join(dataset_dir, TRAIN_CSV))
    val_df   = _read_csv(os.path.join(dataset_dir, VAL_CSV))
    test_dfs = _split_into_blocks(_read_csv(os.path.join(dataset_dir, TEST_CSV)))

    print(f"  train : {len(train_df):>7,} rows  {train_df.index.min()} -> {train_df.index.max()}")
    print(f"  val   : {len(val_df):>7,} rows  {val_df.index.min()} -> {val_df.index.max()}")
    for i, df in enumerate(test_dfs):
        print(f"  test[{i}]: {len(df):>7,} rows  {df.index.min()} -> {df.index.max()}")

    return {'train': train_df, 'val': val_df, 'test': test_dfs}


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


def build_window_features_concat(dfs, appliance, win: int = WIN, stride: int = STRIDE):
    """Build windowed features on each (non-adjacent) DataFrame independently,
    then concatenate -- avoids fabricating a window that straddles the gap
    between two separate test blocks."""
    feats, ys = [], []
    for df in dfs:
        feat, y = build_window_features(df['main'].values, df[appliance].values, win, stride)
        feats.append(feat)
        ys.append(y)
    return np.concatenate(feats, axis=0), np.concatenate(ys, axis=0)


def train_rf_on_appliance(data_dict, appliance_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"\nBuilding windowed features for {appliance_name}...")
    X_tr, y_tr = build_window_features(train_data['main'].values, train_data[appliance_name].values)
    X_va, y_va = build_window_features(val_data['main'].values,   val_data[appliance_name].values)
    X_te, y_te = build_window_features_concat(test_data, appliance_name)

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
        'dataset': 'REDD (APR-new-REDD-dataset/REDD_*.csv, Building 1)',
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
            'testing':    'Building 1, 2011-05-11->05-13 + 2011-05-23->05-25 (4 days, 2 blocks)',
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
    for fname in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            sys.exit(1)

    main()
