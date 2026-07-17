"""
On-duration and time-of-day distribution analysis for REDD, read directly
from the raw NILMTK HDF5 file (APR-new-REDD-dataset/redd.h5), broken down
per split (train / validation / test).

Companion to APR-scripts-UK-dale/analyze_apr_new_house2_distributions.py --
same purpose (ON-duration + time-of-day priors per appliance, per split),
adapted to REDD's data format and 3-second sampling interval.

NILM appliance models (HMM-based and neural) rely on two behavioural priors
per appliance:
    - ON-duration distribution -- how long a device stays on once it switches
      on (e.g. a microwave run is a couple of minutes; a dishwasher/washer-
      dryer cycle can run over an hour; a fridge compressor cycle is short
      and fairly regular).
    - Time-of-day distribution -- when during the day the device tends to be
      used/active.

Dataset: Building 1 of APR-new-REDD-dataset/redd.h5 (has all four target
appliances). Building 1's daily-gap profile has no long clean contiguous
stretch anywhere close to UKDALE's -- REDD is a much shorter, gappier
recording -- so splits here are three separate clean (low-gap) windows,
chronologically ordered and non-overlapping:
    train      : 2011-04-18 -> 2011-04-28  (10 days)
    validation : 2011-04-30 -> 2011-05-03  ( 3 days)
    test       : 2011-05-23 -> 2011-05-25  ( 2 days)
Mains is meter1 + meter2 (REDD's split-phase whole-house power);
washer_dryer is meter10 + meter20 (motor + heating-element sub-meters).

Column mapping (REDD appliance type -> canonical name used here):
    fridge       -> fridge        (meter 5)
    dish washer  -> dishwasher    (meter 6)
    washer dryer -> washing_machine (meters 10 + 20)
    microwave    -> microwave     (meter 11)

Threshold choice: unlike UKDALE House 2 (which needed 20 W for fridge and
30 W for microwave to avoid misreading standby/idle floors as ON), sweeping
5/10/15/20/30 W across all four REDD appliances and all three splits here
shows the ON-time fraction is already stable from 10 W upward for every
appliance/split. A single uniform THRESHOLD_W = 10.0 is therefore used for
all appliances and all splits, consistent with the existing REDD scripts in
this repo (e.g. test_advanced_lnn_v5_redd.py's THRESHOLD = 10.0).

Outputs (written into APR-scripts-REDD/):
    - on_time_and_time_of_day_train.png
    - on_time_and_time_of_day_validation.png
    - on_time_and_time_of_day_test.png
    - on_time_and_time_of_day_comparison.png
    - distribution_summary_by_split.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_DIR   = os.path.join(os.path.dirname(__file__), '..', 'APR-new-REDD-dataset')
H5_FILENAME   = 'redd.h5'
BUILDING      = 1
RESAMPLE_FREQ = '3s'
TIMEZONE      = 'US/Eastern'

APPLIANCES   = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD_W  = 10.0
THRESHOLDS   = {app: THRESHOLD_W for app in APPLIANCES}
SPLITS       = ['train', 'validation', 'test']
SPLIT_COLORS = {'train': 'steelblue', 'validation': 'darkorange', 'test': 'seagreen'}
STEP_SECONDS = 3.0

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
    'test':       ('2011-05-23', '2011-05-25'),
}


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


def load_splits(dataset_dir=DATASET_DIR, building=BUILDING):
    """Load train/validation/test as separate DataFrames (see module docstring)."""
    h5_path = os.path.join(dataset_dir, H5_FILENAME)
    print(f"Loading REDD h5 data from '{h5_path}' (building {building}) ...")

    dfs = {}
    for name, (start, end) in SPLIT_RANGES.items():
        target_index = pd.date_range(
            start=pd.Timestamp(start, tz=TIMEZONE), end=pd.Timestamp(end, tz=TIMEZONE),
            freq=RESAMPLE_FREQ, inclusive='left')
        df = pd.DataFrame(index=target_index)
        df['main'] = _load_channel(h5_path, building, MAINS_METERS, target_index)
        for app, meters in APPLIANCE_METERS.items():
            df[app] = _load_channel(h5_path, building, meters, target_index)
        dfs[name] = df
        n_days = len(df) * STEP_SECONDS / 86400
        print(f"{name:<12} {len(df):,} rows  (~{n_days:.2f} days)")
    return dfs


def extract_on_segments(series, threshold):
    """
    Vectorized run-length encoding of contiguous ON (> threshold) segments.

    Returns:
        durations_min:  array of segment durations in minutes
        start_hours:    array of hour-of-day (0-23, fractional) at segment start
    """
    state  = (series.values > threshold).astype(np.int8)
    padded = np.concatenate(([0], state, [0]))
    diffs  = np.diff(padded)
    starts = np.where(diffs ==  1)[0]
    ends   = np.where(diffs == -1)[0]

    durations_min = (ends - starts) * STEP_SECONDS / 60.0

    if isinstance(series.index, pd.DatetimeIndex):
        start_timestamps = series.index[starts]
        start_hours = start_timestamps.hour + start_timestamps.minute / 60.0
    else:
        start_hours = np.full(len(starts), np.nan)

    return durations_min, np.asarray(start_hours)


def on_time_fraction_by_hour(series, threshold):
    """ON-time fraction per hour-of-day bucket (0-23)."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(0.0, index=range(24))
    state        = series > threshold
    by_hour      = state.groupby(series.index.hour)
    on_counts    = by_hour.sum()
    total_counts = by_hour.count()
    frac = (on_counts / total_counts).reindex(range(24), fill_value=0.0)
    return frac


def summarize(durations_min, span_hours):
    if len(durations_min) == 0:
        return {'count': 0, 'total_on_hours': 0.0, 'pct_of_span': 0.0}
    total_on_hours = float(np.sum(durations_min) / 60.0)
    return {
        'count':          int(len(durations_min)),
        'mean_min':       float(np.mean(durations_min)),
        'median_min':     float(np.median(durations_min)),
        'p25_min':        float(np.percentile(durations_min, 25)),
        'p75_min':        float(np.percentile(durations_min, 75)),
        'p95_min':        float(np.percentile(durations_min, 95)),
        'max_min':        float(np.max(durations_min)),
        'total_on_hours': total_on_hours,
        'pct_of_span':    float(total_on_hours / span_hours * 100),
    }


def analyze_split(df, split_name):
    span_hours = len(df) * STEP_SECONDS / 3600
    results, summary = {}, {}

    for app in APPLIANCES:
        thr = THRESHOLDS[app]
        durations_min, start_hours = extract_on_segments(df[app], thr)
        hour_frac = on_time_fraction_by_hour(df[app], thr)
        results[app] = {'durations_min': durations_min, 'hour_frac': hour_frac}
        summary[app] = summarize(durations_min, span_hours)

    print(f"\n{'='*78}\n{split_name.upper()} -- ON-DURATION SUMMARY\n{'='*78}")
    for app in APPLIANCES:
        s = summary[app]
        thr = THRESHOLDS[app]
        print(f"\n{app}  (threshold={thr:.0f} W):")
        if s['count'] == 0:
            print("  No ON events found.")
            continue
        print(f"  events         : {s['count']:,}")
        print(f"  mean duration  : {s['mean_min']:7.2f} min")
        print(f"  median duration: {s['median_min']:7.2f} min")
        print(f"  p25 / p75      : {s['p25_min']:7.2f} / {s['p75_min']:7.2f} min")
        print(f"  total ON time  : {s['total_on_hours']:7.1f} hours  "
              f"({s['pct_of_span']:.2f}% of split span)")

    print(f"\n{'='*78}\n{split_name.upper()} -- TIME-OF-DAY SUMMARY\n{'='*78}")
    for app in APPLIANCES:
        hour_frac = results[app]['hour_frac']
        if hour_frac.max() == 0:
            print(f"\n{app}: no ON time in this split.")
            continue
        peak_hour = int(hour_frac.idxmax())
        print(f"\n{app}: peak hour = {peak_hour:02d}:00-{peak_hour+1:02d}:00  "
              f"(ON {hour_frac.max()*100:.1f}% of the time in that hour)")

    return results, summary


def plot_split(results, split_name, out_dir=None):
    out_dir = out_dir or os.path.dirname(__file__)
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(14, 4 * len(APPLIANCES)))
    fig.suptitle(f'REDD -- ON-Duration & Time-of-Day ({split_name} split only)', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        thr           = THRESHOLDS[app]
        durations_min = results[app]['durations_min']
        hour_frac     = results[app]['hour_frac']

        ax_dur = axes[row][0]
        if len(durations_min) > 0:
            cap = np.percentile(durations_min, 99) if len(durations_min) > 1 else durations_min[0]
            cap = max(cap, 1.0)
            ax_dur.hist(np.clip(durations_min, 0, cap), bins=30,
                        color=SPLIT_COLORS[split_name], edgecolor='none')
            ax_dur.axvline(np.median(durations_min), color='red', linestyle='--',
                           linewidth=1, label=f"median={np.median(durations_min):.1f} min")
            ax_dur.legend(fontsize=8)
        else:
            ax_dur.text(0.5, 0.5, 'no ON events', ha='center', va='center',
                        transform=ax_dur.transAxes, color='gray')
        ax_dur.set_title(f'{app}  (>{thr:.0f} W) -- ON-duration distribution')
        ax_dur.set_xlabel('Duration (min, capped at p99)')
        ax_dur.set_ylabel('Event count')
        ax_dur.grid(True, alpha=0.3)

        ax_tod = axes[row][1]
        ax_tod.bar(range(24), hour_frac.values * 100,
                   color=SPLIT_COLORS[split_name], width=0.85)
        ax_tod.set_title(f'{app}  (>{thr:.0f} W) -- Time-of-day (% ON per hour)')
        ax_tod.set_xlabel('Hour of day')
        ax_tod.set_ylabel('% time ON')
        ax_tod.set_xticks(range(0, 24, 2))
        ax_tod.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'on_time_and_time_of_day_{split_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved -> {out_path}")


def plot_comparison(all_results, out_dir=None):
    """Overlay train/validation/test per appliance -- duration boxplot + time-of-day lines."""
    out_dir = out_dir or os.path.dirname(__file__)
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(14, 4 * len(APPLIANCES)))
    fig.suptitle('REDD -- Train vs Validation vs Test comparison', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        thr    = THRESHOLDS[app]
        ax_dur = axes[row][0]
        box_data, box_labels, box_colors = [], [], []
        for split in SPLITS:
            durations_min = all_results[split][app]['durations_min']
            if len(durations_min) > 0:
                cap = np.percentile(durations_min, 99) if len(durations_min) > 1 else durations_min[0]
                box_data.append(np.clip(durations_min, 0, max(cap, 1.0)))
            else:
                box_data.append(np.array([]))
            box_labels.append(f"{split}\n(n={len(durations_min)})")
            box_colors.append(SPLIT_COLORS[split])

        bp = ax_dur.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_dur.set_title(f'{app}  (>{thr:.0f} W) -- ON-duration by split')
        ax_dur.set_ylabel('Duration (min, capped at p99)')
        ax_dur.grid(True, alpha=0.3)

        ax_tod = axes[row][1]
        for split in SPLITS:
            hour_frac = all_results[split][app]['hour_frac']
            ax_tod.plot(range(24), hour_frac.values * 100, marker='o', markersize=3,
                        color=SPLIT_COLORS[split], label=split, linewidth=1.5)
        ax_tod.set_title(f'{app}  (>{thr:.0f} W) -- Time-of-day by split')
        ax_tod.set_xlabel('Hour of day')
        ax_tod.set_ylabel('% time ON')
        ax_tod.set_xticks(range(0, 24, 2))
        ax_tod.legend(fontsize=8)
        ax_tod.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'on_time_and_time_of_day_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved -> {out_path}")


def main():
    out_dir = os.path.dirname(__file__)
    dfs = load_splits()

    all_results, all_summaries = {}, {}
    for split in SPLITS:
        results, summary = analyze_split(dfs[split], split)
        all_results[split]   = results
        all_summaries[split] = summary
        plot_split(results, split, out_dir)

    plot_comparison(all_results, out_dir)

    json_summary = {
        split: {
            app: {
                **all_summaries[split][app],
                'threshold_w': THRESHOLDS[app],
                'hour_frac': {
                    str(h): float(all_results[split][app]['hour_frac'][h])
                    for h in range(24)
                },
            }
            for app in APPLIANCES
        }
        for split in SPLITS
    }
    json_path = os.path.join(out_dir, 'distribution_summary_by_split.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2)
    print(f"\nSummary saved -> {json_path}")

    print(f"\n{'='*78}\nCROSS-SPLIT COVERAGE CHECK\n{'='*78}")
    for app in APPLIANCES:
        counts = {split: all_summaries[split][app]['count'] for split in SPLITS}
        flag = "  <-- zero events in at least one split!" if 0 in counts.values() else ""
        print(f"  {app:<18} train={counts['train']:4d}  "
              f"validation={counts['validation']:4d}  test={counts['test']:4d}{flag}")


if __name__ == "__main__":
    main()
