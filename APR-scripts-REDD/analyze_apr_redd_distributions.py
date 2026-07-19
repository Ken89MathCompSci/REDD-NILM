"""
On-duration and time-of-day distribution analysis for REDD, read from the
exported CSVs (APR-new-REDD-dataset/REDD_{train,validation,test}.csv -- see
export_apr_new_redd_dataset_csvs.py), broken down per split (train /
validation / test).

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

Dataset: Building 1 (has all four target appliances). Building 1's daily-gap
profile has no long clean contiguous stretch anywhere close to UKDALE's --
REDD is a much shorter, gappier recording. train/validation are single
clean windows; REDD_test.csv holds TWO separate clean windows concatenated
by row (real timestamps preserved, so there's a genuine ~10-day gap in the
timestamp column between them) -- a single 2-day test block left dishwasher
with too few ON events to evaluate reliably. Every split is processed as a
list of one or more non-adjacent blocks (test's two blocks are recovered by
detecting that gap -- see _split_into_blocks): ON-duration segments are
extracted per block and the resulting duration arrays concatenated
afterwards (never run-length-encoded across the raw concatenation, which
would fabricate a segment spanning the gap between blocks); time-of-day
fractions are safe to tally directly over the concatenated raw series,
since hour-of-day grouping does not assume temporal adjacency. All windows
are chronologically ordered and non-overlapping:
    train      : 2011-04-18 -> 2011-04-28  (10 days)
    validation : 2011-04-30 -> 2011-05-03  ( 3 days)
    test       : 2011-05-11 -> 2011-05-13  ( 2 days)  +
                 2011-05-23 -> 2011-05-25  ( 2 days)  = 4 days total

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

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-REDD-dataset')
TRAIN_CSV   = 'REDD_train.csv'
VAL_CSV     = 'REDD_validation.csv'
TEST_CSV    = 'REDD_test.csv'
TEST_GAP_TOLERANCE = '10s'   # any bigger gap than this in REDD_test.csv marks a block boundary

APPLIANCES   = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD_W  = 10.0
THRESHOLDS   = {app: THRESHOLD_W for app in APPLIANCES}
SPLITS       = ['train', 'validation', 'test']
SPLIT_COLORS = {'train': 'steelblue', 'validation': 'darkorange', 'test': 'seagreen'}
STEP_SECONDS = 3.0


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


def load_splits(dataset_dir=DATASET_DIR):
    """Load train/validation/test, each as a LIST of one or more non-adjacent
    DataFrame blocks (see module docstring) -- test has two (recovered from
    REDD_test.csv via _split_into_blocks); train/validation have one each,
    wrapped in a single-element list for a uniform interface."""
    print(f"Loading REDD CSV data from '{dataset_dir}' ...")

    dfs = {
        'train':      [_read_csv(os.path.join(dataset_dir, TRAIN_CSV))],
        'validation': [_read_csv(os.path.join(dataset_dir, VAL_CSV))],
        'test':       _split_into_blocks(_read_csv(os.path.join(dataset_dir, TEST_CSV))),
    }
    for name, blocks in dfs.items():
        for i, df in enumerate(blocks):
            n_days = len(df) * STEP_SECONDS / 86400
            label = name if len(blocks) == 1 else f'{name}[{i}]'
            print(f"{label:<12} {len(df):,} rows  (~{n_days:.2f} days)")
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


def extract_on_segments_multi(dfs, appliance, threshold):
    """Run extract_on_segments on each (non-adjacent) block independently and
    concatenate the results -- never run-length-encode across the raw
    concatenation, which would fabricate a segment spanning the gap."""
    durations_list, hours_list = [], []
    for df in dfs:
        d, h = extract_on_segments(df[appliance], threshold)
        durations_list.append(d)
        hours_list.append(h)
    return np.concatenate(durations_list), np.concatenate(hours_list)


def on_time_fraction_by_hour_multi(dfs, appliance, threshold):
    """Hour-of-day tally is safe directly over the concatenated raw series
    (it doesn't assume temporal adjacency between blocks)."""
    combined = pd.concat([df[appliance] for df in dfs])
    return on_time_fraction_by_hour(combined, threshold)


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


def analyze_split(dfs, split_name):
    """dfs is a list of one or more non-adjacent DataFrame blocks."""
    span_hours = sum(len(df) for df in dfs) * STEP_SECONDS / 3600
    results, summary = {}, {}

    for app in APPLIANCES:
        thr = THRESHOLDS[app]
        durations_min, start_hours = extract_on_segments_multi(dfs, app, thr)
        hour_frac = on_time_fraction_by_hour_multi(dfs, app, thr)
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
