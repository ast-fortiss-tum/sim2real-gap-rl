#!/usr/bin/env python3
"""
Analyse every run in the chosen dataset folder and print two tables:
 • one row per run
 • one row per sub-folder (group average / max)

Fixes vs. original:
 1. Turn angle comes from heading ∠Δpose, NOT centre-orbit angle.
 2. Heading unwrap state survives pauses.
 3. Optional 1080° cap moved to constant CAP_DEG.
"""

import os
import json
import numpy as np
import argparse
import ast
import pandas as pd


# --------------------------- constants & helpers ----------------------------

SCALE      = 25        # multiply every coordinate by this factor
CAP_DEG    = 1080      # stop SurvivedDeg at 3 laps; set to None to disable
EPS_SPEED  = 1e-6      # avoid /0 when dt tiny


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scale_pts(points, factor=SCALE):
    return [[p[0] * factor, p[1] * factor] for p in points]


def compute_centerline_halfwidths(lanes):
    """Return (centrePts, halfWidths).  Lanes 0&1 are boundaries."""
    if lanes is None or len(lanes) < 2:
        return None, None
    ln0, ln1 = lanes[0], lanes[1]
    n = min(len(ln0), len(ln1))
    centre, half = [], []
    for i in range(n):
        cx = (ln0[i][0] + ln1[i][0]) / 2.0
        cy = (ln0[i][1] + ln1[i][1]) / 2.0
        centre.append((cx, cy))
        half.append(np.hypot(ln0[i][0] - cx, ln0[i][1] - cy))
    return centre, half


def computed_cte(x, y, centre_pts, half_widths):
    """CTE scaled so half-width ⇒ 1.5."""
    if centre_pts is None:
        return 0.0
    dists = [np.hypot(x - cx, y - cy) for cx, cy in centre_pts]
    i     = int(np.argmin(dists))
    hw    = half_widths[i]
    return (dists[i] / hw) * 1.5 if hw > 0 else 0.0


def build_going_intervals(flags, times):
    """Merge consecutive True flags into (start,end] intervals."""
    intervals, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            start = times[i]
            j = i + 1
            while j < n and flags[j]:
                j += 1
            end = times[j] if j < n else times[-1]
            intervals.append((start, end))
            i = j
        else:
            i += 1
    return intervals


def is_going(t, intervals):
    return any(s <= t <= e for s, e in intervals)


# ------------------------------ core routine --------------------------------

def analyse_run(poses_js, moving_js, surv_thresh, safe_threshs):
    # ---- load & scale ------------------------------------------------------
    poses     = poses_js["pose"]
    pose_t    = poses_js["time"]
    lanes     = poses_js["lanes"]

    if isinstance(poses, str):
        poses = ast.literal_eval(poses)
    if isinstance(pose_t, str):
        pose_t = ast.literal_eval(pose_t)
    poses = scale_pts(poses)
    lanes = [scale_pts(l) for l in lanes]

    flags     = moving_js["going"]
    moving_t  = moving_js["time"]
    if isinstance(flags, str):
        flags = ast.literal_eval(flags)
    if isinstance(moving_t, str):
        moving_t = ast.literal_eval(moving_t)

    intervals = build_going_intervals(flags, moving_t)

    # ---- pre-compute lane geometry & track centre -------------------------
    centre_pts, half_w = compute_centerline_halfwidths(lanes)
    cx = np.mean([p[0] for p in lanes[0] + lanes[1]])
    cy = np.mean([p[1] for p in lanes[0] + lanes[1]])

    # ---- accumulators ------------------------------------------------------
    survived_deg   = None        # stop when CTE > surv_thresh (or CAP_DEG)
    safe_turn      = {st: 0.0 for st in safe_threshs}
    cum_turn       = 0.0
    total_turn     = 0.0
    max_cte        = 0.0
    total_speed    = 0.0
    speed_cnt      = 0

    prev_good_idx  = None        # last "going" pose index
    prev_heading   = None

    # ---- iterate over poses -----------------------------------------------
    for i, (pose, t) in enumerate(zip(poses, pose_t)):
        if not is_going(t, intervals):
            continue  # ignore non-going poses entirely

        x, y = pose
        cte  = computed_cte(x, y, centre_pts, half_w)
        max_cte = max(max_cte, cte)

        # heading: direction of motion from previous going pose
        if prev_good_idx is not None:
            dx = x - poses[prev_good_idx][0]
            dy = y - poses[prev_good_idx][1]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:   # no movement
                heading = prev_heading
            else:
                heading = np.arctan2(dy, dx)
        else:
            prev_good_idx = i
            prev_heading  = np.arctan2(
                poses[i+1][1] - y if i+1 < len(poses) else 0,
                poses[i+1][0] - x if i+1 < len(poses) else 1
            )
            continue

        # delta heading (unwrap to keep continuity)
        d = np.degrees(np.unwrap([prev_heading, heading])[1] - prev_heading)
        prev_heading = heading
        prev_good_idx = i

        # accumulate total turn over whole run
        total_turn += abs(d)

        # accumulate survival / safe only until threshold crossed
        if survived_deg is None:
            cum_turn += abs(d)
            for st in safe_threshs:
                if cte <= st:
                    safe_turn[st] += abs(d)
            if (cte > surv_thresh) or (CAP_DEG and cum_turn >= CAP_DEG):
                survived_deg = cum_turn

        # speed
        dt = t - pose_t[prev_good_idx]
        if dt > EPS_SPEED:
            dist = np.hypot(dx, dy)
            total_speed += dist / dt
            speed_cnt  += 1

    survived_deg = survived_deg or cum_turn  # never crossed threshold
    mean_speed   = total_speed / speed_cnt if speed_cnt else 0.0
    safe_pct     = {st: (safe_turn[st]/survived_deg)*100 if survived_deg else 0
                    for st in safe_threshs}

    return survived_deg, safe_pct, mean_speed, max_cte, total_turn


# ------------------------------ CLI wrapper ---------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Analyse runs, compute SurvivedDeg (heading-based), "
                    "safe-angle percentages and mean speed.")
    p.add_argument('--folder',  default='/home/cubos98/catkin_ws/src/Vehicle/results_reality',
                   help="Root path that contains dataset folders")
    p.add_argument('--dataset', default='dataset_final',
                   help="Sub-folder with the runs")
    p.add_argument('--survival_threshold', type=float, default=4.0,
                   help="CTE limit that ends the survival counter")
    p.add_argument('--safe_thresholds',   default='1.5,2.0,3.0',
                   help="Comma-separated CTE thresholds for safe %")
    p.add_argument('--ideal_angle', type=float, default=360.0)
    args = p.parse_args()

    safe_threshs = [float(x) for x in args.safe_thresholds.split(',') if x.strip()]
    ds_path      = os.path.join(args.folder, args.dataset)

    # ------------- discover runs -------------------------------------------
    runs = {}
    for root, _, files in os.walk(ds_path):
        for f in files:
            if f.endswith('_poses.json'):
                base  = f[:-11]
                pfile = os.path.join(root, f)
                mfile = os.path.join(root, f'{base}_moving.json')
                if os.path.exists(mfile):
                    grp = os.path.relpath(root, ds_path) or 'default'
                    runs[os.path.join(grp, base)] = dict(group=grp,
                                                          poses=pfile,
                                                          moving=mfile)
    if not runs:
        print("No runs found.")
        return

    # ------------- process runs --------------------------------------------
    results = []
    for key, files in runs.items():
        try:
            poses_j  = load_json(files['poses'])
            moving_j = load_json(files['moving'])
            surv_deg, safe_pct, spd, maxcte, tot_deg = analyse_run(
                poses_j, moving_j, args.survival_threshold, safe_threshs)

            if tot_deg < 1080:
                continue

            avg_safe = sum(safe_pct.values()) / len(safe_pct)
            trunc    = min(surv_deg, 1080 if CAP_DEG else surv_deg)
            score    = (trunc / args.ideal_angle) * (avg_safe / 100) / 3 * 100

            row = dict(Run=key, Group=files['group'],
                       MeanSpeed=spd, MaxCTE=maxcte,
                       SurvivedDeg=surv_deg, TotalTurn=tot_deg,
                       ScorePerc=score)
            row.update({f'SafePerc_{st}': safe_pct[st] for st in safe_threshs})
            results.append(row)
        except Exception as e:
            print(f"⚠ {key}: {e}")

    if not results:
        print("All runs filtered out (<1080°).")
        return

    df = pd.DataFrame(results).sort_values('Run')
    grp = (df.groupby('Group')
             .agg(MeanSpeed=('MeanSpeed', 'mean'),
                  MaxCTE=('MaxCTE', 'max'),
                  SurvivedDeg=('SurvivedDeg', 'mean'),
                  TotalTurn=('TotalTurn', 'mean'),
                  ScorePerc=('ScorePerc', 'mean'),
                  MinTotalTurn=('TotalTurn', 'min')))
    for st in safe_threshs:
        grp[f'SafePerc_{st}'] = df.groupby('Group')[f'SafePerc_{st}'].mean()
    grp = grp.reset_index()

    print("\nIndividual Performance Table:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nGrouped Performance Table:")
    print(grp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == '__main__':
    main()
