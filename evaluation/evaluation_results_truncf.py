#!/usr/bin/env python3
"""Performance table generator

This script processes driving-run folders that contain
    * <run>_poses.json
    * <run>_moving.json

For each run it computes:
    - cumulative turning until computed CTE first exceeds a survivability threshold
    - total turning
    - safe‑percentage metrics for several CTE thresholds
    - mean speed, maximum CTE
    - an overall score

Runs whose **TotalTurn** is < 1080 deg are ignored (invalid runs).

It then prints:
  • an individual‑run table
  • a grouped summary table (one row per sub‑folder) that now also
    includes **ValidRuns** = how many runs in that group survived
    the 1080‑degree filter.
"""

import os
import json
import numpy as np
import argparse
import ast
import pandas as pd

# -----------------------------------------------------------------------------#
#                                helper functions                              #
# -----------------------------------------------------------------------------#

def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def scale_data(data, scale=25):
    """Scale a list of 2D points by the given factor."""
    return [[p[0] * scale, p[1] * scale] for p in data]


def compute_centerline_and_halfwidths(lanes):
    """Return centerline points and half‑widths from two road‑edge polylines."""
    if lanes is None or len(lanes) < 2:
        return None, None
    lane0, lane1 = lanes[0], lanes[1]
    n = min(len(lane0), len(lane1))
    centerline, half_widths = [], []
    for i in range(n):
        cx = (lane0[i][0] + lane1[i][0]) / 2.0
        cy = (lane0[i][1] + lane1[i][1]) / 2.0
        centerline.append((cx, cy))
        hw = np.hypot(lane0[i][0] - cx, lane0[i][1] - cy)
        half_widths.append(hw)
    return centerline, half_widths


def compute_computed_cte(x, y, centerline, half_widths):
    """Cross‑track error, scaled so hw == 1.5 CTE‑units."""
    if centerline is None:
        return 0.0
    distances = [np.hypot(x - cx, y - cy) for cx, cy in centerline]
    idx = int(np.argmin(distances))
    closest_distance = distances[idx]
    hw = half_widths[idx]
    return (closest_distance / hw) * 1.5 if hw > 0 else 0.0


def analyze_run(poses_data, moving_data, survival_threshold, safe_thresholds):
    """Analyse a single run and return various metrics."""
    # --- poses ---------------------------------------------------------------
    poses = poses_data.get("pose")
    pose_times = poses_data.get("time")
    lanes = poses_data.get("lanes")
    if poses is None or pose_times is None or lanes is None:
        raise ValueError("Poses JSON must contain 'pose', 'time', and 'lanes'.")
    if isinstance(pose_times, str):
        pose_times = ast.literal_eval(pose_times)
    poses = scale_data(poses)
    for i in range(len(lanes)):
        lanes[i] = scale_data(lanes[i])

    # --- moving --------------------------------------------------------------
    going_raw = moving_data.get("going")
    moving_times = moving_data.get("time")
    if going_raw is None or moving_times is None:
        raise ValueError("Moving JSON must contain 'going' and 'time'.")
    if isinstance(going_raw, str):
        going_raw = ast.literal_eval(going_raw)
    if isinstance(moving_times, str):
        moving_times = ast.literal_eval(moving_times)

    # --- build 'going' intervals --------------------------------------------
    intervals = []
    n_moving = len(going_raw)
    if n_moving == 1:
        if going_raw[0]:
            intervals.append((moving_times[0], pose_times[-1]))
    elif n_moving > 1:
        i = 0
        while i < n_moving:
            if going_raw[i]:
                start = moving_times[i]
                j = i + 1
                while j < n_moving and going_raw[j]:
                    j += 1
                end = moving_times[j] if j < n_moving else pose_times[-1]
                intervals.append((start, end))
                i = j
            else:
                i += 1

    def is_going(timestamp):
        return any(s <= timestamp <= e for s, e in intervals)

    # --- road reference ------------------------------------------------------
    centerline, half_widths = compute_centerline_and_halfwidths(lanes)
    all_points = lanes[0] + lanes[1]
    center_x = np.mean([pt[0] for pt in all_points])
    center_y = np.mean([pt[1] for pt in all_points])

    ref_angle = 0.0
    for i in range(len(poses)):
        if is_going(pose_times[i]):
            x0, y0 = poses[i]
            ref_angle = np.arctan2(y0 - center_y, x0 - center_x)
            break

    # --- accumulators --------------------------------------------------------
    survival_turn  = None
    total_turn     = 0.0
    safe_turns     = {st: 0.0 for st in safe_thresholds}
    cumulative_turn = 0.0
    prev_raw_angle = None
    max_cte        = 0.0
    total_speed    = 0.0
    speed_count    = 0
    prev_going_idx = None

    # --- iterate poses -------------------------------------------------------
    for i, (x, y) in enumerate(poses):
        if not is_going(pose_times[i]):
            prev_raw_angle = None
            continue

        # cte
        curr_cte = compute_computed_cte(x, y, centerline, half_widths)
        max_cte = max(max_cte, curr_cte)

        # turning
        current_angle = np.arctan2(y - center_y, x - center_x)
        current_raw = np.degrees(current_angle - ref_angle)
        if prev_raw_angle is None:
            diff = 0.0
        else:
            diff = current_raw - prev_raw_angle
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
        prev_raw_angle = current_raw

        total_turn += abs(diff)
        if survival_turn is None:
            cumulative_turn += abs(diff)
            for st in safe_thresholds:
                if curr_cte <= st:
                    safe_turns[st] += abs(diff)
            if curr_cte > survival_threshold or cumulative_turn > 1080:
                survival_turn = cumulative_turn

        # speed
        if prev_going_idx is not None:
            dt = pose_times[i] - pose_times[prev_going_idx]
            if dt > 0:
                x_prev, y_prev = poses[prev_going_idx]
                total_speed += np.hypot(x - x_prev, y - y_prev) / dt
                speed_count += 1
        prev_going_idx = i

    if survival_turn is None:
        survival_turn = cumulative_turn

    safe_percents = {}
    for st in safe_thresholds:
        percent = (safe_turns[st] / survival_turn) * 100 if survival_turn > 0 else 0.0
        safe_percents[st] = min(percent, 100.0)

    mean_speed = (total_speed / speed_count) if speed_count else 0.0

    return survival_turn, safe_percents, mean_speed, max_cte, total_turn


# -----------------------------------------------------------------------------#
#                                      main                                    #
# -----------------------------------------------------------------------------#

def main():
    parser = argparse.ArgumentParser(
        description="""Generate a performance table from runs.

Computes cumulative turning until computed CTE exceeds the survival threshold
(SurvivedDeg) as well as TotalTurn, safe‑percent metrics, mean speed,
maximum CTE, and an overall score. Only runs with TotalTurn ≥ 1080° are kept.
Results are then grouped by the sub‑folder they reside in.""")
    parser.add_argument('--folder',   type=str,
                        default='/home/cubos98/catkin_ws/src/Vehicle/results_reality',
                        help="Root folder containing dataset folders.")
    parser.add_argument('--dataset',  type=str, default='dataset3',
                        help="Dataset folder to process (e.g., dataset3).")
    parser.add_argument('--survival_threshold', type=float, default=4.0,
                        help="Survival CTE threshold.")
    parser.add_argument('--safe_thresholds',    type=str, default='1.5,2.0,3.0',
                        help="Comma‑separated list of safe CTE thresholds.")
    parser.add_argument('--ideal_angle',        type=float, default=360.0,
                        help="Ideal turning angle for normalization.")
    args = parser.parse_args()

    survival_threshold = args.survival_threshold
    safe_thresholds = [float(x) for x in args.safe_thresholds.split(',') if x.strip()]
    ideal_angle = args.ideal_angle
    dataset_folder = os.path.join(args.folder, args.dataset)

    # --------------------------------------------------------------------- #
    # find runs                                                             #
    # --------------------------------------------------------------------- #
    runs = {}
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('_poses.json'):
                run_name = file.replace('_poses.json', '')
                moving_filename = f"{run_name}_moving.json"
                poses_path  = os.path.join(root, file)
                moving
