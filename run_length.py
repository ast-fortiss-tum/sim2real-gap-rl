#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import ast
import pandas as pd

def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def scale_data(data, scale=25):
    """Scale a list of 2D points by the given factor."""
    return [[p[0] * scale, p[1] * scale] for p in data]

def compute_centerline_and_halfwidths(lanes):
    """
    Compute a centerline and half-widths given lane boundaries.
    Assumes lanes[0] and lanes[1] are the two road boundaries.
    """
    if not lanes or len(lanes) < 2:
        return None, None
    lane0 = lanes[0]
    lane1 = lanes[1]
    n = min(len(lane0), len(lane1))
    centerline = []
    half_widths = []
    for i in range(n):
        cx = (lane0[i][0] + lane1[i][0]) / 2.0
        cy = (lane0[i][1] + lane1[i][1]) / 2.0
        centerline.append((cx, cy))
        hw = np.hypot(lane0[i][0] - cx, lane0[i][1] - cy)
        half_widths.append(hw)
    return centerline, half_widths

def compute_computed_cte(x, y, centerline, half_widths):
    """
    Compute the cross-track error (CTE) for a given (x,y) pose.
    For each point on the centerline, the Euclidean distance is computed and the minimum is selected.
    The distance is scaled so that a distance equal to the half-width corresponds to 1.5 CTE-units.
    """
    if centerline is None:
        return 0
    distances = [np.hypot(x - cx, y - cy) for cx, cy in centerline]
    idx = np.argmin(distances)
    closest_distance = distances[idx]
    hw = half_widths[idx]
    return (closest_distance / hw) * 1.5 if hw > 0 else 0

def compute_total_turn(poses_data, moving_data):
    """
    Compute the total turning (in degrees) for a run and the maximum computed CTE.
    
    - Loads the poses (and lane boundaries) from poses_data.
    - Loads the "going" status and corresponding times from moving_data.
    - Builds intervals during which the car is "going".
    - Computes the road center from lane boundaries and finds a reference angle using
      the first "going" pose.
    - For each pose during a "going" interval, computes the turning angle (relative to the
      road center and offset by the reference angle) and sums up the absolute differences.
    - Also computes the maximum computed CTE encountered during the "going" intervals.
      
    Returns:
      total_turn: cumulative turning (in degrees) over all "going" segments.
      max_cte: maximum computed CTE observed during those segments.
    """
    # --- Load and scale poses data ---
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
    
    # --- Load moving data ---
    going_raw = moving_data.get("going")
    moving_times = moving_data.get("time")
    if going_raw is None or moving_times is None:
        raise ValueError("Moving JSON must contain 'going' and 'time'.")
    if isinstance(going_raw, str):
        going_raw = ast.literal_eval(going_raw)
    if isinstance(moving_times, str):
        moving_times = ast.literal_eval(moving_times)
    
    # --- Build "going" intervals from moving.json ---
    intervals = []
    n_moving = len(going_raw)
    if n_moving == 1:
        if going_raw[0]:
            intervals.append((moving_times[0], pose_times[-1]))
    else:
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

    def is_going(timestamp, intervals):
        for start, end in intervals:
            if start <= timestamp <= end:
                return True
        return False

    # --- Compute road center from lanes ---
    centerline, half_widths = compute_centerline_and_halfwidths(lanes)
    all_points = lanes[0] + lanes[1]
    center_x = np.mean([pt[0] for pt in all_points])
    center_y = np.mean([pt[1] for pt in all_points])
    
    # --- Determine reference angle using first "going" pose ---
    ref_angle = None
    for i in range(len(poses)):
        if is_going(pose_times[i], intervals):
            x0, y0 = poses[i]
            ref_angle = np.arctan2(y0 - center_y, x0 - center_x)
            break
    if ref_angle is None:
        ref_angle = 0

    # --- Accumulate total turning and track maximum CTE ---
    total_turn = 0.0
    max_cte = 0.0
    prev_raw = None
    for i in range(len(poses)):
        if not is_going(pose_times[i], intervals):
            prev_raw = None
            continue

        x, y = poses[i]
        # Compute current CTE.
        curr_cte = compute_computed_cte(x, y, centerline, half_widths)
        if curr_cte > max_cte:
            max_cte = curr_cte

        current_angle = np.arctan2(y - center_y, x - center_x)
        current_raw = np.degrees(current_angle - ref_angle)
        if prev_raw is None:
            prev_raw = current_raw
        else:
            diff = current_raw - prev_raw
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            total_turn += abs(diff)
            prev_raw = current_raw
    return total_turn, max_cte

def main():
    parser = argparse.ArgumentParser(
        description="Compute total turning degrees and maximum CTE for each run in a given folder. "
                    "Each run consists of a _poses.json and its corresponding _moving.json file."
    )
    parser.add_argument('--folder', type=str, default='./results_reality',
                        help="Folder containing run files (_poses.json and _moving.json).")
    args = parser.parse_args()

    folder = args.folder
    results = []
    
    # Recursively find all _poses.json files in the folder.
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('_poses.json'):
                run_name = file.replace('_poses.json', '')
                moving_filename = f"{run_name}_moving.json"
                poses_path = os.path.join(root, file)
                moving_path = os.path.join(root, moving_filename)
                if os.path.exists(moving_path):
                    try:
                        poses_data = load_data(poses_path)
                        moving_data = load_data(moving_path)
                        total_turn, max_cte = compute_total_turn(poses_data, moving_data)
                        results.append({
                            "Run": run_name,
                            "TotalTurn": total_turn,
                            "MaxCTE": max_cte,
                            "Path": os.path.relpath(root, folder)
                        })
                    except Exception as e:
                        print(f"Error processing run {run_name}: {e}")
                else:
                    print(f"Skipping {run_name} because {moving_filename} not found.")
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("TotalTurn", ascending=False)
        print(df.to_string(index=False))
    else:
        print("No valid runs found.")

if __name__ == '__main__':
    main()
