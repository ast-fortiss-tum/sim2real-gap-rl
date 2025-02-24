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
    return [[p[0]*scale, p[1]*scale] for p in data]

def compute_centerline_and_halfwidths(lanes):
    """
    Given lanes (assumed that lanes[0] and lanes[1] are the road boundaries),
    compute a centerline and half-widths.
    The centerline is computed as the midpoint of corresponding points on lane0 and lane1.
    The half-width is the distance from the midpoint to lane0.
    """
    if lanes is None or len(lanes) < 2:
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

def analyze_run(poses_data, moving_data, survival_thresholds, safe_threshold):
    """
    Process a single run and compute:
      - For each survival threshold (a computed CTE value), the cumulative turning angle (in degrees)
        accumulated while the agent is "going" until the computed CTE first exceeds that threshold.
      - Within that period, the percentage of turning (in degrees) during which computed CTE was at or below
        the safe threshold.
      - The mean speed computed from consecutive "going" poses.
      - The maximum computed CTE observed during the run.
    
    This function expects the poses JSON to contain keys "pose", "time", and "lanes", and the moving JSON
    to contain keys "going" and "time". If any of these values are strings (as in your files), they are converted.
    Additionally, all coordinate values are scaled by 25.
    """
    # Load poses data
    poses = poses_data.get("pose")
    times = poses_data.get("time")
    lanes = poses_data.get("lanes")
    if poses is None or times is None or lanes is None:
        raise ValueError("Poses JSON must contain 'pose', 'time', and 'lanes'.")
    if isinstance(times, str):
        times = ast.literal_eval(times)
    poses = scale_data(poses)
    for i in range(len(lanes)):
        lanes[i] = scale_data(lanes[i])
    
    # Load moving data
    going = moving_data.get("going")
    moving_times = moving_data.get("time")
    if going is None or moving_times is None:
        raise ValueError("Moving JSON must contain 'going' and 'time'.")
    if isinstance(going, str):
        going = ast.literal_eval(going)
    if isinstance(moving_times, str):
        moving_times = ast.literal_eval(moving_times)
    
    n_points = min(len(poses), len(going))
    
    # Compute centerline and road center
    centerline, half_widths = compute_centerline_and_halfwidths(lanes)
    all_points = lanes[0] + lanes[1]
    center_x = np.mean([pt[0] for pt in all_points])
    center_y = np.mean([pt[1] for pt in all_points])
    
    # Determine a reference angle using the first sample where the agent is going.
    ref_angle = None
    for i in range(n_points):
        if going[i]:
            x0, y0 = poses[i]
            ref_angle = np.arctan2(y0 - center_y, x0 - center_x)
            break
    if ref_angle is None:
        ref_angle = 0

    # Initialize accumulators.
    survival_turns = {thr: None for thr in survival_thresholds}
    safe_turns = {thr: 0.0 for thr in survival_thresholds}
    cumulative_turn = 0.0
    prev_raw = None
    max_cte = 0.0

    total_speed = 0.0
    speed_count = 0

    for i in range(n_points):
        if not going[i]:
            prev_raw = None
            continue

        x, y = poses[i]
        t = times[i]
        curr_cte = compute_computed_cte(x, y, centerline, half_widths)
        max_cte = max(max_cte, curr_cte)

        # Compute turning angle relative to road center.
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
            cumulative_turn += abs(diff)
            if curr_cte <= safe_threshold:
                for thr in survival_thresholds:
                    if survival_turns[thr] is None:
                        safe_turns[thr] += abs(diff)
            prev_raw = current_raw

        # Compute speed from previous going sample.
        if i > 0 and going[i-1]:
            x_prev, y_prev = poses[i-1]
            dt = t - times[i-1]
            if dt > 0:
                dist = np.hypot(x - x_prev, y - y_prev)
                total_speed += (dist / dt)
                speed_count += 1

        # For each survival threshold, if not yet recorded, check if computed CTE exceeds threshold.
        for thr in survival_thresholds:
            if survival_turns[thr] is None and curr_cte > thr:
                survival_turns[thr] = cumulative_turn

    # For thresholds never exceeded, use final cumulative turning.
    for thr in survival_thresholds:
        if survival_turns[thr] is None:
            survival_turns[thr] = cumulative_turn

    safe_percents = {}
    for thr in survival_thresholds:
        if survival_turns[thr] > 0:
            safe_percents[thr] = (safe_turns[thr] / survival_turns[thr]) * 100
        else:
            safe_percents[thr] = 0.0

    mean_speed = (total_speed / speed_count) if speed_count > 0 else 0.0

    return survival_turns, safe_percents, mean_speed, max_cte

def main():
    parser = argparse.ArgumentParser(
        description="Generate a performance table from runs using poses.json and moving.json files. "
                    "The script computes cumulative turning until computed CTE exceeds given thresholds, "
                    "the percentage of turning that stayed within a safe CTE threshold, the mean speed, and "
                    "the maximum computed CTE."
    )
    parser.add_argument('--folder', type=str, default='/home/cubos98/catkin_ws/src/Vehicle/results_reality/dataset3/',
                        help="Folder containing run files (default: results_reality)")
    parser.add_argument('--survival_thresholds', type=str, default='2.0,100.0,100.0',
                        help="Comma-separated list of survival CTE thresholds.")
    parser.add_argument('--safe_threshold', type=float, default=10,
                        help="CTE threshold below which turning is considered safe.")
    args = parser.parse_args()

    survival_thresholds = [float(val.strip()) for val in args.survival_thresholds.split(',') if val.strip()]
    folder = args.folder

    # Find all runs by looking for files ending with '_poses.json'
    run_files = [f for f in os.listdir(folder) if f.endswith('_poses.json')]
    runs = {}
    for pf in run_files:
        run_name = pf.replace("_poses.json", "")
        moving_filename = f"{run_name}_moving.json"
        poses_path = os.path.join(folder, pf)
        moving_path = os.path.join(folder, moving_filename)
        if os.path.exists(moving_path):
            runs[run_name] = {"poses": poses_path, "moving": moving_path}
        else:
            print(f"Warning: Skipping {run_name} because {moving_filename} not found.")

    results = []
    for run_name, files in runs.items():
        try:
            poses_data = load_data(files["poses"])
            moving_data = load_data(files["moving"])
            survival_turns, safe_percents, mean_speed, max_cte = analyze_run(
                poses_data, moving_data, survival_thresholds, args.safe_threshold
            )
            row = {"Run": run_name, "MeanSpeed": mean_speed, "MaxCTE": max_cte}
            for thr in survival_thresholds:
                row[f"SurvivedDeg_{thr}"] = survival_turns[thr]
                row[f"SafePerc_{thr}"] = safe_percents[thr]
            results.append(row)
        except Exception as e:
            print(f"Error processing run {run_name}: {e}")

    if not results:
        print("No valid runs found in folder.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("Run")
    print("\nPerformance Table:")
    print(df.to_string(index=False))
    # Optionally, save to CSV:
    # df.to_csv("performance_table.csv", index=False)

if __name__ == '__main__':
    main()
