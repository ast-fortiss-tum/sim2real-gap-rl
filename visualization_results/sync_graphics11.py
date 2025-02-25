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

def analyze_run(poses_data, moving_data, survival_threshold, safe_thresholds):
    """
    Process a single run and compute:
      - The cumulative turning angle (in degrees) accumulated while the agent is "going"
        until the computed CTE first exceeds the survival threshold.
      - For each safe threshold (three values), the percentage of turning during that period
        where the computed CTE was at or below that safe threshold.
      - The mean speed computed from consecutive "going" poses.
      - The maximum computed CTE observed during the run.
    
    The poses JSON is expected to contain keys "pose", "time", and "lanes".
    The moving JSON is expected to contain keys "going" and "time".
    
    Timestamps are used to decide which pose points are “going.” The loop stops once the survival threshold is reached.
    """
    # --- Load poses data ---
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
    if n_moving == 0:
        pass  # No data.
    elif n_moving == 1:
        if going_raw[0]:
            intervals.append((moving_times[0], pose_times[-1]))
        # Otherwise, always off.
    else:
        i = 0
        while i < n_moving:
            if going_raw[i]:
                start = moving_times[i]
                j = i + 1
                while j < n_moving and going_raw[j]:
                    j += 1
                if j < n_moving:
                    end = moving_times[j]
                else:
                    end = pose_times[-1]
                intervals.append((start, end))
                i = j
            else:
                i += 1

    def is_going(timestamp, intervals):
        for start, end in intervals:
            if start <= timestamp <= end:
                return True
        return False

    # --- Compute centerline and road center ---
    centerline, half_widths = compute_centerline_and_halfwidths(lanes)
    all_points = lanes[0] + lanes[1]
    center_x = np.mean([pt[0] for pt in all_points])
    center_y = np.mean([pt[1] for pt in all_points])
    
    # --- Determine reference angle using the first "going" pose ---
    ref_angle = None
    for i in range(len(poses)):
        if is_going(pose_times[i], intervals):
            x0, y0 = poses[i]
            ref_angle = np.arctan2(y0 - center_y, x0 - center_x)
            break
    if ref_angle is None:
        ref_angle = 0

    # --- Initialize accumulators ---
    survival_turn = None   # cumulative turning until computed CTE exceeds survival_threshold
    safe_turns = {st: 0.0 for st in safe_thresholds}
    cumulative_turn = 0.0
    prev_raw = None
    max_cte = 0.0

    total_speed = 0.0
    speed_count = 0
    prev_going_idx = None  # index of the last "going" pose

    # --- Process each pose ---
    for i in range(len(poses)):
        if not is_going(pose_times[i], intervals):
            prev_raw = None
            continue

        x, y = poses[i]
        t = pose_times[i]
        curr_cte = compute_computed_cte(x, y, centerline, half_widths)
        #print(curr_cte, "  ", i)  # Debug print.
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
            # Accumulate safe turning only if current CTE is below each safe threshold.
            for st in safe_thresholds:
                if curr_cte <= st:
                    safe_turns[st] += abs(diff)
            prev_raw = current_raw

        # Compute speed using the previous "going" pose.
        if prev_going_idx is not None:
            dt = t - pose_times[prev_going_idx]
            if dt > 0:
                x_prev, y_prev = poses[prev_going_idx]
                dist = np.hypot(x - x_prev, y - y_prev)
                total_speed += (dist / dt)
                speed_count += 1
        prev_going_idx = i

        # Check if survival threshold is reached.
        if survival_turn is None and curr_cte > survival_threshold:
            survival_turn = cumulative_turn
            # Stop processing further poses.
            break

    # If survival threshold was never reached, use the total cumulative turning.
    if survival_turn is None:
        survival_turn = cumulative_turn

    # Calculate safe percentages and cap at 100%
    safe_percents = {}
    for st in safe_thresholds:
        percent = (safe_turns[st] / survival_turn) * 100 if survival_turn > 0 else 0.0
        safe_percents[st] = percent

    mean_speed = (total_speed / speed_count) if speed_count > 0 else 0.0

    return survival_turn, safe_percents, mean_speed, max_cte

def main():
    parser = argparse.ArgumentParser(
        description="Generate a performance table from runs using poses.json and moving.json files. "
                    "Computes cumulative turning until computed CTE exceeds the survival threshold, and for each "
                    "safe threshold (three values), the percentage of turning within that period where CTE was safe. "
                    "Also computes mean speed and maximum CTE."
    )
    parser.add_argument('--folder', type=str,
                        default='/home/cubos98/catkin_ws/src/Vehicle/results_reality/dataset2/',
                        help="Folder containing run files.")
    parser.add_argument('--survival_threshold', type=float, default=4.0,
                        help="The survival CTE threshold (a single float).")
    parser.add_argument('--safe_thresholds', type=str, default='1.5,2.0,3.0',
                        help="Comma-separated list of three safe CTE thresholds.")
    args = parser.parse_args()

    survival_threshold = args.survival_threshold
    safe_thresholds = [float(val.strip()) for val in args.safe_thresholds.split(',') if val.strip()]
    folder = args.folder

    # --- Find all runs ending with '_poses.json'
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
            surv_turn, safe_percents, mean_speed, max_cte = analyze_run(
                poses_data, moving_data, survival_threshold, safe_thresholds
            )
            row = {"Run": run_name, "MeanSpeed": mean_speed, "MaxCTE": max_cte, "SurvivedDeg": surv_turn}
            for st in safe_thresholds:
                row[f"SafePerc_{st}"] = safe_percents[st]
            results.append(row)
        except Exception as e:
            print(f"Error processing run {run_name}: {e}")

    if not results:
        print("No valid runs found in folder.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("Run")

    # --- Group results by the first 17 characters of the run name ---
    df["Group"] = df["Run"].apply(lambda x: x[:17])
    # Build an aggregation dictionary.
    agg_dict = {}
    for col in df.columns:
        if col in ["Run", "Group"]:
            continue
        if col == "MaxCTE":
            agg_dict[col] = ['max', 'std']
        else:
            agg_dict[col] = ['mean', 'std']
    grouped_df = df.groupby("Group").agg(agg_dict).reset_index()
    # Flatten MultiIndex columns.
    grouped_df.columns = ['_'.join(col).strip('_') if col[1] != '' else col[0]
                            for col in grouped_df.columns.values]

    print("\nIndividual Performance Table:")
    print(df.to_string(index=False))
    print("\nGrouped Performance Table (by first 17 characters of run name):")
    print(grouped_df.to_string(index=False))
    # Optionally, save to CSV:
    # grouped_df.to_csv("performance_table_grouped.csv", index=False)

if __name__ == '__main__':
    main()
