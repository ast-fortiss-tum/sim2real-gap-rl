#!/usr/bin/env python3
"""
Locate the run with the largest accumulated heading change, plot it,
and list every try with total-deg + sub-folder location.

Run:
    python plot_largest_run.py
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = "/home/cubos98/catkin_ws/src/Vehicle/results_reality/dataset_final"
OUT_FIG     = "largest_run_circuit.png"
SCALE       = 25        # scale factor for coordinates
TRIM_LIMIT  = None      # e.g. 10800 to clip, or None to plot full run

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os, json, ast
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def scale_pts(xy, factor=SCALE):
    return [[p[0]*factor, p[1]*factor] for p in xy]

def collect_pose_files(root):
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith("_poses.json"):
                yield os.path.join(r, f)

def heading_total_deg(poses, centre):
    cx, cy = centre
    ang = np.unwrap(np.arctan2([y-cy for _,y in poses],
                               [x-cx for x,_ in poses]))
    return abs(np.degrees(ang[-1]-ang[0]))

def trim_after_angle(poses, centre, limit_deg):
    if limit_deg is None or len(poses) < 2:
        return poses
    cx, cy = centre
    ang = np.unwrap(np.arctan2([y-cy for _,y in poses],
                               [x-cx for x,_ in poses]))
    ang_deg = np.degrees(ang - ang[0])
    idx = np.argmax(np.abs(ang_deg) > limit_deg)
    return poses if idx == 0 else poses[:idx]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stats = []  # list of tuples (total_deg, rel_subfolder, basename, poses, lanes, centre)
    for pf in collect_pose_files(DATASET_DIR):
        data = load_json(pf)
        lanes_raw, poses_raw = data["lanes"], data["pose"]

        if isinstance(poses_raw, str):
            poses_raw = ast.literal_eval(poses_raw)
        if isinstance(lanes_raw[0], str):
            lanes_raw = [ast.literal_eval(l) for l in lanes_raw]

        poses = scale_pts(poses_raw)
        lanes = [scale_pts(l) for l in lanes_raw]

        centre = (np.mean([p[0] for p in lanes[0]+lanes[1]]),
                  np.mean([p[1] for p in lanes[0]+lanes[1]]))

        deg = heading_total_deg(poses, centre)

        rel_dir  = os.path.relpath(os.path.dirname(pf), DATASET_DIR)
        rel_dir  = "." if rel_dir == "." else rel_dir  # root folder shown as "."
        basename = os.path.basename(pf)
        stats.append((deg, rel_dir, basename, poses, lanes, centre))

    if not stats:
        print("No *_poses.json files found.")
        return

    # sort high→low by total_deg
    stats.sort(reverse=True, key=lambda t: t[0])

    # report all tries
    print("\nTotal heading change by try (largest first)")
    print("---------------------------------------------------------------")
    for rank, (deg, rel, base, *_rest) in enumerate(stats, 1):
        print(f"{rank:2d}. {rel:<20} {base:<25}  {deg:8.1f}°")

    # best run
    best_deg, best_rel, best_base, poses, lanes, centre = stats[0]
    print("---------------------------------------------------------------")
    print(f"Champion run → {best_rel}/{best_base}  ({best_deg:.1f}°)")

    poses = trim_after_angle(poses, centre, TRIM_LIMIT)

    # ----------------------------------------------------------- plot -------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("dimgray")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])

    lane_col = ["white", "white", "yellow"] + ["gray"]*max(0, len(lanes)-3)
    lane_sty = ["-", "-", "--"]             + ["-"]   *max(0, len(lanes)-3)
    for ln, c, s in zip(lanes, lane_col, lane_sty):
        xs, ys = zip(*ln); ax.plot(xs, ys, color=c, linestyle=s, lw=1)

    xs, ys = zip(*poses)
    ax.plot(xs, ys, color="red", lw=1.3)

    margin = 100
    ax.set_xlim(min(xs)-margin, max(xs)+margin)
    ax.set_ylim(min(ys)-margin, max(ys)+margin)

    ax.set_title(f"{best_rel}/{best_base} – {best_deg:.1f}° total heading change",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Plot saved → {OUT_FIG}")

if __name__ == "__main__":
    main()
