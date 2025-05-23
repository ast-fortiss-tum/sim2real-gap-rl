#!/usr/bin/env python3
"""
First-3-lap plots + “time-inside band” metric for four lane ranges.

Run:
    python plot_all_circuits_inband.py
"""

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATASET_DIR     = "/home/cubos98/catkin_ws/src/Vehicle/results_reality/dataset_final"
OUT_FIG         = "all_runs_circuit2.png"
SCALE           = 25            # metres → centimetres
TRIM_LIMIT_DEG  = 1080          # keep first 3 laps (None → full run)
LANE_RANGES     = [1.0, 1.5, 2.0, 2.5]   # bands to evaluate (in lane widths)

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------

import os, json, ast, numpy as np, matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def scale_pts(seq, factor=SCALE):
    return [[x * factor, y * factor] for x, y in seq]

def collect_runs(root):
    runs = []
    for sub in sorted(os.listdir(root)):
        folder = os.path.join(root, sub)
        if os.path.isdir(folder):
            pose_files = [f for f in os.listdir(folder) if f.endswith("_poses.json")]
            if pose_files:
                runs.append((sub, os.path.join(folder, pose_files[0])))
    return runs

def centreline_halfwidths(l0, l1):
    n = min(len(l0), len(l1))
    centre, half = [], []
    for i in range(n):
        cx = (l0[i][0] + l1[i][0]) / 2
        cy = (l0[i][1] + l1[i][1]) / 2
        centre.append((cx, cy))
        half.append(np.hypot(l0[i][0] - cx, l0[i][1] - cy))
    return centre, half

def normalised_errors(poses, centre, half):
    errs = []
    for x, y in poses:
        dists = [np.hypot(x - cx, y - cy) for cx, cy in centre]
        idx   = int(np.argmin(dists))
        if half[idx] > 0:
            errs.append(dists[idx] / half[idx])
    return np.asarray(errs)

def unwrap_angle(poses, centre):
    cx, cy = centre
    ang = np.unwrap(np.arctan2([y - cy for _, y in poses],
                               [x - cx for x, _ in poses]))
    return ang - ang[0]

def trim_after_angle(poses, centre, limit_deg=TRIM_LIMIT_DEG):
    if limit_deg is None or len(poses) < 2:
        return poses
    deg = np.degrees(unwrap_angle(poses, centre))
    idx = np.argmax(np.abs(deg) > limit_deg)
    return poses if idx == 0 else poses[:idx]

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

runs = collect_runs(DATASET_DIR)
if not runs:
    raise SystemExit("No *_poses.json files found.")

# Console header
header = "run-name".ljust(18)
for r in LANE_RANGES:
    header += f" inside≤{r:<3} ".rjust(12)
print("\n" + header)
print("-" * len(header))

# Plot grid
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for ax, (run, pf) in zip(axes, runs):
    data       = load_json(pf)
    lanes_raw  = data["lanes"]
    poses_raw  = data["pose"]

    if isinstance(poses_raw, str):
        poses_raw = ast.literal_eval(poses_raw)
    if isinstance(lanes_raw[0], str):
        lanes_raw = [ast.literal_eval(l) for l in lanes_raw]

    poses = scale_pts(poses_raw)
    lanes = [scale_pts(l) for l in lanes_raw]

    centre_geom = (np.mean([p[0] for p in lanes[0] + lanes[1]]),
                   np.mean([p[1] for p in lanes[0] + lanes[1]]))

    poses_trim = trim_after_angle(poses, centre_geom, TRIM_LIMIT_DEG)
    centre, half = centreline_halfwidths(lanes[0], lanes[1])
    errs = normalised_errors(poses_trim, centre, half)

    # Percentages for each lane range
    pct = [(np.mean(errs <= r) * 100.0) if errs.size else 0.0 for r in LANE_RANGES]

    # Console row
    row = f"{run:<18}"
    for p in pct:
        row += f"{p:10.2f}% "
    print(row)

    # ------------------ plot ------------------
    ax.set_facecolor("dimgray")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    lane_col = ["white", "white", "yellow"] + ["gray"] * max(0, len(lanes) - 3)
    lane_sty = ["-", "-", "--"]             + ["-"]    * max(0, len(lanes) - 3)
    for ln, col, ls in zip(lanes, lane_col, lane_sty):
        xs, ys = zip(*ln); ax.plot(xs, ys, color=col, ls=ls, lw=1)

    xs, ys = zip(*poses_trim)
    ax.plot(xs, ys, color="red", lw=1.2)

    margin = 100
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_title(f"{run}\ninside ≤1 lane: {pct[0]:.1f}%", fontsize=9)

# Hide unused subplots
for extra in axes[len(runs):]:
    extra.set_visible(False)

fig.suptitle("First 3 laps – time inside multiple lane-width bands", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
fig.savefig(OUT_FIG, dpi=150)
print("\nfigure saved →", OUT_FIG)
