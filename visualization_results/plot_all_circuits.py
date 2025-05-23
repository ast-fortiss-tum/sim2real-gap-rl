#!/usr/bin/env python3
"""
Quality audit for every run inside *dataset_final*:

  • Trim to first three laps (≈ 1080 °)
  • Metrics per run
        – mean L-2 error [cm]
        – “success” score   = (1 − L2/100) · 100  [%]
        – % of poses inside ± {1.0,1.5,2.0,2.5} lane-widths
  • Overview table + annotated circuit plots.

Usage
-----
    python analyse_runs.py
"""

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATASET_DIR    = "/home/cubos98/catkin_ws/src/Vehicle/results_reality/dataset_final"
OUT_FIG        = "all_runs_circuit_final.png"

SCALE          = 25            # metres → centimetres
TRIM_LIMIT_DEG = 1080          # keep first 3 laps
LANE_RANGES    = [1.0, 1.5, 2.0, 2.5]

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------

import os, json, ast, numpy as np, matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def scale_pts(seq, factor=SCALE):
    return [[x * factor, y * factor] for x, y in seq]

def collect_runs(root):
    """Return list of (run_name, poses_json_path)."""
    runs = []
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if os.path.isdir(folder):
            pose_files = [f for f in os.listdir(folder) if f.endswith("_poses.json")]
            if pose_files:
                runs.append((entry, os.path.join(folder, pose_files[0])))
    return runs

# ---------- geometry helpers ----------

def centre_line(l0, l1):
    n = min(len(l0), len(l1))
    return [((l0[i][0] + l1[i][0]) / 2, (l0[i][1] + l1[i][1]) / 2) for i in range(n)]

def centreline_halfwidths(l0, l1):
    n = min(len(l0), len(l1))
    centre, half = [], []
    for i in range(n):
        cx = (l0[i][0] + l1[i][0]) / 2
        cy = (l0[i][1] + l1[i][1]) / 2
        centre.append((cx, cy))
        half.append(np.hypot(l0[i][0] - cx, l0[i][1] - cy))
    return centre, half

# ---------- trimming ----------

def unwrap_angle(poses, centre):
    cx, cy = centre
    ang = np.unwrap(np.arctan2([y - cy for _, y in poses],
                               [x - cx for x, _ in poses]))
    return ang - ang[0]

def trim_first_laps(poses, centre, limit_deg=TRIM_LIMIT_DEG):
    if limit_deg is None or len(poses) < 2:
        return poses
    deg = np.degrees(unwrap_angle(poses, centre))
    idx = np.argmax(np.abs(deg) > limit_deg)
    return poses if idx == 0 else poses[:idx]

# ---------- metrics ----------

def mean_l2_cm(poses, centre_pts):
    """Average Euclidean distance [cm] to nearest centre-line sample."""
    if not poses:
        return 0.0
    d = []
    for x, y in poses:
        dd = [np.hypot(x - cx, y - cy) for cx, cy in centre_pts]
        d.append(min(dd))
    return float(np.mean(d))

def normalised_errors(poses, centre_pts, half):
    errs = []
    for x, y in poses:
        d = [np.hypot(x - cx, y - cy) for cx, cy in centre_pts]
        idx = int(np.argmin(d))
        if half[idx] > 0:
            errs.append(d[idx] / half[idx])
    return np.asarray(errs)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

runs = collect_runs(DATASET_DIR)
if not runs:
    raise SystemExit("No *_poses.json files found.")

# console header
hdr = "run-name".ljust(18) + "L2cm   Succ% "
for r in LANE_RANGES:
    hdr += f"in≤{r:<3}".rjust(8)
print("\n" + hdr)
print("-" * len(hdr))

# prepare plot grid (max 6 runs)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for ax, (run, pf) in zip(axes, runs):
    data       = load_json(pf)
    lanes_raw  = data["lanes"]
    poses_raw  = data["pose"]

    # some logs store lists as strings
    if isinstance(poses_raw, str):
        poses_raw = ast.literal_eval(poses_raw)
    if isinstance(lanes_raw[0], str):
        lanes_raw = [ast.literal_eval(l) for l in lanes_raw]

    poses = scale_pts(poses_raw)
    lanes = [scale_pts(l) for l in lanes_raw]

    # build geometry
    centre_geom = (np.mean([p[0] for p in lanes[0] + lanes[1]]),
                   np.mean([p[1] for p in lanes[0] + lanes[1]]))
    poses_trim = trim_first_laps(poses, centre_geom)

    centre_pts, half = centreline_halfwidths(lanes[0], lanes[1])
    l2_cm   = mean_l2_cm(poses_trim, centre_pts)
    succ    = max(0.0, (1.0 - l2_cm / 100.0) * 100.0)   # simple mapping
    errs    = normalised_errors(poses_trim, centre_pts, half)
    pct_band = [(np.mean(errs <= r) * 100.0) if errs.size else 0.0
                for r in LANE_RANGES]

    # ---- console row ----
    row = f"{run:<18}{l2_cm:6.1f}{succ:8.1f}"
    for p in pct_band:
        row += f"{p:8.1f}"
    print(row)

    # ---- drawing ----
    ax.set_facecolor("dimgray")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    lane_col = ["white", "white", "yellow"] + ["gray"] * max(0, len(lanes) - 3)
    lane_sty = ["-", "-", "--"]             + ["-"]    * max(0, len(lanes) - 3)
    for ln, col, ls in zip(lanes, lane_col, lane_sty):
        xs, ys = zip(*ln)
        ax.plot(xs, ys, color=col, ls=ls, lw=1)

    xs, ys = zip(*poses_trim)
    ax.plot(xs, ys, color="red", lw=1.2)

    m = 100
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_title(f"{run}\nSucc={succ:.1f}%  in≤1 lane={pct_band[0]:.1f}%",
                 fontsize=9)

# hide unused sub-plots
for extra in axes[len(runs):]:
    extra.set_visible(False)

fig.suptitle("First 3 laps – L-2 error & time-in-band metrics", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
fig.savefig(OUT_FIG, dpi=150)
print("\nfigure saved →", OUT_FIG)
