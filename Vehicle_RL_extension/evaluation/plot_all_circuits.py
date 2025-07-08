#!/usr/bin/env python3
import os, json, ast, numpy as np, matplotlib.pyplot as plt

"""
Author: Cristian Cubides
This script evaluates multiple runs of a vehicle navigation system
by analyzing their performance on a predefined circuit.
It trims the runs to the first three laps, computes various metrics,
and generates a summary table along with annotated circuit plots.
It is designed to work with a dataset of runs stored in JSON files,
where each run contains lane definitions and vehicle poses. 

Quality audit for every run inside *dataset_final*:

  • Trim to first three laps (≈1080 °)
  • Metrics per run
        – mean L-2 error [cm]
        – “success” score   = (1 − L2/100) · 100  [%]
        – % of poses inside ± {1.0, 1.5, 2.0, 2.5} lane-widths
  • Overview table + annotated circuit plots (hi-res, larger fonts).
"""

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATASET_DIR    = "./results_reality/dataset_final"
OUT_FIG        = "./figures/all_runs_circuit_final.png"

SCALE          = 25            # metres → centimetres
TRIM_LIMIT_DEG = 1080          # keep first 3 laps (still used for trim)
LANE_RANGES    = [1.0, 1.5, 2.0, 2.5]
MAX_JUMP_CM    = 10            # Max allowed jump between points (in cm)
MAX_NORM_ERROR = 4.0           # If any normalized error > this, the run fails

# --------------------------------------------------------------------------
# Global style tweaks
# --------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":        25,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.titlesize": 35,
})

# --------------------------------------------------------------------------
# Custom labels for your runs (Edit this as you want!)
# --------------------------------------------------------------------------

custom_labels = [
    r"$\bf{Demo}$",
    r"$\bf{Trained\ in:}$ Fake Camera, $\bf{Sees:}$ Camera",
    r"$\bf{Trained\ in:}$ Fake Camera, $\bf{Sees:}$ Fake Camera",
    r"$\bf{Trained\ in:}$ Simulation, $\bf{Sees:}$ Camera",
    r"$\bf{Trained\ in:}$ Simulation, $\bf{Sees:}$ Fake Simulation",
    r"$\bf{Trained\ in:}$ Simulation, $\bf{Sees:}$ Simulation (Twin)"
    # Add/remove/edit as many as you need, one per run
]
# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def scale_pts(seq, factor=SCALE):
    return [[x * factor, y * factor] for x, y in seq]

def collect_runs(root):
    runs = []
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if os.path.isdir(folder):
            pose_files = [f for f in os.listdir(folder) if f.endswith("_poses.json")]
            if pose_files:
                runs.append((entry, os.path.join(folder, pose_files[0])))
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

def filter_large_jumps(poses, max_jump_cm):
    if not poses:
        return poses
    filtered = [poses[0]]
    for p in poses[1:]:
        if np.hypot(p[0] - filtered[-1][0], p[1] - filtered[-1][1]) <= max_jump_cm:
            filtered.append(p)
    return filtered

def mean_l2_cm(poses, centre_pts):
    if not poses:
        return 0.0
    d = []
    for x, y in poses:
        d.append(min(np.hypot(x - cx, y - cy) for cx, cy in centre_pts))
    return float(np.mean(d))

def normalised_errors(poses, centre_pts, half):
    errs = []
    for x, y in poses:
        dists = [np.hypot(x - cx, y - cy) for cx, cy in centre_pts]
        idx   = int(np.argmin(dists))
        if half[idx] > 0:
            errs.append(dists[idx] / half[idx])
    return np.asarray(errs)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

runs = collect_runs(DATASET_DIR)
if not runs:
    raise SystemExit("No *_poses.json files found.")

if len(custom_labels) < len(runs):
    raise ValueError("Not enough custom labels for the runs! Add more to 'custom_labels'.")

hdr = "run-name".ljust(18) + "L2cm   Succ% "
for r in LANE_RANGES:
    hdr += f"in≤{r:<3}".rjust(8)
print("\n" + hdr)
print("-" * len(hdr))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, (ax, (run, pf)) in enumerate(zip(axes, runs)):
    run_label = custom_labels[i]
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
    poses_trim = trim_first_laps(poses, centre_geom)
    poses_filtered = filter_large_jumps(poses_trim, MAX_JUMP_CM)
    centre_pts, half = centreline_halfwidths(lanes[0], lanes[1])
    errs_full = normalised_errors(poses_filtered, centre_pts, half)

    # FAILURE criteria: any error > MAX_NORM_ERROR
    fail_norm_err = np.any(errs_full > MAX_NORM_ERROR)

    if fail_norm_err:
        # Console: red text
        print(f"\033[91m{run_label:<18}FAILURE !!\033[0m")
        title_status = f"FAILURE !!"
        title_color = "red"
    else:
        l2_cm   = mean_l2_cm(poses_filtered, centre_pts)
        succ    = max(0.0, (1.0 - l2_cm / 100.0) * 100.0)
        pct_band = [(np.mean(errs_full <= r) * 100.0) if errs_full.size else 0.0 for r in LANE_RANGES]
        row = f"{run_label:<18}{l2_cm:6.1f}{succ:8.1f}"
        for p in pct_band:
            row += f"{p:8.1f}"
        print(row)
        title_status = f"Success={succ:.1f}%; on lane={pct_band[0]:.1f}%"
        # ... inside your loop after success and pct_band have been calculated:
        title_status = (
            r"$\bf{Success:}$"
            f" {succ:.1f}%; "
            r"$\bf{on\ lane:}$"
            f" {pct_band[0]:.1f}%"
        )

        title_color = "black"

    # Plot (filtered trajectory)
    ax.set_facecolor("dimgray")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    lane_col = ["white", "white", "yellow"] + ["gray"] * max(0, len(lanes) - 3)
    lane_sty = ["-", "-", "--"]             + ["-"]    * max(0, len(lanes) - 3)
    for ln, col, ls in zip(lanes, lane_col, lane_sty):
        xs, ys = zip(*ln)
        ax.plot(xs, ys, color=col, ls=ls, lw=1)
    if poses_filtered:
        xs, ys = zip(*poses_filtered)
        ax.plot(xs, ys, color="red", lw=1.4)

    m = 100
    if poses_filtered:
        ax.set_xlim(min(xs) - m, max(xs) + m)
        ax.set_ylim(min(ys) - m, max(ys) + m)
    # Title in red if FAILURE
    ax.set_title(f"{run_label}\n{title_status}", color=title_color)

for extra in axes[len(runs):]:
    extra.set_visible(False)

#fig.suptitle("Performance: L-2 success & time-in-band metrics")
fig.tight_layout(rect=[0, 0.03, 1, 0.92])

fig.savefig(OUT_FIG, dpi=300)       # 300 DPI
print("\nfigure saved →", OUT_FIG)
