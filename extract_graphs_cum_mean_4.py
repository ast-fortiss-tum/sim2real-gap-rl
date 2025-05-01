#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Five‑algorithm comparison (CLI filter)
Plots are now sorted descending by final reward.
"""

import os, re, math, argparse, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from collections import defaultdict

# ────────────────────────── CLI options ───────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--exclude", nargs="*", default=[],
                    help="algorithms to leave out (space‑separated list)")
args = parser.parse_args()

# ───────────────────────── experiment list ────────────────────────
algos_all = ["ContSAC_STR", "DARC_ON", "DARC_OFF", "ContSAC_t"]
algos = [a for a in algos_all if a not in args.exclude]
if not algos:
    raise SystemExit("All algorithms excluded – nothing to do.")

# ───────────────────────── settings ───────────────────────────────
folder_path   = "/home/cubos98/Desktop/MA/DARAIL/runs/runs_Total"
drop_first_k  = 0                             
base_regex    = re.compile(
    r"(?=.*noise_cfrs_0\.0)(?=.*noise_0\.2)(?=.*bias_0\.5)",
    #r"(?=.*noise_cfrs_0\.0)",
    re.IGNORECASE,
)
algo_pat      = re.compile(rf"^({'|'.join(map(re.escape, algos))})", re.IGNORECASE)
deg_pat       = re.compile(r"degree_([-+]?\d*\.?\d+)", re.IGNORECASE)
tag_map = {
    "DARC":    "Target Env/Rewards",
    "ContSAC": "Target/Env/Rewards",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       7,
    "axes.labelsize":  8,
    "axes.titlesize":  9,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "grid.linestyle": "--",
    "grid.alpha":     0.3,
    "figure.dpi":     300,
    "savefig.dpi":    300,
})

# ──────────────────────── read runs ───────────────────────────────
raw_runs = []
for root, _, files in os.walk(folder_path):
    name = os.path.basename(root)
    if not (base_regex.search(name) and algo_pat.match(name)):
        continue

    algo   = algo_pat.match(name).group(1)
    degree = float(deg_pat.search(name).group(1)) if deg_pat.search(name) else math.nan
    tag    = tag_map["DARC" if algo.lower().startswith("darc") else "ContSAC"]

    steps, vals = [], []
    for ev in [f for f in files if "tfevents" in f]:
        for e in tf.compat.v1.train.summary_iterator(os.path.join(root, ev)):
            if not e.summary: continue
            for v in e.summary.value:
                if v.tag == tag and hasattr(v, "simple_value"):
                    steps.append(e.step)
                    vals.append(v.simple_value)

    if len(steps) <= drop_first_k + 1:
        continue

    idx   = np.argsort(steps)
    arr   = np.array(vals)[idx][drop_first_k:]
    if algo.lower().startswith("darc"):
        arr = arr[:-1]
    curve = np.cumsum(arr)
    raw_runs.append({"algo": algo, "degree": degree, "curve": curve})

if not raw_runs:
    raise SystemExit("No runs matched the filter – check paths / regex.")

# ───────────────── helper: pad to same length ─────────────────────
def pad_to_len(curve_list, length):
    return np.vstack([
        np.pad(c, (0, length - len(c)), constant_values=np.nan)
        for c in curve_list
    ])

# ───────────── Figure 1 data (per algo) ───────────────────────────
group_algo = defaultdict(list)
max_len     = 0
for r in raw_runs:
    group_algo[r["algo"]].append(r["curve"])
    max_len = max(max_len, len(r["curve"]))

stat_algo = {}
for algo, curves in group_algo.items():
    stack = pad_to_len(curves, max_len)
    stat_algo[algo] = (
        np.nanmedian(stack, axis=0),
        np.nanpercentile(stack, 25, axis=0),
        np.nanpercentile(stack, 75, axis=0),
        len(curves),
    )

# ──────── Sort algorithms by their final median reward (desc) ───────
# final median = last entry of the median curve
algos_present = sorted(
    stat_algo.keys(),
    key=lambda a: stat_algo[a][0][-1],
    reverse=True
)
# regenerate color map in this new order
cmap = plt.cm.tab10(np.linspace(0, 1, len(algos_present)))

# ───────────────────────── figure 1: curves ───────────────────────
fig1, ax1 = plt.subplots(figsize=(3, 3))
for color, algo in zip(cmap, algos_present):
    med, q1, q3, n = stat_algo[algo]
    ep = np.arange(1, len(med) + 1)
    ax1.plot(ep, med, "-", color=color, lw=1.4, label=f"{algo} (n={n})")
    ax1.fill_between(ep, q1, q3, color=color, alpha=0.25)

ax1.set_xlim(1, 20)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Median Cumulative Reward")
#ax1.set_title("Cumulative Reward (median ± IQR)")
ax1.grid(True, axis="y")
fig1.legend(*ax1.get_legend_handles_labels(), fontsize=6,
            loc="upper center", bbox_to_anchor=(0.3, 1.2))
fig1.tight_layout()
plt.savefig("plots_final/cumreward_median_IQR_by_algo.pdf", bbox_inches="tight")
plt.show()

# ───────────── figure 2: box‑plot final per algorithm ─────────────
fig2, ax2 = plt.subplots(figsize=(3, 3))
box_data = [[c[-1] for c in group_algo[a]] for a in algos_present]
bp = ax2.boxplot(
    box_data, widths=0.6, patch_artist=True, showmeans=True,
    meanprops=dict(marker="o", markerfacecolor="k", markersize=3, markeredgewidth=0),
)
for patch, color in zip(bp["boxes"], cmap):
    patch.set_facecolor(color); patch.set_alpha(0.5)
    patch.set_edgecolor("k"); patch.set_linewidth(0.8)

ax2.set_xticks(range(1, len(algos_present) + 1))
ax2.set_xticklabels(algos_present, rotation=20, ha="right")
ax2.set_ylabel("Final Cumulative Reward")
#ax2.set_title("Final Reward Distribution per Algorithm")
ax2.grid(axis="y", linestyle="--", alpha=0.3)
fig2.tight_layout()
plt.savefig("plots_final/final_reward_boxplot_by_algo.pdf")
plt.show()

# ───────── gather final‑reward samples per degree × algorithm ─────
deg_algo_vals = defaultdict(lambda: defaultdict(list))
for r in raw_runs:
    deg_algo_vals[r["degree"]][r["algo"]].append(r["curve"][-1])

# ─── Sort degrees by overall mean final reward (desc) ─────────────
degree_means = {
    deg: np.mean([
        v for algo_vals in deg_algo_vals[deg].values() for v in algo_vals
    ])
    for deg in deg_algo_vals
}
degrees_present = sorted(
    degree_means.keys(),
    key=lambda d: degree_means[d],
    reverse=True
)

bar_width = 0.8 / max(1, len(algos_present))

# ─────────── figure 3: mean ± std bar‑chart ───────────────────────
x_centres = np.arange(len(degrees_present))
fig3, ax3 = plt.subplots(figsize=(3, 3))
for j, (algo, color) in enumerate(zip(algos_present, cmap)):
    for i, deg in enumerate(degrees_present):
        vals = deg_algo_vals[deg].get(algo, [])
        if not vals:
            continue
        mean = np.mean(vals)
        std  = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        x = x_centres[i] + (j - (len(algos_present)-1)/2) * bar_width
        ax3.bar(
            x, mean, yerr=std, capsize=2, width=bar_width,
            color=color, alpha=0.7, error_kw=dict(elinewidth=0.8),
        )

ax3.set_xticks(x_centres)
ax3.set_xticklabels([f"{d:g}" for d in degrees_present])
ax3.set_xlabel("degree")
ax3.set_ylabel("Mean Final Reward ± 1 SD")
#ax3.set_title("Mean Final Reward by degree & algorithm")
ax3.grid(axis="y", linestyle="--", alpha=0.3)
ax3.legend(
    [plt.Line2D([0],[0], color=c, lw=6) for c in cmap],
    algos_present, ncol=2, fontsize=5,
    loc="upper center", bbox_to_anchor=(0.5, 1.2)
    #loc="best"
)
fig3.tight_layout()
plt.savefig("plots_final/final_reward_bar_mean_std_by_deg_algo.pdf", bbox_inches="tight")
plt.show()

# ─────────── figure 4: one box‑plot per degree ────────────────────
for deg in degrees_present:
    data, labels, colors = [], [], []
    for algo, color in zip(algos_present, cmap):
        vals = deg_algo_vals[deg].get(algo, [])
        if vals:
            data.append(vals)
            labels.append(algo)
            colors.append(color)
    if not data:
        continue

    fig_d, ax_d = plt.subplots(figsize=(3,3))
    bp_d = ax_d.boxplot(
        data, widths=0.6, patch_artist=True, showmeans=True,
        meanprops=dict(marker="o", markerfacecolor="k",
                       markersize=3, markeredgewidth=0),
    )
    for patch, col in zip(bp_d["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.5)
        patch.set_edgecolor("k"); patch.set_linewidth(0.8)

    ax_d.set_xticks(range(1, len(labels)+1))
    ax_d.set_xticklabels(labels, rotation=20, ha="right")
    ax_d.set_ylabel("Final Cumulative Reward")
    #ax_d.set_title(f"Final Reward Distribution – degree {deg:g}")
    ax_d.grid(axis="y", linestyle="--", alpha=0.3)
    fig_d.tight_layout()
    fname = f"plots_final/final_reward_box_deg_{str(deg).replace('.', 'p')}.pdf"
    plt.savefig(fname)
    plt.show()
