#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DARC – denoiser ON vs OFF
• Figure 1: median curve with inter‑quartile shading
• Figure 2: box‑plot of final cumulative rewards
"""

import os, re, math, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import wilcoxon, ttest_rel

# ─── add once, near other imports ─────────────────────────────────
import tikzplotlib as tpl

# ───────────────────────── settings ───────────────────────────────
folder_path   = "/home/cubos98/Desktop/MA/DARAIL/runs/runs_DARC_denoiser_vs"
tag           = "Target Env/Rewards"           # DARC reward tag
drop_first_k  = 0                              # skip first k episodes
base_regex    = re.compile(
    r'(?=.*noise_cfrs_0\.0)(?=.*noise_0\.2)(?=.*bias_0\.5)(?=.*DARC)',
    re.IGNORECASE,
)

deg_pat = re.compile(r'degree_([-+]?\d*\.?\d+)', re.IGNORECASE)
seed_pat = re.compile(r'seed_(\d+)', re.IGNORECASE)
den_pat  = re.compile(r'use_denoiser_(0|1)', re.IGNORECASE)      # 0=OFF, 1=ON

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ──────────────────────── read runs ───────────────────────────────
raw_runs = []        #  {degree, seed, den, curve}
for root, _, files in os.walk(folder_path):
    name = os.path.basename(root)
    if not base_regex.search(name):
        continue
    m_deg, m_seed, m_den = deg_pat.search(name), seed_pat.search(name), den_pat.search(name)
    if not (m_deg and m_seed and m_den):
        continue
    degree = float(m_deg.group(1))
    seed   = int(m_seed.group(1))
    den    = int(m_den.group(1))            # 0 or 1

    steps, vals = [], []
    for ev in [f for f in files if "tfevents" in f]:
        for e in tf.compat.v1.train.summary_iterator(os.path.join(root, ev)):
            if not e.summary:
                continue
            for v in e.summary.value:
                if v.tag == tag and hasattr(v, "simple_value"):
                    steps.append(e.step)
                    vals.append(v.simple_value)
    if len(steps) <= drop_first_k:
        continue
    idx = np.argsort(steps)
    curve = np.cumsum(np.array(vals)[idx][drop_first_k:])
    raw_runs.append({"degree": degree, "seed": seed, "den": den, "curve": curve})

if not raw_runs:
    raise RuntimeError("No runs matched the filter.")

# ──────────────────────── organise data ───────────────────────────
group = defaultdict(lambda: defaultdict(list))     # degree → den → list[curves]
max_len = 0
for r in raw_runs:
    group[r["degree"]][r["den"]].append(r["curve"])
    max_len = max(max_len, len(r["curve"]))

def pad(curve):          # right‑pad with NaN to max_len
    return np.pad(curve, (0, max_len-len(curve)), constant_values=np.nan)

stat = {}                # degree → den → (median, q1, q3, n)
for deg, ddict in group.items():
    stat[deg] = {}
    for den, curves in ddict.items():
        stack = np.vstack([pad(c) for c in curves])
        med = np.nanmedian(stack, axis=0)
        q1  = np.nanpercentile(stack, 25, axis=0)
        q3  = np.nanpercentile(stack, 75, axis=0)
        stat[deg][den] = (med, q1, q3, len(curves))

degrees = sorted(stat.keys())
colmap  = plt.cm.viridis(np.linspace(0, 1, len(degrees)))

# ──────────────────────── figure 1: curves ────────────────────────
fig, ax = plt.subplots(figsize=(3, 3))
for c, deg in zip(colmap, degrees):
    for den, ls in ((0, "--"), (1, "-")):
        if den not in stat[deg]:
            continue
        med, q1, q3, n = stat[deg][den]
        ep = np.arange(1, len(med)+1)
        ax.plot(ep, med, ls, color=c, linewidth=1.2,
                label=f"deg={deg}, {'ON' if den else 'OFF'} (n={n})")
        ax.fill_between(ep, q1, q3, color=c, alpha=0.25)
ax.set_xlabel("Episode")
ax.set_ylabel("Median Cumulative Reward")
#ax.set_title(f"Cumulative Reward (median ± IQR, skip first {drop_first_k})")
ax.grid(True, axis="y")
fig.legend(*ax.get_legend_handles_labels(), ncol=2,
           bbox_to_anchor=(0.5, 1.02), loc="upper center")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/cumreward_median_IQR.png")
tpl.save("plots/cumreward_median_IQR.tex")
plt.show()

# ─────────────────── figure 2: box‑plot final ─────────────────────
from matplotlib.patches import Patch

#fig2, ax2 = plt.subplots(figsize=(6.2, 2.8))
fig2, ax2 = plt.subplots(figsize=(3, 3))

box_data, box_pos, box_props = [], [], []   # props: (facecolor, hatch)
width = 0.35
for i, (deg, c) in enumerate(zip(degrees, colmap)):
    for j, den in enumerate((0, 1)):        # 0 = OFF, 1 = ON
        finals = [cur[-1] for cur in group[deg][den]]
        if not finals:
            continue
        box_data.append(finals)             # plain Python list
        box_pos.append(i + (j-0.5)*width)
        face = c
        hatch = "/" if den == 0 else ""    # hatched = OFF, solid = ON
        box_props.append((face, hatch))

# all boxes must have the same length → trim to min group size
m = min(len(lst) for lst in box_data)
box_data = [lst[:m] for lst in box_data]

bp = ax2.boxplot(
    box_data,
    positions=box_pos,
    widths=width*0.8,
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker="o", markerfacecolor="k",
                   markersize=3, markeredgewidth=0),
)

# colour & hatch each box according to our list
for patch, (face, hatch) in zip(bp["boxes"], box_props):
    patch.set_facecolor(face)
    patch.set_alpha(0.5)
    patch.set_hatch(hatch)
    patch.set_edgecolor("k")
    patch.set_linewidth(0.8)

ax2.set_xticks(range(len(degrees)))
ax2.set_xticklabels([str(d) for d in degrees])
ax2.set_xlabel("degree")
ax2.set_ylabel("Final Cumulative Reward")
#ax2.set_title("Final Reward Distribution")
ax2.grid(axis="y", linestyle="--", alpha=0.3)

# explicit legend handles
legend_handles = [
    Patch(facecolor="white", edgecolor="k", hatch="/", label="denoiser OFF"),
    Patch(facecolor="white", edgecolor="k", hatch="",   label="denoiser ON")
]
ax2.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=6)

fig2.tight_layout()
plt.savefig("plots/final_reward_boxplot.png")
tpl.save("plots/final_reward_boxplot.tex")
plt.show()


