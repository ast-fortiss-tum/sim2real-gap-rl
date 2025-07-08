#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare DARC vs. SAC on cumulative reward

Outputs:
  • cumreward_darc_vs_sac_lines.pdf
  • cumreward_darc_vs_sac_bar.pdf
  • console Wilcoxon/t‑test summary
"""

import os, re, math, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import wilcoxon, ttest_rel
import time

""" This script extracts training metrics from TensorBoard event files and plots them.
It supports exponential smoothing similar to TensorBoard and creates a global legend
for multiple runs. The script is designed to work with a specific folder structure
and regex filter to select relevant runs based on their names."""

# ------------------------------------------------------------------ #
# 1) Ensure eager execution & regex‑based filters                     #
# ------------------------------------------------------------------ #
tf.compat.v1.enable_eager_execution()          # harmless if already enabled

folder_path = "/home/cubos98/Desktop/MA/DARAIL/runs/runs_Magnum"
#folder_path  = "/home/cubos98/Desktop/MA/DARAIL/runs/runs6"
drop_first_k = 0   # skip first k samples/episodes
base_regex   = re.compile(
    r'(?=.*noise_cfrs_0\.0)(?=.*noise_0\.2)(?=.*bias_0\.5)(?=.*use_denoiser_1)',
    re.IGNORECASE,
)

deg_pat  = re.compile(r'degree_([-+]?\d*\.?\d+)', re.IGNORECASE)
seed_pat = re.compile(r'seed_(\d+)',              re.IGNORECASE)
alg_pat  = re.compile(r'(DARC_ON|ContSAC_STR)',   re.IGNORECASE)

# ------------------------------------------------------------------ #
# 2) Load TF‑event files with TFRecordDataset (eager)                #
# ------------------------------------------------------------------ #

raw_runs = []            # {degree, seed, algo, cum}
for root, _, files in os.walk(folder_path):
    name = os.path.basename(root)
    if not base_regex.search(name):
        continue

    m_deg, m_seed, m_alg = deg_pat.search(name), seed_pat.search(name), alg_pat.search(name)
    if not (m_deg and m_seed and m_alg):
        continue
    degree  = float(m_deg.group(1))
    seed    = int(m_seed.group(1))
    algo    = m_alg.group(1)                       # SAC or DARC

    tag_key = "Target/Env/Rewards" if algo == "ContSAC_STR" else "Target Env/Rewards"

    steps, vals = [], []
    for evfile in [f for f in files if "tfevents" in f]:
        ds = tf.data.TFRecordDataset(os.path.join(root, evfile))
        for rec in ds:
            ev = tf.compat.v1.Event()
            ev.ParseFromString(rec.numpy())
            for v in ev.summary.value:
                if v.tag == tag_key and v.HasField("simple_value"):
                    steps.append(ev.step)
                    vals.append(v.simple_value)

    if len(steps) <= drop_first_k:
        continue
    idx  = np.argsort(steps)
    if algo == "ContSAC_STR":
        # SAC: steps are already sorted
        cum  = np.cumsum(np.array(vals)[idx][drop_first_k:])
    else:
        cum  = np.cumsum(np.array(vals)[idx][drop_first_k:-1])
    raw_runs.append({"degree": degree, "seed": seed, "algo": algo, "cum": cum})

#print(raw_runs)

if not raw_runs:
    raise RuntimeError("No matching runs found.")

# ------------------------------------------------------------------ #
# 3) Group curves & compute mean ± std safely                        #
# ------------------------------------------------------------------ #
group = defaultdict(lambda: defaultdict(list))     # degree -> algo -> [curves]
max_len = 0
for r in raw_runs:
    group[r["degree"]][r["algo"]].append(r["cum"])
    max_len = max(max_len, len(r["cum"]))

def safe_mean_std(stack: np.ndarray):
    """
    Return episode‑wise mean and std without triggering the
    “mean of empty slice / DOF <= 0” warnings.

    Parameters
    ----------
    stack : ndarray, shape = [n_runs, max_len]
        Each row is a cumulative‑reward curve padded with NaNs.

    Returns
    -------
    mean, std : ndarrays, both length = max_len
        NaN where no run has data for that episode.
    """
    if stack.ndim != 2:
        raise ValueError("stack must be 2‑D")

    n_cols = stack.shape[1]
    mean   = np.full(n_cols, np.nan, dtype=float)
    std    = np.full(n_cols, np.nan, dtype=float)

    # columns that have at least one non‑NaN value
    valid_cols = ~np.all(np.isnan(stack), axis=0)
    #print(f"valid_cols = {valid_cols}")
    if valid_cols.any():
        mean[valid_cols] = np.nanmean(stack[:, valid_cols], axis=0)
        print(f"mean = {mean}")
        std [valid_cols] = np.nanstd (stack[:, valid_cols], axis=0)

    return mean, std

stats = {}                 
#print(group.items())

               # degree -> algo -> (mean, std, n)
for deg, adict in group.items():
    #print(f"degree {deg}: {adict.keys()}")
    stats[deg] = {}
    for alg, curves in adict.items():
        #print(f"  {alg}: {len(curves)} runs")
        padded = [np.pad(c, (0, max_len-len(c)), constant_values=np.nan)
                  for c in curves]
        arr = np.vstack(padded)
        #print(arr)

        mean, std = safe_mean_std(arr)
        stats[deg][alg] = (mean, std, len(curves))
        #print("stats alg: ", stats[deg][alg])

# drop degrees missing either algorithm
degrees = sorted([d for d in stats if {"ContSAC_STR","DARC_ON"} <= stats[d].keys()])
if not degrees:
    raise RuntimeError("No degree has both SAC and DARC runs.")

# ------------------------------------------------------------------ #
# 4) FIG 1 – learning curves (solid=DARC, dashed=SAC)                #
# ------------------------------------------------------------------ #
plt.rcParams.update({
    "font.family":"serif", "font.size":6,
    "grid.linestyle":"--", "grid.alpha":0.3,
    "figure.dpi":300, "savefig.dpi":300,
})
fig, ax = plt.subplots(figsize=(4.8,3.3))
colmap = plt.cm.viridis(np.linspace(0,1,len(degrees)))

for color, deg in zip(colmap, degrees):
    for alg, style in (("ContSAC_STR","--"), ("DARC_ON","-")):
        mean, std, n = stats[deg][alg]
        ep = np.arange(1, len(mean)+1)
        label = f"deg={deg}, {alg} (n={n})"
        ax.plot(ep, mean, style, color=color, label=label, linewidth=1)
        ax.fill_between(ep, mean-std, mean+std, color=color, alpha=0.25)

ax.set_xlabel("Episode")
ax.set_ylabel("Mean Cumulative Reward")
ax.set_title(f"Mean ±1σ Cumulative Reward (skip first {drop_first_k})")
ax.grid(True, axis="y")
fig.legend(*ax.get_legend_handles_labels(), loc="center right",
           bbox_to_anchor=(1.02,0.5))
fig.tight_layout(rect=[0,0,0.95,1])
plt.savefig("plots/cumreward_darc_vs_sac_lines.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------ #
# 5) FIG 2 – final reward bar chart                                  #
# ------------------------------------------------------------------ #
bar_w = 0.35
sac_m, darc_m, sac_e, darc_e = [], [], [], []
for deg in degrees:
    sac_m.append (stats[deg]["ContSAC_STR"][0][-1]); darc_m.append(stats[deg]["DARC_ON"][0][-1])
    sac_e.append (stats[deg]["ContSAC_STR"][1][-1]); darc_e.append(stats[deg]["DARC_ON"][1][-1])
x = np.arange(len(degrees))
fig2, ax2 = plt.subplots(figsize=(4.8,3))
ax2.bar(x-bar_w/2, sac_m,  bar_w, yerr=sac_e,  label=r"SAC$_{\mathrm{PFT}}$",  capsize=3)
ax2.bar(x+bar_w/2, darc_m, bar_w, yerr=darc_e, label="DARC", capsize=3)
ax2.set_xticks(x); ax2.set_xticklabels([str(d) for d in degrees])
ax2.set_xlabel("degree"); ax2.set_ylabel("Mean Final Cumulative Reward")
ax2.set_title("Final Cumulative Reward per Degree (mean ±1σ)")
ax2.grid(axis="y", linestyle="--", alpha=0.3); ax2.legend()
plt.tight_layout(); plt.savefig("plots/cumreward_darc_vs_sac_bar.pdf"); plt.show()

print("Difference improvement 0.5: ",(darc_m[0]-sac_m[0])/sac_m[0])
print("Difference improvement 0.65: ",(darc_m[1]-sac_m[1])/sac_m[1])
print("Difference improvement 0.8: ",(darc_m[2]-sac_m[2])/sac_m[2])
# ------------------------------------------------------------------ #
# 6) STATISTICAL TESTS  (paired by seed)                             #
# ------------------------------------------------------------------ #
def cliffs_delta(a, b):
    gt = sum(x>y for x,y in zip(a,b)); lt = sum(x<y for x,y in zip(a,b))
    return (gt-lt)/len(a)

print("\n=== DARC vs SAC (paired on seed, metric = final cumulative reward) ===")
for deg in degrees:
    sac = {r["seed"]: r["cum"][-1] for r in raw_runs
           if r["degree"]==deg and r["algo"]=="ContSAC_STR"}
    dar = {r["seed"]: r["cum"][-1] for r in raw_runs
           if r["degree"]==deg and r["algo"]=="DARC_ON"}
    seeds = sorted(set(sac) & set(dar))
    sac_v = [sac[s] for s in seeds]; dar_v = [dar[s] for s in seeds]
    if len(seeds) < 20:
        stat, p = wilcoxon(dar_v, sac_v,alternative="greater")
        tname = "Wilcoxon"
    else:
        stat, p = ttest_rel(dar_v, sac_v)
        tname = "paired t"
    delta = cliffs_delta(dar_v, sac_v)
    sig = "YES" if p < 0.05 else "NO "
    print(f"degree {deg:>4}: {tname:10s}  p={p:.4f}  Δ={delta:+.3f}  significant? {sig}")
print("======================================================================\n")
