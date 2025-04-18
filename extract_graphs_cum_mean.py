import os, re, math, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from collections import defaultdict

# ------------- SETTINGS (edit as needed) -----------------
folder_path   = "/home/cubos98/Desktop/MA/DARAIL/runs/runs6"
tags          = ["Target Env/Rewards"]
drop_first_k  = 2                          # skip first k episodes
base_regex    = re.compile(
    r'(?=.*noise_cfrs_0\.0)(?=.*noise_0\.2)(?=.*bias_0\.5)(?=.*DARC)', re.IGNORECASE
)
# ---------------------------------------------------------

deg_pat   = re.compile(r'degree_([-+]?\d*\.?\d+)')
seed_pat  = re.compile(r'seed_(\d+)')
den_pat   = re.compile(r'use_denoiser_(0|1)')

# ---------- READ TENSORBOARD EVENT FILES ----------
raw_runs = []   # list of dicts {deg, seed, denoiser, tag -> arrays}

for root, _, files in os.walk(folder_path):
    fname = os.path.basename(root)
    if not base_regex.search(fname):
        continue

    m_deg  = deg_pat.search(fname)
    m_seed = seed_pat.search(fname)
    m_den  = den_pat.search(fname)
    if not (m_deg and m_seed and m_den):
        continue

    degree   = float(m_deg.group(1))
    seed     = int(m_seed.group(1))
    den_flag = int(m_den.group(1))         # 0 = off, 1 = on

    event_files = [f for f in files if "tfevents" in f]
    if not event_files:
        continue

    run = {"degree": degree, "seed": seed, "den": den_flag,
           "metrics": {t: {"s": [], "v": []} for t in tags}}

    for f in event_files:
        for ev in tf.compat.v1.train.summary_iterator(os.path.join(root, f)):
            if not ev.summary:
                continue
            for v in ev.summary.value:
                if v.tag in tags and hasattr(v, "simple_value"):
                    run["metrics"][v.tag]["s"].append(ev.step)
                    run["metrics"][v.tag]["v"].append(v.simple_value)
    raw_runs.append(run)

if not raw_runs:
    raise RuntimeError("No matching runs found under given criteria.")

# -------- COLLECT CUMULATIVE‑REWARD CURVES --------
# by_tag[tag][degree][den] -> list of cumulative arrays (one per seed)
by_tag = {t: defaultdict(lambda: defaultdict(list)) for t in tags}
max_len = {t: 0 for t in tags}

for run in raw_runs:
    d, den = run["degree"], run["den"]
    for tag in tags:
        s   = np.array(run["metrics"][tag]["s"])
        v   = np.array(run["metrics"][tag]["v"])
        if len(s) <= drop_first_k:
            continue
        idx = np.argsort(s)
        c   = np.cumsum(v[idx][drop_first_k:])       # skip first k episodes
        by_tag[tag][d][den].append(c)
        max_len[tag] = max(max_len[tag], len(c))

# -------- MEAN ± STD per episode per group --------
stat_curves = {t: {} for t in tags}
for tag in tags:
    for deg, den_dict in by_tag[tag].items():
        for den, runs in den_dict.items():
            # pad shorter with NaN for stacking
            padded = [np.pad(r, (0, max_len[tag]-len(r)), constant_values=np.nan)
                      for r in runs]
            arr  = np.vstack(padded)
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr,  axis=0)
            stat_curves[tag].setdefault(deg, {})[den] = (mean, std, len(runs))

# ---------------- PLOT LINES ----------------------
plt.rcParams.update({
    "font.family":"serif", "font.size":6,
    "grid.linestyle":"--", "grid.alpha":0.3,
    "figure.dpi":300, "savefig.dpi":300,
})
ncols = 2
nrows = math.ceil(len(tags)/ncols)
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(3.8*ncols, 3*nrows), sharex=False)
axes = axes.flatten()

degrees_sorted = sorted(stat_curves[tags[0]].keys())
colmap = plt.cm.viridis(np.linspace(0,1,len(degrees_sorted)))

for j, tag in enumerate(tags):
    ax = axes[j]
    for color, deg in zip(colmap, degrees_sorted):
        for den in (0,1):
            if den not in stat_curves[tag][deg]:
                continue
            mean, std, N = stat_curves[tag][deg][den]
            ep = np.arange(1, len(mean)+1)
            style = "-" if den else "--"        # solid = on, dashed = off
            label = (f"deg={deg}, "
                     f"{'denoiser ON' if den else 'denoiser OFF'} (n={N})")
            ax.plot(ep, mean, style, color=color, label=label, linewidth=1)
            ax.fill_between(ep, mean-std, mean+std,
                            color=color, alpha=0.25, linewidth=0)
    ax.set_title(tag, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Cumulative Reward")
    ax.grid(True, axis="y")

for k in range(len(tags), len(axes)):
    axes[k].axis("off")

fig.suptitle("Mean ± 1σ Cumulative Reward\n"
             f"(first {drop_first_k} episodes skipped; denoiser effect)",
             y=0.99, fontsize=10, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93])
# place one outside legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right",
           bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
plt.savefig("cumreward_denoiser_lines.pdf", bbox_inches="tight")
plt.show()

# ------------- BAR CHART: final totals -------------
fig2, ax2 = plt.subplots(figsize=(4.5,3))
bar_w = 0.35
x_ticks, x_labels = [], []
off_means, on_means = [], []
off_errs,  on_errs  = [], []

for i, deg in enumerate(degrees_sorted):
    for den in (0,1):
        if den not in stat_curves[tags[0]][deg]:
            continue
        mean, std, _ = stat_curves[tags[0]][deg][den]
        if den==0:
            off_means.append(mean[-1]); off_errs.append(std[-1])
        else:
            on_means.append(mean[-1]);  on_errs.append(std[-1])
    x_ticks.append(i)
    x_labels.append(str(deg))

# bars side by side
x = np.array(x_ticks)
ax2.bar(x-bar_w/2, off_means, bar_w, yerr=off_errs,
        label="denoiser OFF", capsize=3)
ax2.bar(x+bar_w/2, on_means,  bar_w, yerr=on_errs,
        label="denoiser ON",  capsize=3)

ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.set_xlabel("degree")
ax2.set_ylabel("Mean Final Cumulative Reward")
ax2.set_title("Final Cumulative Reward per Degree\n(mean ± 1σ across seeds)")
ax2.grid(axis="y", linestyle="--", alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.savefig("cumreward_denoiser_bar.pdf")
plt.show()


# ==============================================================
#  STATISTICAL SIGNIFICANCE: denoiser ON  vs OFF  (per degree)
# ==============================================================

# -------- Power analysis (paired t‑test approximation) ----------
#from statsmodels.stats.power import TTestPower
#analysis = TTestPower()
#needed_n = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)
#print(f"You need about {int(np.ceil(needed_n))} seeds for 80% power.")

# -------- One‑sided Wilcoxon in your print block -------------#--
#stat, p = wilcoxon(on_vals, off_vals, alternative="greater")

from scipy.stats import wilcoxon, ttest_rel
import itertools

def cliffs_delta(x, y):
    """Compute Cliff's Δ for two paired lists."""
    n = len(x)
    gt = sum((xi > yi) for xi, yi in zip(x, y))
    lt = sum((xi < yi) for xi, yi in zip(x, y))
    return (gt - lt) / n

print("\n===== Statistical comparison: denoiser ON vs OFF =====")
tag_for_test = tags[0]   # choose the metric you care about

for deg in degrees_sorted:
    # --- collect final cumulative reward for each seed ---
    off_vals, on_vals = [], []
    for run in raw_runs:
        if run["degree"] != deg:
            continue
        final_val = np.cumsum(
            np.array(run["metrics"][tag_for_test]["v"])[
                np.argsort(run["metrics"][tag_for_test]["s"])
            ][drop_first_k:]
        )[-1]  # last value of cum‑sum
        if run["den"] == 0:
            off_vals.append(final_val)
        else:
            on_vals.append(final_val)

    # ensure both lists are aligned by seed order
    # (they are if runs used identical seeds)
    if len(off_vals) != len(on_vals) or len(off_vals) == 0:
        print(f"degree {deg}: unmatched seed counts; skipping.")
        continue

    # --- choose test ---
    if len(off_vals) < 20:
        stat, p = wilcoxon(on_vals, off_vals, alternative="greater")
        test_name = "Wilcoxon signed‑rank"
    else:
        stat, p = ttest_rel(on_vals, off_vals, alternative="two-sided")
        test_name = "Paired t‑test"

    # --- effect size (Cliff's Δ) ---
    delta = cliffs_delta(on_vals, off_vals)

    # --- interpretation helper ---
    sig = "YES" if p < 0.05 else "NO "

    print(f"degree {deg:>4}:  {test_name:17s}  p = {p:.4f}   "
          f"Cliff's Δ = {delta:+.3f}   significant? {sig}")

print("=======================================================\n")
