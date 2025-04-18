import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------
#  SETTINGS & CONFIGURATION
# --------------------------
folder_path = "/home/cubos98/Desktop/MA/DARAIL/runs/runs6"      # <<<--- Update to your folder path
tags         = ["Env/Rewards", "Target Env/Rewards"]
run_filter_regex = re.compile(r'(?=.*noise_cfrs_0.0)(?=.*degree_0.5)(?=.*noise_0.2)(?=.*bias_0.5)(?=.*seed_45)', re.IGNORECASE)

smoothing_window = 0                                           # moving‑average window

# --------------------------
#  MATPLOTLIB PUBLICATION SETTINGS
# --------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 5,
    "axes.titlesize": 5,
    "axes.labelsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "xtick.major.pad": 1,
    "ytick.major.pad": 1,
})

# --------------------------
#  HELPER FUNCTION
# --------------------------
def smooth_signal(x, y, k):
    if len(y) < k:
        return np.array(x), np.array(y)
    kernel = np.ones(k) / k
    y_sm = np.convolve(y, kernel, mode="valid")
    x_sm = np.convolve(x, kernel, mode="valid")
    return x_sm, y_sm

# --------------------------
#  READ EVENT FILES
# --------------------------
runs_data = {}
for root, _, files in os.walk(folder_path):
    folder_name = os.path.basename(root)
    if not run_filter_regex.search(folder_name):
        continue

    for f in files:
        if "tfevents" not in f:
            continue
        full_path = os.path.join(root, f)
        run_label = folder_name
        runs_data[run_label] = {tag: {"steps": [], "values": []} for tag in tags}

        print(f"Processing {run_label}: {full_path}")
        try:
            for e in tf.compat.v1.train.summary_iterator(full_path):
                if not e.summary:
                    continue
                for v in e.summary.value:
                    if v.tag in tags and hasattr(v, "simple_value"):
                        runs_data[run_label][v.tag]["steps"].append(e.step)
                        runs_data[run_label][v.tag]["values"].append(v.simple_value)
        except Exception as exc:
            print(f"Error reading {full_path}: {exc}")

if not runs_data:
    raise RuntimeError("No matching event files found.")

# --------------------------
#  PLOT CUMULATED REWARDS (SKIPPING FIRST TWO VALUES)
# --------------------------
num_tags = len(tags)
ncols    = 2
nrows    = (num_tags + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes      = axes.flatten()

global_handles = {}

for i, tag in enumerate(tags):
    ax = axes[i]
    for run_label, data in runs_data.items():
        steps  = np.array(data[tag]["steps"])
        values = np.array(data[tag]["values"])

        if len(steps) <= 2:          # nothing left after skipping
            continue

        # sort by step and **drop the first two items**
        sort_idx   = np.argsort(steps)
        steps      = steps[sort_idx][2:]
        values     = values[sort_idx][2:]

        # cumulative sum
        cum_values = np.cumsum(values)

        # optional smoothing
        if smoothing_window > 1 and len(cum_values) >= smoothing_window:
            s_steps, s_vals = smooth_signal(steps, cum_values, smoothing_window)
            line, = ax.plot(s_steps, s_vals, linewidth=1.0)
        else:
            line, = ax.plot(steps, cum_values, linewidth=1.0)

        global_handles[run_label] = line

    ax.set_title(tag, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cumulative Reward (first two points skipped)")
    ax.grid(True)
    ax.tick_params(axis='both', which='major', pad=1)

# hide unused axes
for j in range(num_tags, len(axes)):
    axes[j].axis('off')

fig.suptitle("Cumulative Training Rewards (ignoring first two values)", fontsize=16,
             fontweight="bold", y=0.95)

handles = list(global_handles.values())
labels  = list(global_handles.keys())
fig.legend(handles, labels, title="Run", loc='lower center',
           ncol=min(1, len(labels)), bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.85])
plt.subplots_adjust(wspace=0.3, hspace=0.8, bottom=0.4)

#plt.savefig("cumulative_reward_skip2.pdf")
print(labels[2], "->", handles[2].get_ydata()[-1])
print(labels[3], "->", handles[3].get_ydata()[-1])
#plt.show()

