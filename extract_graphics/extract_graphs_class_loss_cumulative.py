import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import collections
import scipy.interpolate

from cycler import cycler

"""
SCRIPT FOR SUMMARIZE TENSORBOARD RUNS AND PLOT THEM (Mean ± STD per Degree Across Seeds)
This script extracts data from TensorBoard event files, applies smoothing,
and plots the mean and variance across seeds per degree, with a global legend.
It supports filtering runs based on a regex pattern and allows for
customizing the appearance of the plots, including color-blind friendly palettes.
"""

# Set color-blind friendly palette (Okabe–Ito palette)
plt.rcParams["axes.prop_cycle"] = cycler(color=[
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#D55E00",  # Vermillion
    "#CC79A7"   # Reddish Purple
])

# --------------------------
#  SETTINGS & CONFIGURATION
# --------------------------
folder_path = "/home/cubos98/Desktop/MA/DARAIL/runs/runs_Magnum"  # Update to your folder path
tags = ["Env/Rewards", "Target Env/Rewards", "Loss/Classify Loss"]

run_filter_regex = re.compile(r'(?=.*noise_cfrs_0.0)(?=.*degree_)(?=.*v_noise_0.2_bias_0.5)(?=.*use_denoiser_1)(?=.*seed_)(?=.*_ON_)', re.IGNORECASE)
#run_filter_regex = re.compile(r'(?=.*degree_0.8)', re.IGNORECASE)

# Smoothing factor for TensorBoard-style exponential smoothing (0 = no smoothing)
tensorboard_smoothing = 0.7

# Editable subtitles for each subplot (change these values as desired)
subtitle_map = {
    "Env/Rewards": "SAC Rewards (20 iteractions)",
    "Target Env/Rewards": "DARC Rewards (20 iteractions with Target)",
    "Loss/Classify Loss": "Classification Loss of DARC",
}

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
    "ytick.major.pad": 1
})

# --------------------------
#  HELPER FUNCTION FOR TENSORBOARD-STYLE EXPONENTIAL SMOOTHING
# --------------------------
def smooth_signal(steps, values, smoothing):
    if smoothing == 0 or len(values) == 0:
        return steps, values
    smoothed_values = [values[0]]
    for value in values[1:]:
        smoothed_values.append(smoothed_values[-1] * smoothing + (1 - smoothing) * value)
    return steps, smoothed_values

# --------------------------
#  EXTRACT DATA FROM EVENT FILES
# --------------------------
runs_data = {}
run_counter = 0

for root, dirs, files in os.walk(folder_path):
    folder_name = os.path.basename(root)
    if not run_filter_regex.search(folder_name):
        continue

    for file in files:
        if "tfevents" in file:
            run_counter += 1
            run_label = f"{folder_name}"
            runs_data[run_label] = {tag: {"steps": [], "values": []} for tag in tags}
            event_file_path = os.path.join(root, file)
            print(f"Processing {run_label}: {event_file_path}")
            try:
                for event in tf.compat.v1.train.summary_iterator(event_file_path):
                    if event.summary:
                        for value in event.summary.value:
                            if (value.tag in tags and 
                                hasattr(value, "simple_value") and 
                                value.simple_value is not None):
                                runs_data[run_label][value.tag]["steps"].append(event.step)
                                runs_data[run_label][value.tag]["values"].append(value.simple_value)
            except Exception as e:
                print(f"Error processing {event_file_path}: {e}")

if run_counter == 0:
    print("No matching event files found. Please check your folder path and regex filter.")
    exit(1)

# --------------------------
#  GROUPING: GROUP RUNS BY DEGREE
# --------------------------
def extract_degree(label):
    match = re.search(r'degree_([0-9\.]+)', label)
    return match.group(1) if match else None

def group_by_degree(runs_data):
    groups = collections.defaultdict(list)
    for run_label in runs_data.keys():
        degree = extract_degree(run_label)
        if degree is not None:
            groups[degree].append(run_label)
    return groups

groups = group_by_degree(runs_data)

# --------------------------
#  PLOTTING (MEAN ± STD PER DEGREE)
# --------------------------
num_tags = len(tags)
ncols = 3
nrows = (num_tags + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

fig.suptitle("Training Metrics", fontsize=16, fontweight="bold", y=0.95)

# Unique color for each degree group
color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

for i, tag in enumerate(tags):
    ax = axes[i]
    # For consistent color per degree across tags:
    degree_to_color = {}
    for degree_idx, (degree, run_labels) in enumerate(sorted(groups.items(), key=lambda x: float(x[0]))):
        all_steps = []
        all_values = []

        for run_label in run_labels:
            steps = np.array(runs_data[run_label][tag]["steps"])
            values = np.array(runs_data[run_label][tag]["values"])
            if len(steps) == 0:
                continue
            _, smoothed_values = smooth_signal(steps, values, tensorboard_smoothing)
            all_steps.append(steps)
            all_values.append(np.array(smoothed_values))

        if not all_steps or len(all_steps) < 1:
            continue

        # Find common range for interpolation
        min_step = max(min(steps) for steps in all_steps)
        max_step = min(max(steps) for steps in all_steps)
        if min_step >= max_step:
            continue
        common_steps = np.linspace(min_step, max_step, 300)

        interpolated = []
        for steps, values in zip(all_steps, all_values):
            if len(steps) < 2:
                continue
            f = scipy.interpolate.interp1d(steps, values, kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolated.append(f(common_steps))
        interpolated = np.array(interpolated)
        if interpolated.shape[0] == 0:
            continue

        mean = np.mean(interpolated, axis=0)
        std = np.std(interpolated, axis=0)

        # Pick or assign color
        if degree not in degree_to_color:
            degree_to_color[degree] = next(color_cycle)
        color = degree_to_color[degree]

        # Plot mean and std band
        ax.plot(common_steps, mean, label=f"degree={degree}", color=color, linewidth=1.2)
        ax.fill_between(common_steps, mean-std, mean+std, color=color, alpha=0.25)

    # Titles, labels, grid, etc.
    title_text = subtitle_map.get(tag, tag)
    ax.set_title(title_text, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.tick_params(axis='both', which='major', pad=1)
    ax.legend(title="Degree", fontsize=6)

# Hide any unused subplots
for j in range(num_tags, len(axes)):
    axes[j].axis('off')

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.85])
plt.subplots_adjust(wspace=0.3, hspace=0.8, bottom=0.2)

plt.savefig("plots/training_plots.pdf", format="pdf")
plt.show()
