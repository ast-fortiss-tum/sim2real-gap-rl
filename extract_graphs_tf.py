import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

from cycler import cycler

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
folder_path = "/home/cubos98/Desktop/MA/DARAIL/runs/runs5"  # Update to your folder path
tags = ["Env/Rewards", "Target Env/Rewards", "Loss/Classify Loss"]

run_filter_regex = re.compile(r'(?=.*noise_cfrs_0.0)(?=.*degree_0.8)', re.IGNORECASE)
#run_filter_regex = re.compile(r'(?=.*degree_0.8)', re.IGNORECASE)

# Smoothing factor for TensorBoard-style exponential smoothing (0 = no smoothing)
tensorboard_smoothing = 0.7

# Editable subtitles for each subplot (change these values as desired)
subtitle_map = {
    "Env/Rewards": "SAC Rewards (20 iteractions)",
    "Target Env/Rewards": "DARC Rewards (20 iteractions with Target)",
    "Loss/Classify Loss": "Classification Loss of DARC",
}

# --------------------------
#  MATPLOTLIB SETTINGS
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
    "ytick.major.pad": 1
})

# --------------------------
#  HELPER FUNCTION FOR TENSORBOARD-STYLE EXPONENTIAL SMOOTHING
# --------------------------
def smooth_signal(steps, values, smoothing):
    """
    Applies exponential smoothing similar to TensorBoard.
    smoothed[0] = values[0]
    smoothed[i] = smoothing * smoothed[i-1] + (1 - smoothing) * values[i]
    """
    if smoothing == 0 or len(values) == 0:
        return steps, values
    smoothed_values = [values[0]]
    for value in values[1:]:
        smoothed_values.append(smoothed_values[-1] * smoothing + (1 - smoothing) * value)
    return steps, smoothed_values

# --------------------------
#  HELPER FUNCTION TO EXTRACT VARIABLE PARAMETER PAIRS
# --------------------------
def extract_variable_parameter_pairs(labels):
    """
    Given a list of labels that follow a structure like:
        parameter_1_parameter_2_...
    where parameters and values alternate, this function splits each label by underscores,
    groups tokens in pairs (i.e. parameter name and its value), and then finds which parameter's
    value is variable across all labels.
    
    For each label, only the variable parameter-value pairs are retained (displayed as "param=value").
    If a label does not follow the paired structure, falls back to a simple token-difference extraction.
    """
    if not labels:
        return labels

    # Split each label by underscore.
    split_labels = [label.split('_') for label in labels]
    # Check if every label has an even number of tokens (expected structure)
    if any(len(tokens) % 2 != 0 for tokens in split_labels):
        # Fallback to a simple differing token extraction.
        return extract_variable_parts(labels)
    
    # For each label, make a list of (parameter, value) pairs.
    all_pairs = [ [(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2)]
                  for tokens in split_labels ]
    
    # Assume that all labels have the same parameters (in the same order).
    num_pairs = len(all_pairs[0])
    
    # For each parameter (by position), collect the set of values across all labels.
    variable_param_indices = []
    for i in range(num_pairs):
        values = {pairs[i][1] for pairs in all_pairs}
        if len(values) > 1:
            variable_param_indices.append(i)
    
    # Build new labels for each run using only the variable parameter-value pairs.
    new_labels = []
    for pairs in all_pairs:
        parts = []
        for i in variable_param_indices:
            param, value = pairs[i]
            parts.append(f"{param}={value}")
        new_label = " ".join(parts)
        # Fallback to original label if nothing is variable.
        if new_label == "":
            new_label = "_".join(sum(pairs, ()))
        new_labels.append(new_label)
    
    return new_labels

def extract_variable_parts(labels):
    """
    Fallback helper: Given a list of strings with underscore-separated tokens, 
    keep only the tokens that differ across labels.
    """
    if not labels:
        return labels
    token_lists = [label.split('_') for label in labels]
    max_cols = max(len(tokens) for tokens in token_lists)
    for tokens in token_lists:
        if len(tokens) < max_cols:
            tokens += [""] * (max_cols - len(tokens))
    keep = [False] * max_cols
    for i in range(max_cols):
        col_tokens = [tokens[i] for tokens in token_lists]
        if len(set(col_tokens)) > 1:
            keep[i] = True
    new_labels = []
    for tokens in token_lists:
        diff_tokens = [tokens[i] for i in range(max_cols) if keep[i]]
        joined = "_".join(diff_tokens).strip("_")
        if joined == "":
            joined = "_".join(tokens).strip("_")
        new_labels.append(joined)
    return new_labels

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
            # Preserve full, long run names.
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
#  PLOTTING WITH SMOOTHING, EDITABLE SUBTITLES & GLOBAL LEGEND
# --------------------------
num_tags = len(tags)
ncols = 3
nrows = (num_tags + ncols - 1) // ncols  # Ceiling of rows needed

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

fig.suptitle("Training Metrics", fontsize=16, fontweight="bold", y=0.95)

# Create a color cycle and a mapping for base labels to colors.
color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
base_label_to_color = {}

def get_base_label(run_label):
    """
    Removes a leading 'ContSAC' or 'DARC' token from run_label, if present.
    For example:
      "ContSAC noise_1_lr_4_sp_6" -> "noise_1_lr_4_sp_6"
      "DARC noise_2_lr_4_sp_6"    -> "noise_2_lr_4_sp_6"
    """
    tokens = run_label.split("_", 1)
    if tokens[0] in ["ContSAC", "DARC"]:
        return tokens[1] if len(tokens) > 1 else ""
    return run_label

# For the global legend, store one handle per base label.
global_handles = {}

# --------------------------
#  PLOTTING BLOCK
# --------------------------
for i, tag in enumerate(tags):
    ax = axes[i]
    for run_label, data_dict in runs_data.items():
        steps = data_dict[tag]["steps"]
        values = data_dict[tag]["values"]
        if not steps:
            continue

        # Retrieve the base label (without the "ContSAC"/"DARC" prefix)
        base_label = get_base_label(run_label)

        # Assign a unique color per base label.
        if base_label not in base_label_to_color:
            base_label_to_color[base_label] = next(color_cycle)
        color = base_label_to_color[base_label]

        ls = "-"

        # Apply TensorBoard-style exponential smoothing.
        x_plot, y_plot = smooth_signal(steps, values, tensorboard_smoothing)
        line, = ax.plot(x_plot, y_plot, color=color, linestyle=ls, linewidth=1.0)

        # Store the line handle only once per base label for the legend.
        if base_label not in global_handles:
            global_handles[base_label] = line

    # Set the subplot title using subtitle_map (editable)
    title_text = subtitle_map.get(tag, tag)
    ax.set_title(title_text, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.tick_params(axis='both', which='major', pad=1)

# Hide any unused subplots.
for j in range(num_tags, len(axes)):
    axes[j].axis('off')

# --------------------------
#  PROCESS GLOBAL LEGEND LABELS TO SHOW ONLY VARIABLE PARAMETERS (keeping parameter names)
# --------------------------
original_labels = list(global_handles.keys())
print(original_labels)
final_labels = extract_variable_parameter_pairs(original_labels)

# --------------------------
#  GLOBAL LEGEND
# --------------------------
handles = list(global_handles.values())
fig.legend(handles, final_labels, title="Labels", loc='lower center',
           ncol=min(len(final_labels), 4), bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.85])
plt.subplots_adjust(wspace=0.3, hspace=0.8, bottom=0.2)

plt.savefig("training_plots.pdf", format="pdf")
plt.show()