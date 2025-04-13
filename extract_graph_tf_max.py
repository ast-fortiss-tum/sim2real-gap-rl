import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

# --------------------------
#  SETTINGS & CONFIGURATION
# --------------------------
folder_path = "/home/cubos98/Desktop/MA/DARAIL/runs/runs5"  # Update to your folder path
tags = ["Env/Rewards", "Target Env/Rewards", "Loss/Classify Loss"]

run_filter_regex = re.compile(r'(?=.*noise_cfrs_0.0)(?=.*degree_0.8)', re.IGNORECASE)
#run_filter_regex = re.compile(r'(?=.*degree_0.8)', re.IGNORECASE)
# For exponential smoothing similar to TensorBoard, set tensorboard_smoothing in [0, 1].
# 0 -> no smoothing, higher values -> more smoothing.
tensorboard_smoothing = 0.6

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
    "ytick.major.pad": 1
})

# --------------------------
#  HELPER FUNCTION FOR TENSORBOARD STYLE SMOOTHING
# --------------------------
def smooth_signal(steps, values, smoothing):
    """
    Applies exponential smoothing similar to TensorBoard.
    
    Parameters:
        steps (list or array): The x-axis values (e.g., training steps).
        values (list or array): The y-axis values (metric values).
        smoothing (float): Smoothing factor in [0, 1]. 0 means no smoothing.
    
    Returns:
        steps (list): Unmodified x-axis values.
        smoothed_values (list): Exponentially smoothed y-axis values.
    
    Algorithm:
      smoothed[0] = values[0]
      for i > 0:
          smoothed[i] = smoothing * smoothed[i-1] + (1 - smoothing) * values[i]
    """
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
            # Preserve full, long run names.
            run_label = f"{folder_name}"
            runs_data[run_label] = {tag: {"steps": [], "values": []} for tag in tags}
            event_file_path = os.path.join(root, file)
            print(f"Processing {run_label}: {event_file_path}")
            try:
                for event in tf.compat.v1.train.summary_iterator(event_file_path):
                    if event.summary:
                        for value in event.summary.value:
                            if value.tag in tags and hasattr(value, "simple_value") and value.simple_value is not None:
                                runs_data[run_label][value.tag]["steps"].append(event.step)
                                runs_data[run_label][value.tag]["values"].append(value.simple_value)
            except Exception as e:
                print(f"Error processing {event_file_path}: {e}")

if run_counter == 0:
    print("No matching event files found. Please check your folder path and regex filter.")
    exit(1)

# --------------------------
#  RECREATE THE PLOTS WITH SMOOTHING & GLOBAL LEGEND
# --------------------------
num_tags = len(tags)
ncols = 3
nrows = (num_tags + ncols - 1) // ncols  # Computes the ceiling for the number of rows

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

fig.suptitle("Training Metrics", fontsize=16, fontweight="bold", y=0.95)

# Create a color cycle and a function to parse run labels.
color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
base_label_to_color = {}

def get_base_label(run_label):
    """
    Removes a leading 'ContSAC' or 'DARC' token from the run_label, if present.
    """
    tokens = run_label.split("_", 1)
    if tokens[0] in ["ContSAC", "DARC"]:
        return tokens[1] if len(tokens) > 1 else ""
    return run_label

# For the legend, store only one handle per base label.
global_handles = {}

# --------------------------
#  MODIFIED PLOTTING BLOCK WITH EXPONENTIAL SMOOTHING
# --------------------------
for i, tag in enumerate(tags):
    ax = axes[i]
    for run_label, data_dict in runs_data.items():
        steps = data_dict[tag]["steps"]
        values = data_dict[tag]["values"]
        if not steps:
            continue

        # Parse the base label (strip "ContSAC"/"DARC" if present).
        base_label = get_base_label(run_label)

        # Assign a unique color for this base label (if not already done).
        if base_label not in base_label_to_color:
            base_label_to_color[base_label] = next(color_cycle)
        color = base_label_to_color[base_label]

        # Differentiate line styles for ContSAC vs DARC (optional).
        ls = "-"
        # Apply TensorBoard style exponential smoothing.
        x_plot, y_plot = smooth_signal(steps, values, tensorboard_smoothing)
        line, = ax.plot(x_plot, y_plot, color=color, linestyle=ls, linewidth=1.0)

        # For the legend, only keep one handle per base label.
        if base_label not in global_handles:
            global_handles[base_label] = line

    ax.tick_params(axis='both', which='major', pad=1)
    ax.set_title(tag, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Value")
    ax.grid(True)

# Hide any unused subplots if num_tags < total subplots.
for j in range(num_tags, len(axes)):
    axes[j].axis('off')

# --------------------------
#  GLOBAL LEGEND
# --------------------------
handles = list(global_handles.values())
labels = list(global_handles.keys())
fig.legend(handles, labels, title="Run", loc='lower center',
           ncol=min(len(labels), 1), bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.85])
plt.subplots_adjust(wspace=0.3, hspace=0.8, bottom=0.4)

plt.savefig("training_plots.pdf", format="pdf")
plt.show()
