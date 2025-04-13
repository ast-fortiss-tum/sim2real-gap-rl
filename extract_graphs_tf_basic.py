import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------
#  SETTINGS & CONFIGURATION
# --------------------------
# Folder containing subfolders with one events file per subfolder.
folder_path = "/home/cubos98/Desktop/MA/DARAIL/runs/runs5"  # <<<--- Update to your folder path

# Specify the tags you want to plot.
tags = ["Env/Rewards", "Target Env/Rewards", "Loss/Classify Loss"]

# Regex pattern to include only runs that have both "Cont" and "lin_src1" in the subfolder name.
run_filter_regex = re.compile(r'(?=.*noise_cfrs_0.0)(?=.*degree_0.8)', re.IGNORECASE)
#run_filter_regex = re.compile(r'(?=.*degree_0.5)', re.IGNORECASE)


# Smoothing configuration: set a moving-average window.
smoothing_window = 5  # Adjust as needed

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
    # Increase global tick label padding.
    "xtick.major.pad": 1,
    "ytick.major.pad": 1
})

# --------------------------
#  HELPER FUNCTION FOR SMOOTHING
# --------------------------
def smooth_signal(steps, values, window):
    if len(values) < window:
        return steps, values
    steps_arr = np.array(steps)
    values_arr = np.array(values)
    sort_idx = np.argsort(steps_arr)
    steps_sorted = steps_arr[sort_idx]
    values_sorted = values_arr[sort_idx]
    smoothed_values = np.convolve(values_sorted, np.ones(window) / window, mode='valid')
    smoothed_steps = np.convolve(steps_sorted, np.ones(window) / window, mode='valid')
    return smoothed_steps, smoothed_values

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

# --- New Figure Structure ---
# We choose a grid layout: 2 rows x 2 columns (modify as needed).
num_tags = len(tags)
ncols = 3
nrows = (num_tags + ncols - 1) // ncols  # This computes the ceiling for rows

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()  # Flatten to iterate easily

global_handles = {}  # to store line handles (one per run)

for i, tag in enumerate(tags):
    ax = axes[i]
    for run_label, data_dict in runs_data.items():
        steps = data_dict[tag]["steps"]
        values = data_dict[tag]["values"]
        if steps:
            if smoothing_window > 1 and len(steps) >= smoothing_window:
                smooth_steps, smooth_values = smooth_signal(steps, values, smoothing_window)
                line, = ax.plot(smooth_steps, smooth_values, linestyle='-', linewidth=1.0)
            else:
                line, = ax.plot(steps, values)
            # Store the line handle for the global legend
            global_handles[run_label] = line

    ax.tick_params(axis='both', which='major', pad=1)
    ax.set_title(tag, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Value")
    ax.grid(True)

# If there are any unused subplots (e.g., if num_tags is odd), hide them.
for j in range(num_tags, len(axes)):
    axes[j].axis('off')

# Add a global title at the top
fig.suptitle("Training Metrics", fontsize=16, fontweight="bold", y=0.95)

# Create a global legend using all unique run labels.
handles = list(global_handles.values())
labels = list(global_handles.keys())
fig.legend(handles, labels, title="Run", loc='lower center',
           ncol=min(len(labels), 1), bbox_to_anchor=(0.5, 0.0))

# Adjust layout to provide extra space at top for the global title and bottom for the legend.
plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.85])
plt.subplots_adjust(wspace=0.3, hspace=0.8, bottom=0.4)

plt.savefig("training_plots.pdf", format="pdf")
plt.show()
