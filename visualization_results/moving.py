import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_time(time_raw):
    """
    Parse a time string (which represents a list of floats)
    into an actual Python list of floats.
    """
    try:
        time_values = json.loads(time_raw)
    except json.JSONDecodeError as e:
        print("Error parsing the time string:", e)
        return None
    return time_values

def parse_going(going_raw):
    """
    Parse the 'going' string (which represents a list of booleans)
    into an actual Python list of booleans.
    
    Since the string may contain Python-style booleans (True/False)
    instead of JSON booleans, we replace them.
    """
    going_fixed = going_raw.replace("True", "true").replace("False", "false")
    try:
        going_values = json.loads(going_fixed)
    except json.JSONDecodeError as e:
        print("Error parsing the 'going' string:", e)
        return None
    return going_values

def moving_stats(arr, times, window):
    """
    Compute the moving average and moving standard deviation of an array,
    along with the average time for each window.
    
    Parameters:
      arr    : 1D numpy array of parameter values.
      times  : 1D numpy array of corresponding time values.
      window : integer, the size of the moving window.
    
    Returns:
      t_avg : numpy array of time values for each window (the mean time).
      ma    : numpy array of moving averages.
      std   : numpy array of moving standard deviations.
    """
    if len(arr) < window:
        return None, None, None
    ma = np.convolve(arr, np.ones(window) / window, mode='valid')
    std = np.array([np.std(arr[i:i+window]) for i in range(len(arr) - window + 1)])
    t_avg = np.array([np.mean(times[i:i+window]) for i in range(len(times) - window + 1)])
    return t_avg, ma, std

def plot_parameter_subplot(ax, param_data, going_data, param_key, window_size=100, color=None):
    """
    Plot one parameter (param_key) on the given Axes with a movement partition background.
    
    The procedure is as follows:
      1. Parse the parameter JSON to get its time and the parameter values.
      2. Filter out zero values and compute a rolling (moving) average and standard deviation.
      3. Fill the entire parameter time range with a red background (indicating "not going").
      4. Overlay green backgrounds for intervals from the going JSON where going is True.
      5. Plot the smoothed parameter curve and shade the ±1 standard deviation area.
    
    In addition, the parameter time is shifted so that it starts at 0, and the going times
    are normalized using the parameter file's first timestamp.
    """
    # --- Parse parameter data ---
    time_raw = param_data.get("time")
    if time_raw is None:
        print("Parameter JSON does not contain a 'time' key.")
        return
    time_values = parse_time(time_raw)
    if time_values is None:
        return
    # Shift the parameter time so that the first element is 0.
    time_array = np.array(time_values)
    param0 = time_array[0]
    time_array = time_array - param0
    
    values = param_data.get(param_key)
    if values is None:
        print(f"Parameter '{param_key}' not found in the parameter JSON.")
        return
    if not isinstance(values, list):
        print(f"The value for '{param_key}' is not a list.")
        return
    values_array = np.array(values)
    
    # Remove zero values (and the corresponding times).
    nonzero_mask = values_array != 0
    filtered_times = time_array[nonzero_mask]
    filtered_values = values_array[nonzero_mask]
    if len(filtered_values) < window_size:
        print(f"Not enough nonzero data for '{param_key}' with window size {window_size}.")
        return
    
    t_avg, ma, std = moving_stats(filtered_values, filtered_times, window_size)
    if t_avg is None:
        print(f"Window size {window_size} is too large for '{param_key}'.")
        return

    # --- Background: Determine parameter time range ---
    t_param_min = time_array.min()
    t_param_max = time_array.max()
    
    # Fill the entire parameter time range with red (Not going), zorder low.
    ax.axvspan(t_param_min, t_param_max, facecolor='red', alpha=0.4, label="Not going", zorder=1)
    
    # --- Parse going data ---
    going_time_raw = going_data.get("time")
    if going_time_raw is None:
        print("Going JSON does not contain a 'time' key.")
        return
    going_time_values = parse_time(going_time_raw)
    if going_time_values is None:
        return
    # **IMPORTANT:** Normalize going times using the parameter file's first timestamp.
    going_time_array = np.array(going_time_values) - param0
    
    going_raw = going_data.get("going")
    if going_raw is None:
        print("Going JSON does not contain a 'going' key.")
        return
    going_values = parse_going(going_raw)
    if going_values is None:
        return

    # --- Overlay: For each interval in the going data, if going is True, overlay green ---
    for i in range(len(going_time_array) - 1):
        start = going_time_array[i]
        end = going_time_array[i+1]
        # Clip the interval to the parameter time range.
        interval_start = max(start, t_param_min)
        interval_end = min(end, t_param_max)
        if interval_start < interval_end:
            if going_values[i]:
                ax.axvspan(interval_start, interval_end, facecolor='green', alpha=0.4, 
                           label="Going" if i == 0 else None, zorder=2)
    
    # --- Plot the smoothed parameter curve on top (zorder high) ---
    ax.plot(t_avg, ma, label=f"{param_key} (smoothed)", color=color, zorder=3)
    #ax.fill_between(t_avg, ma - std, ma + std, color=color, alpha=0.1, zorder=3)
    
    ax.set_title(f"{param_key.capitalize()} with Movement Partition (Abs. Value)")
    ax.set_xlabel("Time")
    ax.set_ylabel(param_key.capitalize())
    ax.legend()
    ax.grid(True)

def calculate_cte_below_percentage(param_data, going_data, cte_threshold, speed_threshold):
    """
    Calculate the percentage of time (only during the 'green' intervals from the going data)
    that 'cte' is below a specified threshold, ignoring intervals where speed is below a 
    specified speed threshold.
    
    The function uses:
      - The time, cte, and speed arrays from param_data (which must contain "time", "cte", and "speed").
      - The time and going arrays from going_data (which must contain "time" and "going").
    
    It assumes that:
      1. The parameter time stamps are in order.
      2. The cte and speed values at time[i] represent the interval from time[i] to time[i+1].
      3. The going data intervals are used to decide whether a given time interval is "green" (i.e. moving).
         (For each parameter interval, we use the midpoint to determine its status.)
    
    The function returns the percentage (relative to the total green time in which speed is high)
    during which cte < cte_threshold.
    """
    # --- Parse and normalize parameter data ---
    time_raw = param_data.get("time")
    if time_raw is None:
        print("Parameter JSON does not contain a 'time' key.")
        return None
    time_values = parse_time(time_raw)
    if time_values is None:
        return None
    time_array = np.array(time_values)
    # Shift the parameter time so that it starts at 0.
    param0 = time_array[0]
    time_array = time_array - param0

    # Get cte values.
    cte_values = param_data.get("cte")
    if cte_values is None:
        print("Parameter JSON does not contain a 'cte' key.")
        return None
    cte_array = np.array(cte_values)
    
    # Get speed values.
    speed_values = param_data.get("speed")
    if speed_values is None:
        print("Parameter JSON does not contain a 'speed' key.")
        return None
    speed_array = np.array(speed_values)
    
    # --- Parse and normalize going data ---
    going_time_raw = going_data.get("time")
    if going_time_raw is None:
        print("Going JSON does not contain a 'time' key.")
        return None
    going_time_values = parse_time(going_time_raw)
    if going_time_values is None:
        return None
    # Normalize going time using the parameter file's first timestamp (param0)
    going_time_array = np.array(going_time_values) - param0

    going_raw = going_data.get("going")
    if going_raw is None:
        print("Going JSON does not contain a 'going' key.")
        return None
    going_values = parse_going(going_raw)
    if going_values is None:
        return None

    # --- Compute durations only within "green" (moving) intervals and with sufficient speed ---
    total_green_time = 0.0
    below_threshold_green_time = 0.0

    # For each interval in the parameter time series, we assume the cte and speed values at index i
    # represent the interval from time_array[i] to time_array[i+1]. We use the midpoint to decide if that interval falls in a green period.
    if len(time_array) < 2:
        print("Not enough time samples to compute durations.")
        return None

    for i in range(len(time_array) - 1):
        t_start = time_array[i]
        t_end = time_array[i+1]
        duration = t_end - t_start
        t_mid = (t_start + t_end) / 2.0

        # Find the corresponding going interval for t_mid.
        green = False
        for j in range(len(going_time_array) - 1):
            if going_time_array[j] <= t_mid < going_time_array[j+1]:
                green = going_values[j]  # True if "going", False otherwise.
                break

        # Only consider intervals that are "green" and where speed is at or above the speed threshold.
        if green and speed_array[i] >= speed_threshold:
            total_green_time += duration
            if cte_array[i] < cte_threshold:
                below_threshold_green_time += duration

    if total_green_time == 0:
        print("No green intervals found with speed above the threshold.")
        return None

    percentage = (below_threshold_green_time / total_green_time) * 100
    return percentage


def main(window_size=100):

    # Create an ArgumentParser object.
    parser = argparse.ArgumentParser(description="Example: Select a file using a command-line flag.")
    
    # Define the --file flag that requires a file path.
    parser.add_argument('--name', type=str, required=True, help="Name of experiments file.")
    
    # Parse the command-line arguments.
    args = parser.parse_args()
    
    # Access the file path provided by the user.
    name = args.name

    print("Selected file:", name)

    # Replace these file paths with your actual JSON file paths.
    parameter_json_file = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/{name}_ctes.json'
    going_json_file = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/{name}_moving.json'

    # Load the JSON files.
    param_data = load_data(parameter_json_file)
    going_data = load_data(going_json_file)

    cte_threshold = 2.0
    speed_threshold = 10.0

    perc = calculate_cte_below_percentage(param_data, going_data, cte_threshold=cte_threshold, speed_threshold=speed_threshold)

    if perc is not None:
        print(f"Percentage of green time with cte below {cte_threshold} and speed above {speed_threshold}: {perc:.2f}%")
    
    # Create a single figure with two subplots (one for speed, one for cte).
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot "speed" in the first subplot.
    plot_parameter_subplot(axs[0], param_data, going_data, "speed", window_size=window_size, color="blue")
    # Plot "cte" in the second subplot.
    plot_parameter_subplot(axs[1], param_data, going_data, "cte", window_size=window_size, color="black")

    axs[0].set_ylim(0.0, 300.0)
    axs[1].set_ylim(0.0, 8.0)
    
    plt.tight_layout()
    plt.savefig(f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/figures/{name}_cte{cte_threshold}_speed{speed_threshold}_ws{window_size}_percSuccess_{perc}.png')
    plt.show()

if __name__ == '__main__':
    main(window_size=200)







