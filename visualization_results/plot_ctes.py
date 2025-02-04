import json
import numpy as np
import matplotlib.pyplot as plt

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
    
    t_avg, ma, std = moving_stats(filtered_times, filtered_times, window_size)
    if t_avg is None:
        print(f"Window size {window_size} is too large for '{param_key}'.")
        return

    # --- Background: Determine parameter time range ---
    t_param_min = time_array.min()
    t_param_max = time_array.max()
    
    # Fill the entire parameter time range with red (Not going).
    ax.axvspan(t_param_min, t_param_max, facecolor='red', alpha=0.3, label="Not going", zorder=1)
    
    # --- Parse going data ---
    going_time_raw = going_data.get("time")
    if going_time_raw is None:
        print("Going JSON does not contain a 'time' key.")
        return
    going_time_values = parse_time(going_time_raw)
    if going_time_values is None:
        return
    # Normalize going time using the parameter file's first timestamp.
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
                ax.axvspan(interval_start, interval_end, facecolor='green', alpha=0.3, 
                           label="Going" if i == 0 else None, zorder=2)
    
    # --- Plot the smoothed parameter curve on top ---
    ax.plot(t_avg, ma, label=f"{param_key} (smoothed)", color=color, zorder=3)
    ax.fill_between(t_avg, ma - std, ma + std, color=color, alpha=0.4, zorder=3)
    
    ax.set_title(f"{param_key.capitalize()} with Movement Partition")
    ax.set_xlabel("Time")
    ax.set_ylabel(param_key.capitalize())
    ax.legend()
    ax.grid(True)

def calculate_cte_below_percentage(param_data, threshold):
    """
    Calculate the percentage of time that 'cte' is below a specified threshold.
    
    The function uses the time and cte arrays from param_data (which must contain "time" and "cte").
    It assumes that the time stamps are in order and computes the duration of each interval,
    then sums the durations for which the cte value is below the threshold. The percentage is
    the sum of these durations divided by the total time duration.
    """
    time_raw = param_data.get("time")
    if time_raw is None:
        print("Parameter JSON does not contain a 'time' key.")
        return None
    time_values = parse_time(time_raw)
    if time_values is None:
        return None
    time_array = np.array(time_values)
    # Shift the time so that it starts at 0.
    time_array = time_array - time_array[0]
    
    cte_values = param_data.get("cte")
    if cte_values is None:
        print("Parameter JSON does not contain a 'cte' key.")
        return None
    cte_array = np.array(cte_values)
    
    # Ensure that the number of time intervals is one less than the number of samples.
    # We'll assume that for each interval (from time[i] to time[i+1]), the cte value at time[i]
    # is representative of that interval.
    if len(time_array) < 2:
        print("Not enough time samples to compute durations.")
        return None
    
    total_time = time_array[-1] - time_array[0]
    below_time = 0.0
    
    # For each interval, if cte is below the threshold, add the interval duration.
    for i in range(len(time_array) - 1):
        if cte_array[i] < threshold:
            interval_duration = time_array[i+1] - time_array[i]
            below_time += interval_duration
    
    percentage = (below_time / total_time) * 100
    return percentage

def main(parameter_json_file, going_json_file, window_size=100, cte_threshold=3.0):
    # Load the JSON files.
    param_data = load_data(parameter_json_file)
    going_data = load_data(going_json_file)
    
    # Create a single figure with two subplots (one for speed, one for cte).
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot "speed" in the first subplot.
    plot_parameter_subplot(axs[0], param_data, going_data, "speed", window_size=window_size, color="blue")
    # Plot "cte" in the second subplot.
    plot_parameter_subplot(axs[1], param_data, going_data, "cte", window_size=window_size, color="orange")
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print the percentage of time that cte is below the threshold.
    perc = calculate_cte_below_percentage(param_data, cte_threshold)
    if perc is not None:
        print(f"Percentage of time that cte is below {cte_threshold}: {perc:.2f}%")

if __name__ == '__main__':
    # Replace these file paths with your actual JSON file paths.
    parameter_json_file = '/home/cubos98/catkin_ws/src/Vehicle/results_reality/fake_sim_medium_ctes.json'
    going_json_file = '/home/cubos98/catkin_ws/src/Vehicle/results_reality/fake_sim_medium_moving.json'
    main(parameter_json_file, going_json_file, window_size=50, cte_threshold=3.0)

