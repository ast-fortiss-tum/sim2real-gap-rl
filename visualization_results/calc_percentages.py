


def calculate_cte_below_percentage(param_data, going_data, threshold):
    """
    Calculate the percentage of time (only during the 'green' intervals from the going data)
    that 'cte' is below a specified threshold.
    
    The function uses:
      - The time and cte arrays from param_data (which must contain "time" and "cte").
      - The time and going arrays from going_data (which must contain "time" and "going").
    
    It assumes that:
      1. The parameter time stamps are in order.
      2. The cte value at time[i] represents the interval from time[i] to time[i+1].
      3. The going data intervals are used to decide whether a given time interval is "green" (i.e. moving).
         (For each parameter interval, we use the midpoint to determine its status.)
    
    The function returns the percentage (relative to the total green time) during which cte < threshold.
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
    # Shift parameter time so that it starts at 0.
    param0 = time_array[0]
    time_array = time_array - param0

    cte_values = param_data.get("cte")
    if cte_values is None:
        print("Parameter JSON does not contain a 'cte' key.")
        return None
    cte_array = np.array(cte_values)

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

    # --- Compute durations only within "green" (moving) intervals ---
    total_green_time = 0.0
    below_threshold_green_time = 0.0

    # For each interval in the parameter time series, assume the cte value at index i
    # represents the interval from time_array[i] to time_array[i+1].
    # We use the midpoint of the interval to decide if that interval falls in a green period.
    if len(time_array) < 2:
        print("Not enough time samples to compute durations.")
        return None

    for i in range(len(time_array) - 1):
        t_start = time_array[i]
        t_end = time_array[i+1]
        duration = t_end - t_start
        t_mid = (t_start + t_end) / 2.0

        # Find the corresponding going interval for t_mid.
        # We assume going_time_array is sorted.
        green = False
        for j in range(len(going_time_array) - 1):
            if going_time_array[j] <= t_mid < going_time_array[j+1]:
                green = going_values[j]  # True if "going", False otherwise.
                break

        if green:
            total_green_time += duration
            if cte_array[i] < threshold:
                below_threshold_green_time += duration

    if total_green_time == 0:
        print("No green intervals found.")
        return None

    percentage = (below_threshold_green_time / total_green_time) * 100
    return percentage

# --- Example usage ---
if __name__ == '__main__':
    # Example JSON data (replace with actual file loading as needed)
    example_param_data = {
        "time": "[1000.0, 1001.0, 1002.0, 1003.0, 1004.0]",  # Example timestamps in seconds.
        "cte": "[2.5, 3.2, 2.8, 3.5, 2.9]"  # Example cte values.
    }
    example_going_data = {
        "time": "[1000.0, 1002.0, 1004.0]",  # Going data sampled less frequently.
        "going": "[True, False]"  # Meaning: from 1000 to 1002 is green, then 1002 to 1004 is red.
    }
    
    # Parse the JSON strings if needed (in actual use, you would load them from files).
    # Here we assume they are already in proper JSON string format.
    
    perc = calculate_cte_below_percentage(example_param_data, example_going_data, threshold=3.0)
    if perc is not None:
        print(f"Percentage of green time with cte below 3.0: {perc:.2f}%")
