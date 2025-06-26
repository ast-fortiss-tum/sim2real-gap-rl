import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import ast
from matplotlib.path import Path  # for point-in-polygon testing
from matplotlib.collections import LineCollection  # for multi-colored trace segments
import matplotlib.lines as mlines  # for custom legend handles

"""This script creates an animation of vehicle poses overlaid on lane data, with color coding
 based on the vehicle's status (on-track, off-track, or external intervention). 
 It uses JSON files for input data and matplotlib for visualization."""

def load_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_lanes(ax, lanes):
    """
    Plot the lanes on the provided Axes.
    
    Each lane is a list of points [x, y].  
      - Lane 0 and Lane 1 are drawn as white solid lines  
      - Lane 2 is drawn as a yellow dashed line  
      - Other lanes use default styling  
    (No labels are added for lanes.)
    """
    for lane_index, lane in enumerate(lanes):
        if not isinstance(lane, list):
            print(f"Lane {lane_index} is not a list.")
            continue
        
        x_coords = []
        y_coords = []
        for point in lane:
            try:
                x_coords.append(point[0])
                y_coords.append(point[1])
            except (TypeError, IndexError):
                print(f"Skipping a point in lane {lane_index} due to unexpected format.")
        
        if not (x_coords and y_coords):
            print(f"Lane {lane_index} has no valid points.")
            continue
        
        if lane_index in [0, 1]:
            color = 'white'
            linestyle = '-'  # solid line
        elif lane_index == 2:
            color = 'yellow'
            linestyle = '--'  # dashed line
        else:
            color = None  # Let matplotlib choose
            linestyle = '-'
        
        ax.plot(x_coords, y_coords, color=color, linestyle=linestyle)

def is_inside_lane_area(point, lane0, lane1):
    """
    Check if the given point [x, y] is inside the polygon formed by lane0 and lane1.
    The polygon is constructed by concatenating lane0 with lane1 reversed.
    """
    polygon = lane0 + lane1[::-1]
    path = Path(polygon)
    return path.contains_point(point)

def interpolate_poses_with_time(poses, times, steps_per_segment=10):
    """
    Given a list of poses ([x, y]) and their corresponding times,
    return a list of interpolated tuples (x, y, t).
    """
    if len(poses) < 2:
        return [(poses[0][0], poses[0][1], times[0])]
    
    interpolated = []
    for i in range(len(poses) - 1):
        p0 = poses[i]
        p1 = poses[i+1]
        t0 = times[i]
        t1 = times[i+1]
        for t_frac in np.linspace(0, 1, steps_per_segment, endpoint=False):
            x = p0[0] + t_frac * (p1[0] - p0[0])
            y = p0[1] + t_frac * (p1[1] - p0[1])
            t_interp = t0 + t_frac * (t1 - t0)
            interpolated.append((x, y, t_interp))
    interpolated.append((poses[-1][0], poses[-1][1], times[-1]))
    return interpolated

def get_going_flag(current_time, going_times, going_flags):
    """
    Given the current_time and the list of times and corresponding going flags,
    return the flag for the interval in which current_time falls.
    
    Assumes that going_times is sorted in ascending order.
    """
    idx = np.searchsorted(going_times, current_time, side='right') - 1
    if idx < 0:
        return going_flags[0]
    elif idx >= len(going_flags):
        return going_flags[-1]
    return going_flags[idx]

def animate_pose(ax, poses, times, lanes, going_times, going_flags,
                 metrics_times, metrics_cte, metrics_speed,
                 interval=200, steps_per_segment=10, subsample=1):
    """
    Create an animation on ax using pose data (with time) and display in real time:
      - The animated marker (colored based on on-track/off-track/external intervention)
      - The trace of past positions (segments colored accordingly)
      - The actual metric values (cte and speed) corresponding to the current animation time.
      
    Additionally, when the animation reaches the final frame, it computes and displays
    the "Porcentage on-track" (using samples where going is True).
    The animation stops at the final frame.
    """
    # Subsample the original data if desired.
    poses = poses[::subsample]
    times = times[::subsample]
    # Interpolate (x, y, t) between original points.
    interp_poses = interpolate_poses_with_time(poses, times, steps_per_segment=steps_per_segment)
    
    # Create the animated marker.
    point, = ax.plot([], [], 'o', markersize=12)
    # Create a LineCollection for the trace.
    trace_collection = LineCollection([], linestyle='--', linewidth=1.5, alpha=0.6)
    ax.add_collection(trace_collection)
    
    # Create a text object for percentage display.
    percentage_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                              fontsize=14, color='blue', verticalalignment='top')
    # Create a text object for real-time metrics display.
    metrics_text = ax.text(0.05, 0.90, '', transform=ax.transAxes,
                           fontsize=14, color='purple', verticalalignment='top')
    
    # List to store trace points.
    # Each element is a tuple: (x, y, t, on_track, going_flag)
    trace_points = []
    
    def init():
        point.set_data([], [])
        trace_collection.set_segments([])
        percentage_text.set_text('')
        metrics_text.set_text('')
        return point, trace_collection, percentage_text, metrics_text
    
    def update(frame):
        x, y, t = interp_poses[frame]
        # Determine if the current pose is "on track."
        on_track = is_inside_lane_area([x, y], lanes[0], lanes[1])
        # Get the current "going" flag.
        current_going = get_going_flag(t, going_times, going_flags)
        
        # Set marker color.
        if not current_going:
            point_color = 'black'
        else:
            point_color = 'green' if on_track else 'red'
        point.set_data(x, y)
        point.set_color(point_color)
        
        # Append the current sample.
        trace_points.append((x, y, t, on_track, current_going))
        
        # Build trace segments.
        segments = []
        colors = []
        if len(trace_points) > 1:
            for i in range(len(trace_points) - 1):
                p1 = trace_points[i]
                p2 = trace_points[i+1]
                seg = [[p1[0], p1[1]], [p2[0], p2[1]]]
                segments.append(seg)
                if not p1[4]:
                    seg_color = 'black'
                else:
                    seg_color = 'green' if (p1[3] and p2[3]) else 'red'
                colors.append(seg_color)
        trace_collection.set_segments(segments)
        trace_collection.set_color(colors)
        
        # --- Compute the actual metric values (no moving average) ---
        # Find the index of the metric sample with time closest to (but not greater than) t.
        idx = np.searchsorted(metrics_times, t, side='right') - 1
        if idx < 0:
            metrics_text.set_text("CTE: N/A, Speed: N/A")
        else:
            current_cte = metrics_cte[idx]
            current_speed = metrics_speed[idx]
            metrics_text.set_text(f"CTE: {current_cte:.2f}, Speed: {current_speed:.2f}")
        # --- End metric computation ---
        
        # When on the final frame, compute and display the percentage on-track.
        if frame == len(interp_poses) - 1:
            valid_points = [pt for pt in trace_points if pt[4]]
            if valid_points:
                count_green = sum(1 for pt in valid_points if pt[3])
                percentage = (count_green / len(valid_points)) * 100
            else:
                percentage = 0.0
            percentage_text.set_text(f"Porcentage on-track: {percentage:.1f}%")
        
        return point, trace_collection, percentage_text, metrics_text
    
    # Set repeat=False so the animation stops at the final frame.
    ani = animation.FuncAnimation(
        ax.figure,
        update,
        frames=range(len(interp_poses)),
        init_func=init,
        blit=True,
        interval=interval,
        repeat=False
    )
    return ani

def main():
    parser = argparse.ArgumentParser(
        description="Animation with lane coloring, selective black segments (external intervention), "
                    "and real-time metric values (cte and speed) display."
    )

     # File with lanes, poses, and pose time stamps.
    parser.add_argument('--name', type=str, required=True,
                        help="Name of experiments file for poses and lanes (without extension).")

    args = parser.parse_args()
    name = args.name
    
    # Load the poses JSON file.
    poses_file_path = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/{name}_poses.json'
    going_file_path = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/{name}_moving.json'
    metrics_file_path = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/{name}_ctes.json'


    data = load_data(poses_file_path)
    
    lanes = data.get("lanes")
    poses = data.get("pose")
    poses_time = data.get("time")

    if lanes is None:
        print("The poses JSON does not contain a 'lanes' key.")
        return
    if poses is None:
        print("The poses JSON does not contain a 'pose' key.")
        return
    if poses_time is None:
        print("The poses JSON does not contain a 'time' key.")
        return
    if isinstance(poses_time, str):
        poses_time = ast.literal_eval(poses_time)
    
    # Load the going JSON file.
    going_data = load_data(going_file_path)
    going_list = going_data.get("going")
    going_times = going_data.get("time")
    if going_list is None or going_times is None:
        print("The going JSON must contain 'going' and 'time' keys.")
        return
    if isinstance(going_list, str):
        going_list = ast.literal_eval(going_list)
    if isinstance(going_times, str):
        going_times = ast.literal_eval(going_times)
    
    # Load the metrics JSON file.
    metrics_data = load_data(metrics_file_path)
    metrics_cte = metrics_data.get("cte")
    metrics_speed = metrics_data.get("speed")
    metrics_times = metrics_data.get("time")
    if metrics_cte is None or metrics_speed is None or metrics_times is None:
        print("The metrics JSON must contain 'cte', 'speed', and 'time' keys.")
        return
    if isinstance(metrics_cte, str):
        metrics_cte = ast.literal_eval(metrics_cte)
    if isinstance(metrics_speed, str):
        metrics_speed = ast.literal_eval(metrics_speed)
    if isinstance(metrics_times, str):
        metrics_times = ast.literal_eval(metrics_times)
    
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('grey')
    plot_lanes(ax, lanes)
    
    # Set axis limits based on lanes and pose data.
    all_x = []
    all_y = []
    for lane in lanes:
        if isinstance(lane, list):
            for pt in lane:
                try:
                    all_x.append(pt[0])
                    all_y.append(pt[1])
                except:
                    continue
    for pt in poses:
        try:
            all_x.append(pt[0])
            all_y.append(pt[1])
        except:
            continue
    if all_x and all_y:
        margin = 1
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Lanes and Pose Animation with Going Intervals and Metrics")
    ax.grid(True)
    
    # Create a custom legend for the trace segments.
    on_track_line = mlines.Line2D([], [], color='green', linestyle='--', label='on-track')
    off_track_line = mlines.Line2D([], [], color='red', linestyle='--', label='off-track')
    external_line = mlines.Line2D([], [], color='black', linestyle='--', label='External intervention')
    ax.legend(handles=[on_track_line, off_track_line, external_line], loc='upper right')
    
    # Create the animation.
    ani = animate_pose(ax, poses, poses_time, lanes, going_times, going_list,
                       metrics_times, metrics_cte, metrics_speed,
                       interval=50, steps_per_segment=10, subsample=50)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
