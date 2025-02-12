import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import ast
import time
from matplotlib.path import Path  # for point‐in‐polygon testing
from matplotlib.collections import LineCollection  # for multi‐colored trace segments
import matplotlib.lines as mlines  # for custom legend handles

def load_data(file_path):
    """Load JSON data from a file."""
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

def is_inside_lane_area(point, lane0, lane1):
    """
    Check if the given point [x, y] is inside the polygon formed by lane0 and lane1.
    The polygon is constructed by concatenating lane0 with lane1 reversed.
    """
    polygon = lane0 + lane1[::-1]
    path = Path(polygon)
    return path.contains_point(point)

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

def animate_all(ax1, ax2, ax3, ax4, poses, times, lanes, going_times, going_flags,
                metrics_times, metrics_cte,
                steps_per_segment=10, subsample=1):
    """
    Create a synchronized animation on four axes:
      - ax1: Lane/pose animation (with a moving marker, trace, and on-track percentage)
      - ax2: Dynamic Speed plot (with background partitions for going vs not going)
      - ax3: Dynamic CTE plot (with the same background as ax2)
      - ax4: Driven angle plot (angle in degrees versus time)
    
    Here, speed is computed from the interpolated poses (which now represent positions scaled by 25)
    and then smoothed.
    The x-axes for ax2, ax3 and ax4 update to show the seconds passed.
    
    To avoid shifts when steps_per_segment != 1, the simulation time is linearly remapped from the original times.
    """
    # Subsample poses and times if desired.
    poses = poses[::subsample]
    times = times[::subsample]
    interp_poses = interpolate_poses_with_time(poses, times, steps_per_segment)
    
    total_frames = len(interp_poses)
    t_start = times[0]      # original start time
    t_end = times[-1]       # original end time
    total_sim_time = t_end - t_start

    # --- Compute instantaneous speeds from interp_poses ---
    speeds = [0]  # First value is zero.
    speed_times = []
    for i, (x, y, t) in enumerate(interp_poses):
        speed_times.append(t)
        if i == 0:
            continue
        x_prev, y_prev, t_prev = interp_poses[i-1]
        dt = t - t_prev
        dist = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
        spd = dist / dt if dt > 0 else 0
        speeds.append(spd)
    # Smooth the speeds using a moving average.
    smooth_window = 8  # Adjust as needed.
    speeds_smooth = np.convolve(speeds, np.ones(smooth_window)/smooth_window, mode='same')
    
    # --- Remap simulation time linearly from the original times ---
    def sim_time_for_frame(i):
        return t_start + (i / (total_frames - 1)) * (t_end - t_start)
    
    # Normalize time arrays by subtracting t_start.
    metrics_times_norm = np.array(metrics_times) - t_start
    going_times_norm = np.array(going_times) - t_start
    # For speeds, create a new normalized time array for the total number of frames:
    speed_times_norm = np.linspace(0, total_sim_time, total_frames)
    
    # --- Setup for ax1 (lane/pose animation) ---
    lane_point, = ax1.plot([], [], 'o', markersize=12)
    lane_trace = LineCollection([], linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.add_collection(lane_trace)
    percentage_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes,
                               fontsize=14, color='blue', verticalalignment='top')
    
    # --- Setup for ax2 (Dynamic Speed plot) ---
    speed_line, = ax2.plot([], [], color='blue', label='Speed (smoothed)')
    ax2.set_title("Speed (computed from Pose)")
    ax2.set_xlabel("[s]")
    ax2.set_ylabel("Speed [cm/s]")
    ax2.grid(True)
    
    # --- Setup for ax3 (Dynamic CTE plot) ---
    cte_line, = ax3.plot([], [], color='black', label='CTE')
    ax3.set_title("CTE")
    ax3.set_xlabel("[s]")
    ax3.set_ylabel("CTE [1]")
    ax3.grid(True)
    
    # --- Setup for ax4 (Driven Angle plot) ---
    angle_line, = ax4.plot([], [], color='magenta', label='Driven Angle')
    # Add an indicator for high CTE on the driven angle plot.
    cte_indicator, = ax4.plot([], [], 'o', color='red', markersize=10, label='High CTE')
    # Add a warning text to show when CTE is high.
    warning_text = ax4.text(0.95, 0.95, "", transform=ax4.transAxes,
                            ha='right', va='top', fontsize=12, color='red')
    ax4.set_title("Driven Angle")
    ax4.set_xlabel("[s]")
    ax4.set_ylabel("Angle [deg]")
    ax4.grid(True)
    
    # For storing lane/pose trace points.
    lane_trace_points = []
    # For storing driven angle data as (time, angle) tuples.
    angle_trace = []
    
    # Define a threshold for CTE.
    cte_threshold = 4.0
    
    # --- Compute the geometrical center of the road ---
    # Here we use lanes[0] and lanes[1] as the road boundaries.
    if len(lanes) >= 2:
        all_points = lanes[0] + lanes[1]
        center_x = np.mean([pt[0] for pt in all_points])
        center_y = np.mean([pt[1] for pt in all_points])
    else:
        center_x, center_y = 0, 0

    # --- Compute the reference angle using the car's starting position ---
    start_x, start_y, _ = interp_poses[0]
    ref_angle = np.arctan2(start_y - center_y, start_x - center_x)
    
    # --- Persistent state for driven angle unwrapping ---
    # This dictionary now also holds the previous going state.
    unwrapped_state = {"prev_raw": None, "prev_unwrapped": None, "prev_going": None}

    def init():
        lane_point.set_data([], [])
        lane_trace.set_segments([])
        percentage_text.set_text('')
        speed_line.set_data([], [])
        cte_line.set_data([], [])
        angle_line.set_data([], [])
        cte_indicator.set_data([], [])
        warning_text.set_text("")
        ax2.patches.clear()
        ax3.patches.clear()
        ax4.patches.clear()
        angle_trace.clear()
        # Reset unwrapped state
        unwrapped_state["prev_raw"] = None
        unwrapped_state["prev_unwrapped"] = None
        unwrapped_state["prev_going"] = None
        return lane_point, lane_trace, percentage_text, speed_line, cte_line, angle_line, cte_indicator, warning_text

    def update(frame):
        # Determine simulation time for this frame (linearly mapped)
        sim_time = sim_time_for_frame(frame)
        current_norm = sim_time - t_start  # seconds passed
        
        # --- Update lane/pose animation (ax1) ---
        x, y, _ = interp_poses[frame]
        on_track = is_inside_lane_area([x, y], lanes[0], lanes[1])
        current_going = get_going_flag(sim_time, going_times, going_flags)
        # Determine marker color:
        # - Black if not going (stopped)
        # - Green if going and on-track
        # - Red if going and off-track
        if not current_going:
            point_color = 'black'
        else:
            point_color = 'green' if on_track else 'red'
        lane_point.set_data(x, y)
        lane_point.set_color(point_color)
        lane_trace_points.append((x, y, sim_time, on_track, current_going))
        segments = []
        colors = []
        if len(lane_trace_points) > 1:
            for i in range(len(lane_trace_points)-1):
                p1 = lane_trace_points[i]
                p2 = lane_trace_points[i+1]
                segments.append([[p1[0], p1[1]], [p2[0], p2[1]]])
                if not p1[4]:
                    seg_color = 'black'
                else:
                    seg_color = 'green' if (p1[3] and p2[3]) else 'red'
                colors.append(seg_color)
        lane_trace.set_segments(segments)
        lane_trace.set_color(colors)
        if frame == total_frames - 1:
            valid_points = [pt for pt in lane_trace_points if pt[4]]
            if valid_points:
                count_green = sum(1 for pt in valid_points if pt[3])
                percentage = (count_green / len(valid_points)) * 100
            else:
                percentage = 0.0
            percentage_text.set_text(f"Percentage on-track: {percentage:.1f}%")
        
        # --- Update dynamic Speed plot (ax2) ---
        idx_speed = np.where(speed_times_norm <= current_norm)[0]
        if len(idx_speed) > 0:
            x_speed = speed_times_norm[idx_speed]
            y_speed = np.array(speeds_smooth)[idx_speed]
            speed_line.set_data(x_speed, y_speed)
            ax2.set_xlim(0, current_norm + 0.1)
        else:
            speed_line.set_data([], [])
        
        # --- Update dynamic CTE plot (ax3) ---
        idx_cte = np.where(metrics_times_norm <= current_norm)[0]
        if len(idx_cte) > 0:
            x_cte = metrics_times_norm[idx_cte]
            y_cte = np.array(metrics_cte)[idx_cte]
            cte_line.set_data(x_cte, y_cte)
            ax3.set_xlim(0, current_norm + 0.1)
        else:
            cte_line.set_data([], [])
        
        # --- Update background for ax2, ax3, and ax4 based on going data ---
        for a in (ax2, ax3, ax4):
            a.patches.clear()
            a.axvspan(0, current_norm, facecolor='red', alpha=0.4, zorder=0)
            for i in range(len(going_times_norm)-1):
                start = going_times_norm[i]
                end = going_times_norm[i+1]
                interval_start = max(start, 0)
                interval_end = min(end, current_norm)
                if interval_start < interval_end and going_flags[i]:
                    a.axvspan(interval_start, interval_end, facecolor='green', alpha=0.4, zorder=1)
        
        # --- Update Driven Angle plot (ax4) ---
        # Compute the current raw angle (in degrees) relative to the reference.
        current_angle = np.arctan2(y - center_y, x - center_x)
        raw_delta = np.degrees(current_angle - ref_angle)
        
        # If the vehicle is not going, reset the angle to 0.
        if not current_going:
            unwrapped_state["prev_raw"] = None
            unwrapped_state["prev_unwrapped"] = None
            unwrapped_state["prev_going"] = False
            continuous_angle = 0
        else:
            # When going, if the previous state was not going then restart the unwrapping.
            if unwrapped_state.get("prev_going") is False or unwrapped_state["prev_going"] is None:
                unwrapped_state["prev_raw"] = raw_delta
                unwrapped_state["prev_unwrapped"] = 0
                continuous_angle = 0
            else:
                diff = raw_delta - unwrapped_state["prev_raw"]
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                continuous_angle = unwrapped_state["prev_unwrapped"] + diff
                unwrapped_state["prev_raw"] = raw_delta
                unwrapped_state["prev_unwrapped"] = continuous_angle
            unwrapped_state["prev_going"] = True
        
        # Reverse the sign of the output degrees.
        continuous_angle = -continuous_angle
        angle_trace.append((current_norm, continuous_angle))
        times_angle = [pt[0] for pt in angle_trace]
        angles = [pt[1] for pt in angle_trace]
        angle_line.set_data(times_angle, angles)
        ax4.set_xlim(0, current_norm + 0.1)
        
        # --- High CTE Indicator and Warning in Driven Angle Plot ---
        # Determine current CTE value.
        if len(idx_cte) > 0:
            current_cte = np.array(metrics_cte)[idx_cte][-1]
        else:
            current_cte = 0
        
        if current_cte > cte_threshold:
            cte_indicator.set_data([current_norm], [continuous_angle])
            warning_text.set_text("WARNING: High CTE!")
        else:
            cte_indicator.set_data([], [])
            warning_text.set_text("")
        
        return lane_point, lane_trace, percentage_text, speed_line, cte_line, angle_line, cte_indicator, warning_text

    # --- Create a frame generator to synchronize simulation time to real time ---
    def frame_gen():
        start_real = time.time()
        for i in range(total_frames):
            desired_sim_time = sim_time_for_frame(i)  # absolute simulation time (in seconds)
            target_time = start_real + (desired_sim_time - t_start)
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            yield i

    ani = animation.FuncAnimation(
        ax1.figure,
        update,
        frames=frame_gen,  # use the custom generator
        init_func=init,
        blit=False,
        repeat=False
    )
    return ani

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic synchronized animation: lane/pose plus dynamic CTE, computed (and smoothed) Speed, and driven angle plots synchronized to real time."
    )
    parser.add_argument('--name', type=str, required=True,
                        help="Name of experiments file for poses and lanes (without extension).")
    args = parser.parse_args()
    
    name = args.name
    
    # --- Load poses JSON ---
    poses_file_path = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/{name}_poses.json'
    going_file_path = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/{name}_moving.json'
    metrics_file_path = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/{name}_ctes.json'

    poses_data = load_data(poses_file_path)
    lanes = poses_data.get("lanes")
    poses = poses_data.get("pose")
    poses_time = poses_data.get("time")
    if lanes is None or poses is None or poses_time is None:
        print("Error: poses JSON must contain 'lanes', 'pose', and 'time'.")
        return
    if isinstance(poses_time, str):
        poses_time = ast.literal_eval(poses_time)
    
    # --- Scale positions by 25 ---
    poses = [[p[0]*25, p[1]*25] for p in poses]
    for i in range(len(lanes)):
        lanes[i] = [[p[0]*25, p[1]*25] for p in lanes[i]]
    
    # --- Load going JSON ---
    going_data = load_data(going_file_path)
    going_list = going_data.get("going")
    going_times = going_data.get("time")
    if going_list is None or going_times is None:
        print("Error: going JSON must contain 'going' and 'time'.")
        return
    if isinstance(going_list, str):
        going_list = ast.literal_eval(going_list)
    if isinstance(going_times, str):
        going_times = ast.literal_eval(going_times)
    
    # --- Load metrics JSON (only cte and time now) ---
    metrics_data = load_data(metrics_file_path)
    metrics_cte = metrics_data.get("cte")
    metrics_times = metrics_data.get("time")
    if metrics_cte is None or metrics_times is None:
        print("Error: metrics JSON must contain 'cte' and 'time'.")
        return
    if isinstance(metrics_cte, str):
        metrics_cte = ast.literal_eval(metrics_cte)
    if isinstance(metrics_times, str):
        metrics_times = ast.literal_eval(metrics_times)
    
    # --- Create a figure with four subplots ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
    
    # Draw lanes on ax1 so they appear in the background.
    plot_lanes(ax1, lanes)
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
    if all_x and all_y:
        margin = 1
        ax1.set_xlim(min(all_x)-margin, max(all_x)+margin)
        ax1.set_ylim(min(all_y)-margin, max(all_y)+margin)
    
    # Add legend for ax1 (lane/pose animation)
    stopped_line = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Stopped')
    intrack_line = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=8, label='In-track')
    offtrack_line = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Off-track')
    ax1.legend(handles=[offtrack_line, intrack_line, stopped_line], loc='upper left')
    
    # Optionally, set y-axis limits for ax2 and ax3 manually.
    ax2.set_ylim(0, 110)
    ax3.set_ylim(0, 8)
    ax4.set_ylim(0, 360*4) 

    ax1.set_facecolor('grey')
    
    ani = animate_all(ax1, ax2, ax3, ax4, poses, poses_time, lanes, going_times, going_list,
                      metrics_times, metrics_cte,
                      steps_per_segment=8, subsample=200)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
