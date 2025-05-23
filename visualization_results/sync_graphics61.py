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

from matplotlib.animation import PillowWriter, FFMpegWriter   # Pillow ships with Matplotlib
import matplotlib as mpl

mpl.rcParams['axes.titlesize']     = 10   # subplot title
mpl.rcParams['axes.labelsize']     = 10   # x/y labels
mpl.rcParams['xtick.labelsize']    = 10   # x‐axis tick labels
mpl.rcParams['ytick.labelsize']    = 10   # y‐axis tick labels
mpl.rcParams['legend.fontsize']    = 10

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
                steps_per_segment=10, subsample=1, cte_threshold=3.0):
    """
    Create a synchronized animation on four axes:
      - ax1: Lane/pose animation (with a moving marker, trace, and a text display)
      - ax2: Dynamic Speed plot (with background partitions for going vs not going)
      - ax3: Dynamic CTE plot (with the same background as ax2)
      - ax4: Driven Angle plot (angle in degrees versus time)
    
    Speed is computed from the interpolated poses (positions scaled by 25) and then smoothed.
    The x-axes for ax2, ax3 and ax4 update to show the seconds passed.
    
    In this version, the CTE is calculated from the pose relative to a precomputed centerline.
    By convention, a lateral distance equal to the half-width (distance from the centerline to a lane boundary)
    corresponds to 1.5 cte-units.
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
    going_times_norm = np.array(going_times) - t_start
    # For speeds, create a new normalized time array for the total number of frames:
    speed_times_norm = np.linspace(0, total_sim_time, total_frames)
    
    # --- Precompute a centerline and half-widths from lanes[0] and lanes[1] ---
    if lanes and len(lanes) >= 2:
        lane0_points = lanes[0]
        lane1_points = lanes[1]
        n = min(len(lane0_points), len(lane1_points))
        centerline = []
        half_widths = []
        for i in range(n):
            cx = (lane0_points[i][0] + lane1_points[i][0]) / 2.0
            cy = (lane0_points[i][1] + lane1_points[i][1]) / 2.0
            centerline.append((cx, cy))
            hw = np.hypot(lane0_points[i][0] - cx, lane0_points[i][1] - cy)
            half_widths.append(hw)
    else:
        centerline = None

    # --- Compute the geometrical center of the road ---
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
    unwrapped_state = {"prev_raw": None, "prev_unwrapped": None, "prev_going": None}
    
    # --- For accumulating CTE data computed from the pose ---
    cte_trace = []  # list of tuples (time, computed_cte)
    
    # --- Metrics accumulation for CTE-based evaluation ---
    metrics_stats = {
         "prev_time": None,
         "prev_angle": None,
         "time_in": 0.0,
         "time_total": 0.0,
         "angle_in": 0.0,
         "angle_total": 0.0
    }

    # --- Setup for ax1 (lane/pose animation) ---
    lane_point, = ax1.plot([], [], 'o', markersize=10)
    lane_trace = LineCollection([], linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.add_collection(lane_trace)
    percentage_text = ax1.text(0.55, 0.20, '', transform=ax1.transAxes,
                               fontsize=10, color='blue', verticalalignment='top')
    
    # --- Setup for ax2 (Dynamic Speed plot) ---
    speed_line, = ax2.plot([], [], color='blue', label='Speed (smoothed)')
    ax2.set_title("Speed (computed from Pose)")
    ax2.set_xlabel("[s]")
    ax2.set_ylabel("Speed [cm/s]")
    ax2.grid(True)
    
    # --- Setup for ax3 (Dynamic CTE plot) ---
    cte_line, = ax3.plot([], [], color='black', label='CTE')
    ax3.set_title("CTE (computed from Pose)")
    ax3.set_xlabel("[s]")
    ax3.set_ylabel("CTE [units]")
    ax3.grid(True)
    
    # --- Setup for ax4 (Driven Angle plot) ---
    angle_line, = ax4.plot([], [], color='magenta', label='Driven Angle')
    cte_indicator, = ax4.plot([], [], 'o', color='red', markersize=10, label='High CTE')
    warning_text = ax4.text(0.95, 0.95, "", transform=ax4.transAxes,
                            ha='right', va='top', fontsize=10, color='red')
    lap_text = ax4.text(0.95, 0.85, "", transform=ax4.transAxes,
                        ha='right', va='top', fontsize=14, color='blue')
    
    ax4.set_title("Driven Angle")
    ax4.set_xlabel("[s]")
    ax4.set_ylabel("Angle [deg]")
    ax4.grid(True)
    
    # For storing lane/pose trace points.
    lane_trace_points = []
    # For storing driven angle data as (time, angle) tuples.
    angle_trace = []
    
    def init():
        lane_point.set_data([], [])
        lane_trace.set_segments([])
        percentage_text.set_text('')
        speed_line.set_data([], [])
        cte_line.set_data([], [])
        angle_line.set_data([], [])
        cte_indicator.set_data([], [])
        warning_text.set_text("")
        lap_text.set_text("")
        ax2.patches.clear()
        ax3.patches.clear()
        ax4.patches.clear()
        angle_trace.clear()
        # Reset unwrapped state
        unwrapped_state["prev_raw"] = None
        unwrapped_state["prev_unwrapped"] = None
        unwrapped_state["prev_going"] = None
        return (lane_point, lane_trace, percentage_text, speed_line, cte_line, 
                angle_line, cte_indicator, warning_text, lap_text)

    def update(frame):
        # Determine simulation time for this frame (linearly mapped)
        sim_time = sim_time_for_frame(frame)
        current_norm = sim_time - t_start  # seconds passed
        
        # --- Update lane/pose animation (ax1) ---
        x, y, _ = interp_poses[frame]
        on_track = is_inside_lane_area([x, y], lanes[0], lanes[1])
        current_going = get_going_flag(sim_time, going_times, going_flags)
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
        
        # --- Update dynamic Speed plot (ax2) ---
        idx_speed = np.where(speed_times_norm <= current_norm)[0]
        if len(idx_speed) > 0:
            x_speed = speed_times_norm[idx_speed]
            y_speed = np.array(speeds_smooth)[idx_speed]
            speed_line.set_data(x_speed, y_speed)
            ax2.set_xlim(0, current_norm + 0.1)
        else:
            speed_line.set_data([], [])
        
        # --- Compute CTE from the pose ---
        # Here we compute the lateral deviation relative to the centerline.
        if centerline is not None:
            # Find the closest centerline point.
            distances = [np.hypot(x - cx, y - cy) for cx, cy in centerline]
            idx_closest = np.argmin(distances)
            closest_distance = distances[idx_closest]
            hw = half_widths[idx_closest]
            if hw > 0:
                computed_cte = (closest_distance / hw) * 1.5
            else:
                computed_cte = 0
        else:
            computed_cte = 0
        
        # Append computed cte for plotting.
        cte_trace.append((current_norm, computed_cte))
        times_cte = [pt[0] for pt in cte_trace]
        values_cte = [pt[1] for pt in cte_trace]
        cte_line.set_data(times_cte, values_cte)
        ax3.set_xlim(0, current_norm + 0.1)
        
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
        current_angle = np.arctan2(y - center_y, x - center_x)
        raw_delta = np.degrees(current_angle - ref_angle)
        
        if not current_going:
            unwrapped_state["prev_raw"] = None
            unwrapped_state["prev_unwrapped"] = None
            unwrapped_state["prev_going"] = False
            continuous_angle = 0
        else:
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
        
        continuous_angle = -continuous_angle
        adjusted_angle = abs(continuous_angle)
        lap_count = int(adjusted_angle // 360)
        display_angle = adjusted_angle % 360
        
        angle_trace.append((current_norm, display_angle))
        times_angle = [pt[0] for pt in angle_trace]
        angles = [pt[1] for pt in angle_trace]
        angle_line.set_data(times_angle, angles)
        ax4.set_xlim(0, current_norm + 0.1)
        lap_text.set_text(f"Laps: {lap_count}")
        
        # --- High CTE Indicator and Warning in Driven Angle Plot ---
        if computed_cte > cte_threshold:
            cte_indicator.set_data([current_norm], [display_angle])
            warning_text.set_text("WARNING: High CTE!")
        else:
            cte_indicator.set_data([], [])
            warning_text.set_text("")
        
        # --- Accumulate metrics for CTE-based evaluation ---
        if current_going:
            if metrics_stats["prev_time"] is not None:
                dt = current_norm - metrics_stats["prev_time"]
                d_angle = abs(continuous_angle - metrics_stats["prev_angle"])
                metrics_stats["time_total"] += dt
                metrics_stats["angle_total"] += d_angle
                if computed_cte <= cte_threshold:
                    metrics_stats["time_in"] += dt
                    metrics_stats["angle_in"] += d_angle
            metrics_stats["prev_time"] = current_norm
            metrics_stats["prev_angle"] = continuous_angle
        else:
            metrics_stats["prev_time"] = None
            metrics_stats["prev_angle"] = None
        
        # --- At the final frame, display percentages and overlay the road area in ax1 ---
        if frame == total_frames - 1:
            if metrics_stats["time_total"] > 0:
                time_percentage = (metrics_stats["time_in"] / metrics_stats["time_total"]) * 100
            else:
                time_percentage = 0.0
            if metrics_stats["angle_total"] > 0:
                angle_percentage = (metrics_stats["angle_in"] / metrics_stats["angle_total"]) * 100
            else:
                angle_percentage = 0.0
            percentage_text.set_text(f"Time-based % below CTE: {time_percentage:.1f}%\n"
                                       f"Angle-based % below CTE: {angle_percentage:.1f}%")
            # --- ROAD AREA OVERLAY ON ax1 ---
            if lanes and len(lanes) >= 2:
                lane0_points = lanes[0]
                lane1_points = lanes[1]
                road_polygon = lane0_points + lane1_points[::-1]
                road_poly_x = [p[0] for p in road_polygon]
                road_poly_y = [p[1] for p in road_polygon]
                n = min(len(lane0_points), len(lane1_points))
                centerline_overlay = []
                for i in range(n):
                    cx = (lane0_points[i][0] + lane1_points[i][0]) / 2.0
                    cy = (lane0_points[i][1] + lane1_points[i][1]) / 2.0
                    centerline_overlay.append((cx, cy))
                left_offsets = []
                right_offsets = []
                for i in range(n):
                    cx, cy = centerline_overlay[i]
                    if i < n - 1:
                        dx = centerline_overlay[i+1][0] - centerline_overlay[i][0]
                        dy = centerline_overlay[i+1][1] - centerline_overlay[i][1]
                    else:
                        dx = centerline_overlay[i][0] - centerline_overlay[i-1][0]
                        dy = centerline_overlay[i][1] - centerline_overlay[i-1][1]
                    norm_val = np.hypot(dx, dy)
                    if norm_val == 0:
                        nx, ny = 0, 0
                    else:
                        nx = -dy / norm_val
                        ny = dx / norm_val
                    half_width = np.hypot(lane0_points[i][0] - cx, lane0_points[i][1] - cy)
                    safe_offset = (cte_threshold / 1.5) * half_width
                    left_offsets.append((cx + safe_offset * nx, cy + safe_offset * ny))
                    right_offsets.append((cx - safe_offset * nx, cy - safe_offset * ny))
                safe_polygon = left_offsets + right_offsets[::-1]
                safe_poly_x = [p[0] for p in safe_polygon]
                safe_poly_y = [p[1] for p in safe_polygon]
                ax1.fill(road_poly_x, road_poly_y, color='red', alpha=0.3, zorder=0)
                ax1.fill(safe_poly_x, safe_poly_y, color='green', alpha=0.3, zorder=1)
        
        return (lane_point, lane_trace, percentage_text, speed_line, cte_line, 
                angle_line, cte_indicator, warning_text, lap_text)

    def frame_gen():
        start_real = time.time()
        for i in range(total_frames):
            desired_sim_time = sim_time_for_frame(i)
            target_time = start_real + (desired_sim_time - t_start)
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            yield i

    ani = animation.FuncAnimation(
        ax1.figure,
        update,
        frames=frame_gen,
        init_func=init,
        blit=False,
        repeat=False
    )

    # Instead of using frames=range(total_frames), step by skip:
    skip = 1  # only draw every 2nd frame
    frames_to_use = range(0, total_frames, skip)

    #ani_export = animation.FuncAnimation(ax1.figure, update, frames=range(total_frames), init_func=init, blit=False, repeat=False)
    ani_export = animation.FuncAnimation(ax1.figure, update, frames=frames_to_use, init_func=init, blit=True, repeat=False)

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax1.margins(x=0,y=0)

    return ani, ani_export, frames_to_use

import argparse
import ast
import json
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.gridspec import GridSpec

# … (all your other imports and helper functions above: load_data, plot_lanes, animate_all, etc.) …

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic synchronized animation: lane/pose plus dynamic CTE, speed, and driven angle plots."
    )
    parser.add_argument('--name', type=str, required=True,
                        help="Name of experiments file for poses and lanes (without extension).")
    parser.add_argument('--folder', type=str, required=True,
                        help="Name of experiments folder for poses and lanes (without extension).")
    args = parser.parse_args()

    name = args.name
    folder = args.folder

    # --- Load poses JSON ---
    poses_file = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/dataset2/{folder}/{name}_poses.json'
    poses_data = load_data(poses_file)
    lanes       = poses_data.get("lanes")
    poses       = poses_data.get("pose")
    poses_time  = poses_data.get("time")
    if lanes is None or poses is None or poses_time is None:
        print("Error: poses JSON must contain 'lanes', 'pose', and 'time'.")
        return
    if isinstance(poses_time, str):
        poses_time = ast.literal_eval(poses_time)

    # Scale positions by 25
    poses = [[p[0]*25, p[1]*25] for p in poses]
    for i in range(len(lanes)):
        lanes[i] = [[p[0]*25, p[1]*25] for p in lanes[i]]

    # --- Load going JSON ---
    going_file = f'/home/cubos98/catkin_ws/src/Vehicle/results_reality/dataset2/{folder}/{name}_moving.json'
    going_data  = load_data(going_file)
    going_list  = going_data.get("going")
    going_times = going_data.get("time")
    if going_list is None or going_times is None:
        print("Error: going JSON must contain 'going' and 'time'.")
        return
    if isinstance(going_list, str):
        going_list = ast.literal_eval(going_list)
    if isinstance(going_times, str):
        going_times = ast.literal_eval(going_times)

    # --- Build figure with GridSpec ---
    fig = plt.figure(figsize=(12, 6))
    gs  = GridSpec(nrows=3, ncols=2,
                   figure=fig,
                   width_ratios=[1, 1],
                   wspace=0.3,
                   hspace=0.6)

    ax1 = fig.add_subplot(gs[:, 0])  # big plot on left
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])


    # --- Static background for ax1 ---
    plot_lanes(ax1, lanes)
    all_x = [pt[0] for lane in lanes if isinstance(lane, list) for pt in lane]
    all_y = [pt[1] for lane in lanes if isinstance(lane, list) for pt in lane]
    if all_x and all_y:
        margin = 100
        ax1.set_xlim(min(all_x)-margin, max(all_x)+margin)
        ax1.set_ylim(min(all_y)-margin, max(all_y)+margin)

    # Legend for ax1
    stopped   = mlines.Line2D([], [], color='black', marker='o',
                              linestyle='None', markersize=8, label='Stopped')
    intrack   = mlines.Line2D([], [], color='green', marker='o',
                              linestyle='None', markersize=8, label='In-track')
    offtrack  = mlines.Line2D([], [], color='red', marker='o',
                              linestyle='None', markersize=8, label='Off-track')
    ax1.legend(handles=[offtrack, intrack, stopped], loc='upper left')
    ax1.set_facecolor('grey')

    # Static limits for the right-hand plots
    ax2.set_ylim(0, 200)
    ax3.set_ylim(0, 6)
    ax4.set_ylim(0, 360)

    # --- Build animations ---
    ani, ani_exp, frames_to_use = animate_all(
        ax1, ax2, ax3, ax4,
        poses, poses_time,
        lanes, going_times, going_list,
        steps_per_segment=8,
        subsample=200,
        cte_threshold=3.0
    )

    # --- Export to GIF ---
    desired_secs = 50
    target_fps   = max(1, len(frames_to_use) / desired_secs)
    writer       = PillowWriter(fps=target_fps)
    ani_exp.save(f"final_{folder}_{name}.gif", writer=writer)

if __name__ == '__main__':
    main()