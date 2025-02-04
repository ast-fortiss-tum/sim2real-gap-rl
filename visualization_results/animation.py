import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_lanes(ax, lanes):
    """
    Plot the lanes on the provided Axes.
    
    Each lane is a list of points in the format [x, y].  
    - Lane 0 and Lane 1: white solid lines  
    - Lane 2: yellow dashed line  
    - Other lanes: default styling
    """
    for lane_index, lane in enumerate(lanes):
        if not isinstance(lane, list):
            print(f"Lane {lane_index} is not a list.")
            continue
        
        # Collect x and y coordinates (assuming each point is a list [x, y])
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
        
        # Set custom style based on lane index.
        if lane_index == 0 or lane_index == 1:
            color = 'white'
            linestyle = '-'  # solid line
        elif lane_index == 2:
            color = 'yellow'
            linestyle = '--'  # dashed line
        else:
            color = None  # Let matplotlib choose
            linestyle = '-'
        
        ax.plot(x_coords, y_coords, color=color, linestyle=linestyle, label=f"Lane {lane_index}")

def animate_pose(ax, poses, interval=200):
    """
    Create an animation on the given Axes with the pose data.
    
    The pose is assumed to be a list of [x, y] pairs.  
    Only every 100th point is used for the animation.
    """
    # Subsample the poses to use every 100th point.
    poses = poses[::100]

    # Create the animated point (red circle) with a large marker size.
    point, = ax.plot([], [], 'ro', markersize=12)
    
    def init():
        point.set_data([], [])
        return point,

    def update(frame):
        x, y = poses[frame]
        point.set_data(x, y)
        return point,

    ani = animation.FuncAnimation(
        ax.figure,
        update,
        frames=range(len(poses)),
        init_func=init,
        blit=True,
        interval=interval
    )
    return ani

def main(file_path):
    # Load the JSON data.
    data = load_data(file_path)
    
    # Get lanes and pose from the JSON.
    lanes = data.get("lanes")
    poses = data.get("pose")
    
    if lanes is None:
        print("The JSON file does not contain a 'lanes' key.")
        return
    if poses is None:
        print("The JSON file does not contain a 'pose' key.")
        return
    
    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set a background color so white lines are visible.
    ax.set_facecolor('grey')
    
    # Plot lanes.
    plot_lanes(ax, lanes)
    
    # Gather all x and y coordinates from lanes and pose for proper limits.
    all_x = []
    all_y = []
    for lane in lanes:
        if isinstance(lane, list):
            for point in lane:
                try:
                    all_x.append(point[0])
                    all_y.append(point[1])
                except (TypeError, IndexError):
                    continue
    # Also include pose data.
    for point in poses:
        try:
            all_x.append(point[0])
            all_y.append(point[1])
        except (TypeError, IndexError):
            continue

    if all_x and all_y:
        margin = 1
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Label the axes and add a title.
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Lanes and Pose Animation")
    ax.legend()
    ax.grid(True)
    
    # Create the pose animation.
    ani = animate_pose(ax, poses, interval=200)
    
    # Ensure a tight layout and display the plot.
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Replace with the path to your JSON file.
    json_file = '/home/cubos98/catkin_ws/src/Vehicle/results_reality/real_camera_medium_poses.json'
    main(json_file)

