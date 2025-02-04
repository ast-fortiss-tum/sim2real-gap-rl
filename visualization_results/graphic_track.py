import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_lanes(file_path):
    """
    Load a JSON file that contains a "lanes" key. Each lane is assumed to be a list of points,
    where each point is a dictionary with "x" and "y" coordinates. This function plots each lane,
    applying custom styles for lanes 0, 1, and 2:
      - Lane 0 and 2: white color, solid lines.
      - Lane 1: yellow color, dashed lines.
    """
    # Load JSON data from file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract lanes
    lanes = data.get("lanes")
    if lanes is None:
        print("The JSON file does not contain a 'lanes' key.")
        return

    # Create a new figure. Adjust background if using white lines.
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    # Optionally set a dark background so white lines show up:
    ax.set_facecolor('grey')
    
    # Plot each lane with the specified styles
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
            except (TypeError, KeyError):
                print(f"Skipping a point in lane {lane_index} because it doesn't have the expected format.")
        
        if not (x_coords and y_coords):
            print(f"Lane {lane_index} has no valid points.")
            continue
        
        # Set default style values
        color = None
        linestyle = '-'
        label = f"Lane {lane_index}"
        
        # Apply custom styles for lane 0, 1, and 2
        if lane_index == 0 or lane_index == 1:
            color = 'white'
            linestyle = '-'  # solid line
        elif lane_index == 2:
            color = 'yellow'
            linestyle = '--'  # dashed line
        else:
            # Use default styling for other lanes if needed
            color = None  # None lets Matplotlib choose a color
        
        plt.plot(x_coords, y_coords, color=color, linestyle=linestyle, label=label)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Plot of Lanes Coordinates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def animate_pose(file_path):
    """
    Load a JSON file that contains a "pose" key. The "pose" value is assumed to be a list of [x, y] coordinates.
    This function creates an animation where a point moves along these coordinates.
    """
    # Load JSON data from file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract pose data
    poses = data.get("pose")
    if poses is None:
        print("The JSON file does not contain a 'pose' key.")
        return

    # Ensure that poses is a list of [x, y] pairs.
    if not isinstance(poses, list) or not all(isinstance(p, list) and len(p) >= 2 for p in poses):
        print("The 'pose' key does not contain a valid list of [x, y] pairs.")
        return

    poses = poses[::100]

    # Extract x and y coordinates for setting plot limits
    x_coords = [p[0] for p in poses]
    y_coords = [p[1] for p in poses]

    # Create a plot with appropriate limits and title.
    fig, ax = plt.subplots()
    margin = 1
    ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    ax.set_title("Pose Animation")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Create a point (red circle) with a larger marker size.
    point, = ax.plot([], [], 'ro', markersize=12)  # Increase markersize as needed.

    # Initialization function for the animation.
    def init():
        point.set_data([], [])
        return point,

    # Update function for each frame of the animation.
    def update(frame):
        x, y = poses[frame]
        point.set_data(x, y)
        return point,

    # Create the animation.
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(len(poses)),
        init_func=init,
        blit=True,
        interval=200  # Adjust the interval (milliseconds) to control the animation speed.
    )

    plt.show()

if __name__ == '__main__':
    # Replace 'example.json' with your JSON file path
    json_file = '/home/cubos98/catkin_ws/src/Vehicle/results_reality/fake_camera_inc_poses.json'
    plot_lanes(json_file)
    animate_pose(json_file)
