#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import time

from PIL import Image
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import Float64      # For the sim/cte topic
from mixed_reality.msg import Control
from cv_bridge import CvBridge

from stable_baselines3 import SAC  # Import the Stable-Baselines3 SAC
from training.version7_RL import preprocess_image

# Path to the Stable-Baselines3 SAC model
MODEL_PATH = '/home/cubos98/catkin_ws/src/Vehicle/final_models/very_good_vanilla_m1.zip'

FIXED_THROTTLE = True
STEERING = 0
THROTTLE = 1

model = None
pub_throttle_steering = None
bridge = CvBridge()

prev_time = None

# Global variable to store the latest CTE value and the threshold
current_cte = 0.0
CTE_THRESHOLD = 4.0

def cte_callback(msg):
    """
    Callback for the sim/cte topic.
    Updates the global current_cte variable with the latest cross track error.
    If the absolute value of current_cte exceeds CTE_THRESHOLD, it publishes a
    stop command and then shuts down the node.
    """
    global current_cte, pub_throttle_steering
    current_cte = msg.data
    rospy.loginfo(f"Received CTE: {current_cte}")
    
    if abs(current_cte) > CTE_THRESHOLD:
        rospy.logwarn("CTE threshold exceeded. Sending stop command and shutting down node.")
        
        # Ensure publisher exists. If not, create one.
        if pub_throttle_steering is None:
            pub_throttle_steering = rospy.Publisher("model/throttle_steering", Control, queue_size=10)
            # Allow publisher time to register with the ROS master
            rospy.sleep(0.5)
        
        # Create a stop command message
        stop_msg = Control()
        stop_msg.throttle = 0.0
        stop_msg.steering = 0.0
        stop_msg.brake = True
        stop_msg.reverse = False
        stop_msg.stopping = True

        pub_throttle_steering.publish(stop_msg)
        
        # Give some time for the message to be sent before shutting down
        rospy.sleep(0.5)
        rospy.signal_shutdown("CTE threshold exceeded")

def new_image(msg):
    """
    Callback for the image topic.
    Processes the image using the SAC model and publishes control commands.
    No processing is done if the CTE threshold has been exceeded.
    """
    global prev_time, FIXED_THROTTLE, model, pub_throttle_steering, bridge, current_cte

    # If CTE threshold is exceeded, skip processing new images.
    if abs(current_cte) > CTE_THRESHOLD:
        return

    now = time.time()
    if prev_time is not None and (now - prev_time) < 0.00:
        return
    prev_time = now

    # Convert ROS image to OpenCV image and preprocess it
    image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    image = preprocess_image(image)

    # Predict actions using the SAC model
    action, _ = model.predict(image, deterministic=True)
    steering = action[0]

    if FIXED_THROTTLE:
        throttle = 0.8

    # Create and populate the control command message
    if pub_throttle_steering is None:
        pub_throttle_steering = rospy.Publisher("model/throttle_steering", Control, queue_size=10)
    
    cmd_msg = Control()
    cmd_msg.throttle = throttle
    cmd_msg.steering = steering
    cmd_msg.brake = False
    cmd_msg.reverse = False
    cmd_msg.stopping = False

    pub_throttle_steering.publish(cmd_msg)

def model_node():
    """
    Initializes the node, loads the SAC model, and sets up the subscribers.
    """
    global MODEL_PATH, model

    rospy.loginfo(f"Loading model: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH, device="cpu")  # Load the Stable-Baselines3 SAC model
    rospy.loginfo("Model loaded successfully.")

    rospy.init_node("model_node", anonymous=True)

    # Subscribe to the image topic and the sim/cte topic
    rospy.Subscriber("sim/image", SensorImage, new_image)
    rospy.Subscriber("sim/cte", Float64, cte_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        model_node()
    except rospy.ROSInterruptException:
        pass
