#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import time

from PIL import Image
from sensor_msgs.msg import Image as SensorImage
from mixed_reality.msg import Control
from cv_bridge import CvBridge

from stable_baselines3 import SAC  # Import the Stable-Baselines3 SAC
from version7_RL import preprocess_image

# Path to the Stable-Baselines3 SAC model
MODEL_PATH = '/home/cubos98/catkin_ws/src/Vehicle/sac_donkeycar_checkpoints/sac_donkeycar_70000_steps.zip'

FIXED_THROTTLE = True
STEERING = 0
THROTTLE = 1

model = None
pub_throttle_steering = None
bridge = CvBridge()

def new_image(msg):
    global FIXED_THROTTLE
    global model
    global pub_throttle_steering
    global bridge

    # Convert ROS image to OpenCV image
    image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
    #image = image[None, ...]
    image = preprocess_image(image)

    # Predict actions using the SAC model
    action, _ = model.predict(image, deterministic=True)
    print(f"Action: {action}")
    steering = action[0]

    if FIXED_THROTTLE:
        throttle = 0.5

    # Publish throttle and steering commands
    if pub_throttle_steering is None:
        pub_throttle_steering = rospy.Publisher("model/throttle_steering", Control, queue_size=10)
    msg = Control(throttle, steering, False, False, False)
    pub_throttle_steering.publish(msg)

def model_node():
    global MODEL_PATH
    global model

    print(f"Loading model: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH)  # Load the Stable-Baselines3 SAC model
    print("Model loaded successfully.")

    rospy.init_node("model_node", anonymous=True)
    rate = rospy.Rate(3)  # Set the desired frequency to 3 Hz
    rospy.Subscriber("/sim/image", SensorImage, new_image)

    while not rospy.is_shutdown():
        # Let ROS handle the callbacks and maintain a 3 Hz loop
        rate.sleep()

if __name__ == '__main__':
    try:
        model_node()
    except rospy.ROSInterruptException:
        pass
