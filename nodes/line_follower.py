#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image as SensorImage
from geometry_msgs.msg import Twist
from mixed_reality.msg import Control, WaypointList, Waypoint, SimPose
from cv_bridge import CvBridge
import math

class LineFollower:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('line_follower', anonymous=True)

        # Create a CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Subscribe to the image topic (replace 'camera/image' with your actual image topic)
        rospy.Subscriber("/sim/image", SensorImage, self.image_callback)

        # Publisher for car control commands (steering and throttle)
        self.pub_throttle_steering = rospy.Publisher("model/throttle_steering", Control, queue_size=10)
        rate = rospy.Rate(50)
        
        # Control parameters
        self.steering_gain = 0.004  # Tuning parameter for steering sensitivity
        self.speed = 0.2  # Constant speed for throttle control

    def image_callback(self, msg):
        # Convert the ROS image message to an OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        height, width, _ = frame.shape
        frame_botton_half = frame[0:height // 2, :]  # Only the botton half

        # Process the image to find the yellow line
        steering_angle = self.process_image(frame_botton_half)

        # Create and publish the control command
        #steering_angle = np.clip(steering_angle, -1.0, 1.0)
        if steering_angle is not None:
            steering_angle = max(-1, min(steering_angle, 1))
            self.pub_throttle_steering.publish(Control(self.speed, steering_angle, False, False, False))

        else:
            self.pub_throttle_steering.publish(Control(0.0 , 0.0, True, True, True))

    def process_image(self, frame):
        # Convert the image to HSV color space for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the yellow color range for masking
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create a binary mask where yellow is detected
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Calculate moments of the binary image to find the center of the yellow line
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            # Calculate the center of the yellow line
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            # Calculate the steering angle based on the position of the yellow line
            # Center of the image frame
            height, width, _ = frame.shape
            center_x = width // 2

            # Error is the difference between the line position and the center of the image
            error_x = cx - center_x

            # Calculate the steering angle (proportional control)
            steering_angle = self.steering_gain * error_x
            return steering_angle
        else:
            # If no yellow line is detected, return a zero steering angle
            return None

    def run(self):
        # Spin the node to keep it active
        rospy.spin()

if __name__ == '__main__':
    try:
        line_follower = LineFollower()
        line_follower.run()
    except rospy.ROSInterruptException:
        pass
