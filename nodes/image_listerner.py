#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageListener:
    def __init__(self):
        self.bridge = CvBridge()
        # Replace '/camera/image_raw' with your image topic
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Now cv_image is a NumPy array with BGR data
        # Convert BGR to RGB if needed
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Access RGB data
        height, width, channels = rgb_image.shape
        rospy.loginfo("Image size: {}x{}, Channels: {}".format(width, height, channels))

        # Example: Get RGB values at the center pixel
        x = width // 2
        y = height // 2
        (r, g, b) = rgb_image[y, x]
        rospy.loginfo("RGB at ({}, {}): ({}, {}, {})".format(x, y, r, g, b))

        # Perform any additional processing here

def main():
    rospy.init_node('image_listener', anonymous=True)
    listener = ImageListener()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()
