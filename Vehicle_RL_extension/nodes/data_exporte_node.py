#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as SensorImage
from mixed_reality.msg import SimPose, Floats, Control, Obstacles
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
import cv2

class DataExporter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('data_exporter_node', anonymous=True)
        
        # Create a CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Flags to track if the first value has been received
        self.image_received = False
        self.speed_received = False
        self.euler_received = False

        # Subscribe to the topics
        rospy.Subscriber('/sim/image', SensorImage, self.image_callback)
        rospy.Subscriber('/sim/speed', Float64, self.speed_callback)
        rospy.Subscriber('/sim/euler', SimPose, self.euler_callback)

        # Keep the node running until all data is received
        rospy.spin()

    def image_callback(self, msg):
        if not self.image_received:
            try:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                # Save the image to a file
                cv2.imwrite('first_image_received.jpg', cv_image)
                rospy.loginfo('First image received and saved as first_image_received.jpg')
                
                self.image_received = True
                self.check_completion()
            except CvBridgeError as e:
                rospy.logerr('Failed to convert image: %s', e)

    def speed_callback(self, msg):
        if not self.speed_received:
            speed_value = msg.data
            rospy.loginfo('First speed value received: %f', speed_value)
            
            # Save the speed value to a file
            with open('first_speed_received.txt', 'w') as f:
                f.write('Speed: {}\n'.format(speed_value))
            
            self.speed_received = True
            self.check_completion()

    def euler_callback(self, msg):
        if not self.euler_received:
            euler_angles = {'x': msg.x, 'y': msg.y, 'z': msg.z}
            rospy.loginfo('First Euler angles received: %s', euler_angles)
            
            # Save the Euler angles to a file
            with open('first_euler_received.txt', 'w') as f:
                f.write('Euler Angles:\n')
                f.write('Roll (x): {}\n'.format(msg.x))
                f.write('Pitch (y): {}\n'.format(msg.y))
                f.write('Yaw (z): {}\n'.format(msg.z))
            
            self.euler_received = True
            self.check_completion()

    def check_completion(self):
        if self.image_received and self.speed_received and self.euler_received:
            rospy.loginfo('All first values received and exported.')
            rospy.signal_shutdown('Data export completed.')

if __name__ == '__main__':
    try:
        DataExporter()
    except rospy.ROSInterruptException:
        pass
