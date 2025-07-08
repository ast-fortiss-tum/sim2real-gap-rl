#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError



video_source = 'http://team10.local:8890/video'




def camera_node():
    print("Starting camera node")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("____________________________________")
        print("[CAMERA CLIENT]: Could not open video source.")
    print("Video source successfully opened")
    

    rospy.init_node("camera_node", anonymous=True)
    pub_camera = None
    bridge = CvBridge()

    stored = False

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            print("____________________________________")
            print("[CAMERA CLIENT]: Failed to capture frame.")
            break
        real_image = cv2.resize(frame, (320, 240))
        if not stored:
            cv2.imwrite("dummy_photo.jpg",real_image)
            stored = True
        try:
            image_message = bridge.cv2_to_imgmsg(real_image, encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        #publish camera image
        if pub_camera is None:
            pub_camera = rospy.Publisher("camera", Image,queue_size=10)
        pub_camera.publish(image_message)
        #print("published image")

    print("[CAMERA CLIENT]: QUIT")







if __name__ == '__main__':
    try:
        camera_node()
    except rospy.ROSInterruptException:
        pass