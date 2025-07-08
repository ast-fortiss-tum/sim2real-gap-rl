#!/usr/bin/env python3

import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image as SensorImage

from PIL import Image
import time
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError



real_image = None
sim_image = None
bridge = CvBridge()

def new_camera_image(msg):
    global real_image
    global bridge
    try:
        real_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        print(e)


def new_simulator_image(msg):
    global sim_image
    sim_image = Image.frombytes(mode="RGBA", size=[msg.width, msg.height], data=msg.data)






def mixer_node():
    global sim_image
    global real_image
    print("Starting image mixing node")

    rospy.init_node("mixer_node", anonymous=True)
    rospy.Subscriber("camera", SensorImage, new_camera_image)
    rospy.Subscriber("sim/image", SensorImage, new_simulator_image)
    pub_mixed = rospy.Publisher("mixed_image", SensorImage, queue_size=10)

    while not rospy.is_shutdown():
        #TODO: checar si este sleep es necesario
        time.sleep(0.01)
        if sim_image is not None and real_image is not None:
            open_cv_image=np.array(sim_image.copy())
            # open_cv_image = np.array(open_cv_image)
            rgb = np.asarray(open_cv_image, dtype=np.float32)[:, :, :3]
            alpha = np.asarray(open_cv_image, dtype=np.float32)[:, :, 3]

            resized_frame = cv2.resize(real_image, (320, 240))
            img = resized_frame

            # Convert the image to BGR format for display
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Create a mask from the alpha channel (convert to 3 channels)
            mask = np.repeat(alpha[:, :, np.newaxis], 3, axis=2) / 255.0

            # Perform alpha blending
            blended = (bgr * mask + img * (1 - mask)).astype(np.uint8)

            # Publish blended image
            #print(f"blended type {type(blended)}")
            #print(blended.shape)
            #blended = cv2.resize(blended, (320,240))
            #cv2.imshow("blended", blended)
            #cv2.waitKey(0)
            try:
                image_message = bridge.cv2_to_imgmsg(blended, encoding="bgr8")
                print("image converted")
                pub_mixed.publish(image_message)
                print("published")
            except CvBridgeError as e:
                print(e)

            #pub_mixed.publish(blended)


            
            '''
            image_int = np.asarray(blended, dtype=np.uint8)
            if send:
                #NOTA: send = REMOTA_VISUALIZATION
                #TODO: preguntar que hace REMOTE_VISUALIZATION (why would it be off in the fisrt place?)
                # Compress the image as JPEG
                _, compressed_data = cv2.imencode('.jpg', image_int, [cv2.IMWRITE_JPEG_QUALITY, 40])
                # print(len(compressed_data))
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

                # Send the compressed image data via UDP
                sock.sendto(compressed_data, (ip, port))
            '''

        elif sim_image is not None:
            open_cv_image=np.array(sim_image.copy())
            # open_cv_image = np.array(open_cv_image)
            rgb = np.asarray(open_cv_image, dtype=np.float32)[:, :, :3]
            alpha = np.asarray(open_cv_image, dtype=np.float32)[:, :, 3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Create a mask from the alpha channel (convert to 3 channels)
            mask = np.repeat(alpha[:, :, np.newaxis], 3, axis=2) / 255.0
            blended = (bgr * mask + [0,0,0] * (1 - mask)).astype(np.uint8)
        

            # Publish "blended" image
            #pub_mixed.publish(blended)

            
            '''
            image_int = np.asarray(blended, dtype=np.uint8)
            if send:
                # Compress the image as JPEG
                _, compressed_data = cv2.imencode('.jpg', image_int, [cv2.IMWRITE_JPEG_QUALITY, 40])
                # print(len(compressed_data))
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

                # Send the compressed image data via UDP
                sock.sendto(compressed_data, (ip, port))
            '''


    print("[IMAGE MIXING NODE]: QUIT")













if __name__ == '__main__':
    try:
        mixer_node()
    except rospy.ROSInterruptException:
        pass