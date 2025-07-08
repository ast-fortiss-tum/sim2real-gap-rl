#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from mixed_reality.msg import SimPose, Obstacles
import socket
import time
import math
import struct
import re
import tf


class ViconTrackerUtils:
    def __init__(self):
        self.FRAME_NUMBER_OFFSET = 0
        self.FRAME_NUMBER_SIZE = 4
        self.ITEMS_IN_BLOCK_SIZE = 1
        self.OBJECT_ID_SIZE = 1
        self.ITEM_DATA_SIZE_SIZE = 2
        self.OBJECT_NAME_SIZE = 24
        self.TRANSLATION_SIZE = 8
        self.ROTATION_SIZE = 8
        self.OBJECT_ID_OFFSET = self.FRAME_NUMBER_OFFSET + self.FRAME_NUMBER_SIZE + self.ITEMS_IN_BLOCK_SIZE
        self.ITEM_DATA_SIZE_OFFSET = self.OBJECT_ID_OFFSET + self.OBJECT_ID_SIZE
        self.OBJECT_NAME_OFFSET = self.ITEM_DATA_SIZE_OFFSET + self.ITEM_DATA_SIZE_SIZE
        self.TRANSLATION_X_OFFSET = 32
        self.TRANSLATION_Y_OFFSET = 40
        self.TRANSLATION_Z_OFFSET = 48
        self.ROTATION_X_OFFSET = 56
        self.ROTATION_Y_OFFSET = 64
        self.ROTATION_Z_OFFSET = 72

    def parse_translation(self, data):
        translation_x = struct.unpack("<d", data[self.TRANSLATION_X_OFFSET:self.TRANSLATION_X_OFFSET + 8])[0]
        translation_y = struct.unpack("<d", data[self.TRANSLATION_Y_OFFSET:self.TRANSLATION_Y_OFFSET + 8])[0]
        translation_z = struct.unpack("<d", data[self.TRANSLATION_Z_OFFSET:self.TRANSLATION_Z_OFFSET + 8])[0]
        return translation_x, translation_y, translation_z

    def parse_rotation(self, data):
        rotation_x_rad = struct.unpack("<d", data[self.ROTATION_X_OFFSET:self.ROTATION_X_OFFSET + 8])[0]
        rotation_y_rad = struct.unpack("<d", data[self.ROTATION_Y_OFFSET:self.ROTATION_Y_OFFSET + 8])[0]
        rotation_z_rad = struct.unpack("<d", data[self.ROTATION_Z_OFFSET:self.ROTATION_Z_OFFSET + 8])[0]
        return rotation_x_rad, rotation_y_rad, rotation_z_rad

    def parse_frame_number(self, data):
        frame_number = struct.unpack("<I", data[self.FRAME_NUMBER_OFFSET:self.FRAME_NUMBER_OFFSET + self.FRAME_NUMBER_SIZE])[0]
        return frame_number

    def parse_object_id(self, data):
        object_id = struct.unpack("<B", data[self.OBJECT_ID_OFFSET:self.OBJECT_ID_OFFSET + self.OBJECT_ID_SIZE])[0]
        return object_id

    def parse_object_name(self, data):
        object_name = data[self.OBJECT_NAME_OFFSET:self.OBJECT_NAME_OFFSET + self.OBJECT_NAME_SIZE].decode().rstrip('\x00')
        # Replace invalid characters and convert to lowercase
        #object_name = re.sub(r'[^a-zA-Z0-9_]', '_', object_name).lower()
        object_name = re.sub(r'[^a-zA-Z0-9_]', '_', object_name)
        return object_name


class For_convertion_utils():
    def __init__(self,size_factor,x_map_shift,y_map_shift,angle_shift):
        self.size_factor=size_factor
        self.x_map_shift=x_map_shift
        self.y_map_shift=y_map_shift
        self.angle_shift=angle_shift
    
    def real2sim_xyzypr(self,pose,orientation):
        x=pose[0]*self.size_factor+self.x_map_shift
        y=pose[1]*self.size_factor+self.y_map_shift
        angle=-orientation[2]+self.angle_shift
        return [x,y,angle]

    def sim2real_xyzypr(self,pose,orientation):
        x=(pose[0]-self.x_map_shift)/self.size_factor
        y=(pose[2]-self.x_map_shift)/self.size_factor
        angle=-(orientation[1]-self.angle_shift)
        return [x,y,angle]
    
    def real2sim_xyp(self,pose):
        x,y,p=pose
        x=x*self.size_factor+self.x_map_shift
        y=y*self.size_factor+self.y_map_shift
        angle=-p+self.angle_shift
        return [x,y,angle]

    def sim2real_xyp(self,pose):
        x,y,p=pose
        x=(x-self.x_map_shift)/self.size_factor
        y=(y-self.y_map_shift)/self.size_factor
        angle=-(p-self.angle_shift)
        return [x,y,angle]

SIZE_FACTOR=7.33
X_MAP_SHIFT=48
Y_MAP_SHIFT=50
ANGLE_SHIFT=0


class Status:
    def __init__(self):
        self.is_running = True
        self.real_pose = [0.0, 0.0, 0.0]
        self.real_orientation = [0.0, 0.0, 0.0]
        self.real_speed = 0.0


def tracking_node():
    print("Tracking node running")
    rospy.init_node('tracking_node', anonymous=True)

    status = Status()
    last_update_time = time.time()
    last_position = None
    tracking_ip = '0.0.0.0'
    tracking_port = 51001
    for_conversions = For_convertion_utils(SIZE_FACTOR,X_MAP_SHIFT,Y_MAP_SHIFT,ANGLE_SHIFT)

    # Initialize socket connection
    vicon = ViconTrackerUtils()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind((tracking_ip, tracking_port))
    print("connection established successfully")
    rate = rospy.Rate(20)  # 50 Hz

    pose_pub = None
    speed_pub = None
    obstacle_pub = None
    while not rospy.is_shutdown() and status.is_running:
        data, _ = sock.recvfrom(256)
        #print("got data")
        #print(data)
        object_name = vicon.parse_object_name(data)  # Get the object name
        if pose_pub is None:
            pose_pub = rospy.Publisher('donkey/pose', PoseStamped, queue_size=10)
        if speed_pub is None:
            speed_pub = rospy.Publisher("donkey/speed", Float64, queue_size=10)
        if obstacle_pub is None:
            obstacle_pub = rospy.Publisher("obstacles", Obstacles, queue_size=10)

        #define vars of donkey
        translation = None
        rotation = None
        #translation = vicon.parse_translation(data)
        #rotation = vicon.parse_rotation(data)

        obstacles = []
        items_in_block = struct.unpack("<B", data[4:5])[0]
        #print(f"There are {items_in_block} items in this data block")
        object_base_offset = 5
        for object in range(0,items_in_block):
            #print(f"BASE OFFSET: {object_base_offset}")
            data_size_offset = object_base_offset + 1
            name_offset = data_size_offset + 2
            trans_x_offset = name_offset + 24

            #Get the object's data
            item_data_size = struct.unpack("<H", data[data_size_offset : data_size_offset+2])[0]
            #print(f"item data size: {item_data_size}")
            name = data[name_offset : name_offset+24].decode().rstrip('\x00')
            #print(f"The name of the object is {name}")
            translation_x = struct.unpack("<d", data[trans_x_offset:trans_x_offset + 8])[0]
            translation_y = struct.unpack("<d", data[trans_x_offset + 8:trans_x_offset + 16])[0]
            translation_z = struct.unpack("<d", data[trans_x_offset + 16:trans_x_offset + 24])[0]
            rotation_x_rad = struct.unpack("<d", data[trans_x_offset + 24:trans_x_offset + 32])[0]
            rotation_y_rad = struct.unpack("<d", data[trans_x_offset + 32:trans_x_offset + 40])[0]
            rotation_z_rad = struct.unpack("<d", data[trans_x_offset + 40:trans_x_offset + 48])[0]

            if name=="Donkey":
                translation = [translation_x, translation_y, translation_z]
                rotation = [rotation_x_rad, rotation_y_rad, rotation_z_rad]

                x = translation[0] / 1000
                y = translation[1] / 1000
                z = translation[2] / 1000
                status.real_pose = [x, y, z]
                status.real_orientation = rotation

                if last_position is not None:
                    current_time = time.time()
                    time_diff = current_time - last_update_time
                    if time_diff > 0:  # Ensure time difference is nonzero to avoid division by zero
                        # Calculate speed components
                        speed_x = (x - last_position[0]) / time_diff
                        speed_y = (y - last_position[1]) / time_diff
                        speed_z = (z - last_position[2]) / time_diff
                        speed = math.sqrt(speed_x ** 2 + speed_y ** 2)
                        status.real_speed = speed
                        #publish real speed of the car
                        speed_pub.publish(speed)

                # Update last position and last update time
                last_position = [x, y, z]
                last_update_time = time.time()

                # Convert rotation to quaternion
                quaternion = tf.transformations.quaternion_from_euler(status.real_orientation[0],
                                                                    status.real_orientation[1],
                                                                    status.real_orientation[2])

                # Publish the combined pose and orientation
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "map"  # or any other frame
                pose_msg.pose.position.x = status.real_pose[0]
                pose_msg.pose.position.y = status.real_pose[1]
                pose_msg.pose.position.z = status.real_pose[2]
                pose_msg.pose.orientation.x = quaternion[0]
                pose_msg.pose.orientation.y = quaternion[1]
                pose_msg.pose.orientation.z = quaternion[2]
                pose_msg.pose.orientation.w = quaternion[3]

                pose_pub.publish(pose_msg)
            else:
                x,y,_ = for_conversions.real2sim_xyp([translation_x/1000,translation_y/1000,0])
                obstacles.append(SimPose(name,x,y,translation_z/1000,0,0,0))

            object_base_offset += 3+item_data_size

        obstacle_pub.publish(obstacles)

        

        #time.sleep(1)
        #rate.sleep()

    print("[TRACKING CLIENT]: QUIT")


if __name__ == '__main__':
    try:
        tracking_node()
    except rospy.ROSInterruptException:
        pass