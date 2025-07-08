#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int64, String
from mixed_reality.msg import SimPose, WaypointList, Waypoint, Obstacles

import json
import math
import time


THRESHOLD = 7

simulator_pose = [0.0,0.0,0.0]
current_waypoint_index = 0
current_lane = "right"
obstacles = {}
""" Example obstacle dictionary:
{
    "barrel": [8,3]
    "cone": [9,7.3]
}
"""

def new_obstacle(msg):
    global obstacles
    for obstacle in msg.data:
        obstacles[obstacle.name] = [obstacle.x, obstacle.y]

def new_current_waypoint_index(msg):
    global current_waypoint_index
    current_waypoint_index = msg.data

def new_sim_pose(msg):
    global simulator_pose
    simulator_pose = [msg.x,msg.z,msg.yaw]    #TODO:check if pitch is appropriate (it's not used though, so it's not super important)

def new_lane(msg):
    global current_lane
    if msg.data in ["left", "right", "center"]:
        current_lane = msg.data


def obstacleAvoidance_node():
    global THRESHOLD
    global obstacles
    global current_waypoint_index
    global current_lane

    #Load lane data from JSON map file
    f = open("map.json", "r")
    map_data = json.loads(f.read())
    f.close()
    sim_waypoint_list_left =  map_data["left_lane"]
    sim_waypoint_list_right = map_data["right_lane"]
    sim_waypoint_list_center = map_data["sim_waypoint_list"]



    rospy.init_node("obstacleAvoidance_node.py", anonymous=True)
    rospy.Subscriber("obstacles", Obstacles, new_obstacle)
    rospy.Subscriber("control/current_waypoint_index", Int64, new_current_waypoint_index)
    rospy.Subscriber("sim/euler", SimPose, new_sim_pose)
    rospy.Subscriber("current_lane", String, new_lane)
    pub_waypoint = rospy.Publisher("waypoints", WaypointList, queue_size=10)



    while not rospy.is_shutdown():
        debug_msg = ""
        for obstacle in obstacles.copy():
            closest_lane = None
            min_distance = float('inf')
            # Slice waypoints based on current waypoint index
            current_index = current_waypoint_index   #TODO: careful with current_waypoint_index being uninitialized in case there are no waypoints yet
            left_waypoints = sim_waypoint_list_left[current_index:]
            right_waypoints = sim_waypoint_list_right[current_index:]

            # Find closest lane
            for lane, waypoints in [("left", left_waypoints), ("right", right_waypoints)]:
                for waypoint in waypoints:
                    distance = math.sqrt((obstacles[obstacle][0] - waypoint[0]) ** 2 + (obstacles[obstacle][1] - waypoint[1]) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_lane = lane

            debug_msg = debug_msg + str(obstacle)+ " in " +str(closest_lane)+" lane, current lane: " +str(current_lane)+"\n"


            ego_x, ego_y = simulator_pose[0], simulator_pose[1]
            #print(f"Car position: f{[ego_x,ego_y]}\nObstacle position: {obstacles[obstacle][0:2]}")
            distance = math.sqrt((obstacles[obstacle][0] - ego_x) ** 2 + (obstacles[obstacle][1] - ego_y) ** 2)

            debug_msg = debug_msg+"Distance is " +str(distance)+"\n"

            if distance < THRESHOLD and closest_lane and closest_lane == current_lane:
                print(f"\nFound {obstacle} in pose: {obstacles[obstacle]}, Lane: {closest_lane}")
                print(f"current_waypoint index {current_waypoint_index}")
                print("Changing lane")
                if closest_lane == "right":
                    wp_list = list(map(lambda pair:Waypoint(pair[0], pair[1]), sim_waypoint_list_center))
                    pub_waypoint.publish(False,wp_list)
                    time.sleep(0.3)
                    wp_list = list(map(lambda pair:Waypoint(pair[0], pair[1]), sim_waypoint_list_left))
                    pub_waypoint.publish(False,wp_list)
                    current_lane = "left"
                elif closest_lane == "left":
                    wp_list = list(map(lambda pair:Waypoint(pair[0], pair[1]), sim_waypoint_list_center))
                    pub_waypoint.publish(False,wp_list)
                    time.sleep(0.3)
                    wp_list = list(map(lambda pair:Waypoint(pair[0], pair[1]), sim_waypoint_list_right))
                    pub_waypoint.publish(False,wp_list)
                    current_lane = "right"
            time.sleep(0.01)
        #print(debug_msg)
            



if __name__ == '__main__':
    try:
        obstacleAvoidance_node()
    except rospy.ROSInterruptException:
        pass