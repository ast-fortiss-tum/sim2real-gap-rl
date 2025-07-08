#!/usr/bin/env python

from __future__ import print_function

from mixed_reality.srv import GetEnvironment, GetEnvironmentResponse
import rospy

def handle_get_environment(req):
    #TODO: read from road definition file, somehow also include the tracked obstacles
    
    return GetEnvironmentResponse("To do")

def environment_server():
    rospy.init_node('environment')
    s = rospy.Service('get_environment', GetEnvironment, handle_get_environment)
    print("Ready to send environment information.")
    rospy.spin()

if __name__ == "__main__":
    environment_server()



