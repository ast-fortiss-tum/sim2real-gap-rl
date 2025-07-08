#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, String, Bool
from mixed_reality.msg import Control


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def calculate(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output



#TODO: move the following to a config file later
SIZE_FACTOR=7.33

#variables that might be changed through ROS later
speed_threshold_min = 0.4
simulated_speed = 0.0
throttle_multiplier = 0.3
driving = False

def new_speed_threshold_min(msg):
    global speed_threshold_min
    speed_threshold_min = msg.data
    print(f"\nTARGET SPEED: {speed_threshold_min}")

def new_is_running(msg):
    global is_running
    is_running = msg.data

def new_state(msg):
    global state
    state = msg.data

def new_simulated_speed(msg):
    global simulated_speed
    simulated_speed = msg.data

def new_control(msg):
    global driving
    driving = not (msg.stopping or msg.reverse or msg.brake)


def throttle_sim():
    global throttle_multiplier
    global driving
    global speed_threshold_min
    global simulated_speed
    print("simulated throttle node running")

    rospy.init_node('throttle_node_sim', anonymous=True)
    rospy.Subscriber('keyboard/speed', Float64, new_speed_threshold_min)
    rospy.Subscriber('sim/speed', Float64, new_simulated_speed)
    rospy.Subscriber("control/throttle_steering", Control, new_control)
    throttle_pub = None
    rate = rospy.Rate(50)

    pid_controller = PIDController(kp=0.5, ki=0.0001, kd=0.1)

    while not rospy.is_shutdown():
        if throttle_pub is None:
            throttle_pub = rospy.Publisher('throttle_sim/multiplier', Float64, queue_size=10)

        if driving:
            throttle_correction = pid_controller.calculate(speed_threshold_min, simulated_speed/SIZE_FACTOR)
            print(f"\nSpeed:\t{round(simulated_speed/SIZE_FACTOR, 3)}\nTarget:\t{speed_threshold_min}\nCorrection:\t{round(throttle_correction, 3)}")
                
            current_throttle = throttle_multiplier
            current_throttle += throttle_correction
            
            throttle_multiplier = min(current_throttle,0.39)
            throttle_multiplier = max(throttle_multiplier,0)
            throttle_pub.publish(throttle_multiplier)
            print(f"Multiplier: {round(throttle_multiplier,3)}")

        rate.sleep()




if __name__ == '__main__':
    try:
        throttle_sim()
    except rospy.ROSInterruptException:
        pass