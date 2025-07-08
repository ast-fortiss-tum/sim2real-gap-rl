#!/usr/bin/env python3






#BIG TODO: DO NOT UPDATE THE GLOBAL THROTTLE VALUE WITH THE THROTTLE MULTIPLIER

















import asyncio
import rospy
from mixed_reality.msg import Control
from std_msgs.msg import Float64, Bool
import websockets
import json
import time
import asyncio

uri = "ws://team10.local:8887/wsDrive"
websocket = None

throttle=0
steering=0
throttle_multiplier = 0.3
stopping = False
brake = False
reverse = False
going = False

CHECKDONE = False

def new_throttle_steering(msg):
    global throttle
    global steering
    global stopping
    global brake
    global reverse

    throttle = msg.throttle
    steering = msg.steering

    #throttle*=throttle_multiplier
    if abs(steering)>1:
        steering = 0.99 if steering>0 else -0.99
    stopping = msg.stopping
    brake = msg.brake
    reverse=msg.reverse

def new_multiplier(msg):
    global throttle_multiplier
    throttle_multiplier = msg.data

async def send_commands(websocket):
    global throttle_multiplier
    global throttle
    global steering
    global stopping
    global brake
    global reverse

    if stopping or brake:
        command = {
            "throttle": throttle*throttle_multiplier,
            "angle": steering
        }
        message = json.dumps(command)
        await websocket.send(message)
        time.sleep(0.3)
        command = {
            "throttle": 0,
            "angle": steering
        }
        message = json.dumps(command)
        await websocket.send(message)
    elif reverse:
        #throttle*=3
        command = {
            "throttle": throttle*throttle_multiplier*3,
            "angle": steering
        }
        message = json.dumps(command)
        await websocket.send(message)
    else:
        command = {
            "throttle": throttle*throttle_multiplier,
            "angle": steering
        }
        message = json.dumps(command)
        await websocket.send(message)

def new_going(msg):
    global going
    command = "going" if msg.data else "stopping"
    print(f"Received {command} command")
    going = msg.data



async def car_node():
    global throttle
    global steering
    global throttle_multiplier
    global going
    print("Starting car node")

    rospy.init_node("car_node",anonymous=True)
    rospy.Subscriber("control/throttle_steering", Control, new_throttle_steering)
    rospy.Subscriber("throttle/multiplier", Float64, new_multiplier)
    rospy.Subscriber("/going", Bool, new_going)
    rate = rospy.Rate(20)   #Rate is set to 20Hz because the car reports running at 20Hz

    global websocket

    connected = False
    while not connected:
        try:
            async with websockets.connect(uri) as websocket:
                connected = True
                print("connection established succesfully!")

                while not rospy.is_shutdown():
                    if going:
                        await send_commands(websocket)
                        print(f"throttle {throttle}\nsteering: {steering}\nmultiplier {throttle_multiplier}")
                    rate.sleep()
                    """
                    command = {
                        "throttle": throttle,
                        "angle": steering
                    }
                    message = json.dumps(command)
                    print(message)
                    print(f"multiplier {throttle_multiplier}")
                    await websocket.send(message)
                    rate.sleep()
                    """

                print("Rospy shutdown, stopping car")
                command = {
                    "throttle": 0.0,
                    "angle": 0.0
                }
                message = json.dumps(command)
                await websocket.send(message)
                print("Stop command sent")
                await websocket.close()
                print("Connection closed")
        except Exception as e:
            print(e)
    
    
    rospy.spin()

    

        

    
        


    
        



    




if __name__ == '__main__':
    try:
        asyncio.get_event_loop().run_until_complete(car_node())
    except rospy.ROSInterruptException:
        pass
