#!/usr/bin/env python3

import rospy
from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
import time

conf = {"exe_path": "/home/cubos98/Desktop/MA/sim/sdsim_2/sim_NT.x86_64", "port": 9091, "start_delay": 5.0}

def open_executable():
    if "exe_path" in conf:
        proc = DonkeyUnityProcess()
        # the unity sim server will bind to the host ip given
        proc.start(conf["exe_path"], host="0.0.0.0", port=conf["port"])

        # wait for simulator to startup and begin listening
        time.sleep(conf["start_delay"])

        # start simulation com
        viewer = DonkeyUnitySimContoller(conf=conf)

        # Note: for some RL algorithms, it would be better to normalize the action space to [-1, 1]
        # and then rescale to proper limtis
        # steering and throttle

        # wait until the car is loaded in the scene
        viewer.wait_until_loaded()

if __name__ == '__main__':
    try:
        open_executable()
    except rospy.ROSInterruptException:
        pass
