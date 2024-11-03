#!/usr/bin/env python3

import rospy
import gym
from gym import spaces
import numpy as np
from mixed_reality.msg import Control
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

class ROSCustomEnv(gym.Env):
    def __init__(self):
        super(ROSCustomEnv, self).__init__()
        # Initialize ROS node
        rospy.init_node('rl_agent_node', anonymous=True)

        # Define action and observation spaces
        # Action space: [throttle, steering], values between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation space: [throttle, steering, brake, reverse, stopping]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # Initialize variables
        self.current_state = np.zeros(5)
        self.done = False

        # Set up ROS publishers and subscribers
        self.control_pub = rospy.Publisher('/control_topic', Control, queue_size=1)
        self.state_sub = rospy.Subscriber('/state_topic', Control, self.state_callback)

        # For synchronization
        self.rate = rospy.Rate(10)  # 10 Hz

        # Wait for the ROS topics to be ready
        rospy.sleep(2.0)

    def state_callback(self, msg):
        # Update current state from ROS messages
        self.current_state = np.array([
            msg.throttle,
            msg.steering,
            float(msg.brake),
            float(msg.reverse),
            float(msg.stopping)
        ])

    def step(self, action):
        # Apply action: Publish control commands to ROS
        control_msg = Control()
        control_msg.throttle = float(action[0])
        control_msg.steering = float(action[1])
        control_msg.brake = False
        control_msg.reverse = False
        control_msg.stopping = False
        self.control_pub.publish(control_msg)

        # Wait for the new state to be received
        rospy.sleep(0.1)  # Adjust sleep time as needed

        # Get the new observation
        obs = self.current_state.copy()

        # Compute reward
        reward = self.compute_reward(obs, action)

        # Check if the episode is done
        done = self.check_done_condition(obs)

        # Additional info (optional)
        info = {}

        return obs, reward, done, info

    def reset(self):
        # Reset the environment to an initial state
        # Implement any necessary reset logic here
        # For example, send a reset command to the simulator or robot
        rospy.loginfo("Resetting environment")
        self.done = False
        self.current_state = np.zeros(5)

        # Wait for the system to reset
        rospy.sleep(1.0)

        return self.current_state.copy()

    def compute_reward(self, state, action):
        # Define your reward function
        # Placeholder example: Encourage the agent to minimize the throttle and steering values
        reward = - (abs(state[0]) + abs(state[1]))
        return reward

    def check_done_condition(self, state):
        # Define when the episode is done
        # Placeholder example: End episode if stopping flag is True
        if state[4]:  # If 'stopping' is True
            rospy.loginfo("Episode finished")
            return True
        return False

def main():
    # Initialize the environment
    env = ROSCustomEnv()

    # Create the SAC agent
    model = SAC('MlpPolicy', env, verbose=1)

    # Optionally, set up callbacks for saving models
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/',
                                             name_prefix='sac_model')

    # Train the agent
    try:
        model.learn(total_timesteps=10000, callback=checkpoint_callback)
    except rospy.ROSInterruptException:
        pass

    # Save the trained agent
    model.save("sac_agent")

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()
