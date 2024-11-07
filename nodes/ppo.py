"""
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines
"""
import argparse
import uuid

import gym
import gym_donkeycar  # Import this before gym.make
from stable_baselines3 import PPO

if __name__ == "__main__":
    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]

    env_id = env_list[1]

    conf = {
        "exe_path": "/home/cubos98/Desktop/MA/sim/sim_vehicle.x86_64",
        "host": "127.0.0.1",
        "port": 9091,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "PPO",
        "country": "USA",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "frame_skip": 1,
        "cam_resolution": (240, 320, 4),
        "log_level": 20,
        "steer_limit": 1.0,
        "throttle_min": 0.0,
        "throttle_max": 1.0,
    }

    # make gym env
    env = gym.make(env_list[0], conf=conf)

    # create cnn policy
    model = PPO("CnnPolicy", env, verbose=1)

    # set up model in learning mode with goal number of timesteps to complete
    model.learn(total_timesteps=10000)

    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        try:
            env.render()
        except Exception as e:
            print(e)
            print("failure in render, continuing...")

        if done:
            obs = env.reset()

        if i % 100 == 0:
            print("saving...")
            model.save("ppo_donkey")

    # Save the agent
    model.save("ppo_donkey")
    print("done training")

env.close()