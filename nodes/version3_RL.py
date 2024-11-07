import argparse
import uuid

import gym
import gym_donkeycar  # Import this before gym.make
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

if __name__ == "__main__":
    # Initialize the Donkey Car environment
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

    env_id = env_list[1]  # Choose your desired environment

    conf = {
        "exe_path": "/home/cubos98/Desktop/MA/sim/sim_vehicle.x86_64",
        "host": "127.0.0.1",
        "port": 9091,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "SAC",
        "country": "USA",
        "bio": "Learning to drive w SAC RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "frame_skip": 1,
        "cam_resolution": (240, 320, 4),  # Adjusted resolution if needed
        "log_level": 20,
        "steer_limit": 1.0,
        "throttle_min": 0.0,
        "throttle_max": 1.0,
    }

    # Create the gym environment
    def make_env():
        env = gym.make(env_id, conf=conf)
        return env

    # Wrap the environment in a DummyVecEnv and VecTransposeImage for image observations
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)  # Transpose to channel-first (C, H, W)

    # Create SAC model with a CNN policy
    model = SAC("CnnPolicy", env, verbose=1, buffer_size=1000)

    # Train the model
    model.learn(total_timesteps=10000)

    # Testing the trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Render the environment (may not work with vectorized envs)
        try:
            env.envs[0].render()
        except Exception as e:
            print(e)
            print("Failure in render, continuing...")

        if done[0]:
            obs = env.reset()

        if i % 100 == 0:
            print("Saving model...")
            model.save("sac_donkey")

    # Save the final model
    model.save("sac_donkey")
    print("Done training")

    env.close()
