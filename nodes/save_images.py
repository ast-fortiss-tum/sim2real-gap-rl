#!/usr/bin/env python3
"""
Save Images at Regular Intervals in DonkeyCar Environment Using an SAC Policy

This script uses a pre-trained SAC policy to control the vehicle in the DonkeyCar environment. It saves images
from the environment at regular intervals during evaluation.

Author: Cristian Cubides-Herrera
Date: 2024-11-23
"""

# ================================
# 1. Imports
# ================================

# Standard Libraries
import os
import time
import uuid

# Third-Party Libraries
import gym
import numpy as np
from PIL import Image
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local Modules
import gym_donkeycar
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from version7_RL import DonkeyCarConfig, CustomDonkeyEnv, preprocess_image

# ================================
# 2. Save Images Function
# ================================

def save_images_from_policy(model, env, save_path, delay=1.0):
    """
    Save images from the DonkeyCar environment at a regular interval while using an SAC policy.

    Args:
        model: The trained SAC model.
        env (gym.Env): The DonkeyCar environment instance.
        save_path (str): Path to save the captured images.
        delay (float): Time interval (in seconds) between saving images.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    obs = env.reset()
    done = False
    image_count = 0

    print(f"Starting to save images to {save_path} every {delay} second(s)...")

    for i in range(5000):  # Run for a maximum of 1000 steps

        # Extract the image from the observation (assuming it is the observation output)
        if obs.shape[-1] == 4:  # RGBA image
            obs = obs[:, :, :3]  # Convert to RGB if necessary  

        print(obs.shape)
        time.sleep(15)

        

        # Convert the image to PIL format for saving
        pil_image = Image.fromarray(obs)

        # Save the image with a timestamp and count
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = os.path.join(save_path, f"image_{timestamp}_{image_count}.jpg")
        pil_image.save(image_path)
        print(f"Saved image: {image_path}")     

        # Predict the action using the SAC model
        obs = preprocess_image(obs)

        action, _ = model.predict(obs, deterministic=True)

        # Step in the environment
        obs, _, done, _ = env.step(action)

        # Increment the image counter
        image_count += 1

        # Wait for the specified delay
        if done:
            print("Episode finished.")
            obs = env.reset()
            image_count = 0
        time.sleep(delay)

# ================================
# 3. Main Function
# ================================

def main():
    """
    Main function to initialize the environment, load the SAC model, and start saving images.
    """
    # Initialize configuration
    config = DonkeyCarConfig()

    # Create the environment
    def make_env():
        """
        Creates and returns a wrapped DonkeyCar environment.
        """
        env = CustomDonkeyEnv(level=config.env_list[1], conf=config.env_config)
        return env

    env = DummyVecEnv([make_env])  # Vectorized environment

    # Check if a VecNormalize checkpoint exists
    vec_normalize_path = "vecnormalize_53.pkl"
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)

    # Load the trained SAC model
    model_path = "sac_donkeycar_final.zip"
    model = SAC.load(model_path)
    print(f"Successfully loaded model from checkpoint: {model_path}")

    # Path to save the images
    save_path = "./saved_images"

    # Start saving images
    save_images_from_policy(model, env, save_path, delay=1.0)

    # Close the environment
    env.close()
    print("Environment closed.")

# ================================
# 4. Entry Point
# ================================

if __name__ == "__main__":
    main()