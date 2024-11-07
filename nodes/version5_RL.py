import gym
import gym_donkeycar
import time
import random

import argparse
import numpy as np
import uuid
from gym_donkeycar.envs.donkey_env import DonkeyEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import cv2
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Define the configuration for the Donkey Car environment

env_list = [
        "warehouse",
        "generated_road",
        "sparkfun_avc",
        "generated_track",
        "roboracingleague_1",
    ]

env_config = {
    "exe_path": "/home/cubos98/Desktop/MA/sim/sim_vehicle.x86_64",  
    "host": "127.0.0.1",
    "port": 9091,
    "conf": {
        "car_name": "SAC",
        "race": False,
        "racer_name": "SAC_agent",
        "country": "Germany",
        "bio": "Learning to drive with SAC",
        "guid": str(uuid.uuid4()),
        "random_seed": random.randint(0, 10000),  # Use a random seed
        "max_cte": 10,
        "frame_skip": 1,
        "cam_resolution": (240, 320, 4),
        "log_level": 20,
        #"steer_limit": 1.0,
        #"throttle_min": 0.0,
        #"throttle_max": 1.0,
    }
}

max_cte = env_config["conf"]["max_cte"]

# Preprocess each input image for steering policy learning
def preprocess_image(observation):
    # Convert RGBA to RGB if necessary
    if observation.shape[2] == 4:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB)

    # Convert RGB to YUV color space (NVIDIA network uses YUV)
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2YUV)

    # Resize to (66, 200), which is typical for NVIDIA's network
    observation = cv2.resize(observation, (80, 60), interpolation=cv2.INTER_AREA)

    # Normalize the image (similar to NVIDIA's normalization)
    observation = observation / 255.0  # Scale pixel values to [0, 1]

    # Add channel dimension for PyTorch
    observation = np.transpose(observation, (2, 0, 1))  # Channel-first for PyTorch
    return observation

# Custom environment with image preprocessing
class CustomDonkeyEnv(DonkeyEnv):
    def __init__(self, level, conf):
        super(CustomDonkeyEnv, self).__init__(level=level, conf=conf)
        #self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 66, 200), dtype=np.float32)
        #self.observation_space = gym.spaces.Box(0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(0, self.VAL_PER_PIXEL, shape = (3, 60, 80), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

    def step(self, action):
        observation, reward, done, info = super().step([action[0], 0.5])  # Constant throttle
        obs = preprocess_image(observation)

        """
        info = {
            "pos": (self.x, self.y, self.z),
            "cte": self.cte,
            "speed": self.speed,
            "forward_vel": self.forward_vel,
            "hit": self.hit,
            "gyro": (self.gyro_x, self.gyro_y, self.gyro_z),
            "accel": (self.accel_x, self.accel_y, self.accel_z),
            "vel": (self.vel_x, self.vel_y, self.vel_z),
            "lidar": (self.lidar),
            "car": (self.roll, self.pitch, self.yaw),
            "last_lap_time": self.last_lap_time,
            "lap_count": self.lap_count,
        }
        """

        # Extract necessary information for reward calculation
        
        speed = info.get('speed', 0)
        cte = info.get('cte', 0)  # Cross-track error (distance from center of lane)
        collision = info.get('collision', False)
        """
        """
        # Extract necessary information for reward calculation
        speed = info['speed']
        cte = info['cte'] # Cross-track error (distance from center of lane)
        collision = info['hit']
        
        # Custom reward function
        #reward = speed * (1 - abs(cte/max_cte))  # Base reward that incentivizes speed and low CTE
        reward = 1 - abs(cte/max_cte)  # Base reward that incentivizes speed and low CTE
        
        # Penalize for off-track (high CTE) and collisions
        if abs(cte) > max_cte*0.8:  # If car is significantly off the center of the lane
            reward += -1
        if collision:
            reward += -2  # Large penalty for collisions

        if done: 
            reward -= 1  # Penalty for episode termination   
        
        # Further encourage staying close to center
        if abs(cte) < max_cte*0.2:
            reward += 5  # Bonus for staying very close to center

        return obs, reward, done, info

    def reset(self):
        observation = super().reset()
        return preprocess_image(observation)

# Custom feature extractor based on NVIDIA's CNN structure
class NvidiaCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(NvidiaCNNExtractor, self).__init__(observation_space, features_dim=1)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            #nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(640, 100), nn.ReLU(),
            #nn.Linear(100, 50), nn.ReLU(),
            #nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)  # Output is steering only
        )

    def forward(self, observations):
        return self.cnn(observations)

# Define the environment and the model with the custom CNN
def make_env():
    env = CustomDonkeyEnv(level=env_list[1], conf=env_config)
    return env

env = DummyVecEnv([make_env])

policy_kwargs = dict(
    features_extractor_class=NvidiaCNNExtractor,
)


"""
model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./sac_donkeycar_tensorboard/",
    buffer_size=100000,
)
"""

model = SAC(
    policy="MlpPolicy",                    # The policy architecture; e.g., 'MlpPolicy' for a multi-layer perceptron policy.
    env=env,                                # The environment object.
    learning_rate=3e-4,                     # Learning rate for the optimizer (default: 3e-4).
    buffer_size=100000,                     # Size of the replay buffer (number of experiences stored).
    learning_starts=100,                    # Minimum steps of interaction before learning starts (default: 100).
    batch_size=256,                         # Batch size for each gradient update (default: 256).
    tau=0.005,                              # Target smoothing coefficient (soft update) (default: 0.005).
    gamma=0.99,                             # Discount factor for reward (default: 0.99).
    train_freq=1,                           # Frequency of model updates (can be set to a tuple like (1, "step")).
    gradient_steps=1,                       # Number of gradient steps taken after each rollout (default: 1).
    action_noise=None,                      # Noise added to actions for exploration; None for deterministic SAC.
    optimize_memory_usage=False,            # Memory optimization for large replay buffers (default: False).
    ent_coef="auto",                        # Entropy regularization coefficient; controls exploration vs exploitation.
    target_update_interval=1,               # Frequency (in updates) to update the target network (default: 1).
    target_entropy="auto",                  # Target entropy; "auto" will adjust automatically.
    use_sde=False,                          # Whether to use State-Dependent Exploration (SDE) (default: False).
    sde_sample_freq=-1,                     # Frequency for sampling a new noise matrix when using SDE.
    use_sde_at_warmup=False,                # Whether to use SDE during warm-up (default: False).
    policy_kwargs=policy_kwargs,            # Additional arguments for the policy network, e.g., custom feature extractors.
    verbose=1,                              # Verbosity level; 0: none, 1: info, 2: debug.
    seed=None,                              # Random seed for reproducibility.
    device="auto",                          # Device for PyTorch tensors; 'auto', 'cpu', or 'cuda'.
    tensorboard_log="./sac_donkeycar_tensorboard/", # Directory for tensorboard logging.

)


checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./sac_donkeycar_checkpoints/',
    name_prefix='sac_donkeycar'
)

# Train the model
model.learn(total_timesteps=10000, callback=checkpoint_callback)

# Save the final model
model.save("sac_donkeycar")

# Close the environment
env.close()
