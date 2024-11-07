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
        "steer_limit": 1.0,
        "throttle_min": 0.0,
        "throttle_max": 1.0,
    }
}

# Custom environment
class CustomDonkeyEnv(DonkeyEnv):
    def __init__(self, level, conf):
        super(CustomDonkeyEnv, self).__init__(level = level, conf=conf)

        # Customize the observation space
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, 60, 80),  # Grayscale image shape
            dtype=np.uint8,
        )

        state_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict({
            'image': image_space,
            'state': state_space
        })

        # Customize the action space to steering only
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5]),  # Steering only
            high=np.array([0.5]),
            dtype=np.float32,
        )

    def step(self, action):
        # action is steering only
        steering = action[0]
        throttle = 0.5  # Constant throttle value (adjust as needed)

        # Combine steering and throttle into the action expected by parent class
        full_action = np.array([steering, throttle])

        # Call the parent class's step method with the full action
        observation, reward, done, info = super(CustomDonkeyEnv, self).step(full_action)

        # Preprocess the observation
        obs = self.preprocess_observation(observation, info)

        # Customize the reward function (adjust as per your requirements)
        speed = info.get('speed', 0)
        cte = info.get('cte', 0)
        reward = speed * (1 - abs(cte))

        if done:
            if info.get('offtrack', False):
                reward -= 100
            if info.get('collision', False):
                reward -= 100

        return obs, reward, done, info

    def reset(self):
        observation = super(CustomDonkeyEnv, self).reset()
        info = {'speed': 0.0, 'cte': 0.0}
        obs = self.preprocess_observation(observation, info)
        return obs

    def preprocess_observation(self, observation, info):
        # Process the image
        if observation.shape[2] == 4:
            # Convert RGBA to RGB
            observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB)

        # Convert RGB to grayscale
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # Resize the image
        observation = cv2.resize(observation, (80, 60), interpolation=cv2.INTER_AREA)

        # Add channel dimension
        observation = np.expand_dims(observation, axis=0)

        # Get additional state information
        speed = np.array([info.get('speed', 0)], dtype=np.float32)
        cte = np.array([info.get('cte', 0)], dtype=np.float32)

        state = np.concatenate((speed, cte))

        # Return a dictionary observation
        return {'image': observation, 'state': state}

# Custom feature extractor (same as before)
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        self.image_space = observation_space.spaces['image']
        self.state_space = observation_space.spaces['state']

        # CNN for image processing
        n_input_channels = self.image_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # MLP for state processing
        self.mlp = nn.Sequential(
            nn.Linear(self.state_space.shape[0], 64),
            nn.ReLU(),
        )

        # Compute the output dimensions
        with torch.no_grad():
            sample_image = torch.zeros(1, *self.image_space.shape)
            cnn_output_dim = self.cnn(sample_image).shape[1]

        # Total features dimension
        self._features_dim = cnn_output_dim + 64

    def forward(self, observations):
        # Extract image and state from observations
        image = observations['image'].float()
        state = observations['state'].float()

        # Pass through CNN and MLP
        cnn_output = self.cnn(image)
        mlp_output = self.mlp(state)

        # Concatenate features
        features = torch.cat((cnn_output, mlp_output), dim=1)

        return features

# Function to create the environment
def make_env():
    #env = CustomDonkeyEnv(args.env_name, conf=env_config)
    env = CustomDonkeyEnv(level=env_list[1], conf=env_config)
    print("----------------------------------------")
    print(env.viewer.handler.SceneToLoad)
    time.sleep(4)
    return env

# Create the environment
env = DummyVecEnv([make_env])

# Policy keyword arguments
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
)

# Create the SAC model with a custom policy
model = SAC(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./sac_donkeycar_tensorboard/",
    buffer_size=10000,
)

# Create a callback to save the model every 10,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./sac_donkeycar_checkpoints/',
    name_prefix='sac_donkeycar'
)

# Train the model
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the final model
model.save("sac_donkeycar")

# Close the environment
env.close()


