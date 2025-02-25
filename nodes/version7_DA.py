#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Implementation for Line Following in DonkeyCar Environment
Now with data augmentation: noise is added and images are randomly flipped horizontally.
If the image is flipped, the steering action is inverted.

Author: Cristian Cubides-Herrera
Date: 2024-11-23
Modified to remove CycleGAN transformation by [Your Name]
"""

# =========================================
# 1. Imports
# =========================================
import argparse
import random
import uuid
import time
import os
import sys

# Optionally comment this out if it's not needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# If using donkeycar:
# from gym_donkeycar.envs.donkey_env import DonkeyEnv
# If you have local donkeycar environment code, import from there:
from gym_donkeycar.envs.donkey_env import DonkeyEnv

# =========================================
# 2. Configuration Classes
# =========================================

class DonkeyCarConfig:
    """
    Configuration parameters for the DonkeyCar environment.
    """
    env_list = [
        "warehouse",
        "generated_road",
        "sparkfun_avc",
        "generated_track",
        "roboracingleague_1",
        "sandbox_track",
    ]

    env_config = {
        "exe_path": "/home/cubides/MA/DonkeyCar/sdsim_2/sim_NT.x86_64",
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 5.0,
        "max_cte": 2.5,
        "frame_skip": 1,
        "cam_resolution": (120, 160, 3),
        "steer_limit": 1.0,
        "throttle_min": 0.0,
        "throttle_max": 1.0,
        "conf": {
            "car_name": "SAC",
            "race": False,
            "racer_name": "SAC_agent",
            "country": "Germany",
            "bio": "Learning to drive with SAC",
            "guid": str(uuid.uuid4()),
            "random_seed": random.randint(0, 10000),
            "max_cte": 2.5,
            "frame_skip": 1,
            "cam_resolution": (240, 320, 4),
        }
    }
    max_cte = env_config["max_cte"]

# =========================================
# 3. Custom Environment with Flip & Noise Augmentation
# =========================================

class CustomDonkeyEnv(DonkeyEnv):
    """
    Custom DonkeyCar environment with data augmentation (random horizontal flip and Gaussian noise)
    and a tailored reward function for line tracking. If the image is flipped, the corresponding
    steering action is inverted.
    """
    def __init__(self, level: str, conf: dict, throttle: int, verbose=0):
        """
        Args:
            level (str): environment level name
            conf (dict): config dictionary
            verbose (int): if > 0, prints additional logs
        """
        super(CustomDonkeyEnv, self).__init__(level=level, conf=conf)
        self.verbose = verbose

        self.throttle = throttle
        self.distance_traveled = 0.0
        self.prev_pos = None  # will store (x, y, z)

        # Define the observation space for (3, 60, 80) in [0,1]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, 60, 80),
            dtype=np.float32
        )

        # Define the action space: only steering ([-0.9, 0.9])
        self.action_space = gym.spaces.Box(
            low=np.array([-0.9]),
            high=np.array([0.9]),
            dtype=np.float32
        )

        # Environment parameters
        self.max_cte = conf["max_cte"]
        self.max_steering_angle = 0.5  # used for normalizing steering penalty and for augmentation correction
        self.target_speed = 0.8  # Not used here

        # Remove CycleGAN model loading; we use the original image only.
        # Initialize augmentation parameters (for the very first observation)
        self.last_aug_params = {'flip': False, 'noise_std': 0.0}

    def step(self, action: np.ndarray):
        # ------------------------------------------------------------
        # BEFORE sending the action to the simulator, adjust it to
        # “undo” the augmentation applied to the observation the agent saw.
        # ------------------------------------------------------------
        aug = self.last_aug_params
        adjusted_action = action[0]
        # If the image was flipped horizontally, reverse the steering sign.
        if aug.get('flip', False):
            adjusted_action = -adjusted_action

        # Now pass the adjusted action (and throttle) to the simulator.
        observation, original_reward, done, info = super().step([adjusted_action, self.throttle])

        # ------------------------------------------------------------
        # AFTER the simulator step, preprocess the new observation.
        # The _preprocess_image method applies data augmentation and updates self.last_aug_params.
        # ------------------------------------------------------------
        obs = self._preprocess_image(observation)

        # Update distance traveled and handle episode termination.
        x, y, z = info.get("pos")
        current_pos = np.array([x, y, z])
        if self.prev_pos is not None:
            step_dist = np.linalg.norm(current_pos - self.prev_pos)
            self.distance_traveled += step_dist
        self.prev_pos = current_pos

        if done:
            info["distance_traveled"] = self.distance_traveled
            self.distance_traveled = 0.0
            self.prev_pos = None

        # Compute custom reward using the adjusted action.
        cte = info.get('cte', 0.0)
        speed = info.get('speed', 0.0)
        collision = info.get('hit', False)
        reward = self.compute_reward(cte, adjusted_action, speed, collision, done)

        return obs, reward, done, info

    def reset(self):
        observation = super().reset()
        obs = self._preprocess_image(observation)
        # Reset distance each time we start a new episode.
        self.distance_traveled = 0.0
        self.prev_pos = None
        return obs

    def _preprocess_image(self, observation: np.ndarray) -> np.ndarray:
        """
        Preprocesses the raw observation: resizes to (80,60), applies data augmentation
        (random horizontal flip and Gaussian noise), and returns a (C, H, W) float32 array.
        Also, stores the augmentation parameters in self.last_aug_params for action correction.
        """
        # Convert RGBA to RGB if needed
        if observation.shape[2] == 4:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB)

        # Convert to PIL image (using the original image without any CycleGAN transformation)
        out_pil = Image.fromarray(observation, "RGB")

        # Resize to (80,60)
        out_pil = out_pil.resize((80, 60), Image.Resampling.LANCZOS)

        # ------------------------------
        # DATA AUGMENTATION
        # ------------------------------
        aug_params = {}
        # 1. Random horizontal flip (with probability 0.5)
        if random.random() < 0.5:
            out_pil = out_pil.transpose(Image.FLIP_LEFT_RIGHT)
            aug_params['flip'] = True
        else:
            aug_params['flip'] = False

        # 2. Convert to numpy array and normalize to [0,1]
        obs_np = np.asarray(out_pil, dtype=np.float32) / 255.0

        # 3. Add Gaussian noise (e.g. std=0.05)
        noise_std = 0.05
        noise = np.random.normal(0, noise_std, obs_np.shape)
        obs_np = np.clip(obs_np + noise, 0, 1)
        aug_params['noise_std'] = noise_std

        # Store these augmentation parameters for action correction in the next step.
        self.last_aug_params = aug_params

        # 4. Convert from (H, W, C) to (C, H, W)
        obs_np = np.transpose(obs_np, (2, 0, 1))
        return obs_np

    def compute_reward(self, cte, steering_angle, speed, collision, done):
        """
        Compute custom reward for line-tracking.
        """
        # Normalized CTE
        cte_normalized = cte / self.max_cte
        cte_reward = 1.0 - abs(cte_normalized)
        cte_reward = np.clip(cte_reward, 0.0, 1.0)

        # Steering penalty
        steering_penalty = abs(steering_angle) / self.max_steering_angle
        steering_penalty = np.clip(steering_penalty, 0.0, 1.0)

        # Speed reward
        speed_error = abs(speed - self.target_speed) / self.target_speed
        speed_reward = 1.0 - speed_error
        speed_reward = np.clip(speed_reward, 0.0, 1.0)

        # Collision penalty
        collision_penalty = -100.0 if (collision or done) else 0.0

        # Weighted sum of terms (example weighting)
        total_reward = (1.0 * cte_reward
                        - 0.0 * steering_penalty
                        + 0.0 * speed_reward
                        + 0.0 * collision_penalty)

        return total_reward

# =========================================
# 4. Custom Feature Extractor
# =========================================
class NvidiaCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor inspired by NVIDIA's architecture for image inputs.
    """
    def __init__(self, observation_space: gym.spaces.Box):
        super(NvidiaCNNExtractor, self).__init__(observation_space, features_dim=100)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # (24, 28, 38)
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),  # (36, 12, 17)
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),  # (48, 4, 7)
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # (64, 2, 5)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 5, 100),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)

# =========================================
# 5. Custom Callbacks
# =========================================

class AverageRewardCallback(BaseCallback):
    """
    Logs and prints average reward per episode and also logs
    total distance traveled in each episode.
    """
    def __init__(self, verbose: int = 1):
        super(AverageRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_distances = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info.keys():
                episode_reward = info['episode']['r']
                distance = info.get('distance_traveled', 0.0)

                self.episode_rewards.append(episode_reward)
                self.episode_distances.append(distance)

                avg_reward = np.mean(self.episode_rewards)
                avg_distance = np.mean(self.episode_distances)

                if self.verbose > 0:
                    print(
                        f"Episode {len(self.episode_rewards)}: "
                        f"Reward = {episode_reward:.2f}, "
                        f"Average Reward = {avg_reward:.2f}, "
                        f"Distance = {distance:.2f}, "
                        f"Average Distance = {avg_distance:.2f}"
                    )

                self.logger.record("env/episode_reward", episode_reward)
                self.logger.record("env/episode_distance", distance)
                self.logger.record("env/avg_reward", avg_reward)
                self.logger.record("env/avg_distance", avg_distance)

        return True

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            final_avg_reward = np.mean(self.episode_rewards)
            final_avg_distance = np.mean(self.episode_distances)
            print(
                f"Training completed. "
                f"Final average reward: {final_avg_reward:.2f}, "
                f"Final average distance: {final_avg_distance:.2f}"
            )
            
class SavePreprocessedObservationCallback(BaseCallback):
    """
    Saves a post-processed observation to disk every 'save_freq' steps.
    """
    def __init__(self, save_freq=5000, save_dir="./saved_observations", verbose=1):
        super(SavePreprocessedObservationCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        current_obs = self.locals["new_obs"]
        if self.n_calls % self.save_freq == 0:
            obs = current_obs[0]
            obs = np.transpose(obs, (1, 2, 0))
            obs = (obs * 255.0).clip(0, 255).astype(np.uint8)
            obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

            filename = os.path.join(self.save_dir, f"step_{self.n_calls}.png")
            cv2.imwrite(filename, obs_bgr)

            if self.verbose > 0:
                print(f"[SaveObsCallback] Saved post-processed observation at step={self.n_calls} to {filename}")
        return True

class SACLossLogger(BaseCallback):
    """
    Logs SAC loss components to TensorBoard, if available.
    """
    def __init__(self, log_dir: str = "./sac_losses_tensorboard/", verbose: int = 0):
        super(SACLossLogger, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir)
        self.step_count = 0

    def _on_step(self) -> bool:
        if hasattr(self.model, 'policy'):
            policy = self.model.policy
            critic_loss = getattr(policy, 'critic_loss', None)
            actor_loss = getattr(policy, 'actor_loss', None)
            entropy_loss = getattr(policy, 'alpha_loss', None)

            if critic_loss is not None:
                self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.step_count)
            if actor_loss is not None:
                self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.step_count)
            if entropy_loss is not None:
                self.writer.add_scalar('Loss/Entropy', entropy_loss.item(), self.step_count)

            self.step_count += 1
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
        if self.verbose > 0:
            print("Training completed. Losses logged to TensorBoard.")

# =========================================
# 6. Main Training Code
# =========================================

def main():
    # Hyperparameters for checkpoint names:
    custom_lr = 5.3e-4
    custom_ent_coef = 0.10
    custom_tau = 0.02
    custom_gamma = 0.99
    custom_batch_size = 256
    total_timesteps = 1000000
    throttle = 0.45

    param_str = f"lr{custom_lr}_ent{custom_ent_coef}_tau{custom_tau}_gamma{custom_gamma}_bs{custom_batch_size}_throttle{throttle}"
    name_prefix = f"model_Ch_{param_str}"

    config = DonkeyCarConfig()

    # Set random seed for reproducibility
    seed = config.env_config["conf"]["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        def _init():
            env = CustomDonkeyEnv(level=config.env_list[1], conf=config.env_config, throttle=throttle, verbose=0)
            env = Monitor(env)
            return env
        return _init

    env = DummyVecEnv([make_env()])

    # Policy kwargs to use the custom CNN
    policy_kwargs = dict(
        features_extractor_class=NvidiaCNNExtractor,
    )

    average_reward_callback = AverageRewardCallback(verbose=1)
    sac_loss_logger = SACLossLogger(log_dir="./sac_losses_tensorboard/", verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./sac_donkeycar_checkpoints/',
        name_prefix=name_prefix,
        verbose=2
    )
    save_observation_callback = SavePreprocessedObservationCallback(
        save_freq=5000,
        save_dir="./saved_preprocessed_observations",
        verbose=1
    )
    callbacks = CallbackList([
        average_reward_callback,
        # sac_loss_logger,
        checkpoint_callback,
        # save_observation_callback
    ])

    checkpoint_path = "/home/cubides/MA/DonkeyCar/ROS_DonkeyCar/src/nodes/sac_donkeycar_checkpoints/model_Ch_lr0.00053_ent0.25_tau0.02_gamma0.99_bs256_throttle0.45_150000_steps.zip"

    try:
        model = SAC.load(checkpoint_path, env=env)
        print(f"Successfully loaded model from checkpoint: {checkpoint_path}")
        model.learning_rate = custom_lr
        model.ent_coef = custom_ent_coef

    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}. Training from scratch.")
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=custom_lr,
            buffer_size=70000,
            learning_starts=1,
            batch_size=custom_batch_size,
            tau=custom_tau,
            gamma=custom_gamma,
            train_freq=1,
            gradient_steps=1,
            ent_coef=custom_ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device="auto",
            tensorboard_log="./sac_donkeycar_tensorboard/",
        )

    print(f"Starting training with parameters:")
    print(f"  Learning Rate : {model.learning_rate}") #
    print(f"  Ent Coef      : {model.ent_coef}")
    print(f"  Batch Size    : {model.batch_size}")
    print(f"  Tau           : {model.tau}")
    print(f"  Gamma         : {model.gamma}")
    print(f"  Throttle      : {throttle}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )

    final_model_name = f"{name_prefix}_final_{total_timesteps}"
    model.save(final_model_name)
    print(f"Model saved as '{final_model_name}'.")

    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
