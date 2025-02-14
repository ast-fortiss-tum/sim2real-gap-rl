#!/usr/bin/env python
import torch
import torch.nn as nn
import torch as th
import gym

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from architectures.gaussian_policy import ContGaussianPolicy

# =============================================================================
# 1. Define a Custom Features Extractor
# =============================================================================
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that processes raw observations into
    a latent feature vector. Modify the network architecture as needed.
    """
    def __init__(self, observation_space, features_dim: int = 64):
        # The parent class needs to know the number of features (here, features_dim)
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        # For example, if the observation space is a Box with shape (n,)
        input_dim = observation_space.shape[0]

        # Define a simple MLP architecture. Modify as needed.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

# =============================================================================
# 2. Create a Custom SAC Policy Using the Custom Feature Extractor
# =============================================================================
class CustomSACPolicy(SACPolicy):
    """
    Custom SAC Policy that uses the defined CustomFeaturesExtractor.
    """
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )

# =============================================================================
# 3. Main Script: Instantiate Environment, SAC Agent, and Load Weights
# =============================================================================
def main():
    # Replace 'YourEnv-v0' with the id of your Gym environment.
    env = gym.make("YourEnv-v0")
    
    # Instantiate the SAC agent with the custom policy.
    # You can pass additional hyperparameters if needed.
    model = SAC(ContGaussianPolicy, env, verbose=1)
    
    # =============================================================================
    # 4. Load Your Custom-Trained Weights from policy.pth
    # =============================================================================
    # Load the saved state dictionary. This file should match the architecture.
    state_dict = torch.load("policy.pth")
    
    # If your file was saved as a dictionary containing a key like "state_dict",
    # then extract it. Otherwise, assume the file is directly the state dictionary.
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Load the state dictionary into the policy network.
    model.policy.load_state_dict(state_dict, strict=False)
    
    # =============================================================================
    # 5. Test the Loaded Model
    # =============================================================================
    # Reset the environment and obtain an initial observation.
    obs = env.reset()
    # Predict an action using the loaded model.
    action, _states = model.predict(obs)
    print("Predicted action:", action)
    
    # =============================================================================
    # 6. (Optional) Save the Full Model in SB3 Format
    # =============================================================================
    # This will create a full model checkpoint (ZIP archive) that includes
    # additional information (e.g., hyperparameters, training state) so that
    # you can later load it using SAC.load("full_model.zip", env=env)
    model.save("full_model.zip")
    print("Model saved as 'full_model.zip'.")

if __name__ == '__main__':
    main()
