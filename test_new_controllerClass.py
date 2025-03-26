import os
import torch
import numpy as np
import gymnasium as gym
from models.sac import ContSAC, set_global_seed
from utils import ZFilter  # Your running mean filter
from environments.get_customized_envs import get_simple_linear_env
from commonpower.control.controllers import RLControllerSAC_Customized  # The class we implemented above

# ----------------------------
# Set up the environment & model
# ----------------------------
seed = 42
set_global_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Create the environment (must match your training settings)
env = get_simple_linear_env(seed, rl=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Create your running mean filter (normalization object)
running_mean = ZFilter((state_dim,), clip=20)

# Define network configurations (should be the same as during training)
policy_config = {
    "input_dim": [state_dim],
    "architecture": [
        {"name": "linear1", "size": 256},
        {"name": "linear2", "size": 256},
        {"name": "split1", "sizes": [action_dim, action_dim]}
    ],
    "hidden_activation": "relu",
    "output_activation": "none"
}
value_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [
        {"name": "linear1", "size": 256},
        {"name": "linear2", "size": 256},
        {"name": "linear2", "size": 1}
    ],
    "hidden_activation": "relu",
    "output_activation": "none"
}

# Instantiate your SAC model (e.g., ContSAC)
sac_model = ContSAC(
    policy_config=policy_config,
    value_config=value_config,
    env=env,
    device="cpu",
    log_dir="latest_runs",
    running_mean=running_mean,
    warmup_games=10,
    batch_size=64,
    lr=0.0001,
    ent_adj=False,
    n_updates_per_train=1,
    max_steps=24,
    seed=seed
)

# ----------------------------
# Wrap the SAC model in your controller class
# ----------------------------
# Create the controller instance by passing your trained sac_model.
# Optionally, specify a load_path if you want to load a saved model.
controller = RLControllerSAC_Customized(sac_model=sac_model, load_path="DARC__20250325_041917_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids/DARC__20250325_041917_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids")

# ----------------------------
# Save the model (after training)
# ----------------------------
# This will save the policy, twin Q-network, and running_mean filter in the specified folder.
#controller.save(save_path="saved_weights/my_saved_model_folder")

# ----------------------------
# Load the model (for inference or further evaluation)
# ----------------------------
# When you load, the controller uses the load_path that was set during initialization.
controller.load(env, device="cpu")

# ----------------------------
# Use the controller to predict actions
# ----------------------------
# For example, reset the environment and predict an action for the first observation.
obs, _ = env.reset()
action = controller.predict_action(obs, deterministic=True)
print("Predicted action:", action)
