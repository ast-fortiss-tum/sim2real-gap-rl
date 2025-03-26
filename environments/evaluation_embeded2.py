#!/usr/bin/env python3
from datetime import timedelta, datetime
import os
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------
# ENVIRONMENT CREATION FUNCTIONS (provided)
# ----------------------------
from environments.smartgrid_env import SmartGrid_Linear, SmartGrid_Nonlinear, SmartGrid_TwoHouses

def get_simple_linear_env(seed, rl=True):
    """
    Factory function to create a linear SmartGrid environment.
    Returns a gym.Env instance configured via SmartGrid_Linear.
    """
    env_instance = SmartGrid_Linear(
        rl=rl,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        seed=seed,
        params_battery={"rho": 0.1, "p_lim": 1.5}
    )
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    return env_instance.env

# ----------------------------
# COMMONPOWER & OTHER IMPORTS
# ----------------------------
from commonpower.modelling import ModelHistory
from commonpower.core import System
from commonpower.control.controllers import OptimalController, RLControllerSAC_Customized
from commonpower.control.runners import DeploymentRunner
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.utils.helpers import get_adjusted_cost

# ----------------------------
# SAC-BASED MODEL IMPORTS (your code)
# ----------------------------
from models.sac import ContSAC, set_global_seed
from utils import ZFilter  # Your running mean filter

# ----------------------------
# SIMULATION PARAMETERS
# ----------------------------
seed = 42
eval_seed = 5
horizon = timedelta(hours=24)
fixed_start = datetime(2016, 11, 27)

# ----------------------------
# Set random seeds for reproducibility
# ----------------------------
set_global_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ----------------------------
# Create the Environment
# ----------------------------
env = get_simple_linear_env(seed, rl=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# ----------------------------
# Instantiate the Running Mean Filter
# ----------------------------
running_mean = ZFilter((state_dim,), clip=20)

# ----------------------------
# Define Network Configurations (must match training)
# ----------------------------
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

# ----------------------------
# Instantiate Your SAC Model (YOUR_SAC_MODEL_INSTANCE) Using ContSAC
# ----------------------------
YOUR_SAC_MODEL_INSTANCE = ContSAC(
    policy_config=policy_config,
    value_config=value_config,
    env=env,
    device="cpu",
    log_dir="latest_runs",
    running_mean=running_mean,
    warmup_games=10,
    batch_size=64,
    lr=0.0001,
    gamma=0.99,
    tau=0.003,
    alpha=0.2,
    ent_adj=False,
    target_update_interval=1,
    n_games_til_train=1,
    n_updates_per_train=1,
    max_steps=24,
    seed=seed
)
print("YOUR_SAC_MODEL_INSTANCE created successfully!")

# ----------------------------
# Wrap the SAC Model in the Customized Controller
# ----------------------------
customized_controller = RLControllerSAC_Customized(
    sac_model=YOUR_SAC_MODEL_INSTANCE,
    load_path="saved_weights/my_custom_model_folder"
)
# Save (if desired) and then load the controller’s parameters
customized_controller.save(save_path="/home/cubos98/Desktop/MA/DARAIL/saved_weights/DARC__20250325_061946_lr0.0008_noise0_seed42_broken1_break_src0_Smart_Grids/DARC__20250325_061946_lr0.0008_noise0_seed42_broken1_break_src0_Smart_Grids/policy")
customized_controller.load(env, device="cpu")

# ----------------------------
# Instantiate the Optimal Controller
# ----------------------------
optimal_controller = OptimalController("global")
# (Assume the OptimalController is properly configured within your CommonPower framework.)

# ----------------------------
# Deployment Evaluation Function
# ----------------------------
def evaluate_controller(controller, env, num_games=10, base_seed=42):
    """
    Evaluate the given controller over num_games episodes.
    Each episode resets the environment with seed (base_seed + episode).
    Returns average total reward, average per-timestep reward, and standard deviation per timestep.
    """
    rewards_all = []
    # For each episode, reset the env with a new seed and run until done.
    for episode in tqdm(range(num_games), desc="Evaluating controller"):
        current_seed = base_seed + episode
        obs, _ = env.reset(seed=current_seed)
        # Apply the running_mean normalization (assumed stored in the SAC model)
        obs = running_mean(obs)
        done = False
        episode_rewards = []
        while not done:
            action = controller.predict_action(obs, deterministic=False)
            obs, reward, done, _, _ = env.step(action)
            obs = running_mean(obs)
            episode_rewards.append(reward)
        rewards_all.append(episode_rewards)
    rewards_all = np.array(rewards_all)  # shape: (episodes, timesteps)
    avg_rewards = np.mean(rewards_all, axis=0)
    std_rewards = np.std(rewards_all, axis=0)
    avg_total_reward = np.mean(np.sum(rewards_all, axis=1))
    return avg_total_reward, avg_rewards, std_rewards

# ----------------------------
# Deploy Both Controllers Using DeploymentRunner
# ----------------------------
# For the customized controller:
custom_history = ModelHistory([env])
custom_deployer = DeploymentRunner(
    sys=None,  # When using a pre-built grid environment, sys can be omitted or set appropriately.
    global_controller=customized_controller,
    wrapper=SingleAgentWrapper,
    forecast_horizon=horizon,
    control_horizon=horizon,
    history=custom_history,
    seed=eval_seed
)
custom_deployer.run(n_steps=24, fixed_start=fixed_start)
custom_power_import_cost = custom_history.get_history_for_element("Trading1", name='cost')
custom_dispatch_cost = custom_history.get_history_for_element("MultiFamilyHouse", name='cost')
custom_total_cost = [
    (custom_power_import_cost[t][0], custom_power_import_cost[t][1] + custom_dispatch_cost[t][1])
    for t in range(len(custom_power_import_cost))
]
custom_soc = custom_history.get_history_for_element("ESS1", name="soc")

# For the optimal controller:
optimal_history = ModelHistory([env])
optimal_deployer = DeploymentRunner(
    sys=None,
    global_controller=optimal_controller,
    wrapper=SingleAgentWrapper,
    forecast_horizon=horizon,
    control_horizon=horizon,
    history=optimal_history,
    seed=eval_seed
)
optimal_deployer.run(n_steps=24, fixed_start=fixed_start)
optimal_power_import_cost = optimal_history.get_history_for_element("Trading1", name='cost')
optimal_dispatch_cost = optimal_history.get_history_for_element("MultiFamilyHouse", name='cost')
optimal_total_cost = [
    (optimal_power_import_cost[t][0], optimal_power_import_cost[t][1] + optimal_dispatch_cost[t][1])
    for t in range(len(optimal_power_import_cost))
]
optimal_soc = optimal_history.get_history_for_element("ESS1", name="soc")

# ----------------------------
# Plot Cost Comparison Between Optimal and Customized Controllers
# ----------------------------
plt.figure()
plt.plot(
    range(len(custom_total_cost)), [x[1] for x in custom_total_cost],
    label="Cost Customized Controller", marker="s"
)
plt.plot(
    range(len(optimal_total_cost)), [x[1] for x in optimal_total_cost],
    label="Cost Optimal Controller", marker="o"
)
plt.xticks(
    ticks=range(len(custom_power_import_cost)),
    labels=[x[0] for x in custom_power_import_cost], rotation=45
)
plt.xlabel("Timestamp")
plt.ylabel("Cost")
plt.title("Comparison of Household Cost")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Plot SOC (State-of-Charge) Comparison
# ----------------------------
plt.figure()
plt.plot(
    range(len(custom_soc)), [x[1] for x in custom_soc],
    label="SOC Customized Controller", marker="s"
)
plt.plot(
    range(len(optimal_soc)), [x[1] for x in optimal_soc],
    label="SOC Optimal Controller", marker="o"
)
plt.xticks(
    ticks=range(len(custom_soc)),
    labels=[x[0] for x in custom_soc], rotation=45
)
plt.xlabel("Timestamp")
plt.ylabel("State of Charge (SOC)")
plt.title("Comparison of Battery SOC")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Compare Daily Cost
# ----------------------------
cost_day_custom = sum(get_adjusted_cost(custom_history, None))
cost_day_optimal = sum(get_adjusted_cost(optimal_history, None))
print(f"Daily cost:\n Customized Controller: {cost_day_custom}\n Optimal Controller: {cost_day_optimal}")
