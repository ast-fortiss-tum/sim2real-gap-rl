#!/usr/bin/env python3
import pathlib
import wandb
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime, timedelta

# Import the environment creation function from your customized environments module.
# We assume that get_simple_linear_env now returns the full SmartGrid_Linear instance.
from environments.get_customized_envs import get_simple_linear_env

# Import models, policies, networks, etc.
from models.sac import ContSAC
from models.darc import DARC
from utils import ZFilter
from architectures.gaussian_policy import ContGaussianPolicy
from architectures.value_networks import ContTwinQNet

from commonpower.modelling import ModelHistory
from commonpower.core import System, Node, Bus
from commonpower.models.busses import *
from commonpower.models.components import *
from commonpower.models.powerflow import *
from commonpower.control.controllers import (
    RLControllerSB3,
    OptimalController,
    RLControllerSAC_Customized,
    RLBaseController
)
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.control.safety_layer.penalties import DistanceDependingPenalty
from commonpower.control.runners import SingleAgentTrainer, DeploymentRunner
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.control.logging.loggers import TensorboardLogger
from commonpower.control.configs.algorithms import SB3MetaConfig, SB3PPOConfig, ContSACConfig
from commonpower.data_forecasting import CSVDataSource, LookBackForecaster, PerfectKnowledgeForecaster, DataProvider, ConstantDataSource
from commonpower.utils.helpers import get_adjusted_cost
from commonpower.utils.param_initialization import RangeInitializer
from stable_baselines3 import PPO, SAC
import tensorboard

# ----------------------------
# Helper functions
# ----------------------------
def run_optimal_control(sys, m1, n1, e1, fixed_start, horizon, eval_seed):
    """Run deployment using the Optimal Controller."""
    oc_history = ModelHistory([sys])
    oc_deployer = DeploymentRunner(
        sys=sys,
        global_controller=OptimalController('global'),
        forecast_horizon=horizon,
        control_horizon=horizon,
        history=oc_history,
        seed=eval_seed
    )
    oc_deployer.run(n_steps=24, fixed_start=fixed_start)
    print("Optimal controller used:", oc_deployer.controllers)
    return oc_history

def run_custom_deployment(sys, m1, n1, e1, fixed_start, horizon, eval_seed, alg_config, agent_name):
    """Run deployment using a customized SAC controller (e.g., ContSAC or DARC)."""
    agent = RLControllerSAC_Customized(
        name=agent_name,
        safety_layer=ActionProjectionSafetyLayer(
            penalty=DistanceDependingPenalty(penalty_factor=0.001)
        ),
    )
    history = ModelHistory([sys])
    deployer = DeploymentRunner(
        sys=sys,
        global_controller=agent,
        alg_config=alg_config,
        wrapper=SingleAgentWrapper,
        forecast_horizon=horizon,
        control_horizon=horizon,
        history=history,
        seed=eval_seed
    )
    deployer.run(n_steps=24, fixed_start=fixed_start)
    print(f"{agent_name} controller used:", deployer.controllers)
    return history

def extract_cost_soc(history, m1, n1, e1):
    """Extract the total cost and SOC history from a ModelHistory instance."""
    power_import_cost = history.get_history_for_element(m1, name='cost')
    dispatch_cost = history.get_history_for_element(n1, name='cost')
    total_cost = [
        (power_import_cost[t][0], power_import_cost[t][1] + dispatch_cost[t][1])
        for t in range(len(power_import_cost))
    ]
    soc = history.get_history_for_element(e1, name="soc")
    return total_cost, soc

def plot_comparisons(oc_cost, cont_cost, darc_cost, oc_soc, cont_soc, darc_soc):
    """Plot the cost and SOC comparisons between controllers."""
    # Cost Comparison
    plt.figure()
    plt.plot(range(len(oc_cost)), [x[1] for x in oc_cost], label="Optimal Control", marker="o")
    plt.plot(range(len(cont_cost)), [x[1] for x in cont_cost], label="CustomContSAC", marker="s")
    plt.plot(range(len(darc_cost)), [x[1] for x in darc_cost], label="CustomDARC", marker="^")
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("Cost")
    plt.title("Comparison of Household Cost")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # SOC Comparison
    plt.figure()
    plt.plot(range(len(oc_soc)), [x[1] for x in oc_soc], label="Optimal Control SOC", marker="o")
    plt.plot(range(len(cont_soc)), [x[1] for x in cont_soc], label="CustomContSAC SOC", marker="s")
    plt.plot(range(len(darc_soc)), [x[1] for x in darc_soc], label="CustomDARC SOC", marker="^")
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("State of Charge (SOC)")
    plt.title("Comparison of Battery SOC")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main script
# ----------------------------
def main():
    # Simulation configuration
    n_hours = 24
    horizon = timedelta(hours=n_hours)
    fixed_start = datetime(2016, 11, 27)
    training_seed = 42  # for reproducibility in training
    eval_seed = 5       # evaluation seed

    # Use the factory function to create the smart grid environment.
    # The factory function now returns the full SmartGrid instance.
    env_instance = get_simple_linear_env(training_seed, rl=True)
    
    # Extract system and key components from the instance.
    sys = env_instance.sys
    n1 = env_instance.n1
    m1 = env_instance.m1
    e1 = env_instance.e1

    sys.pprint()
    print("Controllers:", sys.controllers)

    # ----------------------------
    # Configuration for Customized SAC Controllers
    # ----------------------------
    policy_config = {
        "input_dim": [126],
        "architecture": [
            {"name": "linear1", "size": 256},
            {"name": "linear2", "size": 256},
            {"name": "split1", "sizes": [1, 1]}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    value_config = {
        "input_dim": [127],
        "architecture": [
            {"name": "linear1", "size": 256},
            {"name": "linear2", "size": 256},
            {"name": "linear2", "size": 1}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }

    # Define the saved paths for each customized controller (update these paths as needed)
    saved_path_cont_sac = (
        "/home/cubos98/Desktop/MA/DARAIL/saved_weights/ContSAC__20250325_042457_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids"
    )
    saved_path_darc = (
        "/home/cubos98/Desktop/MA/DARAIL/saved_weights/DARC__20250325_041917_lr0.0008_noise0_seed42_"
        "lin_src1_varietyv_degree0.5_Smart_Grids/DARC__20250325_041917_lr0.0008_noise0_seed42_"
        "lin_src1_varietyv_degree0.5_Smart_Grids"
    )

    alg_config_custom_cont_sac = ContSACConfig(
        policy_config=policy_config,
        value_config=value_config,
        action_range=(-1.5, 1.5),
        device="cpu",
        load_path=saved_path_cont_sac,
        running_mean=ZFilter((126,), clip=20),
        policy=ContGaussianPolicy(policy_config, (-1.5, 1.5)),
        twin_q=ContTwinQNet(value_config),
        target_twin_q=ContTwinQNet(value_config),
        algorithm=None,
        seed=42
    )

    alg_config_custom_darc = ContSACConfig(
        policy_config=policy_config,
        value_config=value_config,
        action_range=(-1.5, 1.5),
        device="cpu",
        load_path=saved_path_darc,
        running_mean=ZFilter((126,), clip=20),
        policy=ContGaussianPolicy(policy_config, (-1.5, 1.5)),
        twin_q=ContTwinQNet(value_config),
        target_twin_q=ContTwinQNet(value_config),
        algorithm=None,
        seed=42
    )

    # ----------------------------
    # Run Deployments
    # ----------------------------
    oc_history = run_optimal_control(sys, m1, n1, e1, fixed_start, horizon, eval_seed)
    custom_cont_sac_history = run_custom_deployment(
        sys, m1, n1, e1, fixed_start, horizon, eval_seed, alg_config_custom_cont_sac, "CustomContSAC"
    )
    custom_darc_history = run_custom_deployment(
        sys, m1, n1, e1, fixed_start, horizon, eval_seed, alg_config_custom_darc, "CustomDARC"
    )

    # ----------------------------
    # Extract Histories and Compare
    # ----------------------------
    oc_total_cost, oc_soc = extract_cost_soc(oc_history, m1, n1, e1)
    cont_total_cost, cont_soc = extract_cost_soc(custom_cont_sac_history, m1, n1, e1)
    darc_total_cost, darc_soc = extract_cost_soc(custom_darc_history, m1, n1, e1)

    plot_comparisons(oc_total_cost, cont_total_cost, darc_total_cost, oc_soc, cont_soc, darc_soc)

    # Daily cost comparison
    daily_cost_oc = sum(get_adjusted_cost(oc_history, sys))
    daily_cost_cont = sum(get_adjusted_cost(custom_cont_sac_history, sys))
    daily_cost_darc = sum(get_adjusted_cost(custom_darc_history, sys))
    print("Daily cost:")
    print(f" Optimal Control: {daily_cost_oc}")
    print(f" CustomContSAC:   {daily_cost_cont}")
    print(f" CustomDARC:      {daily_cost_darc}")

if __name__ == "__main__":
    main()
