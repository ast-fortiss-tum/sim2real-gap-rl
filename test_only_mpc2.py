#!/usr/bin/env python3
import pathlib
import wandb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm

# Import the environment creation function from your customized environments module.
# We assume that get_new_all_eff_env now returns the full SmartGrid_Linear instance.
from environments.get_customized_envs import get_new_all_eff_env

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

from environments.get_customized_envs import (
    get_simple_linear_env, get_new_soc_env, get_new_charge_env, 
    get_new_discharge_env, get_new_all_eff_env, get_new_limited_capacity_env, 
    get_new_limited_plim_env, get_twoHouses_env
)

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

def extract_cost_soc(history, m1, n1, e1):
    """
    Extract the total cost and SOC history from a ModelHistory instance,
    excluding the last value of each.
    """
    power_import_cost = history.get_history_for_element(m1, name='cost')
    dispatch_cost = history.get_history_for_element(n1, name='cost')
    total_cost = [
        (power_import_cost[t][0], power_import_cost[t][1] + dispatch_cost[t][1])
        for t in range(len(power_import_cost) - 1)  # Exclude the last value
    ]
    soc = history.get_history_for_element(e1, name="soc")[:-1]  # Exclude the last value
    return total_cost, soc

def plot_comparisons(oc_cost, oc_soc):
    """Plot the cost and SOC comparisons between controllers."""
    # Cost Comparison
    plt.figure()
    plt.plot(range(len(oc_cost)), [x[1] for x in oc_cost], label="Optimal Control", marker="o")
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("Cost")
    plt.title("Comparison of Household Cost")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # SOC Comparison
    plt.figure()
    plt.plot(range(len(oc_soc)), [-x[1] for x in oc_soc], label="Optimal Control SOC", marker="o")
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("State of Charge (SOC)")
    plt.title("Comparison of Battery SOC")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_mpc(sys, m1, n1, e1, fixed_start, horizon, num_games=10, base_seed=42):
    """
    Evaluate the MPC (Optimal Controller) on the provided system.
    
    Args:
      sys: The system/environment.
      m1, n1, e1: Elements from the system used for cost and SOC extraction.
      fixed_start: The fixed starting datetime, if any.
      horizon: Forecast and control horizon.
      num_games: Number of evaluation episodes.
      base_seed: Base seed to vary the experiment initialization.
      
    Returns:
      avg_daily_cost: Average daily cost over all experiments.
      all_daily_costs: List of daily costs for each experiment.
      std_daily_cost: Standard deviation of the daily cost.
      avg_cost_per_time: Average cost per timestep (averaged over experiments).
      std_cost_per_time: Standard deviation of cost per timestep.
    """
    all_daily_costs = []
    cost_series_all = []
    for episode in tqdm(range(num_games), desc="Evaluating MPC episodes"):
        current_seed = base_seed + episode  # Increment seed for each run
        oc_history = run_optimal_control(sys, m1, n1, e1, fixed_start, horizon, current_seed)
        cost_series = get_adjusted_cost(oc_history, sys)  # List of cost per timestep
        daily_cost = sum(cost_series)
        all_daily_costs.append(daily_cost)
        cost_series_all.append(cost_series)
    avg_daily_cost = np.mean(all_daily_costs)
    std_daily_cost = np.std(all_daily_costs)
    
    # Convert cost_series_all to a numpy array for per-timestep statistics.
    cost_series_all = np.array(cost_series_all)
    avg_cost_per_time = np.mean(cost_series_all, axis=0)
    std_cost_per_time = np.std(cost_series_all, axis=0)
    
    return avg_daily_cost, all_daily_costs, std_daily_cost, avg_cost_per_time, std_cost_per_time

def main():
    eval_seed = 43

    SG = get_new_all_eff_env(degree=0.5, rl=False, seed=eval_seed)
    sys = SG.sys
    m1 = SG.m1
    n1 = SG.n1
    e1 = SG.e1

    horizon = timedelta(hours=24)
    fixed_start = None

    # Evaluate MPC (Optimal Controller) over multiple experiments
    (avg_daily_cost, all_daily_costs, std_daily_cost,
     avg_cost_per_time, std_cost_per_time) = evaluate_mpc(sys, m1, n1, e1, fixed_start, horizon,
                                                         num_games=10, base_seed=42)

    print("MPC Evaluation Results:")
    print("Average Daily Cost:", avg_daily_cost)
    print("Daily Costs for each experiment:", all_daily_costs)
    print("Standard Deviation of Daily Cost:", std_daily_cost)
    print("Average Cost per Timestep:", avg_cost_per_time)

    # Plot the standard deviation of cost per timestep
    plt.figure()
    plt.plot(std_cost_per_time, marker='o', label="Std of Cost per Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Standard Deviation of Cost")
    plt.title("Standard Deviation of Cost per Timestep over MPC Episodes")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optionally, run one deployment to extract and plot cost and SOC comparisons.
    oc_history = run_optimal_control(sys, m1, n1, e1, fixed_start, horizon, eval_seed)
    oc_total_cost, oc_soc = extract_cost_soc(oc_history, m1, n1, e1)
    plot_comparisons(oc_total_cost, oc_soc)

if __name__ == "__main__":
    main()
