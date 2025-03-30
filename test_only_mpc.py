
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

def main():

    eval_seed = 43

    SG = get_new_all_eff_env(degree=0.5, rl=False, seed=eval_seed)
    sys = SG.sys
    m1 = SG.m1
    n1 = SG.n1
    e1 = SG.e1

    horizon = timedelta(hours=24)
    #fixed_start = datetime.strptime("27.11.2016", "%d.%m.%Y")
    fixed_start = None
    # ----------------------------

    oc_history = run_optimal_control(sys, m1, n1, e1, fixed_start, horizon, eval_seed)

    # ----------------------------
    # Extract Histories and Compare
    # ----------------------------
    oc_total_cost, oc_soc = extract_cost_soc(oc_history, m1, n1, e1)

    plot_comparisons(oc_total_cost, oc_soc)

    # Daily cost comparison
    daily_cost_oc = sum(get_adjusted_cost(oc_history, sys))

    print("Daily cost:")
    print(f" Optimal Control: {daily_cost_oc}")

if __name__ == "__main__":
    main()
