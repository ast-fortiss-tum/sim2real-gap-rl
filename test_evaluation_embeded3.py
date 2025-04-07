#!/usr/bin/env python3
import argparse
import pathlib
import wandb
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime, timedelta

# --------------------------------------------------
# Environment creation functions
# --------------------------------------------------
from environments.get_customized_envs import (
    get_simple_linear_env,
    get_new_soc_env,
    get_new_charge_env,
    get_new_discharge_env,
    get_new_all_eff_env,
    get_new_limited_capacity_env,
    get_new_limited_plim_env,
    get_twoHouses_env
)

# --------------------------------------------------
# Models, policies, networks, etc.
# --------------------------------------------------
from models.sac import ContSAC
from models.darc import DARC
from utils import ZFilter
from architectures.gaussian_policy import ContGaussianPolicy
from architectures.value_networks import ContTwinQNet

# --------------------------------------------------
# CommonPower modules for control, logging, etc.
# --------------------------------------------------
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
N_STEPS = 24

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic deployment simulation for Smart Grid control experiments."
    )
    # Simulation parameters
    parser.add_argument("--fixed-start", type=str, default=None,
                        help="Fixed start date for simulation (YYYY-MM-DD)")
    parser.add_argument("--n_hours", type=int, default=24,
                        help="Number of simulation hours")
    parser.add_argument("--training_seed", type=int, default=42,
                        help="Training seed for environment creation")
    parser.add_argument("--eval_seed", type=int, default=42,
                        help="Evaluation seed for deployment")
    # Environment hyperparameters for non-broken (one-house) experiments.
    parser.add_argument("--broken", action="store_true",
                        help="Set this flag for a broken (two-house) environment")
    parser.add_argument("--lin_src", type=int, default=None,
                        help="For one-house experiments, set to 1 if the source should be linear (required if broken==False)")
    parser.add_argument("--variety-name", type=str, default=None,
                        help="Name of variety ('s', 'c', 'd', 'v...', 'lc', 'lp') (required if broken==False)")
    parser.add_argument("--degree", type=float, default=None,
                        help="Degree parameter for variety (required for s, c, d, v... types)")
    parser.add_argument("--capacity", type=float, default=3.0,
                        help="Battery capacity (required for variety 'lc')")
    parser.add_argument("--p_lim", type=float, default=1.5,
                        help="Battery power limit (required for variety 'lp')")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise level")
    # For broken (two-house) experiments.
    parser.add_argument("--break_src", type=int, default=None,
                        help="For broken experiments, which house to break: 1 for source, 0 for target (required if broken==True)")
    # Model selection: choose which deployment to run.
    parser.add_argument("--model", type=str, default="all",
                        choices=["optimal", "custom_cont_sac", "custom_darc", "all"],
                        help="Which model deployment to run")
    # Base path for saved models and agent names.
    parser.add_argument("--pre_path", type=str,
                        default="/home/cubos98/Desktop/MA/DARAIL/saved_weights/saved_models_experiments_2/",
                        help="Base path for saved models")
    parser.add_argument("--agent1", type=str, default="CustomContSAC",
                        help="Name for the first custom agent")
    parser.add_argument("--agent2", type=str, default="CustomDARC",
                        help="Name for the second custom agent")
    
    args = parser.parse_args()
    
    # Conditional checks for non-broken experiments.
    if not args.broken:
        missing = []
        if args.lin_src is None:
            missing.append("--lin_src")
        if args.variety_name is None:
            missing.append("--variety-name")
        if args.degree is None and (args.variety_name in ['s', 'c', 'd'] or (args.variety_name and args.variety_name.startswith('v'))):
            missing.append("--degree")
        if missing:
            parser.error("For non-broken experiments (broken==False) the following arguments are required: " + ", ".join(missing))
    else:
        if args.break_src is None:
            parser.error("For broken experiments (broken==True), --break_src is required.")
    return args

def create_environments(args, rl):
    """
    Create source and target environments based on the experiment hyperparameters.
    For non-broken (one-house) experiments, the variety-name determines which modified
    environment to use alongside the simple linear one.
    For broken (two-house) experiments, get_twoHouses_env is used with the damaged_battery flag.
    The parameter 'rl' is passed to set the mode (True for training, False for MPC/deployment).
    """
    # For one-house experiments:
    if not args.broken:
        if args.lin_src == 1:
            source_env = get_simple_linear_env(args.training_seed, rl=rl, fixed_start=args.fixed_start)
            if args.variety_name == 's':
                target_env = get_new_soc_env(args.degree, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'c':
                target_env = get_new_charge_env(args.degree, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'd':
                target_env = get_new_discharge_env(args.degree, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'v':
                target_env = get_new_all_eff_env(args.degree, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'lc':
                target_env = get_new_limited_capacity_env(args.capacity, 1.5, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'lp':
                target_env = get_new_limited_plim_env(args.capacity, args.p_lim, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
        else:
            # When lin_src == 0, swap source and target roles.
            target_env = get_simple_linear_env(args.training_seed, rl=rl, fixed_start=args.fixed_start)
            if args.variety_name == 's':
                source_env = get_new_soc_env(args.degree, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'c':
                source_env = get_new_charge_env(args.degree, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'd':
                source_env = get_new_discharge_env(args.degree, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name.startswith('v'):
                source_env = get_new_all_eff_env(args.degree, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'lc':
                source_env = get_new_limited_capacity_env(args.capacity, 1.5, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            elif args.variety_name == 'lp':
                source_env = get_new_limited_plim_env(args.capacity, args.p_lim, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
    else:
        # For two-house (broken) experiments.
        if args.break_src == 1:
            source_env = get_twoHouses_env(True, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            target_env = get_twoHouses_env(False, args.training_seed, rl=rl, fixed_start=args.fixed_start)
        else:
            source_env = get_twoHouses_env(False, args.training_seed, rl=rl, fixed_start=args.fixed_start)
            target_env = get_twoHouses_env(True, args.training_seed, rl=rl, fixed_start=args.fixed_start)
    return source_env, target_env

def run_optimal_control(sys, m1, n1, e1, fixed_start, horizon, eval_seed):
    """Run deployment using the Optimal Controller."""
    oc_history = ModelHistory([sys])
    sys.controllers = {}
    print("Controllers:", sys.controllers)
    oc_deployer = DeploymentRunner(
        sys=sys,
        global_controller=OptimalController('global'),
        forecast_horizon=horizon,
        control_horizon=horizon,
        history=oc_history,
        seed=eval_seed
    )
    print("Optimal Controller seed:", oc_deployer.seed)
    oc_deployer.run(n_steps=N_STEPS, fixed_start=fixed_start)
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
    deployer.run(n_steps=N_STEPS, fixed_start=fixed_start)
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
    args = parse_args()

    # Simulation configuration.
    n_hours = args.n_hours
    horizon = timedelta(hours=n_hours)
    if args.fixed_start is not None:
        fixed_start = datetime.fromisoformat(args.fixed_start)
    else:
        fixed_start = args.fixed_start
    training_seed = args.training_seed
    eval_seed = args.eval_seed

    # Create environments dynamically.
    # For training the environment is created with rl=True.
    env_SAC, _ = create_environments(args, rl=True)
    env_DARC, _ = create_environments(args, rl=True)
    # For MPC deployment, we create the environment with rl=False.
    env_deploy, _ = create_environments(args, rl=False)
    
    # Extract system and key components from the training environment.
    sys_darc = env_DARC.sys
    n1_darc = env_DARC.n1
    m1_darc = env_DARC.m1
    e1_darc = env_DARC.e1

    sys_sac = env_SAC.sys
    n1_sac = env_SAC.n1
    m1_sac = env_SAC.m1
    e1_sac = env_SAC.e1

    # For deployment, we use the MPC environment.
    sys_mpc = env_deploy.sys
    n1_mpc = env_deploy.n1
    m1_mpc = env_deploy.m1
    e1_mpc = env_deploy.e1

    #sys1.pprint()
    #print("Controllers:", sys.controllers)
    print("MPC controllers:", sys_mpc.controllers)

    # ----------------------------
    # Network Architecture & Model Configuration
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

    # Build saved model paths dynamically.
    fixed_start_str = args.fixed_start  # Further formatting can be applied if desired.
    pre_path = args.pre_path
    saved_path_cont_sac = (pre_path + f"ContSAC_test_run__fs{fixed_start_str}_lr0.0008_noise{args.noise}_seed{training_seed}_lin_src{args.lin_src}_variety{args.variety_name}_degree{args.degree}_Smart_Grids")
    saved_path_darc = (pre_path + f"DARC_test_run__fs{fixed_start_str}_lr0.0008_noise{args.noise}_seed{training_seed}_lin_src{args.lin_src}_variety{args.variety_name}_degree{args.degree}_Smart_Grids")

    # Create algorithm configurations for the custom controllers.
    alg_config_custom_cont_sac = ContSACConfig(
        policy_config=policy_config,
        value_config=value_config,
        action_range=(-args.p_lim, args.p_lim),
        device="cpu",
        load_path=saved_path_cont_sac,
        running_mean=ZFilter((126,), clip=20),
        policy=ContGaussianPolicy(policy_config, (-args.p_lim, args.p_lim)),
        twin_q=ContTwinQNet(value_config),
        target_twin_q=ContTwinQNet(value_config),
        algorithm=None,
        seed=training_seed
    )

    alg_config_custom_darc = ContSACConfig(
        policy_config=policy_config,
        value_config=value_config,
        action_range=(-args.p_lim, args.p_lim),
        device="cpu",
        load_path=saved_path_darc,
        running_mean=ZFilter((126,), clip=20),
        policy=ContGaussianPolicy(policy_config, (-args.p_lim, args.p_lim)),
        twin_q=ContTwinQNet(value_config),
        target_twin_q=ContTwinQNet(value_config),
        algorithm=None,
        seed=training_seed
    )

    # ----------------------------
    # Model Deployment (Dynamic Selection)
    # ----------------------------
    results = {}
    if args.model in ["optimal", "all"]:
        print("\nRunning Optimal Controller deployment...")
        oc_history = run_optimal_control(sys_mpc, m1_mpc, n1_mpc, e1_mpc, fixed_start, horizon, eval_seed)
        results["optimal"] = oc_history

    if args.model in ["custom_cont_sac", "all"]:
        print(f"\nRunning {args.agent1} deployment...")
        cont_history = run_custom_deployment(
            sys_sac, m1_sac, n1_sac, e1_sac, fixed_start, horizon, eval_seed,
            alg_config_custom_cont_sac, args.agent1
        )
        results["custom_cont_sac"] = cont_history

    if args.model in ["custom_darc", "all"]:
        print(f"\nRunning {args.agent2} deployment...")
        darc_history = run_custom_deployment(
            sys_darc, m1_darc, n1_darc, e1_darc, fixed_start, horizon, eval_seed,
            alg_config_custom_darc, args.agent2
        )
        results["custom_darc"] = darc_history

    # ----------------------------
    # Result Extraction and Comparison
    # ----------------------------
    if len(results) > 1:
        # Extract histories for comparison.
        oc_total_cost, oc_soc = extract_cost_soc(results.get("optimal", list(results.values())[0]), m1_mpc, n1_mpc, e1_mpc)
        cont_total_cost, cont_soc = extract_cost_soc(results.get("custom_cont_sac", list(results.values())[0]), m1_sac, n1_sac, e1_sac)
        darc_total_cost, darc_soc = extract_cost_soc(results.get("custom_darc", list(results.values())[0]), m1_darc, n1_darc, e1_darc)
        plot_comparisons(oc_total_cost, cont_total_cost, darc_total_cost, oc_soc, cont_soc, darc_soc)
        
        daily_cost_oc = sum(get_adjusted_cost(results.get("optimal", list(results.values())[0]), sys_mpc))
        daily_cost_cont = sum(get_adjusted_cost(results.get("custom_cont_sac", list(results.values())[0]), sys_sac))
        daily_cost_darc = sum(get_adjusted_cost(results.get("custom_darc", list(results.values())[0]), sys_darc))

        print("\nDaily cost comparison:")
        print(f" Optimal Control: {daily_cost_oc}")
        print(f" {args.agent1}:   {daily_cost_cont}")
        print(f" {args.agent2}:      {daily_cost_darc}")
    #else:
        #key = list(results.keys())[0]
        #daily_cost = sum(get_adjusted_cost(results[key], sys))
        #print(f"\nDaily cost for {key}: {daily_cost}")

if __name__ == "__main__":
    main()
