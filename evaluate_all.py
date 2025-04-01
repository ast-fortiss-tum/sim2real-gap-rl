#!/usr/bin/env python3
import argparse
import os
import pickle
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta

# Import your model classes and utility functions.
from models.sac import ContSAC, set_global_seed
from models.darc import DARC, DARC_two  # Import both DARC and DARC_two
from environments.get_customized_envs import (
    get_simple_linear_env, get_new_soc_env, get_new_charge_env, 
    get_new_discharge_env, get_new_all_eff_env, get_new_limited_capacity_env, 
    get_new_limited_plim_env, get_twoHouses_env
)
from utils import ZFilter  # Assuming ZFilter is defined in utils

# Additional imports for MPC evaluation.
from commonpower.modelling import ModelHistory
from commonpower.control.runners import DeploymentRunner
from commonpower.control.controllers import OptimalController
from commonpower.utils.helpers import get_adjusted_cost

#############################
# Helper functions for MPC Evaluation
#############################
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
        for t in range(len(power_import_cost))
    ]
    soc = history.get_history_for_element(e1, name="soc")
    return total_cost, soc

def evaluate_mpc(sys, m1, n1, e1, fixed_start, horizon, num_games=10, base_seed=42):
    """
    Evaluate the MPC (Optimal Controller) on the provided system.
    
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
        cost_series = get_adjusted_cost(oc_history, sys)[:-1]  # List of cost per timestep
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

#############################
# Folder Construction Helpers
#############################
def construct_save_model_path(args, prefix):
    fs = args.fixed_start
    if args.fixed_start is not None:
        fs = args.fixed_start.replace('.', '-')
    filename = f"{prefix}_{args.save_file_name}_fs{fs}_lr{args.lr}_noise{args.noise}_seed{args.seed}_"
    if args.broken == 0:
        filename += f"lin_src{args.lin_src}_variety{args.variety_name}_"
        if args.variety_name in ['s', 'c', 'd'] or args.variety_name.startswith('v'):
            filename += f"degree{args.degree}_"
        elif args.variety_name == 'lc':
            filename += f"cap{args.capacity}_"
        elif args.variety_name == 'lp':
            filename += f"p_lim{args.p_lim}_"
        filename += args.env_name
    else:
        filename += f"broken{args.broken}_break_src{args.break_src}_{args.env_name}"
    return os.path.join(args.save_model, filename)

def construct_log_dir(args, prefix):
    base_log_dir = "eval_runs_2"
    fs = args.fixed_start
    if args.fixed_start is not None:
        fs = args.fixed_start.replace('.', '-')
    if args.broken == 0:
        log_subfolder = (f"{prefix}_fs_{fs}_lin_src_{args.lin_src}_variety_{args.variety_name}_"
                         f"noise_{args.noise}_seed_{args.seed}_")
        if args.variety_name in ['s', 'c', 'd'] or args.variety_name.startswith('v'):
            log_subfolder += f"degree_{args.degree}_"
        elif args.variety_name == 'lc':
            log_subfolder += f"cap_{args.capacity}_"
        elif args.variety_name == 'lp':
            log_subfolder += f"p_lim_{args.p_lim}_"
        log_subfolder += args.env_name
    else:
        log_subfolder = (f"{prefix}_fs_{fs}_broken_{args.broken}_break_src_{args.break_src}_"
                         f"noise_{args.noise}_seed_{args.seed}_{args.env_name}")
    return os.path.join(base_log_dir, log_subfolder)

#############################
# Environment Selection Helper
#############################
def select_environment(args):
    """
    Selects and creates the source and target environments based on the hyperparameters.
    """
    if args.broken == 0:
        if args.lin_src is None or args.variety_name is None or args.degree is None:
            raise ValueError("For non-broken experiments, --lin_src, --variety-name, and --degree must be provided.")
        if args.lin_src == 1:
            source_env = get_simple_linear_env(args.seed, fixed_start=args.fixed_start).env
            if args.variety_name == 's':
                target_env = get_new_soc_env(args.degree, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'c':
                target_env = get_new_charge_env(args.degree, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'd':
                target_env = get_new_discharge_env(args.degree, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name.startswith('v'):
                target_env = get_new_all_eff_env(args.degree, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'lc':
                target_env = get_new_limited_capacity_env(args.capacity, args.p_lim, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'lp':
                target_env = get_new_limited_plim_env(args.capacity, args.p_lim, args.seed, fixed_start=args.fixed_start).env
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
        else:
            target_env = get_simple_linear_env(args.seed, fixed_start=args.fixed_start).env
            if args.variety_name == 's':
                source_env = get_new_soc_env(args.degree, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'c':
                source_env = get_new_charge_env(args.degree, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'd':
                source_env = get_new_discharge_env(args.degree, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name.startswith('v'):
                source_env = get_new_all_eff_env(args.degree, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'lc':
                source_env = get_new_limited_capacity_env(args.capacity, args.p_lim, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'lp':
                source_env = get_new_limited_plim_env(args.capacity, args.p_lim, args.seed, fixed_start=args.fixed_start).env
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
        return source_env, target_env
    else:
        if args.break_src is None:
            raise ValueError("For two-house experiments, --break_src must be provided.")
        if args.break_src == 1:
            source_env = get_twoHouses_env(damaged_battery=True, seed=args.seed, fixed_start=args.fixed_start).env
            target_env = get_twoHouses_env(damaged_battery=False, seed=args.seed, fixed_start=args.fixed_start).env
        else:
            source_env = get_twoHouses_env(damaged_battery=False, seed=args.seed, fixed_start=args.fixed_start).env
            target_env = get_twoHouses_env(damaged_battery=True, seed=args.seed, fixed_start=args.fixed_start).env
        return source_env, target_env

#############################
# Evaluation Function for RL models
#############################
def evaluate_model(model, env, num_games=10, base_seed=42, deterministic=True):
    """
    Evaluate a given model on the provided environment.
    Returns:
      avg_total_reward, avg_rewards_per_timestep, std_rewards_per_timestep
    """
    model.policy.eval()
    model.twin_q.eval()
    rewards_all = []
    for episode in tqdm(range(num_games), desc="Evaluating episodes"):
        current_seed = base_seed + episode
        state, _ = env.reset(seed=current_seed)
        state = model.running_mean(state)
        done = False
        episode_rewards = []
        while not done:
            action = model.get_action(state, deterministic=deterministic)
            next_state, reward, done, _, _ = env.step(action)
            next_state = model.running_mean(next_state)
            episode_rewards.append(reward)
            state = next_state
        rewards_all.append(episode_rewards[:-1])
    rewards_all = np.array(rewards_all)
    avg_rewards_per_timestep = np.mean(rewards_all, axis=0)
    std_rewards_per_timestep = np.std(rewards_all, axis=0)
    avg_total_reward = np.mean(np.sum(rewards_all, axis=1))
    return avg_total_reward, avg_rewards_per_timestep, std_rewards_per_timestep

#############################
# Main Testing Script
#############################
def parse_args():
    parser = argparse.ArgumentParser(description="Test ContSAC vs DARC Models")
    
    # Folder/hyperparameter settings
    parser.add_argument('--save-model', type=str, default="",
                        help='Base folder for saving/loading models')
    parser.add_argument('--save_file_name', type=str, default='',
                        help='File name identifier (appended in folder name)')
    parser.add_argument('--lr', type=float, default=0.0008,
                        help='Learning rate')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--env-name', type=str, default="Smart_Grids",
                        help='Name of environment (Smart_Grids only)')
    parser.add_argument('--fixed-start', type=str, default=None,
                        help='Fixed start date (format DD.MM.YYYY)')
    parser.add_argument('--broken', type=int, choices=[0, 1], default=0,
                        help='Broken flag: 0 for one-house, 1 for two-house')
    parser.add_argument('--lin_src', type=int, default=1,
                        help='For non-broken experiments: 1 for source linear, 0 for target')
    parser.add_argument('--variety-name', type=str, default="v",
                        help="Variety name (e.g., 's', 'c', 'd', 'v', 'lc', 'lp')")
    parser.add_argument('--degree', type=float, default=0.5,
                        help='Degree parameter for variety (if applicable)')
    parser.add_argument('--capacity', type=float, default=3.0,
                        help='Capacity (if applicable)')
    parser.add_argument('--p_lim', type=float, default=1.5,
                        help='Power limit (if applicable)')
    parser.add_argument('--break_src', type=int, default=None,
                        help='For broken experiments, which environment to break: 1 for source, 0 for target')
    
    # Model folder paths
    parser.add_argument('--contsac_model_folder', type=str, default="",
                        help='(Optional) Folder path for saved ContSAC model')
    parser.add_argument('--darc_model_folder', type=str, default="",
                        help='(Optional) Folder path for saved DARC model')
    
    parser.add_argument('--num-games', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--eval-seed', type=int, default=126,
                        help='Base seed for evaluation (incremented per episode)')
    parser.add_argument('--stochastic-eval', action='store_true', 
                        help='If set, evaluation will use stochastic actions')
    
    # Option to run MPC evaluation as well.
    parser.add_argument('--run-mpc', action='store_true',
                        help='If set, also run MPC (Optimal Control) evaluation')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select environments based on hyperparameters.
    source_env, target_env = select_environment(args)
    
    # For RL evaluation, use the target environment.
    env = target_env
    
    # Compute model folder paths if not provided.
    if not args.contsac_model_folder:
        args.contsac_model_folder = construct_save_model_path(args, prefix="saved_models_experiments_2/ContSAC_test_run")
    if not args.darc_model_folder:
        args.darc_model_folder = construct_save_model_path(args, prefix="saved_models_experiments_2/DARC_test_run")
    
    log_dir_DARC = construct_log_dir(args, prefix="DARC")
    log_dir_ContSAC = construct_log_dir(args, prefix="ContSAC")

    print("Using ContSAC model folder:", args.contsac_model_folder)
    print("Using DARC model folder:", args.darc_model_folder)
    
    # Set seeds for reproducibility.
    set_global_seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    
    # Get dimensions from the environment.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    soc_dim = 1 if args.broken == 0 else 2

    # Define network architectures.
    policynet_size = 256  
    classifier_size = 32  
    policy_config = {
        "input_dim": [state_dim],
        "architecture": [
            {"name": "linear1", "size": policynet_size},
            {"name": "linear2", "size": policynet_size},
            {"name": "split1", "sizes": [action_dim, action_dim]}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    value_config = {
        "input_dim": [state_dim + action_dim],
        "architecture": [
            {"name": "linear1", "size": policynet_size},
            {"name": "linear2", "size": policynet_size},
            {"name": "linear2", "size": 1}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    sa_config = {
        "input_dim": [soc_dim + action_dim],
        "architecture": [
            {"name": "linear1", "size": classifier_size},
            {"name": "linear2", "size": 2}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    sas_config = {
        "input_dim": [soc_dim * 2 + action_dim],
        "architecture": [
            {"name": "linear1", "size": classifier_size},
            {"name": "linear2", "size": 2}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    
    running_mean = ZFilter((state_dim,), clip=20)
    device = "cpu"
    
    # Instantiate the models.
    contsac_model = ContSAC(
        policy_config=policy_config,
        value_config=value_config,
        env=env,
        device=device,
        log_dir=log_dir_ContSAC,
        running_mean=running_mean,
        warmup_games=2,
        batch_size=12,
        lr=args.lr,
        ent_adj=False,
        n_updates_per_train=1,
        max_steps=24,
        seed=args.eval_seed
    )

    if args.broken == 0:
        darc_model = DARC(
            policy_config=policy_config,
            value_config=value_config,
            sa_config=sa_config,
            sas_config=sas_config,
            source_env=env,
            target_env=env,
            device=device,
            running_mean=running_mean,
            log_dir=log_dir_DARC,
            warmup_games=2,
            batch_size=12,
            lr=args.lr,
            ent_adj=True,
            n_updates_per_train=1,
            max_steps=24,
            seed=args.eval_seed
        )
    else:
        darc_model = DARC_two(
            policy_config=policy_config,
            value_config=value_config,
            sa_config=sa_config,
            sas_config=sas_config,
            source_env=env,
            target_env=env,
            device=device,
            running_mean=running_mean,
            log_dir=log_dir_DARC,
            warmup_games=2,
            batch_size=12,
            lr=args.lr,
            ent_adj=True,
            n_updates_per_train=1,
            max_steps=24,
            seed=args.eval_seed
        )

    # Load the saved weights.
    contsac_model.load_model(args.contsac_model_folder, device)
    darc_model.load_model(args.darc_model_folder, device)
    
    # Evaluate RL models.
    evaluation_deterministic = not args.stochastic_eval
    contsac_total, contsac_rewards_ts, contsac_std_ts = evaluate_model(
        contsac_model, env,
        num_games=args.num_games,
        base_seed=args.eval_seed,
        deterministic=evaluation_deterministic
    )
    darc_total, darc_rewards_ts, darc_std_ts = evaluate_model(
        darc_model, env,
        num_games=args.num_games,
        base_seed=args.eval_seed,
        deterministic=evaluation_deterministic
    )
    
    print("\nEvaluation over {} episodes:".format(args.num_games))
    print("ContSAC Average Total Reward: {:.2f}".format(contsac_total))
    print("DARC Average Total Reward:    {:.2f}".format(darc_total))
    
    # Optionally, run MPC (Optimal Control) evaluation.
    if args.run_mpc:
        print("\nRunning MPC (Optimal Control) Evaluation...")
        # Create MPC evaluation environment based on the same selection logic.
        if args.broken == 0:
            if args.lin_src == 1:
                if args.variety_name == 's':
                    SG = get_new_soc_env(args.degree, args.eval_seed, fixed_start=args.fixed_start, rl=False)
                elif args.variety_name == 'c':
                    SG = get_new_charge_env(args.degree, args.eval_seed, fixed_start=args.fixed_start, rl=False)
                elif args.variety_name == 'd':
                    SG = get_new_discharge_env(args.degree, args.eval_seed, fixed_start=args.fixed_start, rl=False)
                elif args.variety_name.startswith('v'):
                    SG = get_new_all_eff_env(args.degree, args.eval_seed, fixed_start=args.fixed_start, rl=False)
                elif args.variety_name == 'lc':
                    SG = get_new_limited_capacity_env(args.capacity, args.p_lim, args.eval_seed, fixed_start=args.fixed_start, rl=False)
                elif args.variety_name == 'lp':
                    SG = get_new_limited_plim_env(args.capacity, args.p_lim, args.eval_seed, fixed_start=args.fixed_start, rl=False)
                else:
                    raise ValueError("Unknown variety name: " + args.variety_name)
            else:
                SG = get_simple_linear_env(args.eval_seed, fixed_start=args.fixed_start, rl=False)
        else:
            if args.break_src == 1:
                SG = get_twoHouses_env(damaged_battery=True, seed=args.eval_seed, fixed_start=args.fixed_start, rl=False)
            else:
                SG = get_twoHouses_env(damaged_battery=False, seed=args.eval_seed, fixed_start=args.fixed_start, rl=False)
        
        # Extract components for MPC evaluation.
        sys = SG.sys
        m1 = SG.m1
        n1 = SG.n1
        e1 = SG.e1
        horizon = timedelta(hours=24)
        
        (avg_daily_cost, all_daily_costs, std_daily_cost,
         avg_cost_per_time, std_cost_per_time) = evaluate_mpc(sys, m1, n1, e1, args.fixed_start, horizon,
                                                             num_games=args.num_games, base_seed=args.eval_seed)
        print("MPC Evaluation Results:")
        print("Average Daily Cost:", avg_daily_cost)
        print("Daily Costs for each experiment:", all_daily_costs)
        print("Standard Deviation of Daily Cost:", std_daily_cost)
        print("Average Cost per Timestep:", avg_cost_per_time)
        
        # Also, run one deployment to extract cost and SOC (for reference)
        oc_history = run_optimal_control(sys, m1, n1, e1, args.fixed_start, horizon, args.eval_seed)
        oc_total_cost, oc_soc = extract_cost_soc(oc_history, m1, n1, e1)
        print("Optimal Control Cost:", [x[1] for x in oc_total_cost])
        
        # --- Prepare Data for Plots ---
        # For MPC, convert cost to reward for per-timestep comparison.
        mpc_rewards = -avg_cost_per_time  
        # Create timesteps arrays (assume RL and MPC evaluations have same number of timesteps)
        timesteps_rl = np.arange(1, len(contsac_rewards_ts) + 1)
        timesteps_mpc = np.arange(1, len(mpc_rewards) + 1)
        
        # --- Reward Comparison Plot ---
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps_rl, contsac_rewards_ts, label="ContSAC Reward", marker='o', color='blue')
        plt.fill_between(timesteps_rl, contsac_rewards_ts - contsac_std_ts, contsac_rewards_ts + contsac_std_ts, alpha=0.3, color='blue')
        plt.plot(timesteps_rl, darc_rewards_ts, label="DARC Reward", marker='o', color='red')
        plt.fill_between(timesteps_rl, darc_rewards_ts - darc_std_ts, darc_rewards_ts + darc_std_ts, alpha=0.3, color='red')
        plt.plot(timesteps_mpc, mpc_rewards, label="MPC Reward", marker='o', color='green')
        plt.fill_between(timesteps_mpc, mpc_rewards - std_cost_per_time, mpc_rewards + std_cost_per_time, alpha=0.3, color='green')
        plt.xlabel("Timestep")
        plt.ylabel("Reward (higher is better)")
        plt.title("Reward Comparison of Controllers")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Build a figure filename that encodes the key hyperparameters.
        # (This follows a similar format to the model folder names.)
        fs = args.fixed_start
        if args.fixed_start is not None:
            fs = args.fixed_start.replace('.', '-')
        fig_filename = f"{args.save_file_name}_fs{fs}_lr{args.lr}_noise{args.noise}_seed{args.seed}_"
        if args.broken == 0:
            fig_filename += f"lin_src{args.lin_src}_variety{args.variety_name}_"
            if args.variety_name in ['s','c','d'] or args.variety_name.startswith('v'):
                fig_filename += f"degree{args.degree}_"
            elif args.variety_name == 'lc':
                fig_filename += f"cap{args.capacity}_"
            elif args.variety_name == 'lp':
                fig_filename += f"p_lim{args.p_lim}_"
            fig_filename += args.env_name
        else:
            fig_filename += f"broken{args.broken}_break_src{args.break_src}_{args.env_name}"
        
        # Build a reward figure filename.
        fs = args.fixed_start
        if args.fixed_start is not None:
            fs = args.fixed_start.replace('.', '-')
            
        reward_filename = f"Cost_all_Eval_seed{args.eval_seed}_{fig_filename}.png"
        reward_figure_path = os.path.join("figures", reward_filename)
        plt.savefig(reward_figure_path)
        print(f"Cost figure saved to {reward_figure_path}")
        plt.close()
        
        # --- Cumulative Total Cost Comparison Plot ---
        # For RL models, total cost is cumulative sum of (-reward)
        cum_cost_contsac = np.cumsum(-contsac_rewards_ts)
        cum_cost_darc = np.cumsum(-darc_rewards_ts)
        # For MPC, cumulative cost is cumulative sum of avg_cost_per_time
        cum_cost_mpc = np.cumsum(avg_cost_per_time)
        
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps_rl, cum_cost_contsac, label="ContSAC Total Cost", linestyle='--', marker='x', color='blue')
        plt.plot(timesteps_rl, cum_cost_darc, label="DARC Total Cost", linestyle='--', marker='x', color='red')
        plt.plot(timesteps_mpc, cum_cost_mpc, label="MPC Total Cost", linestyle='--', marker='x', color='green')
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Total Cost (lower is better)")
        plt.title("Cumulative Total Cost Comparison of Controllers")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Ensure the figures directory exists.
        #os.makedirs("figures", exist_ok=True)
        #plt.show()
        
        # Build a cost figure filename.
        cost_filename = f"Cumulative_cost_all_Eval_seed{args.eval_seed}_{fig_filename}.png"
        cost_figure_path = os.path.join("figures", cost_filename)
        plt.savefig(cost_figure_path)
        print(f"Cumulative cost figure saved to {cost_figure_path}")
        plt.close()
        
        # --- Save Average Rewards ---
        # For RL models, we already have average total rewards.
        # For MPC, we convert average daily cost to average reward (reward = -cost)
        mpc_avg_reward = -avg_daily_cost
        # Create an array with the order: [DARC, ContSAC, MPC]
        avg_rewards_array = np.array([darc_total, contsac_total, mpc_avg_reward])
        print("Average rewards array (DARC, ContSAC, MPC):", avg_rewards_array)
        avg_rewards_filename = f"average_rewards_Eval_seed{args.eval_seed}_{fig_filename}.npy"
        avg_rewards_path = os.path.join("figures", avg_rewards_filename)
        np.save(avg_rewards_path, avg_rewards_array)
        print(f"Average rewards saved to {avg_rewards_path}")
    
if __name__ == '__main__':
    main()
