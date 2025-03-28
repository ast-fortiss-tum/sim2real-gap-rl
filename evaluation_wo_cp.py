#!/usr/bin/env python3
import argparse
import os
import pickle
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar

# Import your model classes and utility functions.
from models.sac import ContSAC, set_global_seed
from models.darc import DARC, DARC_two  # Import both DARC and DARC_two
from environments.get_customized_envs import (
    get_simple_linear_env, get_new_soc_env, get_new_charge_env, 
    get_new_discharge_env, get_new_all_eff_env, get_new_limited_capacity_env, 
    get_new_limited_plim_env, get_twoHouses_env
)
from utils import ZFilter  # Assuming ZFilter is defined in utils

#############################
# Folder Construction Helpers
#############################
def construct_save_model_path(args, prefix):
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
    For non-broken experiments, the selection depends on --lin_src and --variety-name.
    For broken (two-house) experiments, it depends on --break_src.
    """
    if args.broken == 0:
        # Ensure mandatory hyperparameters for one-house experiments are provided.
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
# Evaluation Function
#############################
def evaluate_model(model, env, num_games=10, base_seed=42, deterministic=True):
    """
    Evaluate a given model on the provided environment.
    
    Args:
      model: The RL model to evaluate.
      env: The Gym environment.
      num_games: Number of evaluation episodes.
      base_seed: Base seed to vary the episode initialization.
      deterministic: Whether to use deterministic actions (recommended for evaluation).
      
    Returns:
      avg_total_reward: Average total reward across episodes.
      avg_rewards_per_timestep: Average reward per timestep across episodes.
      std_rewards_per_timestep: Standard deviation of reward per timestep.
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
            # Use the provided 'deterministic' flag for evaluation.
            action = model.get_action(state, deterministic=deterministic)
            next_state, reward, done, _, _ = env.step(action)
            next_state = model.running_mean(next_state)
            episode_rewards.append(reward)
            state = next_state
        rewards_all.append(episode_rewards)
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
    
    # Folder/hyperparameter settings (used for selecting the folder)
    parser.add_argument('--save-model', type=str, default="saved_weights",
                        help='Base folder for saving/loading models')
    parser.add_argument('--save_file_name', type=str, default='test_run',
                        help='File name identifier (appended in folder name)')
    parser.add_argument('--lr', type=float, default=0.0008,
                        help='Learning rate')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--env-name', type=str, default="Smart_Grids",
                        help='Name of environment (Smart_Grids only)')
    parser.add_argument('--fixed-start', type=str, default="27.11.2016",
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
    
    # Optional: override model folder paths directly.
    parser.add_argument('--contsac_model_folder', type=str, default="",
                        help='(Optional) Folder path for saved ContSAC model')
    parser.add_argument('--darc_model_folder', type=str, default="",
                        help='(Optional) Folder path for saved DARC model')
    
    parser.add_argument('--num-games', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--eval-seed', type=int, default=126,
                        help='Base seed for evaluation (incremented per episode)')
    # If set, evaluation will use stochastic actions instead of deterministic ones.
    parser.add_argument('--stochastic-eval', action='store_true', 
                        help='If set, evaluation will use stochastic actions (not recommended for final reporting)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select environments based on hyperparameters.
    source_env, target_env = select_environment(args)
    
    # For evaluation, we use the target environment.
    env = target_env
    
    # If model folder paths are not provided, compute them from hyperparameters.
    if not args.contsac_model_folder:
        args.contsac_model_folder = construct_save_model_path(args, prefix="ContSAC")
    if not args.darc_model_folder:
        args.darc_model_folder = construct_save_model_path(args, prefix="DARC")
    
    log_dir_DARC = construct_log_dir(args, prefix="DARC")
    log_dir_ContSAC = construct_log_dir(args, prefix="ContSAC")

    print("Using ContSAC model folder:", args.contsac_model_folder)
    print("Using DARC model folder:", args.darc_model_folder)
    
    # Set seeds for reproducibility.
    set_global_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get dimensions from the environment.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Determine the soc dimension based on whether the experiment is broken.
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
    # Update classifier network configurations based on soc_dim.
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
    
    # Instantiate the normalization filter.
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
        seed=args.seed
    )

    # Instantiate the DARC model using the appropriate class.
    if args.broken == 0:
        darc_model = DARC(
            policy_config=policy_config,
            value_config=value_config,
            sa_config=sa_config,
            sas_config=sas_config,
            source_env=env,  # for evaluation, we use the same env
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
            seed=args.seed
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
            seed=args.seed
        )

    # Load the saved weights.
    contsac_model.load_model(args.contsac_model_folder, device)
    darc_model.load_model(args.darc_model_folder, device)
    
    # Evaluate each model.
    # Recommended: use deterministic actions during evaluation for reproducibility.
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
    
    # Plot average reward vs timestep.
    timesteps = np.arange(1, len(contsac_rewards_ts) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, contsac_rewards_ts, label="ContSAC", color="blue")
    plt.fill_between(timesteps, contsac_rewards_ts - contsac_std_ts, contsac_rewards_ts + contsac_std_ts,
                     color="blue", alpha=0.3)
    plt.plot(timesteps, darc_rewards_ts, label="DARC", color="red")
    plt.fill_between(timesteps, darc_rewards_ts - darc_std_ts, darc_rewards_ts + darc_std_ts,
                     color="red", alpha=0.3)
    plt.xlabel("Timestep")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Timestep during Evaluation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Build a figure filename that encodes the key hyperparameters.
    # (This follows a similar format to the model folder names.)
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
    
    # Ensure the figures directory exists.
    os.makedirs("figures", exist_ok=True)
    figure_path = os.path.join("figures", f"reward_vs_timestep_{fig_filename}.png")
    plt.savefig(figure_path)
    print(f"Figure saved to {figure_path}")
    plt.show()

if __name__ == '__main__':
    main()

"""
Example for calling:
python3 test_models.py \
  --env-name Smart_Grids \
  --fixed-start 27.11.2016 \
  --broken 0 \
  --lin_src 0 \
  --variety-name lc \
  --capacity 2 \
  --degree 0.5 \
  --lr 0.0008 \
  --noise 0 \
  --seed 42 \
  --save-model "" \
  --save_file_name "" \
  --eval-seed 126 \
  --num-games 10
"""
