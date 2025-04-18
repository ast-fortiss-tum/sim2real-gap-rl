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
from models.darc import DARC
from environments.get_customized_envs import *
from utils import ZFilter  # Assuming ZFilter is defined in utils

def evaluate_model(model, env, num_games=10, base_seed=42):
    """
    Runs evaluation episodes on the provided environment using the model.
    For each episode, the environment is reset with a new seed (base_seed + episode index)
    so that starting conditions vary.
    
    Returns:
      - avg_total_reward: average episode return.
      - avg_rewards_per_timestep: average reward per timestep (averaged across episodes).
      - std_rewards_per_timestep: standard deviation of reward per timestep across episodes.
    """
    model.policy.eval()
    model.twin_q.eval()
    rewards_all = []
    
    # Loop over episodes with a progress bar.
    for episode in tqdm(range(num_games), desc="Evaluating episodes"):
        # Use a different seed for each episode
        current_seed = base_seed + episode
        state, _ = env.reset(seed=current_seed)
        state = model.running_mean(state)
        done = False
        episode_rewards = []
        while not done:
            action = model.get_action(state, deterministic=False)
            next_state, reward, done, _, _ = env.step(action)
            next_state = model.running_mean(next_state)
            episode_rewards.append(reward)
            state = next_state
        rewards_all.append(episode_rewards)
        
    # Convert rewards to a numpy array (num_episodes x episode_length)
    rewards_all = np.array(rewards_all)
    # Compute average instantaneous reward (across episodes) at each timestep.
    avg_rewards_per_timestep = np.mean(rewards_all, axis=0)
    std_rewards_per_timestep = np.std(rewards_all, axis=0)
    # Compute average total reward per episode.
    avg_total_reward = np.mean(np.sum(rewards_all, axis=1))
    return avg_total_reward, avg_rewards_per_timestep, std_rewards_per_timestep

def main():
    parser = argparse.ArgumentParser(description="Compare ContSAC vs DARC Models")
    parser.add_argument('--contsac-model-folder', type=str, default="/saved_models_experiments/ContSAC_test_run__20250327_045047_lr0.0008_noise0.0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids",
                        help='Folder name under saved_weights for the saved ContSAC model')
    parser.add_argument('--darc-model-folder', type=str, default="/saved_models_experiments/DARC_test_run__20250327_044227_lr0.0008_noise0.0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids/saved_models_experiments/DARC_test_run__20250327_044227_lr0.0008_noise0.0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids",
                        help='Folder name under saved_weights for the saved DARC model')
    parser.add_argument('--num-games', type=int, default=10,
                        help='Number of evaluation episodes per model')
    parser.add_argument('--seed', type=int, default=126,
                        help='Base seed for evaluation (will be incremented for each episode)')
    args = parser.parse_args()

    # Set seeds for reproducibility.
    set_global_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the evaluation environment.
    #env = get_new_all_eff_env(0.2, args.seed, rl=True).env
    env = get_new_all_eff_env(0.5, args.seed, rl = True).env
    
    # Get dimensions from the environment.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Define network architectures (should match training settings).
    policynet_size = 256  # example value
    classifier_size = 32  # for DARC classifiers
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
    # For DARC, define classifier network configs.
    soc_dim = 1  # For one-house experiments.
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
    
    # Instantiate the normalization filter (running mean).
    running_mean = ZFilter((state_dim,), clip=20)

    device = "cpu"  # Change to "cuda" if using GPU

    # Instantiate the ContSAC model.
    contsac_model = ContSAC(
        policy_config=policy_config,
        value_config=value_config,
        env=env,
        device=device,
        log_dir="eval_runs",
        running_mean=running_mean,
        warmup_games=10,
        batch_size=64,
        lr=0.0001,
        ent_adj=False,
        n_updates_per_train=1,
        max_steps=24,
        seed=args.seed
    )
    
    # Instantiate the DARC model.
    darc_model = DARC(
        policy_config=policy_config,
        value_config=value_config,
        sa_config=sa_config,
        sas_config=sas_config,
        source_env=env,
        target_env=env,
        device=device,
        savefolder="dummy",  # not used during evaluation
        running_mean=running_mean,
        log_dir="eval_runs",
        warmup_games=10,
        batch_size=64,
        lr=0.0001,
        ent_adj=True,
        n_updates_per_train=1,
        max_steps=24,
        seed=args.seed
    )
    
    # Load the saved weights for both models.
    contsac_model.load_model(args.contsac_model_folder, device)
    darc_model.load_model(args.darc_model_folder, device)
    
    # Evaluate each model. Each episode will use a new seed.
    contsac_total, contsac_rewards_ts, contsac_std_ts = evaluate_model(contsac_model, env,
                                                                       num_games=args.num_games,
                                                                       base_seed=args.seed)
    darc_total, darc_rewards_ts, darc_std_ts = evaluate_model(darc_model, env,
                                                               num_games=args.num_games,
                                                               base_seed=args.seed)
    
    print("\nEvaluation over {} episodes:".format(args.num_games))
    print("ContSAC Average Total Reward: {:.2f}".format(contsac_total))
    print("DARC Average Total Reward:    {:.2f}".format(darc_total))
    
    # Plot average reward vs timestep for both models with standard deviation as colored area.
    timesteps = np.arange(1, len(contsac_rewards_ts) + 1)
    plt.figure(figsize=(10, 6))
    
    # Plot ContSAC mean and std deviation area.
    plt.plot(timesteps, contsac_rewards_ts, label="ContSAC", color="blue")
    plt.fill_between(timesteps, contsac_rewards_ts - contsac_std_ts, contsac_rewards_ts + contsac_std_ts,
                     color="blue", alpha=0.3)
    
    # Plot DARC mean and std deviation area.
    plt.plot(timesteps, darc_rewards_ts, label="DARC", color="red")
    plt.fill_between(timesteps, darc_rewards_ts - darc_std_ts, darc_rewards_ts + darc_std_ts,
                     color="red", alpha=0.3)
    
    plt.xlabel("Timestep")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Timestep during Evaluation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/reward_vs_timestep.png")
    plt.show()

if __name__ == '__main__':
    main()
