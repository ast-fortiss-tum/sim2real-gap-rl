#!/usr/bin/env python3
import gymnasium as gym
from datetime import datetime, timedelta
from models.darc import DARC, DARC_two
from models.sac import ContSAC
from models.mpc import MPC
from utils import ZFilter, EMAZFilter
from environments.get_customized_envs import (get_new_soc_env, get_new_charge_env, 
                   get_new_discharge_env, get_new_all_eff_env, 
                   get_new_limited_capacity_env, get_new_limited_plim_env, get_twoHouses_env,
                   get_simple_linear_env)
import argparse
from gymnasium.envs.registration import register
import os
import time

# -----------------------------------------------------------------------------
# Registration of the Smart Grid environments with Gymnasium.
# -----------------------------------------------------------------------------

register(
    id='Smart_Grids_Linear-v0',
    entry_point='environments.smartgrid_env:make_smartgrid_linear',
    max_episode_steps=24,
)

def parse_args():
    parser = argparse.ArgumentParser()

    # Saving and training hyperparameters
    parser.add_argument('--save-model', type=str, default="",
                        help='Base path for saving the model')
    parser.add_argument('--train-steps', type=int, default=24*30,
                        help='Number of training steps')
    parser.add_argument('--max-steps', type=int, default=24,
                        help='Maximum steps per episode')
    parser.add_argument('--save_file_name', type=str, default='',
                        help='File name to append to saved model path')
    parser.add_argument('--lr', type=float, default=8e-4,
                        help='Learning rate')
    parser.add_argument('--bs', type=int, default=12,
                        help='Batch size')
    parser.add_argument('--update', type=int, default=1,
                        help='Number of updates per training iteration')
    parser.add_argument('--deltar', type=float, default=1,
                        help='Delta r scale')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup steps')

    # Environment settings (only Smart Grid supported)
    parser.add_argument('--env-name', type=str, default="Smart_Grids",
                        help='Name of environment to use (only Smart_Grids supported)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for environment')

    # Mandatory: Broken flag (0: one-house, 1: two-house)
    parser.add_argument('--broken', type=int, choices=[0,1], required=True,
                        help='Set to 1 for a broken (two-house) environment; 0 for one-house')

    # For non-broken experiments (one-house), these arguments are mandatory:
    parser.add_argument('--lin_src', type=int, default=None,
                        help='For non-broken experiments, which environment is linear: '
                        '1 for source, 0 for target (mandatory when broken==0)')
    parser.add_argument('--variety-name', type=str, default=None,
                        help="Name of variety ('s', 'c', 'd', 'v...', 'lc', 'lp') (mandatory when broken==0)")
    parser.add_argument('--degree', type=float, default=None,
                        help='Degree parameter for variety (in (0,1]) (mandatory when broken==0)')
    parser.add_argument('--noise', type=float, default=None,
                        help='Noise scale (mandatory when broken==0)')
    parser.add_argument('--capacity', type=int, default=None,
                        help='Capacity of the battery in Wh (mandatory when broken==0)')
    parser.add_argument('--p_lim', type=float, default=None,
                        help='Power limit of the battery in W (mandatory when broken==0)')

    # For broken experiments (two-house), this argument is mandatory:
    parser.add_argument('--break_src', type=int, default=None,
                        help='For broken experiments, which environment to break: '
                        '1 for source, 0 for target (mandatory when broken==1)')

    # Network architecture parameters (always needed)
    parser.add_argument('--policynet', type=int, default=256,
                        help='Size of policy network layers')
    parser.add_argument('--classifier', type=int, default=32,
                        help='Size of classifier network layers')

    args = parser.parse_args()

    # Conditional checks: ensure that the correct parameters are provided
    if args.broken == 0:
        missing = []
        if args.lin_src is None:
            missing.append("--lin_src")
        if args.variety_name is None:
            missing.append("--variety-name")
        if args.degree and args.capacity and args.p_lim is None:
            missing.append("--degree or --capacity or --p_lim")
        if missing:
            parser.error("When broken == 0, the following arguments are required: " + ", ".join(missing))
    else:  # args.broken == 1
        if args.break_src is None:
            parser.error("When broken == 1, --break_src is required.")
    return args

def construct_save_model_path(args, prefix="DARC"):
    currentDateAndTime = datetime.now()
    date_str = currentDateAndTime.strftime("%Y%m%d_%H%M%S")
    # Build a base filename with common parameters, model prefix, and seed value.
    filename = f"{prefix}_{args.save_file_name}_{date_str}_lr{args.lr}_noise{args.noise}_seed{args.seed}_"
    if args.broken == 0:
        # One-house experiments: include linear source flag and variety.
        filename += f"lin_src{args.lin_src}_variety{args.variety_name}_"
        # Depending on the variety, add the relevant parameter.
        if args.variety_name in ['s', 'c', 'd'] or args.variety_name.startswith('v'):
            filename += f"degree{args.degree}_"
        elif args.variety_name == 'lc':
            filename += f"cap{args.capacity}_"
        elif args.variety_name == 'lp':
            filename += f"p_lim{args.p_lim}_"
        filename += args.env_name
    else:
        # Two-house (broken) experiments.
        filename += f"broken{args.broken}_break_src{args.break_src}_{args.env_name}"
    return os.path.join(args.save_model, filename)

def construct_log_dir(args, prefix="DARC"):
    base_log_dir = "runs"
    os.makedirs(base_log_dir, exist_ok=True)
    currentDateAndTime = datetime.now()
    date_str = currentDateAndTime.strftime("%Y%m%d_%H%M%S")
    if args.broken == 0:
        log_subfolder = (f"{prefix}_lin_src_{args.lin_src}_variety_{args.variety_name}_"
                         f"noise_{args.noise}_seed_{args.seed}_")
        if args.variety_name in ['s', 'c', 'd'] or args.variety_name.startswith('v'):
            log_subfolder += f"degree_{args.degree}_"
        elif args.variety_name == 'lc':
            log_subfolder += f"cap_{args.capacity}_"
        elif args.variety_name == 'lp':
            log_subfolder += f"p_lim_{args.p_lim}_"
        log_subfolder += f"{args.env_name}_{date_str}"
    else:
        log_subfolder = (f"{prefix}_broken_{args.broken}_break_src_{args.break_src}_"
                         f"noise_{args.noise}_seed_{args.seed}_{args.env_name}_{date_str}")
    log_dir = os.path.join(base_log_dir, log_subfolder)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def main():
    args = parse_args()
    # Build paths for DARC/DARC_two with "DARC" prefix.
    save_model_path = construct_save_model_path(args, prefix="DARC")
    log_dir = construct_log_dir(args, prefix="DARC")
    print("Saving DARC model to:", save_model_path)
    print("TensorBoard logs for DARC will be saved in:", log_dir)

    # Environment creation (Smart Grid only):
    if args.env_name != "Smart_Grids":
        raise ValueError("This script only supports Smart_Grids environments.")
        
    if args.broken == 0:
        # One-house setting with variety modifications.
        if args.lin_src == 1:
            source_env = get_simple_linear_env(args.seed)
            if args.variety_name == 's':
                target_env = get_new_soc_env(args.degree, args.seed)
            elif args.variety_name == 'c':
                target_env = get_new_charge_env(args.degree, args.seed)
            elif args.variety_name == 'd':
                target_env = get_new_discharge_env(args.degree, args.seed)
            elif args.variety_name.startswith('v'):
                target_env = get_new_all_eff_env(args.degree, args.seed)
            elif args.variety_name == 'lc':
                target_env = get_new_limited_capacity_env(args.capacity, 1.5, args.seed)
            elif args.variety_name == 'lp':
                target_env = get_new_limited_plim_env(3, args.p_lim, args.seed)
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
        else:
            target_env = get_simple_linear_env(args.seed)
            if args.variety_name == 's':
                source_env = get_new_soc_env(args.degree, args.seed)
            elif args.variety_name == 'c':
                source_env = get_new_charge_env(args.degree, args.seed)
            elif args.variety_name == 'd':
                source_env = get_new_discharge_env(args.degree, args.seed)
            elif args.variety_name.startswith('v'):
                source_env = get_new_all_eff_env(args.degree, args.seed)
            elif args.variety_name == 'lc':
                source_env = get_new_limited_capacity_env(args.capacity, 1.5, args.seed)
            elif args.variety_name == 'lp':
                source_env = get_new_limited_plim_env(3, args.p_lim, args.seed)
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
    else:
        # Two-house setting (broken environment).
        if args.break_src == 1:
            source_env = get_twoHouses_env(damaged_battery=True, seed=args.seed)
            target_env = get_twoHouses_env(damaged_battery=False, seed=args.seed)
        else:
            source_env = get_twoHouses_env(damaged_battery=False, seed=args.seed)
            target_env = get_twoHouses_env(damaged_battery=True, seed=args.seed)

    # Get dimensions from the source environment.
    state_dim = source_env.observation_space.shape[0]
    action_dim = source_env.action_space.shape[0]
    
    # When broken==0, we only use one-house soc input; when broken==1, two values.
    soc_dim = 1 if args.broken == 0 else 2

    print("State dimension:", state_dim)
    print("Action dimension:", action_dim)

    policy_config = {
        "input_dim": [state_dim],
        "architecture": [
            {"name": "linear1", "size": args.policynet},
            {"name": "linear2", "size": args.policynet},
            {"name": "split1", "sizes": [action_dim, action_dim]}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    value_config = {
        "input_dim": [state_dim + action_dim],
        "architecture": [
            {"name": "linear1", "size": args.policynet},
            {"name": "linear2", "size": args.policynet},
            {"name": "linear2", "size": 1}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    sa_config = {
        "input_dim": [soc_dim + action_dim],
        "architecture": [
            {"name": "linear1", "size": args.classifier},
            {"name": "linear2", "size": 2}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    sas_config = {
        "input_dim": [soc_dim * 2 + action_dim],
        "architecture": [
            {"name": "linear1", "size": args.classifier},
            {"name": "linear2", "size": 2}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }

    running_state = ZFilter((state_dim,), clip=20)
    #running_state = EMAZFilter((state_dim,), clip=20)

    # Instantiate DARC (or DARC_two) using the constructed paths.
    if args.broken == 0:
        model = DARC(
            policy_config, value_config, sa_config, sas_config,
            source_env, target_env, "cpu", ent_adj=True,
            n_updates_per_train=args.update, lr=args.lr,
            max_steps=args.max_steps, batch_size=args.bs,
            savefolder=save_model_path, running_mean=running_state,
            if_normalize=False, delta_r_scale=args.deltar,
            noise_scale=args.noise, warmup_games=args.warmup,
            log_dir=log_dir, seed=args.seed
        )
    else:
        model = DARC_two(
            policy_config, value_config, sa_config, sas_config,
            source_env, target_env, "cpu", ent_adj=True,
            n_updates_per_train=args.update, lr=args.lr,
            max_steps=args.max_steps, batch_size=args.bs,
            savefolder=save_model_path, running_mean=running_state,
            if_normalize=False, delta_r_scale=args.deltar,
            noise_scale=args.noise, warmup_games=args.warmup,
            log_dir=log_dir, seed=args.seed
        )

    model.train(args.train_steps, deterministic=False)
    model.save_model(save_model_path)

    # -------------------------------------------------------------------------
    # Also train a vanilla ContSAC model with the same arguments but with its own paths.
    # The vanilla model will use a log_dir and save path that include "ContSAC" in their names.
    # -------------------------------------------------------------------------

    vanilla_save_model_path = construct_save_model_path(args, prefix="ContSAC")
    vanilla_log_dir = construct_log_dir(args, prefix="ContSAC")

    print("Saving Vanilla SAC model to:", vanilla_save_model_path)
    print("TensorBoard logs for vanilla SAC will be saved in:", vanilla_log_dir)

    model_vanilla = ContSAC(policy_config, value_config, target_env, 
        "cpu", log_dir=vanilla_log_dir, running_mean=running_state,
        warmup_games=args.warmup, batch_size=args.bs, lr=args.lr, 
        ent_adj=False, n_updates_per_train=args.update, max_steps=args.max_steps,
        seed=args.seed
    )

    model_vanilla.train(int(args.train_steps/10), deterministic=False)
    model_vanilla.save_model(vanilla_save_model_path)

    mpc_save_model_path = construct_save_model_path(args, prefix="MPC")

    ############ TEST ####################
    mpc = MPC(target_env, save_dir=mpc_save_model_path,seed=args.seed)
    mpc.train()
    #Save is automatic with run in mpc

if __name__ == '__main__':
    main()
