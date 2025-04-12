#!/usr/bin/env python3
import gymnasium as gym
from datetime import datetime, timedelta
#from models.darc import DARC, DARC_two
#from models.sac import ContSAC
#from models.darc_denoise import DARC, DARC_two
from models.darc_refactored import DARC_one, DARC_two
from models.sac_refactored import ContSAC
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
    parser.add_argument('--train-steps', type=int, default=200,
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
    #parser.add_argument('--fixed-start', type=str, default="27.11.2016",
    #                    help='Fixed start date for the environment (format DD.MM.YYYY)')

    parser.add_argument('--fixed-start', type=str, default=None,
                        help='Fixed start date for the environment (format DD.MM.YYYY)')

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
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise scale (mandatory when broken==0)')
    parser.add_argument('--capacity', type=float, default=3.0,
                        help='Capacity of the battery in Wh (mandatory when broken==0)')
    parser.add_argument('--p_lim', type=float, default=1.5,
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
    
    parser.add_argument('--bias', type=float, default=0.5,
                        help='Bias for unknown noise')
    
    parser.add_argument('--use_denoiser', type=int, default=1,
                        help='Use denoiser (1) or not (0)')
    
    parser.add_argument('--s_t_ratio', type=int, default=10,
                        help='Ratio between source and target training steps')
    
    parser.add_argument('--noise_cfrs', type=float, default=0.2,
                        help='Noise scale for the denoiser (if used)')

    args = parser.parse_args()

    # Conditional checks: ensure that the correct parameters are provided
    if args.broken == 0:
        missing = []
        if args.lin_src is None:
            missing.append("--lin_src")
        if args.variety_name is None:
            missing.append("--variety-name")
        if args.degree is None and args.capacity is None and args.p_lim is None:
            missing.append("--degree, --capacity, or --p_lim")
        if missing:
            parser.error("When broken == 0, the following arguments are required: " + ", ".join(missing))
    else:  # args.broken == 1
        if args.break_src is None:
            parser.error("When broken == 1, --break_src is required.")
    return args

def construct_save_model_path(args, prefix="DARC"):
    # Replace dots in fixed_start for a cleaner filename.
    fs = args.fixed_start
    if args.fixed_start is not None:
        fs = args.fixed_start.replace('.', '-')
    # Build a filename solely from the hyperparameters.
    filename = f"{prefix}_{args.save_file_name}_fs{fs}_lr{args.lr}_noise{args.noise}_bias{args.bias}_seed{args.seed}_noise_cfrs_{args.noise_cfrs}_use_denoiser_{args.use_denoiser}_"
    if args.broken == 0:
        # One-house experiments: include linear source flag and variety.
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
    if False:
        # Append smoother name if one is used.
        smoother = ""
        if args.fft:
            smoother += "FFT"
        if args.kalman:
            smoother += "KF"
        if args.learn_smoother:
            smoother += "Learnable"
        if smoother != "":
            filename += f"_{smoother}"

    return os.path.join(args.save_model, filename)

def construct_log_dir(args, prefix="DARC"):
    base_log_dir = "runs5"  # Change if needed !!
    #os.makedirs(base_log_dir, exist_ok=True)
    fs = args.fixed_start
    if args.fixed_start is not None:
        fs = args.fixed_start.replace('.', '-')
    if args.broken == 0:
        log_subfolder = (f"{prefix}_{args.save_file_name}_fs_{fs}_lin_src_{args.lin_src}_variety_{args.variety_name}_"
                         f"noise_{args.noise}_bias_{args.bias}_seed_{args.seed}_noise_cfrs_{args.noise_cfrs}_use_denoiser_{args.use_denoiser}_")
        if args.variety_name in ['s', 'c', 'd'] or args.variety_name.startswith('v'):
            log_subfolder += f"degree_{args.degree}_"
        elif args.variety_name == 'lc':
            log_subfolder += f"cap_{args.capacity}_"
        elif args.variety_name == 'lp':
            log_subfolder += f"p_lim_{args.p_lim}_"
        log_subfolder += f"{args.env_name}"
    else:
        log_subfolder = (f"{prefix}_fs_{fs}_broken_{args.broken}_break_src_{args.break_src}_"
                         f"noise_{args.noise}_seed_{args.seed}_noise_cfrs_{args.noise_cfrs}_use_denoiser_{args.use_denoiser}_{args.env_name}")
    
    if False:
        # Append smoother name if one is used.
        smoother = ""
        if args.fft:
            smoother += "FFT"
        if args.kalman:
            smoother += "KF"
        if args.learn_smoother:
            smoother += "Learnable"
        if smoother != "":
            log_subfolder += f"_{smoother}"
    log_dir = os.path.join(base_log_dir, log_subfolder)
    #os.makedirs(log_dir, exist_ok=True)
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
                target_env = get_new_limited_capacity_env(args.capacity, 1.5, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'lp':
                target_env = get_new_limited_plim_env(3, args.p_lim, args.seed, fixed_start=args.fixed_start).env
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
                source_env = get_new_limited_capacity_env(args.capacity, 1.5, args.seed, fixed_start=args.fixed_start).env
            elif args.variety_name == 'lp':
                source_env = get_new_limited_plim_env(3, args.p_lim, args.seed, fixed_start=args.fixed_start).env
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
    else:
        # Two-house setting (broken environment).
        if args.break_src == 1:
            source_env = get_twoHouses_env(damaged_battery=True, seed=args.seed, fixed_start=args.fixed_start).env
            target_env = get_twoHouses_env(damaged_battery=False, seed=args.seed, fixed_start=args.fixed_start).env
        else:
            source_env = get_twoHouses_env(damaged_battery=False, seed=args.seed, fixed_start=args.fixed_start).env
            target_env = get_twoHouses_env(damaged_battery=True, seed=args.seed, fixed_start=args.fixed_start).env

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
        model = DARC_one(
            policy_config, value_config, sa_config, sas_config,
            source_env, target_env, "cpu", ent_adj=True,
            n_updates_per_train=args.update, lr=args.lr,
            max_steps=args.max_steps, batch_size=args.bs,
            running_mean=running_state,
            if_normalize=False, delta_r_scale=args.deltar, s_t_ratio=args.s_t_ratio,
            noise_scale=args.noise, bias=args.bias, warmup_games=args.warmup,
            log_dir=log_dir, seed=args.seed, use_denoiser=args.use_denoiser, noise_cfrs=args.noise_cfrs
        )
    else:
        model = DARC_two(
            policy_config, value_config, sa_config, sas_config,
            source_env, target_env, "cpu", ent_adj=True,
            n_updates_per_train=args.update, lr=args.lr,
            max_steps=args.max_steps, batch_size=args.bs,
            running_mean=running_state,
            if_normalize=False, delta_r_scale=args.deltar, s_t_ratio=args.s_t_ratio,
            noise_scale=args.noise, bias=args.bias, warmup_games=args.warmup,
            log_dir=log_dir, seed=args.seed, use_denoiser=args.use_denoiser, noise_cfrs=args.noise_cfrs
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

    noise_indices = [100, 226] if args.broken == 1 else [100]

    train_steps_SAC = int(args.train_steps / args.s_t_ratio)

    model_vanilla = ContSAC(
        policy_config, value_config, target_env, 
        "cpu", log_dir=vanilla_log_dir, running_mean=running_state, noise_scale=args.noise, bias=args.bias,
        warmup_games=args.warmup, batch_size=args.bs, lr=args.lr, 
        ent_adj=False, n_updates_per_train=args.update, max_steps=args.max_steps,
        seed=args.seed, noise_indices=noise_indices, use_denoiser=args.use_denoiser
    )

    model_vanilla.train(train_steps_SAC, deterministic=False)
    model_vanilla.save_model(vanilla_save_model_path)

    ############ NO NECESSARY ####################
    # mpc_save_model_path = construct_save_model_path(args, prefix="MPC")
    # mpc = MPC(target_env_mpc, save_dir=mpc_save_model_path, seed=args.seed)
    # mpc.train()
    # Save is automatic with run in mpc

if __name__ == '__main__':
    main()
