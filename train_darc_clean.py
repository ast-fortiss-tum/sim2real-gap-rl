#!/usr/bin/env python3
import gymnasium as gym
from datetime import datetime, timedelta
from models.darc import DARC
from utils import ZFilter
from environments.get_customized_envs import (get_new_soc_env, get_new_charge_env, 
                   get_new_discharge_env, get_new_all_eff_env, 
                   get_new_limited_capacity_env, get_new_limited_plim_env)
import argparse
from gymnasium.envs.registration import register
import os

# -----------------------------------------------------------------------------
# Registration of the Smart Grid environments with Gymnasium.
# -----------------------------------------------------------------------------

register(
    id='Smart_Grids_Linear-v0',
    entry_point='environments.smartgrid_env:make_smartgrid_linear',
    max_episode_steps=24,
)

register(
    id='Smart_Grids_TwoHouses_Normal-v0',
    entry_point='environments.smartgrid_env:make_smartgrid_twohouses_normal',
    max_episode_steps=24,
)

register(
    id='Smart_Grids_TwoHouses_Damage-v0',
    entry_point='environments.smartgrid_env:make_smartgrid_twohouses_damaged_battery',
    max_episode_steps=24,
)

def parse_args():
    parser = argparse.ArgumentParser()
    # Saving and training hyperparameters
    parser.add_argument('--save-model', type=str, default="",
                        help='Base path for saving the model')
    parser.add_argument('--train-steps', type=int, default=1000,
                        help='Number of training steps')
    parser.add_argument('--max-steps', type=int, default=24,
                        help='Maximum steps per episode')
    parser.add_argument('--save_file_name', type=str, default='',
                        help='File name to append to saved model path')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--bs', type=int, default=12,
                        help='Batch size')
    parser.add_argument('--update', type=int, default=1,
                        help='Number of updates per training iteration')
    parser.add_argument('--deltar', type=float, default=1,
                        help='Delta r scale')

    # Environment settings (only Smart Grid supported)
    parser.add_argument('--env-name', type=str, default="Smart_Grids",
                        help='Name of environment to use (only Smart_Grids supported)')

    # Smart Grid broken environment settings:
    # Broken == 1: Two-house setting (one battery damaged)
    # Broken == 0: One-house setting with battery variety modifications.
    parser.add_argument('--lin_src', type=int, default=1,
                        help='For non broken experiments, which environment to behave linear. 1 for source, 0 for target')
    parser.add_argument('--broken', type=int, default=1,
                        help='Use a broken environment (1: two-house setting) or a one-house setting (0)')
    parser.add_argument('--break_src', type=int, default=1,
                        help='For broken experiments, which environment to break: 1 for source, 0 for target')

    # One-house variety parameters (only used when broken == 0)
    # variety-name: 's' (soc), 'c' (charge), 'd' (discharge),
    # 'v...' (all efficiencies), 'lc' (limited capacity), or 'lp' (limited p)
    parser.add_argument('--variety-name', type=str, default="s",
                        help='Name of variety (s, c, d, v..., lc, lp)')
    parser.add_argument('--degree', type=float, default=0.5,
                        help='Degree parameter for variety (in (0,1])')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise scale')

    # Network architecture parameters
    parser.add_argument('--policynet', type=int, default=256,
                        help='Size of policy network layers')
    parser.add_argument('--classifier', type=int, default=32,
                        help='Size of classifier network layers')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup steps')

    return parser.parse_args()

def construct_save_model_path(args):
    currentDateAndTime = datetime.now()
    date_str = currentDateAndTime.strftime("%Y%m%d_%H%M%S")
    save_model_path = args.save_model + args.save_file_name + '_' + date_str + '_' + str(args.lr) + '_'
    if args.broken == 0:
        save_model_path += args.variety_name + '_' + str(args.degree) + '_' + args.env_name
    else:
        save_model_path += str(args.break_src) + '_' + args.env_name
    return save_model_path

def construct_log_dir(args):
    """
    Constructs a unique log directory for TensorBoard inside runs/ based on key settings.
    """
    base_log_dir = "runs"
    os.makedirs(base_log_dir, exist_ok=True)
    currentDateAndTime = datetime.now()
    date_str = currentDateAndTime.strftime("%Y%m%d_%H%M%S")
    if args.broken == 1:
        log_subfolder = f"broken_{args.broken}_broken_src_{args.break_src}_{args.env_name}_{date_str}"
    else:
        log_subfolder = f"lin_src_{args.lin_src}_variety_{args.variety_name}_degree_{args.degree}_{args.env_name}_{date_str}"
    log_dir = os.path.join(base_log_dir, log_subfolder)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def main():
    args = parse_args()
    save_model_path = construct_save_model_path(args)
    log_dir = construct_log_dir(args)
    print("Saving model to:", save_model_path)
    print("TensorBoard logs will be saved in:", log_dir)

    # Environment creation (Smart Grid only):
    if args.env_name != "Smart_Grids":
        raise ValueError("This script only supports Smart_Grids environments.")
        
    if args.broken == 0:
        # One-house setting with variety modifications.
        if args.lin_src == 1:
            source_env = gym.make("Smart_Grids_Linear-v0")
            if args.variety_name == 's':
                target_env = get_new_soc_env(args.degree, args.env_name)
            elif args.variety_name == 'c':
                target_env = get_new_charge_env(args.degree, args.env_name)
            elif args.variety_name == 'd':
                target_env = get_new_discharge_env(args.degree, args.env_name)
            elif args.variety_name.startswith('v'):
                target_env = get_new_all_eff_env(args.degree, args.env_name)
            elif args.variety_name == 'lc':
                target_env = get_new_limited_capacity_env(3, 2.0, args.env_name)
            elif args.variety_name == 'lp':
                target_env = get_new_limited_plim_env(3, 2.0, args.env_name)
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
        else:
            target_env = gym.make("Smart_Grids_Linear-v0")
            if args.variety_name == 's':
                source_env = get_new_soc_env(args.degree, args.env_name)
            elif args.variety_name == 'c':
                source_env = get_new_charge_env(args.degree, args.env_name)
            elif args.variety_name == 'd':
                source_env = get_new_discharge_env(args.degree, args.env_name)
            elif args.variety_name.startswith('v'):
                source_env = get_new_all_eff_env(args.degree, args.env_name)
            elif args.variety_name == 'lc':
                source_env = get_new_limited_capacity_env(3, 2.0, args.env_name)
            elif args.variety_name == 'lp':
                source_env = get_new_limited_plim_env(3, 2.0, args.env_name)
            else:
                raise ValueError("Unknown variety name: " + args.variety_name)
    else:
        # Two-house setting (broken environment).
        if args.break_src == 1:
            source_env = gym.make("Smart_Grids_TwoHouses_Damage-v0")
            target_env = gym.make("Smart_Grids_TwoHouses_Normal-v0")
        else:
            source_env = gym.make("Smart_Grids_TwoHouses_Normal-v0")
            target_env = gym.make("Smart_Grids_TwoHouses_Damage-v0")

    # Get dimensions from the source environment.
    state_dim = source_env.observation_space.shape[0]
    action_dim = source_env.action_space.shape[0]

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
        #"input_dim": [state_dim + action_dim],
        "input_dim": [1 + action_dim],
        "architecture": [
            {"name": "linear1", "size": args.classifier},
            {"name": "linear2", "size": 2}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    sas_config = {
        #"input_dim": [state_dim * 2 + action_dim],
        "input_dim": [2 + action_dim],
        "architecture": [
            {"name": "linear1", "size": args.classifier},
            {"name": "linear2", "size": 2}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }

    running_state = ZFilter((state_dim,), clip=1)
    model = DARC(
        policy_config, value_config, sa_config, sas_config,
        source_env, target_env, "cpu", ent_adj=True,
        n_updates_per_train=args.update, lr=args.lr,
        max_steps=args.max_steps, batch_size=args.bs,
        savefolder=save_model_path, running_mean=running_state ,
        if_normalize=False, delta_r_scale=args.deltar,
        noise_scale=args.noise, warmup_games=args.warmup,
        log_dir=log_dir
    )

    model.train(args.train_steps, deterministic=False)
    model.save_model(save_model_path)

if __name__ == '__main__':
    main()

"""
ex:
python3 train_darc_clean.py --env-name Smart_Grids --broken 1 --break_src 1 --lr 0.0001 --train-steps 4000 ...
"""

"""
Broken == 1: Two house setting: One battery works ideal (lineal or non lineal) and the other does not work (efficiencies = 0)
broken_src == 1: Source is broken and target is sound
broken_src == 0: Source is sound and target is broken

Broken == 0: One house setting:
variety-name: s (soc), c (charge), d (discharge), v1 (all 3 efficiencies are set to some value), v2, v3, ...
degree: float (0,1] 
It means that each efficiency is set to a value: i.e --variety-name s --degree 0.5 means that the soc efficiency is set to 0.5.

linear: soc_{t+1} = soc_t + p_t
non-linear: soc_{t+1} = etas * soc_{t} + etas * p_{ec} * p_t + \\frac{1}{etad} * (1 - p_{ec}) * p_t
"""

"""
first trials using only 3 diferent degrees for each efficiency: end up in 9 different settings.
"""