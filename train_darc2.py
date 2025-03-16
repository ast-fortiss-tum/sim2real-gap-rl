#!/usr/bin/env python3
import gymnasium as gym
import datetime
from models.darc import DARC
from models.sac import ContSAC
from environments.broken_joint import BrokenJointEnv, BrokenJointEnv2
from utils import *  # Ensure this imports get_source_env, get_new_gravity_env, etc.
from datetime import datetime
import argparse
from gymnasium.envs.registration import register

# -----------------------------------------------------------------------------
# Registration of the Smart Grid environments with Gymnasium.
# -----------------------------------------------------------------------------

register(
    id='Smart_Grids_Linear-v0',
    entry_point='environments.smartgrid_env:make_smartgrid_linear',
    max_episode_steps=24,
)

register(
    id='Smart_Grids_Nonlinear-v0',
    entry_point='environments.smartgrid_env:make_smartgrid_nonlinear',
    max_episode_steps=24,
)

def parse_args():
    parser = argparse.ArgumentParser()
    # Saving and training hyperparameters
    parser.add_argument('--save-model', type=str, default="",
                        help='Base path for saving the model')
    parser.add_argument('--train-steps', type=int, default=4000,
                        help='Number of training steps')
    parser.add_argument('--max-steps', type=int, default=1000,
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

    # Environment settings
    parser.add_argument('--env-name', type=str, default="Reacher-v2",
                        help='Name of environment to use')

    # Broken environment settings
    parser.add_argument('--broken', type=int, default=1,
                        help='Whether to use a broken environment (1) or not (0)')
    parser.add_argument('--break_src', type=int, default=1,
                        help='For broken experiments, which environment to break: 1 for source, 0 for target')
    parser.add_argument('--break_joint', type=int, default=0,
                        help='Which joint to break')
    parser.add_argument('--normalize', type=int, default=1,
                        help='Whether to normalize actions (1) or not (0)')
    parser.add_argument('--broken-p', type=float, default=1.0,
                        help='Broken parameter (e.g., probability or factor)')

    # Variety parameters (used when broken==0)
    parser.add_argument('--variety-name', type=str, default="g",
                        help='Name of variety (e.g., gravity, density, friction)')
    parser.add_argument('--degree', type=float, default=0.5,
                        help='Degree parameter for variety')
    parser.add_argument('--noise', type=float, default=0.2,
                        help='Noise scale')

    # Network architecture parameters
    parser.add_argument('--policynet', type=int, default=256,
                        help='Size of policy network layers')
    parser.add_argument('--classifier', type=int, default=32,
                        help='Size of classifier network layers')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Number of warmup steps')

    return parser.parse_args()

def construct_save_model_path(args):
    currentDateAndTime = datetime.now()
    date_str = currentDateAndTime.strftime("%Y%m%d_%H%M%S")
    # Build a path that includes base, provided file name, date, lr, and other key parameters.
    save_model_path = args.save_model + args.save_file_name + '_' + date_str + '_' + str(args.lr) + '_'
    if args.broken_p != 1:
        save_model_path += str(args.broken_p) + '_'
    if args.broken == 0:
        save_model_path += args.variety_name + '_' + str(args.degree) + '_' + args.env_name
    else:
        save_model_path += str(args.break_src) + '_' + str(args.break_joint) + '_' + args.env_name
    return save_model_path

def main():
    args = parse_args()
    save_model_path = construct_save_model_path(args)
    print("Saving model to:", save_model_path)

    # Environment creation:
    if args.broken == 0:
        # Variety branch: use utility functions to create source and target environments.
        source_env = get_source_env(args.env_name)
        if args.variety_name == 'g':
            target_env = get_new_gravity_env(args.degree, args.env_name)
        elif args.variety_name == 'd':
            target_env = get_new_density_env(args.degree, args.env_name)
        elif args.variety_name == 'f':
            target_env = get_new_friction_env(args.degree, args.env_name)
        else:
            raise ValueError("Unknown variety name: " + args.variety_name)
    else:
        # Broken branch:
        if args.env_name == "Smart_Grids":
            # Special case for Smart Grid: use registered environments.
            if args.break_src == 1:
                # Source = linear, Target = nonlinear.
                source_env = gym.make("Smart_Grids_Linear-v0")
                target_env = gym.make("Smart_Grids_Nonlinear-v0")
            else:
                # break_src == 0: Reverse roles.
                source_env = gym.make("Smart_Grids_Nonlinear-v0")
                target_env = gym.make("Smart_Grids_Linear-v0")
        else:
            # For non–Smart Grid environments.
            if args.break_src == 1:
                source_env = BrokenJointEnv(gym.make(args.env_name, render_mode="human"), [args.break_joint], args.broken_p)
                target_env = gym.make(args.env_name)
            else:
                source_env = gym.make(args.env_name)
                target_env = BrokenJointEnv(gym.make(args.env_name), [args.break_joint], args.broken_p)

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
        "input_dim": [state_dim + action_dim],
        "architecture": [
            {"name": "linear1", "size": args.classifier},
            {"name": "linear2", "size": 2}
        ],
        "hidden_activation": "relu",
        "output_activation": "none"
    }
    sas_config = {
        "input_dim": [state_dim * 2 + action_dim],
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
        savefolder=save_model_path, running_mean=None,
        if_normalize=args.normalize, delta_r_scale=args.deltar,
        noise_scale=args.noise, warmup_games=args.warmup
    )

    model.train(args.train_steps, deterministic=False)
    model.save_model(save_model_path)

if __name__ == '__main__':
    main()



"""
python3 train_darc.py --env-name Smart_Grids --broken 1 --break_src 1 --lr 0.0001 --train-steps 4000 ...
"""