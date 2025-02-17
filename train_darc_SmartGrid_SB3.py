#import gym
import gymnasium as gym

import datetime
#from models.darc import DARC
#from gym.wrappers import Monitor
from models.darc_from_sac2 import DARC_SB3
#from models.darc_SB3 import DARC
from models.sac import ContSAC
from environments.SmartGrid import *
from architectures.gaussian_policy import ContGaussianPolicy
from utils import *
#from envs import *
from datetime import datetime
from utils import *
import argparse
from commonpower.control.runners import SingleAgentTrainer, DeploymentRunner
from stable_baselines3 import SAC, PPO


parser = argparse.ArgumentParser()

parser.add_argument('--save-model', type=str, default="",
                    help='name of Mujoco environement')
parser.add_argument('--train-steps', type=int, default=100,
                    help='name of Mujoco environement')
parser.add_argument('--max-steps', type=int, default=24,
                    help='name of Mujoco environement')
parser.add_argument('--save_file_name', type=str, default='Smart_Grids',
                    help='name of Mujoco environement')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='name of Mujoco environement')
parser.add_argument('--bs', type=int, default=256,
                    help='name of Mujoco environement')
parser.add_argument('--update', type=int, default=1,
                    help='name of Mujoco environement')
parser.add_argument('--deltar', type=float, default=1,
                    help='name of Mujoco environement')

# env 
parser.add_argument('--env-name', type=str, default="Smart_Grids",
                    help='name of Mujoco environement')

parser.add_argument('--normalize', type=int, default=1,
                    help='break which joint')

parser.add_argument('--noise', type=float, default=0.0,
                    help='name of Mujoco environement')

parser.add_argument('--policynet', type=int, default=256,
                    help='break which joint')
parser.add_argument('--classifier', type=int, default=32,
                    help='break which joint')

parser.add_argument('--warmup', type=int, default=24,
                    help='break which joint')

args = parser.parse_args()

env_name = args.env_name
save_model_path = args.save_model
train_steps = args.train_steps

currentDateAndTime = datetime.datetime.now()
date = currentDateAndTime.strftime("%Y:%M:%D").split(':')[-1]
save_model_path += args.save_file_name
save_model_path += '_'
save_model_path += date
save_model_path += '_'
save_model_path += str(args.lr)
save_model_path += '_'

save_model_path += str(env_name)

source_grid = SmartGrid_Linear(
    rl = True,
    horizon=timedelta(hours=24),
    frequency=timedelta(minutes=60),
    fixed_start="27.11.2016",
    capacity=3,
    data_path="./data/1-LV-rural2--1-sw",
    params_battery={"rho": 0.1, "p_lim": 2.0}
)

target_grid = SmartGrid_Nonlinear(
    rl = True,
    horizon=timedelta(hours=24),
    frequency=timedelta(minutes=60),
    fixed_start="27.11.2016",
    capacity=3,
    data_path="./data/1-LV-rural2--1-sw",
    params_battery={"rho": 0.1, "p_lim": 2.0, "etac": 0.6, "etad": 0.7, "etas": 0.8}
)

source_env = source_grid.env
target_env = target_grid.env

source_env._max_episode_steps = 24
target_env._max_episode_steps = 24

state_dim = source_env.observation_space.shape[0]
action_dim = source_env.action_space.shape[0]

policy_config = {
    "input_dim": [state_dim],
    "architecture": [{"name": "linear1", "size": args.policynet},
                     {"name": "linear2", "size": args.policynet},
                    #  {"name": "linear3", "size": 128},
                     {"name": "split1", "sizes": [action_dim, action_dim]}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
value_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": args.policynet},
                     {"name": "linear2", "size": args.policynet},
                    #  {"name": "linear3", "size": 128},
                     {"name": "linear2", "size": 1}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
sa_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": args.classifier},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
sas_config = {
    "input_dim": [state_dim * 2 + action_dim],
    "architecture": [{"name": "linear1", "size": args.classifier},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": "none"
}

running_state = ZFilter((state_dim,), clip=2)

#model = DARC_SB3(policy_config, value_config, sa_config, sas_config, source_env, target_env, "cpu", ent_adj=True,\
#             n_updates_per_train=args.update,lr=args.lr,\
#             max_steps=args.max_steps,batch_size=args.bs,\
#             savefolder=save_model_path,running_mean=running_state,if_normalize = args.normalize, delta_r_scale = args.deltar,noise_scale = args.noise, warmup_games = args.warmup)
print(source_env.action_space)
model = DARC_SB3(
    policy="MlpPolicy",           # Use your custom policy if registered, e.g. "CustomGaussianPolicy"
    env=source_env,               # Source environment
    target_env=target_env,        # Target environment for domain adaptation
    sa_config=sa_config,          # Configuration for the SA classifier
    sas_config=sas_config,        # Configuration for the SAS classifier
    delta_r_scale=1.0,            # Scaling factor for reward correction
    s_t_ratio=10,                 # How often (in episodes) to collect a target rollout
    noise_scale=0.1,              # Scale of noise added to classifier inputs
    learning_rate=3e-4,           # SAC learning rate
    buffer_size=10000,           # Replay buffer size
    batch_size=12,                # Batch size for updates
    gradient_steps=1,             # Number of gradient steps per rollout collection
    verbose=1                     # Verbosity level for logging
)

model.learn(total_timesteps = 550*24, deterministic=False)
model.save(save_model_path)
