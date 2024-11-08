import gym
# import gymnasium as gym

from models.darc import DARC
from models.sac import ContSAC
from environments.broken_joint import BrokenJointEnv,BrokenJointEnv2
from utils import *
from envs import *
from datetime import datetime
from zfilter import *


parser = argparse.ArgumentParser()

parser.add_argument('--save-model', type=str, default="",
                    help='name of Mujoco environement')
parser.add_argument('--train-steps', type=int, default=5000,
                    help='name of Mujoco environement')
parser.add_argument('--max-steps', type=int, default=1000,
                    help='name of Mujoco environement')
parser.add_argument('--save-file-name', type=str, default='',
                    help='name of Mujoco environement')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='name of Mujoco environement')


# env 
parser.add_argument('--env-name', type=str, default="Reacher-v2",
                    help='name of Mujoco environement')

# broken
parser.add_argument('--broken', type=int, default=1,
                    help='whether broken env')
parser.add_argument('--break_src', type=int, default=1,
                    help='whether break src env or target env')
parser.add_argument('--break_joint', type=int, default=0,
                    help='break which joint')

parser.add_argument('--normalize', type=int, default=1,
                    help='break which joint')
parser.add_argument('--broken-p', type=float, default=1.0,
                    help='name of Mujoco environement')


# variety
parser.add_argument('--variety-name', type=str, default="d",
                    help='name of Mujoco environement')
parser.add_argument('--variety-degree', type=float, default=0.5,
                    help='name of Mujoco environement')


args = parser.parse_args()


env_name = args.env_name
variety_name = args.variety_name
degree = args.variety_degree
save_model_path = args.save_model
train_steps = args.train_steps

currentDateAndTime = datetime.now()
date = currentDateAndTime.strftime("%Y:%M:%D").split(':')[-1]
save_model_path += args.save_file_name
save_model_path += '_'
save_model_path += date
save_model_path += '_'

save_model_path += str(args.lr)
save_model_path += '_'

if args.broken_p != 1:
    save_model_path += str(args.broken_p)
    save_model_path += '_'

if args.broken == 0:
    save_model_path += variety_name
    save_model_path += '_'
    save_model_path += str(degree)
    save_model_path += '_'
    save_model_path += str(env_name)
    source_env = get_source_env(env_name)
    if variety_name == 'g':
        target_env = get_new_gravity_env(degree,env_name)
    elif variety_name == 'd':
        target_env = get_new_density_env(degree, env_name)
    elif variety_name == 'f':
        target_env = get_new_friction_env(degree, env_name)
else:
    save_model_path += str(args.break_src) 
    save_model_path += '_' 
    save_model_path += str(args.break_joint)
    save_model_path += '_'
    save_model_path += str(env_name)
    if args.break_src == 1:
        source_env =  gym.make(env_name)
        target_env = gym.make(env_name)
    else:
        source_env = gym.make(env_name)
        target_env = BrokenJointEnv(gym.make(env_name), [args.break_joint],args.broken_p)   
# env._max_episode_steps = 3000
state_dim = source_env.observation_space.shape[0]
action_dim = source_env.action_space.shape[0]

policy_config = {
    "input_dim": [state_dim],
    "architecture": [{"name": "linear1", "size": 256},
                     {"name": "linear2", "size": 256},
                    #  {"name": "linear3", "size": 128},
                     {"name": "split1", "sizes": [action_dim, action_dim]}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
value_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": 256},
                     {"name": "linear2", "size": 256},
                    #  {"name": "linear3", "size": 128},
                     {"name": "linear2", "size": 1}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
sa_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": 64},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": "none"
}
sas_config = {
    "input_dim": [state_dim * 2 + action_dim],
    "architecture": [{"name": "linear1", "size": 64},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": "none"
}

running_state = ZFilter((state_dim,), clip=5)
model = DARC(policy_config, value_config, sa_config, sas_config, source_env, target_env, "cuda", ent_adj=True,\
             n_updates_per_train=1,lr=args.lr,\
             max_steps=args.max_steps,batch_size=256,\
             savefolder=save_model_path,running_mean=running_state,if_normalize = args.normalize)
# model = ContSAC(policy_config, value_config, source_env, "cpu", ent_adj=True,
#              n_updates_per_train=1,lr=3e-4,max_steps=200,batch_size=256)
# model.train(3500) 
# model.load_model("Ant-v2-DARC2-400_0123", "cuda")
model.train(train_steps, deterministic=False)
model.save_model(save_model_path)
# model.eval(100)
