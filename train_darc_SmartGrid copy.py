#import gym
import gymnasium as gym

import datetime
from models.darc import DARC
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
from stable_baselines3 import SAC


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

source_env = SmartGrid_Linear(
    horizon=timedelta(hours=24),
    frequency=timedelta(minutes=60),
    fixed_start="27.11.2016",
    capacity=3,
    data_path="./data/1-LV-rural2--1-sw",
    params_battery={"rho": 0.1, "p_lim": 2.0}
).env

target_env = SmartGrid_Nonlinear(
    horizon=timedelta(hours=24),
    frequency=timedelta(minutes=60),
    fixed_start="27.11.2016",
    capacity=3,
    data_path="./data/1-LV-rural2--1-sw",
    params_battery={"rho": 0.1, "p_lim": 2.0, "etac": 0.6, "etad": 0.7, "etas": 0.8}
).env

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

#running_state = ZFilter((state_dim,), clip=2)
#model = DARC(policy_config, value_config, sa_config, sas_config, source_env, target_env, "cpu", ent_adj=True,\
#             n_updates_per_train=args.update,lr=args.lr,\
#             max_steps=args.max_steps,batch_size=args.bs,\
#             savefolder=save_model_path,running_mean=running_state,if_normalize = args.normalize, delta_r_scale = args.deltar,noise_scale = args.noise, warmup_games = args.warmup)

#model.train(train_steps, deterministic=False)  # posible change to trainer.run() .... implementing DARC as SAC in the commonpower framework to train with safety layer. 
#model.save_model(save_model_path)           # Ignore safety layer for now !!!!!!!!!!!

#ToDo: Add plot to see performance on differen days maybe?? compare against the baseline MPC

model = SAC(ContGaussianPolicy, target_env, verbose=1)
state_dict = torch.load("./saved_weights/_02/100/policy")  #.pth

# If your file was saved as a dictionary containing a key like "state_dict",
# then extract it. Otherwise, assume the file is directly the state dictionary.
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

# Load the state dictionary into the policy network.
model.policy.load_state_dict(state_dict, strict=False)
model_path = "./saved_weights/_02/100/SB3/SAC_myModel.zip"
model.save(model_path)

# we use pydantic classes for configuration of algorithms
alg_config = SB3MetaConfig(
    total_steps=550*24,
    seed=42,
    algorithm=SAC,
    penalty_factor=0.001,
    algorithm_config=SB3PPOConfig(
        n_steps=24,
        learning_rate=0.0008,
        batch_size=12,
    )  # default hyperparameters for PPO
)

agent2 = RLControllerSB3(
    name="pretrained_agent", 
    safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=alg_config.penalty_factor)),
    pretrained_policy_path=model_path
)

sys = target_env.sys 

oc_model_history = ModelHistory([sys])
rl_model_history = ModelHistory([sys])

rl_deployer = DeploymentRunner(
    sys=sys, 
    global_controller=agent2,  
    alg_config=alg_config,
    wrapper=SingleAgentWrapper,
    forecast_horizon=timedelta(hours=24),
    control_horizon=timedelta(hours=24),
    history=rl_model_history,
    seed=42
)

oc_deployer = DeploymentRunner(
    sys=sys, 
    global_controller=OptimalController('global'), 
    forecast_horizon=timedelta(hours=24),
    control_horizon=timedelta(hours=24),
    history=oc_model_history,
    seed=42
)
oc_deployer.run(n_steps=24, fixed_start="27.11.2016")
# we retrieve logs for the system cost
oc_power_import_cost = oc_model_history.get_history_for_element(target_env.m1, name='cost') # cost for buying electricity
oc_dispatch_cost = oc_model_history.get_history_for_element(target_env.n1, name='cost') # cost for operating the components in the household
oc_total_cost = [(oc_power_import_cost[t][0], oc_power_import_cost[t][1] + oc_dispatch_cost[t][1]) for t in range(len(oc_power_import_cost))]
oc_soc = oc_model_history.get_history_for_element(target_env.e1, name="soc") # state of charge

rl_power_import_cost = rl_model_history.get_history_for_element(target_env.m1, name='cost') # cost for buying electricity
rl_dispatch_cost = rl_model_history.get_history_for_element(target_env.n1, name='cost') # cost for operating the components in the household
rl_total_cost = [(rl_power_import_cost[t][0], rl_power_import_cost[t][1] + rl_dispatch_cost[t][1]) for t in range(len(rl_power_import_cost))]
rl_soc = rl_model_history.get_history_for_element(target_env.e1, name="soc") # state of charge of the battery

# plotting the cost of RL agent and optimal controller
plt.plot(range(len(rl_total_cost)), [x[1] for x in rl_total_cost], label="Cost RL")
plt.plot(range(len(oc_total_cost)), [x[1] for x in oc_total_cost], label="Cost optimal control")
plt.xticks(ticks=range(len(rl_power_import_cost)), labels=[x[0] for x in rl_power_import_cost])
plt.xticks(rotation=45)
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title("Comparison of household cost for RL and optimal controller")
plt.tight_layout()
plt.legend()
plt.show()