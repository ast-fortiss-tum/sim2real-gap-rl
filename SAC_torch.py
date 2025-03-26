import torch
import torch.nn as nn
from torch import distributions
from gym import spaces
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import Schedule
import copy
from architectures.utils import Model

#########################
# Dummy Model (Replace with your actual model)
#########################
    



#########################
# Custom Actor: ContGaussianPolicy
#########################
class ContGaussianPolicy(nn.Module):
    def __init__(self, model_config, action_range):
        super(ContGaussianPolicy, self).__init__()
        self.model = Model(model_config)
        action_low, action_high = action_range
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
        self.action_scale = torch.as_tensor((action_high - action_low) / 2, dtype=torch.float32, device=device)
        self.action_bias = torch.as_tensor((action_high + action_low) / 2, dtype=torch.float32, device=device)

    def forward(self, states, transform=False, use_sde=False, **kwargs):
        # Forward pass: get mu and log_std from the model
        mu, log_std = self.model(states)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def sample(self, states, transform=False):
        mu, log_std = self.forward(states, transform=transform)
        stds = torch.exp(log_std)
        dist = distributions.Normal(mu, stds)
        u = dist.rsample()  # reparameterization trick
        action = torch.tanh(u)
        log_prob = dist.log_prob(u)
        # Correction for Tanh squashing
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        # Scale the action to the desired range
        scaled_action = (action * self.action_scale + self.action_bias).clamp(self.action_low, self.action_high)
        return scaled_action, log_prob, mu

#########################
# Helper: Build a Custom Critic
#########################
def build_custom_critic(observation_space: spaces.Box, action_space: spaces.Box, net_arch: list):
    # The critic takes the concatenated observation and action as input.
    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]
    input_dim = obs_dim + act_dim
    # Create an MLP with the desired hidden layers and a single output (Q-value)
    mlp = create_mlp(input_dim, 1, net_arch)
    return nn.Sequential(*mlp)

#########################
# Custom SAC Policy
#########################
class CustomContGaussianPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        model_config: dict,
        action_range: tuple,
        **kwargs
    ):
        self.model_config = model_config
        self.action_range = action_range
        super(CustomContGaussianPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        # Build the custom actor network
        self.actor = ContGaussianPolicy(self.model_config, self.action_range)
        #model_path = "/home/cubos98/Desktop/MA/DARAIL/saved_weights/_02/SmartGrids/25_0.0001_Smart_Grids/100/policy.pth"
        #state_dict = torch.load(model_path, weights_only=True)  #.pth

        # If your file was saved as a dictionary containing a key like "state_dict",
        # then extract it. Otherwise, assume the file is directly the state dictionary.
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Load the state dictionary into the policy network.
        model.policy.load_state_dict(state_dict, strict=False)
        model.policy.eval()

        self.actor = model.policy
        
        # Build the critic network with a custom architecture, e.g., [256, 256]
        net_arch = [256, 256]
        self.critic = build_custom_critic(self.observation_space, self.action_space, net_arch)
        
        # Set up optimizers for actor and critic
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        
        # Create a target critic network by deep-copying the critic
        self.critic_target = copy.deepcopy(self.critic)
        self._update_target(1.0)

    def _update_target(self, tau: float) -> None:
        """Custom polyak update: target_param = tau * param + (1 - tau) * target_param."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def forward(self, obs, deterministic: bool = False, use_sde: bool = False, **kwargs):
        # Call the actor's forward
        return self.actor(obs, transform=False, use_sde=use_sde, **kwargs)

    def _predict(self, obs, deterministic: bool = False):
        # Use the actor's sample method to get an action.
        action, _, _ = self.actor.sample(obs, transform=False)
        return action

#########################
# Example Usage
#########################
if __name__ == "__main__":

    import numpy as np
    from stable_baselines3 import SAC

    # Create a sample continuous environment
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

    model = DARC(policy_config, value_config, sa_config, sas_config, source_env, target_env, "cpu", ent_adj=True,\
                n_updates_per_train=args.update,lr=args.lr,\
                max_steps=args.max_steps,batch_size=args.bs,\
                savefolder=save_model_path,running_mean=running_state,if_normalize = args.normalize, delta_r_scale = args.deltar,noise_scale = args.noise, warmup_games = args.warmup)

    env = model.env
    observation_space = env.observation_space
    action_space = env.action_space

    # Define model configuration for the custom actor.
    model_config = policy_config
    # Action range as tuple: (low, high)
    action_range = (action_space.low, action_space.high)

    # Define a learning rate schedule as a lambda that returns a constant.
    lr_schedule = lambda _: 0.0003

    # Create the SAC agent with our custom policy.
    agent = SAC(
        policy=CustomContGaussianPolicy,
        #policy = model.policy,
        env=env,
        verbose=1,
        policy_kwargs=dict(
            model_config=model_config,
            action_range=action_range,
        ),
        learning_rate=lr_schedule,
    )

    # Test a forward pass:
    obs , _ = env.reset()
    print("Observation:", obs)
    time.sleep(10)
    obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32)
    action = agent.policy._predict(obs_tensor)
    print("Predicted action:", action.detach().cpu().numpy())
