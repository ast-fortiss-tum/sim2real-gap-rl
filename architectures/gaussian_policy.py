import torch
from torch import nn, distributions
from architectures.utils import Model

"""
This module defines two types of Gaussian policies for reinforcement learning:
1. DiscreteGaussianPolicy: For discrete action spaces, using a softmax layer to compute action probabilities.
2. ContGaussianPolicy: For continuous action spaces, using a Gaussian distribution to sample actions and compute log probabilities.
Both policies inherit from nn.Module and implement methods for forward pass and action sampling.
"""

class DiscreteGaussianPolicy(nn.Module):
    def __init__(self, model_config):
        super(DiscreteGaussianPolicy, self).__init__()
        self.model = Model(model_config)

    def forward(self, states):
        action_probs = torch.softmax(self.model(states), dim=1)
        return action_probs

    def sample(self, states):
        action_probs = torch.softmax(self.forward(states), dim=1)
        action_dists = distributions.Categorical(action_probs)
        rand_actions = action_dists.sample()
        _, actions = torch.max(action_probs, dim=1)
        return rand_actions, action_probs, actions

class ContGaussianPolicy(nn.Module):
    def __init__(self, model_config, action_range):
        super(ContGaussianPolicy, self).__init__()
        self.model = Model(model_config)

        action_low, action_high = action_range
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        self.action_low = torch.tensor(action_low).to(device)
        self.action_high = torch.tensor(action_high).to(device)

        self.action_scale = torch.as_tensor((action_high - action_low) / 2, dtype=torch.float32)
        self.action_bias = torch.as_tensor((action_high + action_low) / 2, dtype=torch.float32)

    def forward(self, states,transform = False):
        mu, log_std = self.model(states)
        if transform:
            mu = self.transform_layer(mu)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def sample(self, states,transform = False):
        mus, log_stds = self.forward(states, transform)
        stds = torch.exp(log_stds)
        dist = distributions.Normal(mus, stds)
        
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_low).clamp_(self.action_low, self.action_high).float(), log_prob, (mus * self.action_low).clamp_(self.action_low, self.action_high).float()

    def to(self, *args, **kwargs):
        device = args[0]
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.model = self.model.to(device)
        return super(ContGaussianPolicy, self).to(device)
