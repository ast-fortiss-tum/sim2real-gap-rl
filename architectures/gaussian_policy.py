import torch
from torch import nn, distributions
from architectures.utils import Model


# n_states -> n_actions (none)
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


# n_states -> split: n_actions * 2 (none)
# class ContGaussianPolicy(nn.Module):
#     def __init__(self, model_config, action_range, transform = False):
#         super(ContGaussianPolicy, self).__init__()
#         self.model = Model(model_config)

#         action_low, action_high = action_range
#         device = 'cpu'
#         if torch.cuda.is_available():
#             device = 'cuda'

#         self.action_low = torch.tensor(action_low).to(device)
#         self.action_high = torch.tensor(action_high).to(device)

#         self.action_scale = torch.as_tensor((action_high - action_low) / 2, dtype=torch.float32)
#         self.action_bias = torch.as_tensor((action_high + action_low) / 2, dtype=torch.float32)

#     def forward(self, states):
#         mu, log_std = self.model(states)
#         log_std = torch.clamp(log_std, min=-20, max=2)
#         return mu, log_std

#     # def sample(self, states):
#     #     mus, log_stds = self.forward(states)
#     #     stds = torch.exp(log_stds)

#     #     normal_dists = distributions.Normal(mus, stds)
#     #     outputs = normal_dists.rsample()
#     #     tanh_outputs = torch.tanh(outputs)
#     #     actions = self.action_scale * tanh_outputs + self.action_bias
#     #     mean_actions = self.action_scale * torch.tanh(mus) + self.action_bias

#     #     log_probs = normal_dists.log_prob(outputs)
#     #     # https://arxiv.org/pdf/1801.01290.pdf appendix C
#     #     log_probs -= torch.log(
#     #         self.action_scale * (torch.ones_like(tanh_outputs, requires_grad=False) - tanh_outputs.pow(2)) + 1e-6)
#     #     log_probs = log_probs.sum(1, keepdim=True)

#     #     return actions, log_probs, mean_actions
#     def sample(self, states):
#         mus, log_stds = self.forward(states)
#         stds = torch.exp(log_stds)
#         dist = distributions.Normal(mus, stds)
        
#         # dist = self(states)
#         # Reparameterization trick
#         u = dist.rsample()
#         action = torch.tanh(u)
#         log_prob = dist.log_prob(value=u)
#         # Enforcing action bounds
#         log_prob -= torch.log(1 - action ** 2 + 1e-6)
#         log_prob = log_prob.sum(-1, keepdim=True)
#         return (action * self.action_low).clamp_(self.action_low, self.action_high), log_prob, (mus * self.action_low).clamp_(self.action_low, self.action_high)


#     def to(self, *args, **kwargs):
#         device = args[0]
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         self.model = self.model.to(device)
#         return super(ContGaussianPolicy, self).to(device)

class ContGaussianPolicy(nn.Module):
    def __init__(self, model_config, action_range):
        super(ContGaussianPolicy, self).__init__()
        self.model = Model(model_config)

        action_low, action_high = action_range
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        # self.transform_layer = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(action_dim, action_dim)
        # )

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

    # def sample(self, states):
    #     mus, log_stds = self.forward(states)
    #     stds = torch.exp(log_stds)

    #     normal_dists = distributions.Normal(mus, stds)
    #     outputs = normal_dists.rsample()
    #     tanh_outputs = torch.tanh(outputs)
    #     actions = self.action_scale * tanh_outputs + self.action_bias
    #     mean_actions = self.action_scale * torch.tanh(mus) + self.action_bias

    #     log_probs = normal_dists.log_prob(outputs)
    #     # https://arxiv.org/pdf/1801.01290.pdf appendix C
    #     log_probs -= torch.log(
    #         self.action_scale * (torch.ones_like(tanh_outputs, requires_grad=False) - tanh_outputs.pow(2)) + 1e-6)
    #     log_probs = log_probs.sum(1, keepdim=True)

    #     return actions, log_probs, mean_actions
    def sample(self, states,transform = False):
        mus, log_stds = self.forward(states, transform)
        stds = torch.exp(log_stds)
        dist = distributions.Normal(mus, stds)
        
        # dist = self(states)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_low).clamp_(self.action_low, self.action_high), log_prob, (mus * self.action_low).clamp_(self.action_low, self.action_high)


    def to(self, *args, **kwargs):
        device = args[0]
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.model = self.model.to(device)
        return super(ContGaussianPolicy, self).to(device)
