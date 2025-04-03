import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import higher  # For differentiable inner-loop updates
import numpy as np

# ---------- SAC Actor Network Definition ---------- #
class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
    
    def sample(self, x):
        # x: shape (batch, obs_dim)
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist

# ---------- SAC Twin Q Network (Simple Version) ---------- #
class TwinQNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(TwinQNet, self).__init__()
        # For simplicity, assume one-hot encoding for discrete actions
        self.n_actions = n_actions
        self.fc1 = nn.Linear(obs_dim + n_actions, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, obs, action):
        # Convert scalar action to one-hot vector.
        action_onehot = F.one_hot(action.long(), num_classes=self.n_actions).float()
        x = torch.cat([obs, action_onehot], dim=1)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q

# ---------- Helper Functions to Collect Episodes ---------- #
def collect_sac_episode(agent, env, max_steps=200):
    """
    Collect an episode in env using the agent's current policy.
    Returns a list of tuples: (state, action, reward, log_prob).
    """
    episode = []
    # Gymnasium reset returns (obs, info)
    obs, _ = env.reset()
    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        action, log_prob, _ = agent.policy.sample(obs_tensor)
        action_item = action.item()
        # Gymnasium step returns (next_obs, reward, terminated, truncated, info)
        next_obs, reward, terminated, truncated, _ = env.step(action_item)
        done = terminated or truncated
        episode.append((obs, action_item, reward, log_prob))
        obs = next_obs
        if done:
            break
    return episode

def compute_sac_actor_loss(agent, episode):
    """
    Compute the SAC actor (policy) loss over an episode.
    Using the loss: L_actor = α * log π(a|s) - Q(s,a)
    averaged over the episode.
    """
    loss = 0
    for (state, action, reward, log_prob) in episode:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        action_tensor = torch.tensor([action], dtype=torch.float32).to(agent.device)
        q_val = agent.twin_q(state_tensor, action_tensor)
        loss += agent.alpha * log_prob - q_val.squeeze()  # scalar loss per step
    return loss / len(episode)

# ---------- MAML Fine-Tuning for SAC Policy ---------- #
def maml_finetuning_sac(agent, env, inner_lr=0.01, meta_lr=0.001, 
                        num_inner_updates=1, num_train_episodes=5, 
                        num_val_episodes=5, num_meta_iterations=50):
    """
    Fine-tune agent.policy (the SAC actor) on env using MAML.
    We assume agent is a SAC agent pre-trained on env1.
    The twin Q network is used in the loss computation but held fixed.
    """
    meta_optimizer = optim.Adam(agent.policy.parameters(), lr=meta_lr)
    
    for meta_iter in range(num_meta_iterations):
        # Create an inner-loop optimizer for the policy.
        inner_optimizer = optim.SGD(agent.policy.parameters(), lr=inner_lr)
        # Use higher to create a differentiable copy of the policy.
        with higher.innerloop_ctx(agent.policy, inner_optimizer, copy_initial_weights=True) as (fpolicy, diffopt):
            # --- Inner Loop: Fast Adaptation ---
            inner_loss = 0
            for _ in range(num_train_episodes):
                episode = collect_sac_episode(agent, env)
                loss = compute_sac_actor_loss(agent, episode)
                inner_loss += loss
            inner_loss /= num_train_episodes
            # Perform inner-loop updates.
            for _ in range(num_inner_updates):
                diffopt.step(inner_loss)
            
            # --- Outer Loop: Meta Update ---
            meta_loss = 0
            for _ in range(num_val_episodes):
                episode = collect_sac_episode(agent, env)
                loss = compute_sac_actor_loss(agent, episode)
                meta_loss += loss
            meta_loss /= num_val_episodes
            
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            
            print(f"Meta Iteration {meta_iter+1}/{num_meta_iterations}, Meta Loss: {meta_loss.item():.4f}")

# ---------- SAC Agent Class Incorporating Policy and Twin Q ---------- #
class SACAgent:
    def __init__(self, env, device):
        self.device = device
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.policy = Policy(obs_dim, n_actions).to(device)
        self.twin_q = TwinQNet(obs_dim, n_actions).to(device)
        # For simplicity, we set a fixed entropy regularization parameter.
        self.alpha = 0.2
        # (In a full SAC agent, you'd include replay buffers, target networks, etc.)
    
    def get_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _ = self.policy.sample(state_tensor)
        return action.item()

# ---------- Main Script: Transfer from env1 to env2 with MAML Fine-Tuning ---------- #
if __name__ == '__main__':
    # Create two Gymnasium environments: env1 (source) and env2 (target)
    env1 = gym.make('CartPole-v1')
    env2 = gym.make('CartPole-v1')
    # For demonstration, we assume env2 has different dynamics.
    # In practice, you might modify some environment parameters to simulate a shift.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Assume we have pre-trained the SAC agent on env1.
    agent = SACAgent(env1, device)
    # (You could load pre-trained weights here.)
    
    print("Starting MAML fine-tuning on env2...")
    maml_finetuning_sac(agent, env2, inner_lr=0.01, meta_lr=0.001, 
                        num_inner_updates=1, num_train_episodes=5, 
                        num_val_episodes=5, num_meta_iterations=50)
    
    # Test the adapted policy on env2.
    test_rewards = []
    for _ in range(10):
        episode = collect_sac_episode(agent, env2)
        total_reward = sum([reward for (_, _, reward, _) in episode])
        test_rewards.append(total_reward)
    print("Average Reward on env2 after MAML adaptation:", np.mean(test_rewards))
