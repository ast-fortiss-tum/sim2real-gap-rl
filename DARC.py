import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter

# Define Classifier Networks
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize environments
source_env = gym.make("SourceEnv")
target_env = gym.make("TargetEnv")

# Hyperparameters
num_iterations = 1000
batch_size = 64
lr = 1e-3
reward_scale = 1.0

# Initialize classifiers
state_action_dim = source_env.observation_space.shape[0] + source_env.action_space.shape[0]
classifier_sas = Classifier(state_action_dim + source_env.observation_space.shape[0])
classifier_sa = Classifier(state_action_dim)
optimizer_sas = optim.Adam(classifier_sas.parameters(), lr=lr)
optimizer_sa = optim.Adam(classifier_sa.parameters(), lr=lr)

# Replay buffers
source_buffer = []
target_buffer = []

# TensorBoard setup
writer = SummaryWriter(log_dir="runs/DARC")

# Collect initial data
def rollout(env, policy, buffer, n_steps=1000):
    state = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        buffer.append((state, action, next_state, reward, done))
        state = next_state if not done else env.reset()

# Rollout source and target environments
rollout(source_env, None, source_buffer, n_steps=1000)
rollout(target_env, None, target_buffer, n_steps=200)

# Update classifiers
def train_classifier(buffer, classifier, optimizer):
    states, actions, next_states, labels = [], [], [], []
    for state, action, next_state, _, _ in buffer:
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        labels.append(0 if buffer == source_buffer else 1)

    inputs = torch.tensor(np.hstack((states, actions, next_states)), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    optimizer.zero_grad()
    outputs = classifier(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# Reinforcement learning with modified rewards
policy = SAC("MlpPolicy", source_env, verbose=1)

for iteration in range(num_iterations):
    # Periodically update classifiers
    if iteration % 10 == 0:
        loss_sas = train_classifier(source_buffer, classifier_sas, optimizer_sas)
        loss_sa = train_classifier(source_buffer, classifier_sa, optimizer_sa)
        writer.add_scalar("Loss/Classifier_SAS", loss_sas, iteration)
        writer.add_scalar("Loss/Classifier_SA", loss_sa, iteration)

    # Modify rewards using classifiers
    modified_rewards = []
    for state, action, next_state, reward, done in source_buffer:
        sa_input = torch.tensor(np.hstack((state, action)), dtype=torch.float32)
        sas_input = torch.tensor(np.hstack((state, action, next_state)), dtype=torch.float32)

        log_prob_target_sas = torch.log(classifier_sas(sas_input.unsqueeze(0))[0, 1])
        log_prob_target_sa = torch.log(classifier_sa(sa_input.unsqueeze(0))[0, 1])

        delta_r = (log_prob_target_sas - log_prob_target_sa).item()
        modified_reward = reward + reward_scale * delta_r
        modified_rewards.append(modified_reward)

    # Train policy on source domain with modified rewards
    for state, action, next_state, reward, done in source_buffer:
        source_env.step(action)  # Apply training using SAC

    # Log SAC rewards to TensorBoard
    mean_reward = np.mean(modified_rewards)
    writer.add_scalar("Reward/Modified", mean_reward, iteration)

    print(f"Iteration {iteration} completed with mean modified reward: {mean_reward}")

# Save the final model
policy.save("DARC_policy")
print("Training complete and model saved.")

writer.close()

