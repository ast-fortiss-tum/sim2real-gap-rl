import os
import pickle
import numpy as np
import torch
import random
from torch.nn import functional
from torch.optim import Adam

from architectures.gaussian_policy import ContGaussianPolicy
from architectures.value_networks import ContTwinQNet
from architectures.utils import polyak_update
from replay_buffer import ReplayBuffer
from tensor_writer import TensorWriter

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ContSAC:
    def __init__(self, policy_config, value_config, env, device, log_dir="", running_mean=None,
                 noise_scale=0.0, bias=0.0, memory_size=1e5, warmup_games=10, batch_size=64, lr=0.0001, 
                 gamma=0.99, tau=0.003, alpha=0.2, ent_adj=False, target_update_interval=1, 
                 n_games_til_train=1, n_updates_per_train=1, max_steps=200, seed=None,
                 noise_indices=None, use_denoiser=1):
        # Set the global seed if provided for reproducibility.
        if seed is not None:
            set_global_seed(seed)
        
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise_scale = noise_scale  # Parameter for observational noise.
        self.bias = bias

        # Create log directory for TensorWriter.
        path = 'runs/' + log_dir
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = TensorWriter(path)

        self.memory_size = memory_size
        self.warmup_games = warmup_games
        self.memory = ReplayBuffer(memory_size, batch_size)

        self.env = env
        self.action_range = (env.action_space.low, env.action_space.high)
        self.policy = ContGaussianPolicy(policy_config, self.action_range).to(self.device)
        self.policy_opt = Adam(self.policy.parameters(), lr=lr)
        self.running_mean = running_mean

        self.twin_q = ContTwinQNet(value_config).to(self.device)
        self.twin_q_opt = Adam(self.twin_q.parameters(), lr=lr)
        self.target_twin_q = ContTwinQNet(value_config).to(self.device)
        polyak_update(self.twin_q, self.target_twin_q, 1)

        self.tau = tau
        self.n_until_target_update = target_update_interval
        self.n_games_til_train = n_games_til_train
        self.n_updates_per_train = n_updates_per_train
        self.max_steps = max_steps

        self.alpha = alpha
        self.ent_adj = ent_adj
        if ent_adj:
            self.target_entropy = -len(self.action_range)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = Adam([self.log_alpha], lr=lr)

        self.total_train_steps = 0

        # New parameters for noise handling and denoising.
        # If noise_indices is provided (e.g., [100] or [100, 226]), noise is only applied at these indices.
        self.noise_indices = noise_indices

        # Option to use a denoiser.
        self.use_denoiser = use_denoiser
        if self.use_denoiser:
            # Assume the denoiser was trained with input_dim=1.
            from online_denoising_AE import OnlineDenoisingAutoencoder
            self.denoiser = OnlineDenoisingAutoencoder(input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1).to(self.device)
            self.denoiser.load_state_dict(torch.load("best_online_denoising_autoencoder.pth", 
                                                     map_location=self.device, weights_only=True))
            self.denoiser.eval()

    def add_obs_noise(self, obs):
        """
        Adds Gaussian noise to the observation.
        If self.noise_indices is specified, noise is added only to those indices (each index gets a random noise sample plus bias);
        otherwise, noise is added elementwise.
        """
        obs_noisy = np.copy(obs)
        if self.noise_indices is not None:
            for idx in self.noise_indices:
                noise = np.random.normal(0, self.noise_scale) + self.bias
                obs_noisy[idx] = obs_noisy[idx] + noise
        else:
            obs_noisy = obs + np.random.normal(0, self.noise_scale, size=obs.shape)
        return obs_noisy

    def denoise_observation(self, obs, buffers):
        """
        If self.use_denoiser is True, for each index specified in self.noise_indices
        the current noisy measurement is appended to a per-index buffer. The buffer is then fed
        to the online denoiser (which expects a sequence input) and the recovered value is used to replace
        the noisy measurement in the observation.
        
        Args:
            obs (np.ndarray): The current observation.
            buffers (dict): A dictionary mapping each index (int) to a list that accumulates measurements.
        Returns:
            obs_denoised (np.ndarray): Updated observation with denoised values at the specified indices.
        """
        obs_denoised = np.copy(obs)
        for idx in self.noise_indices:
            buffers[idx].append(obs[idx])
            acc = buffers[idx]  # Accumulated list of noisy observations for this index.
            # Create a tensor of shape (1, T, 1) where T=len(acc).
            acc_tensor = torch.tensor(acc, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                denoised_seq, _ = self.denoiser.forward_online(acc_tensor)
            # Use the recovered value of the last time step.
            denoised_val = denoised_seq[0, -1, :].cpu().numpy()[0]
            obs_denoised[idx] = denoised_val
        return obs_denoised

    def get_action(self, state, deterministic=False, transform=False):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            if deterministic:
                _, _, action = self.policy.sample(state_tensor, transform)
            else:
                action, _, _ = self.policy.sample(state_tensor, transform)
            return action.detach().cpu().numpy()[0]

    def train_step(self, states, actions, rewards, next_states, done_masks):
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
            rewards = torch.as_tensor(rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
            done_masks = torch.as_tensor(done_masks[:, np.newaxis], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_states)
            next_q = self.target_twin_q(next_states, next_action)[0]
            v = next_q - self.alpha * next_log_prob
            expected_q = rewards + done_masks * self.gamma * v

        # Q-network update.
        q_val, pred_q1, pred_q2 = self.twin_q(states, actions)
        q_loss = functional.mse_loss(pred_q1, expected_q) + functional.mse_loss(pred_q2, expected_q)

        self.twin_q_opt.zero_grad()
        q_loss.backward()
        self.twin_q_opt.step()

        # Policy update.
        s_action, s_log_prob, _ = self.policy.sample(states)
        policy_loss = self.alpha * s_log_prob - self.twin_q(states, s_action)[0]
        policy_loss = policy_loss.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        if self.ent_adj:
            alpha_loss = -(self.log_alpha * (s_log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()

        if self.total_train_steps % self.n_until_target_update == 0:
            polyak_update(self.twin_q, self.target_twin_q, self.tau)

        return {'Loss/Policy Loss': policy_loss,
                'Loss/Q Loss': q_loss,
                'Stats/Avg Q Val': q_val.mean(),
                'Stats/Avg Q Next Val': next_q.mean(),
                'Stats/Avg Alpha': self.alpha.item() if self.ent_adj else self.alpha}

    def train(self, num_games, deterministic=False):
        self.policy.train()
        self.twin_q.train()
        for i in range(num_games):
            total_reward = 0
            n_steps = 0
            done = False

            # Reset environment and get initial observation.
            state = self.env.reset()[0]
            state = self.running_mean(state)
            if self.noise_scale > 0:
                state = self.add_obs_noise(state)
            # If denoiser is used, initialize a buffer dictionary for each noise index.
            if self.use_denoiser and self.noise_indices is not None:
                denoise_buffers = {idx: [] for idx in self.noise_indices}
                state = self.denoise_observation(state, denoise_buffers)
            else:
                denoise_buffers = None

            while not done:
                if self.total_train_steps <= self.warmup_games:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(state, deterministic)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.running_mean(next_state)
                if self.noise_scale > 0:
                    next_state = self.add_obs_noise(next_state)
                # If denoiser is enabled, update the observation using the accumulated noisy values.
                if denoise_buffers is not None:
                    next_state = self.denoise_observation(next_state, denoise_buffers)

                done_mask = 1.0 if n_steps == self.env._max_episode_steps - 1 else float(not done)
                self.memory.add(state, action, reward, next_state, done_mask)

                n_steps += 1
                total_reward += reward
                state = next_state
                if n_steps > self.max_steps:
                    break

            if i >= self.warmup_games:
                self.writer.add_scalar('Env/Rewards', total_reward, i)
                self.writer.add_scalar('Env/N_Steps', n_steps, i)
                if i % self.n_games_til_train == 0:
                    for _ in range(n_steps * self.n_updates_per_train):
                        self.total_train_steps += 1
                        s, a, r, s_, d = self.memory.sample()
                        train_info = self.train_step(s, a, r, s_, d)
                        self.writer.add_train_step_info(train_info, i)
                    self.writer.write_train_step()

            print("index: {}, steps: {}, total_rewards: {}".format(i, n_steps, total_reward))

    def eval(self, num_games, render=True):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            state = self.env.reset()[0]
            state = self.running_mean(state)
            if self.noise_scale > 0:
                state = self.add_obs_noise(state)
            # Initialize denoising buffers for evaluation if needed.
            if self.use_denoiser and self.noise_indices is not None:
                denoise_buffers = {idx: [] for idx in self.noise_indices}
                state = self.denoise_observation(state, denoise_buffers)
            else:
                denoise_buffers = None

            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state, deterministic=True)
                next_state, reward, done, _ , _ = self.env.step(action)
                # Handle cases where next_state may be a tuple.
                if isinstance(next_state, (list, tuple)):
                    next_state = next_state[0]
                next_state = self.running_mean(next_state)
                if self.noise_scale > 0:
                    next_state = self.add_obs_noise(next_state)
                if denoise_buffers is not None:
                    next_state = self.denoise_observation(next_state, denoise_buffers)
                total_reward += reward
                state = next_state

            self.writer.add_scalar('Eval/Reward', total_reward, i)
            reward_all += total_reward

        avg_reward = reward_all / num_games
        self.writer.add_scalar('Eval/Avg Reward', avg_reward, num_games)
        print("Average Eval Reward:", avg_reward)
        return avg_reward

    def save_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), path + '/policy')
        torch.save(self.twin_q.state_dict(), path + '/twin_q_net')
        pickle.dump(self.running_mean, open(path + '/running_mean', 'wb'))

    def load_model(self, folder_name, device):
        prepath = '/home/cubos98/Desktop/MA/DARAIL'
        path = prepath + '/saved_weights/' + folder_name
        self.policy.load_state_dict(torch.load(path + '/policy', map_location=torch.device(device), weights_only=True))
        self.twin_q.load_state_dict(torch.load(path + '/twin_q_net', map_location=torch.device(device), weights_only=True))
        polyak_update(self.twin_q, self.target_twin_q, 1)
        polyak_update(self.twin_q, self.target_twin_q, 1)
        self.running_mean = pickle.load(open(path + '/running_mean', "rb"))
