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

# ---------------- EM & Kalman Helper Functions ---------------- #

def EM_Kalman_seq(obs_seq, act_seq, Q_init, R_init, num_iters=10):
    """
    A simplified EM algorithm for a sequence of observations and actions.
    We assume a linear model: 
        x_{t+1} = x_t + u_t + w_t   with w ~ N(0,Q)
        z_t = x_t + v_t            with v ~ N(0,R)
    Here, A, B, H are identity matrices.
    
    Parameters:
      obs_seq: list of observations (each a 1D numpy array).
      act_seq: list of control inputs (actions) used for transitions.
      Q_init:  initial guess for process noise covariance (d x d array).
      R_init:  initial guess for measurement noise covariance (d x d array).
      num_iters: number of EM iterations.
      
    Returns:
      Q_new, R_new: updated noise covariances.
    """
    T = len(obs_seq)
    d = obs_seq[0].shape[0]
    # Set A, B, H as identity matrices.
    A = np.eye(d)
    B = np.eye(d)
    H = np.eye(d)
    x0 = obs_seq[0].reshape(d, 1)
    P0 = np.eye(d)
    Q = Q_init.copy()
    R = R_init.copy()

    # For simplicity, we perform only a forward pass (filter) instead of full smoothing.
    x_filt = [None] * T
    P_filt = [None] * T
    x_pred = [None] * T
    P_pred = [None] * T
    x_filt[0] = x0
    P_filt[0] = P0

    for t in range(1, T):
        # Prediction: x_pred = A*x_{t-1} + B*u_{t-1}
        x_pred[t] = x_filt[t-1] + act_seq[t-1].reshape(d, 1)
        P_pred[t] = P_filt[t-1] + Q
        # Update:
        y = obs_seq[t].reshape(d, 1) - x_pred[t]
        S = P_pred[t] + R
        K = P_pred[t] @ np.linalg.inv(S)
        x_filt[t] = x_pred[t] + K @ y
        P_filt[t] = (np.eye(d) - K) @ P_pred[t]

    # M-Step: Update Q and R based on filtered estimates.
    Q_new = np.zeros((d, d))
    for t in range(1, T):
        diff = x_filt[t] - (x_filt[t-1] + act_seq[t-1].reshape(d, 1))
        Q_new += diff @ diff.T  # (ignoring the covariance terms for simplicity)
    Q_new /= (T - 1)

    R_new = np.zeros((d, d))
    for t in range(T):
        resid = obs_seq[t].reshape(d, 1) - x_filt[t]
        R_new += resid @ resid.T + P_filt[t]
    R_new /= T

    return Q_new, R_new

# ---------------- Modified SAC Agent with Kalman Filter ---------------- #

class ContSAC_kalman:
    def __init__(self, policy_config, value_config, env, device, log_dir="", running_mean=None,
                 noise_scale=0.0, memory_size=1e5, warmup_games=10, batch_size=64, lr=0.0001, gamma=0.99, 
                 tau=0.003, alpha=0.2, ent_adj=False, target_update_interval=1, n_games_til_train=1, 
                 n_updates_per_train=1, max_steps=200, seed=None):
        # Set the global seed if provided for reproducibility
        if seed is not None:
            set_global_seed(seed)
        
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise_scale = noise_scale  # observational noise level

        path = 'runs/' + log_dir
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = TensorWriter(path)

        self.memory_size = memory_size
        self.warmup_games = warmup_games
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)

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
        self.gamma = gamma
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

        # ----- Initialize Kalman filter variables for denoising observations -----
        # For this toy example, we assume the state dimension equals the observation dimension.
        # We use a very simple model: x_{t+1} = x_t + u_t + w_t, z_t = x_t + v_t.
        self.kalman_initialized = False
        self.kalman_x = None  # current filtered state estimate (as column vector)
        self.kalman_P = None  # current state covariance estimate
        self.kalman_Q = None  # process noise covariance (to be learned)
        self.kalman_R = None  # measurement noise covariance (to be learned)
        # We'll set initial guesses when starting an episode.
        self.kalman_init_P_scale = 1.0

        # To run EM, we store the episode's filtered observations and controls.
        self.episode_obs = []     # list of 1D numpy arrays (filtered state estimates)
        self.episode_actions = [] # list of actions (as numpy arrays)

    def add_obs_noise(self, obs):
        """Adds Gaussian noise to the observation."""
        return obs + np.random.normal(0, self.noise_scale, size=obs.shape)

    def kalman_update(self, obs, control):
        """
        Perform a single-step Kalman update.
        Model assumptions:
          x_pred = x_prev + control   (since A=I, B=I)
          P_pred = P_prev + Q
          Innovation: y = obs - x_pred
          Kalman Gain: K = P_pred * inv(P_pred + R)
          Updated state: x_new = x_pred + K*y
          Updated covariance: P_new = (I - K)*P_pred
        Inputs:
          obs: current noisy observation (1D numpy array)
          control: control input (action) used at the previous step (1D numpy array)
        """
        # Ensure obs and control are column vectors.
        obs = obs.reshape(-1, 1)
        if control is None:
            control = np.zeros_like(obs)
        else:
            control = control.reshape(-1, 1)
        d = obs.shape[0]
        I = np.eye(d)
        # Prediction step:
        x_pred = self.kalman_x + control
        P_pred = self.kalman_P + self.kalman_Q
        # Update step:
        S = P_pred + self.kalman_R
        K = P_pred @ np.linalg.inv(S)
        y = obs - x_pred
        x_new = x_pred + K @ y
        P_new = (I - K) @ P_pred
        # Update the filter’s internal state.
        self.kalman_x = x_new
        self.kalman_P = P_new
        return x_new.flatten()  # return as 1D array

    def update_kalman_params_em(self):
        """
        Run the EM algorithm on the episode’s stored data (filtered observations and actions)
        to update the process and measurement noise covariances (Q and R).
        """
        if len(self.episode_obs) < 2:
            return  # Not enough data to update.
        Q_new, R_new = EM_Kalman_seq(self.episode_obs, self.episode_actions, self.kalman_Q, self.kalman_R, num_iters=10)
        self.kalman_Q = Q_new
        self.kalman_R = R_new
        print("Updated Kalman Q:\n", self.kalman_Q)
        print("Updated Kalman R:\n", self.kalman_R)

    def get_action(self, state, deterministic=False, transform=False):
        with torch.no_grad():
            state = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            if deterministic:
                _, _, action = self.policy.sample(state, transform)
            else:
                action, _, _ = self.policy.sample(state, transform)
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

        # Q backpropagation
        q_val, pred_q1, pred_q2 = self.twin_q(states, actions)
        q_loss = functional.mse_loss(pred_q1, expected_q) + functional.mse_loss(pred_q2, expected_q)

        self.twin_q_opt.zero_grad()
        q_loss.backward()
        self.twin_q_opt.step()

        # Policy backpropagation
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
            # Apply observational noise (if any)
            if self.noise_scale > 0:
                state = self.add_obs_noise(state)
            # ----- Initialize Kalman filter for the episode -----
            d = state.shape[0]
            self.kalman_x = state.reshape(d, 1)  # use the first (noisy) observation as initial estimate
            self.kalman_P = np.eye(d) * self.kalman_init_P_scale
            # Initial guesses for Q and R (if not already set)
            if self.kalman_Q is None:
                self.kalman_Q = np.eye(d) * 0.01  # initial process noise guess
            if self.kalman_R is None:
                self.kalman_R = np.eye(d) * (self.noise_scale ** 2 if self.noise_scale > 0 else 0.1)
            # For the first step, there is no previous control so set it to zero.
            prev_control = np.zeros_like(state)
            # Initialize episode storage for EM update.
            self.episode_obs = [state.copy()]   # store initial filtered state
            self.episode_actions = [prev_control.copy()]

            while not done:
                # Use the current filtered state for action selection.
                action = self.get_action(self.kalman_x.flatten(), deterministic)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.running_mean(next_state)
                if self.noise_scale > 0:
                    next_state = self.add_obs_noise(next_state)
                # Kalman update: use the noisy observation and previous action as control.
                filtered_state = self.kalman_update(next_state, prev_control)
                # Store the filtered state and the control used (action from previous step)
                self.episode_obs.append(filtered_state.copy())
                self.episode_actions.append(action.copy())
                # Add transition to replay buffer (use filtered state)
                done_mask = 1.0 if n_steps == self.env._max_episode_steps - 1 else float(not done)
                self.memory.add(self.kalman_x.flatten(), action, reward, filtered_state, done_mask)
                n_steps += 1
                total_reward += reward
                # Update for next iteration: the new filtered state becomes the current state,
                # and the current action becomes the control for the next step.
                self.kalman_x = filtered_state.reshape(d, 1)
                prev_control = action
                state = next_state
                if n_steps > self.max_steps:
                    break

            # At the end of the episode, update Kalman filter noise parameters via EM.
            self.update_kalman_params_em()

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

            print("Episode: {}, steps: {}, total_reward: {}".format(i, n_steps, total_reward))

    def eval(self, num_games, render=True):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            state = self.env.reset()[0]
            state = self.running_mean(state)
            if self.noise_scale > 0:
                state = self.add_obs_noise(state)
            # Initialize Kalman filter for evaluation.
            d = state.shape[0]
            self.kalman_x = state.reshape(d, 1)
            self.kalman_P = np.eye(d) * self.kalman_init_P_scale
            prev_control = np.zeros_like(state)
            total_reward = 0
            done = False
            while not done:
                action = self.get_action(self.kalman_x.flatten(), deterministic=True)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.running_mean(next_state)
                if self.noise_scale > 0:
                    next_state = self.add_obs_noise(next_state)
                filtered_state = self.kalman_update(next_state, prev_control)
                prev_control = action
                total_reward += reward
                self.kalman_x = filtered_state.reshape(d, 1)
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

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
