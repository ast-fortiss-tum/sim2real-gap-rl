import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import time
import pickle
import random
from collections import deque

from architectures.utils import Model, gen_noise
from replay_buffer import ReplayBuffer
from models.sac import ContSAC, set_global_seed
from models.kalman_filter import LearnableKalmanFilter, FixedKalmanFilter
from models.fft_filter import LearnableFFTFilter, FixedFFTFilter     

class DARC(ContSAC):
    def __init__(self, policy_config, value_config, sa_config, sas_config,
                 source_env, target_env, device, running_mean,
                 log_dir="", memory_size=1e5, warmup_games=50, batch_size=64,
                 lr=0.0001, gamma=0.99, tau=0.003, alpha=0.2, ent_adj=False,
                 delta_r_scale=1.0, s_t_ratio=10, noise_scale=1.0,
                 target_update_interval=1, n_games_til_train=1, n_updates_per_train=1,
                 decay_rate=0.99, max_steps=200, if_normalize=True, print_on=False,
                 use_kf=False, use_fft=False, smoothers_learnable=False, seed=42):
        # Set global seed if provided.
        if seed is not None:
            set_global_seed(seed)
        # Call the ContSAC constructor using the source environment.
        super(DARC, self).__init__(policy_config, value_config, source_env, device,
                                     log_dir, running_mean, noise_scale,
                                     memory_size, warmup_games, batch_size, lr, gamma,
                                     tau, alpha, ent_adj, target_update_interval,
                                     n_games_til_train, n_updates_per_train, max_steps)
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale

        self.source_env = source_env
        self.target_env = target_env
        self.warmup_games = warmup_games
        self.n_games_til_train = n_games_til_train
        self.if_normalize = if_normalize
        self.print_on = print_on
        self.max_steps = max_steps
        self.source_step = 0
        self.target_step = 0

        # Instantiate classifiers.
        self.sa_classifier = Model(sa_config).to(self.device)
        self.sas_adv_classifier = Model(sas_config).to(self.device)

        self.source_memory = ReplayBuffer(memory_size, batch_size)
        self.target_memory = ReplayBuffer(memory_size, batch_size)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.policy_opt, step_size=1, gamma=decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.twin_q_opt, step_size=1, gamma=decay_rate)

        # Group classifier parameters.
        classifier_params = list(self.sa_classifier.parameters()) + list(self.sas_adv_classifier.parameters())
        self.use_kf = use_kf
        self.use_fft = use_fft
        if self.use_kf:
            # Use dimension 1 because only element 100 is filtered.
            if smoothers_learnable:
                self.kf = LearnableKalmanFilter(1).to(device)
            else:
                self.kf = FixedKalmanFilter(1).to(device)
            # We'll keep a separate KF state for each sample in a batch.
            self.kf_states = {}    # key: row_key, value: current state tensor [1]
            self.kf_covs = {}      # key: row_key, value: current covariance [1,1]
            if smoothers_learnable:
                classifier_params += list(self.kf.parameters())
        if self.use_fft:
            # For FFT, we maintain a rolling buffer for the 100th element.
            if smoothers_learnable:
                self.fft_filter = LearnableFFTFilter(signal_length=10).to(device)
            else:
                self.fft_filter = FixedFFTFilter(signal_length=10).to(device)
            self.fft_buffers = {}  # key: row_key, value: deque(maxlen=10)
            if smoothers_learnable:
                classifier_params += list(self.fft_filter.parameters())
        self.classifier_opt = Adam(classifier_params, lr=lr)
        self.scheduler_sa_classifier_opt = torch.optim.lr_scheduler.StepLR(self.classifier_opt, step_size=1, gamma=decay_rate)
        self.scheduler_sas_adv_classifier_opt = torch.optim.lr_scheduler.StepLR(self.classifier_opt, step_size=1, gamma=decay_rate)

    def log_smoother_params(self, epoch):
        if self.use_fft and hasattr(self.fft_filter, "mask_param"):
            mask = self.fft_filter.mask_param
            self.writer.add_scalar("fft_mask_mean", mask.mean().item(), epoch)
            self.writer.add_scalar("fft_mask_std", mask.std().item(), epoch)
            mask_image = mask.detach().unsqueeze(0).unsqueeze(0)  # [1, 1, length]
            self.writer.add_image("fft_mask", mask_image, epoch, dataformats="CHW")
        if self.use_kf and hasattr(self.kf, "A"):
            A, B, H = self.kf.A, self.kf.B, self.kf.H
            log_Q_diag, log_R_diag = self.kf.log_Q_diag, self.kf.log_R_diag
            self.writer.add_scalar("kf_A_mean", A.mean().item(), epoch)
            self.writer.add_scalar("kf_A_std", A.std().item(), epoch)
            self.writer.add_scalar("kf_B_mean", B.mean().item(), epoch)
            self.writer.add_scalar("kf_B_std", B.std().item(), epoch)
            self.writer.add_scalar("kf_H_mean", H.mean().item(), epoch)
            self.writer.add_scalar("kf_H_std", H.std().item(), epoch)
            self.writer.add_scalar("kf_log_Q_diag_mean", log_Q_diag.mean().item(), epoch)
            self.writer.add_scalar("kf_log_R_diag_mean", log_R_diag.mean().item(), epoch)
        self.writer.flush()

    def step_optim(self):
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_sa_classifier_opt.step()
        self.scheduler_sas_adv_classifier_opt.step()

    def reset_kf(self, init_state):
        init_state_t = torch.tensor(init_state, dtype=torch.float32, device=self.device)
        return init_state_t, torch.eye(init_state_t.shape[0], device=self.device)

    def apply_denoising(self, obs, u=None):
        """
        Applies denoising to a single observation vector (shape: [state_dim]),
        injecting noise only into element 100 and updating it via the chosen smoother.
        If using the Kalman filter, the control input u (a scalar tensor) is used.
        """
        noise = torch.zeros_like(obs)
        noise[100] = torch.randn(1, device=self.device) * self.noise_scale
        noisy_obs = obs + noise
        denoised_obs = noisy_obs.clone()
        noisy_val = noisy_obs[100]  # scalar
        if self.use_fft:
            filtered_val = self.fft_filter(noisy_val.unsqueeze(0)).squeeze(0)
            denoised_obs[100] = filtered_val
        elif self.use_kf:
            sample = noisy_val.unsqueeze(0)  # shape [1]
            # Use provided u in KF update.
            if self.kf_states.get("single") is None:
                self.kf_states["single"], self.kf_covs["single"] = self.reset_kf(sample)
            self.kf_states["single"], self.kf_covs["single"] = self.kf(
                self.kf_states["single"], self.kf_covs["single"], sample, u=u)
            denoised_obs[100] = self.kf_states["single"].squeeze(0)
        else:
            denoised_obs[100] = noisy_val
        return denoised_obs

    def apply_denoising_batch(self, obs, is_source=True, u=None):
        """
        Batch-friendly version for observations of shape [batch_size, state_dim].
        A separate rolling buffer (for FFT) or KF state is maintained for each row.
        Optionally, if u is provided, it is a tensor of shape [batch_size] and is passed to the KF update.
        """
        batch_size, state_dim = obs.shape
        denoised_obs = obs.clone()
        noise = torch.zeros_like(obs)
        noise[:, 100] = torch.randn(batch_size, device=self.device) * self.noise_scale
        noisy_obs = obs + noise

        if self.use_fft and not hasattr(self, "fft_buffers"):
            self.fft_buffers = {}
        if self.use_kf and not hasattr(self, "kf_states"):
            self.kf_states = {}
            self.kf_covs = {}

        for i in range(batch_size):
            row_key = f"{'src' if is_source else 'tgt'}_{i}"
            val_noisy = noisy_obs[i, 100]
            if self.use_fft:
                if row_key not in self.fft_buffers:
                    self.fft_buffers[row_key] = deque(maxlen=10)
                self.fft_buffers[row_key].append(float(val_noisy.item()))
                if len(self.fft_buffers[row_key]) < 2:
                    denoised_obs[i, 100] = val_noisy
                    continue
                buffer_tensor = torch.tensor(list(self.fft_buffers[row_key]), dtype=torch.float32, device=self.device)
                filtered_series = self.fft_filter(buffer_tensor)
                denoised_obs[i, 100] = filtered_series[-1]
            elif self.use_kf:
                if row_key not in self.kf_states:
                    self.kf_states[row_key], self.kf_covs[row_key] = self.reset_kf(val_noisy.unsqueeze(0))
                sample = val_noisy.unsqueeze(0)
                # If u is provided for this batch, extract the i-th control input.
                u_val = u[i].unsqueeze(0) if u is not None else None
                x_prev = self.kf_states[row_key]
                P_prev = self.kf_covs[row_key]
                x_new, P_new = self.kf(x_prev, P_prev, sample, u=u_val)
                self.kf_states[row_key] = x_new
                self.kf_covs[row_key] = P_new
                denoised_obs[i, 100] = x_new.squeeze(0)
            else:
                denoised_obs[i, 100] = val_noisy
        return denoised_obs

    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        t_states, t_actions, _, t_next_states, _, game_count = args

        s_states = torch.as_tensor(s_states, dtype=torch.float32, device=self.device)
        s_actions = torch.as_tensor(s_actions, dtype=torch.float32, device=self.device)
        s_rewards = torch.as_tensor(s_rewards[:, np.newaxis], dtype=torch.float32, device=self.device)
        s_next_states = torch.as_tensor(s_next_states, dtype=torch.float32, device=self.device)
        s_done_masks = torch.as_tensor(s_done_masks[:, np.newaxis], dtype=torch.float32, device=self.device)
        t_states = torch.as_tensor(t_states, dtype=torch.float32, device=self.device)
        t_actions = torch.as_tensor(t_actions, dtype=torch.float32, device=self.device)
        t_next_states = torch.as_tensor(t_next_states, dtype=torch.float32, device=self.device)

        # For the classifier branch, reapply denoising.
        # Here, we use s_actions[:,0] and t_actions[:,0] as the control input u.
        s_states_denoised = self.apply_denoising_batch(s_states, is_source=True, u=s_actions[:,0])
        t_states_denoised = self.apply_denoising_batch(t_states, is_source=False, u=t_actions[:,0])

        s_sa_inputs = torch.cat([s_states_denoised[:, 100].unsqueeze(1),
                                 s_actions[:, 0].unsqueeze(1)], dim=1)
        s_sas_inputs = torch.cat([s_states_denoised[:, 100].unsqueeze(1),
                                  s_actions[:, 0].unsqueeze(1),
                                  s_next_states[:, 100].unsqueeze(1)], dim=1)
        t_sa_inputs = torch.cat([t_states_denoised[:, 100].unsqueeze(1),
                                 t_actions[:, 0].unsqueeze(1)], dim=1)
        t_sas_inputs = torch.cat([t_states_denoised[:, 100].unsqueeze(1),
                                  t_actions[:, 0].unsqueeze(1),
                                  t_next_states[:, 100].unsqueeze(1)], dim=1)

        s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale, s_sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(s_sas_inputs + gen_noise(self.noise_scale, s_sas_inputs, self.device))
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs, self.device))

        sa_log_probs = torch.log(torch.softmax(s_sa_logits, dim=1) + 1e-12)
        sas_log_probs = torch.log(torch.softmax(s_sas_logits + s_sa_logits, dim=1) + 1e-12)
        delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
        if game_count >= 2 * self.warmup_games:
            s_rewards = s_rewards + self.delta_r_scale * delta_r.unsqueeze(1)

        loss_function = nn.CrossEntropyLoss()
        label_zero = torch.zeros((s_sa_logits.shape[0],), dtype=torch.int64, device=self.device)
        label_one = torch.ones((t_sa_logits.shape[0],), dtype=torch.int64, device=self.device)
        classify_loss = (loss_function(s_sa_logits, label_zero) +
                         loss_function(t_sa_logits, label_one) +
                         loss_function(s_sas_logits, label_zero) +
                         loss_function(t_sas_logits, label_one))

        self.classifier_opt.zero_grad()
        classify_loss.backward()
        self.classifier_opt.step()

        train_info = {'Loss/Classify Loss': classify_loss,
                      'Stats/Avg Delta Reward': delta_r.mean(),
                      'Stats/Avg Source SA Acc': 1 - torch.argmax(s_sa_logits, dim=1).double().mean(),
                      'Stats/Avg Source SAS Acc': 1 - torch.argmax(s_sas_logits, dim=1).double().mean(),
                      'Stats/Avg Target SA Acc': torch.argmax(t_sa_logits, dim=1).double().mean(),
                      'Stats/Avg Target SAS Acc': torch.argmax(t_sas_logits, dim=1).double().mean()}

        s_states_rl = torch.as_tensor(s_states.cpu().detach().numpy(), dtype=torch.float32, device=self.device)
        s_actions_rl = s_actions
        s_rewards_rl = s_rewards
        s_next_states_rl = torch.as_tensor(s_next_states.cpu().detach().numpy(), dtype=torch.float32, device=self.device)
        s_done_masks_rl = s_done_masks

        rl_train_info = super(DARC, self).train_step(s_states_rl, s_actions_rl, s_rewards_rl, s_next_states_rl, s_done_masks_rl)
        train_info.update(rl_train_info)

        if game_count % 10 == 0:
            self.log_smoother_params(game_count)

        return train_info

    def train(self, num_games, deterministic=False):
        self.policy.train()
        self.twin_q.train()
        self.sa_classifier.train()
        self.sas_adv_classifier.train()
        for i in range(num_games):
            source_reward, source_step = self.simulate_env(i, "source", deterministic)
            if i < self.warmup_games or i % self.s_t_ratio == 0:
                target_reward, target_step = self.simulate_env(i, "target", deterministic)
                self.writer.add_scalar('Target Env/Rewards', target_reward, i)
                self.writer.add_scalar('Target Env/N_Steps', target_step, i)
                print("TARGET: index: {}, steps: {}, total_rewards: {}".format(i, target_step, target_reward))
            if i >= self.warmup_games:
                self.writer.add_scalar('Source Env/Rewards', source_reward, i)
                self.writer.add_scalar('Source Env/N_Steps', source_step, i)
                if i % self.n_games_til_train == 0:
                    for _ in range(source_step * self.n_updates_per_train):
                        self.total_train_steps += 1
                        s_s, s_a, s_r, s_s_, s_d = self.source_memory.sample()
                        t_s, t_a, t_r, t_s_, t_d = self.target_memory.sample()
                        train_info = self.train_step(s_s, s_a, s_r, s_s_, s_d, t_s, t_a, t_r, t_s_, t_d, i)
                        self.writer.add_train_step_info(train_info, i)
                    self.writer.write_train_step()
                if i % 100 == 0:
                    print('src', self.eval_src(10))
                    print('tgt', self.eval_tgt(10))
            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))

    def simulate_env(self, game_count, env_name, deterministic):
        if env_name == "source":
            env = self.source_env
            memory = self.source_memory
        elif env_name == "target":
            env = self.target_env
            memory = self.target_memory
        else:
            raise Exception("Env name not recognized")
        total_rewards = 0
        n_steps = 0
        done = False
        state = env.reset()[0]
        if self.if_normalize:
            state = self.running_mean(state[0])
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        is_source = (env_name == "source")
        state_denoised = self.apply_denoising_batch(state_tensor, is_source=is_source, u=None)
        state = state_denoised[0].cpu().detach().numpy()
        while not done:
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                action = self.get_action(state, deterministic)
            next_state, reward, done, _, _ = env.step(action)
            if self.if_normalize:
                next_state = self.running_mean(next_state)
            next_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # In simulation, we assume no control input (or u can be extracted from verified actions)
            next_denoised = self.apply_denoising_batch(next_tensor, is_source=is_source, u=None)
            next_state = next_denoised[0].cpu().detach().numpy()
            done_mask = 1.0 if n_steps == env._max_episode_steps - 1 else 0.0
            if n_steps == self.max_steps:
                done = True

            ver_actions = env.env.sys.verified_actions
            ver_penalties = env.env.sys.verification_penalties
            ver_actions_values = [node_data['p']
                                  for agent in ver_actions.values()
                                  for node_data in agent.values() if 'p' in node_data]
            ver_penalties_values = [ver_penalties.values()]
            ver_actions_values = [value for arr in ver_actions_values for value in arr]
            ver_penalties_values = [value for dv in ver_penalties_values for value in list(dv)]
            memory.add(state, ver_actions_values, reward, next_state, done_mask)
            if env_name == "source":
                self.source_step += 1
            else:
                self.target_step += 1
            total_rewards += reward
            n_steps += 1
            state = next_state
        return total_rewards, n_steps

    def eval_src(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            state = self.source_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state[0])
            st_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            st_denoised = self.apply_denoising_batch(st_tensor, is_source=True, u=None)
            state = st_denoised[0].cpu().detach().numpy()
            done = False
            total_reward = 0
            step = 0
            while not done:
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _, _ = self.source_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                nxt_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                nxt_denoised = self.apply_denoising_batch(nxt_tensor, is_source=True, u=None)
                next_state = nxt_denoised[0].cpu().detach().numpy()
                total_reward += reward
                state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
            self.writer.add_scalar('Eval/Source Reward', total_reward, i)
            reward_all += total_reward
        avg_reward = reward_all / num_games
        self.writer.add_scalar('Eval/Source Avg Reward', avg_reward, num_games)
        return avg_reward

    def eval_tgt(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            state = self.target_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state[0])
            st_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            st_denoised = self.apply_denoising_batch(st_tensor, is_source=False, u=None)
            state = st_denoised[0].cpu().detach().numpy()
            done = False
            total_reward = 0
            step = 0
            while not done:
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _, _ = self.target_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                nxt_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                nxt_denoised = self.apply_denoising_batch(nxt_tensor, is_source=False, u=None)
                next_state = nxt_denoised[0].cpu().detach().numpy()
                total_reward += reward
                state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
            self.writer.add_scalar('Eval/Target Reward', total_reward, i)
            reward_all += total_reward
        avg_reward = reward_all / num_games
        self.writer.add_scalar('Eval/Target Avg Reward', avg_reward, num_games)
        return avg_reward

    def eval(self, num_games, render=True):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            state = self.env.reset()[0]
            state = self.running_mean(state)
            st_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            st_denoised = self.apply_denoising_batch(st_tensor, is_source=True, u=None)
            state = st_denoised[0].cpu().detach().numpy()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state, deterministic=True)
                next_state, reward, done, _ , _ = self.env.step(action)
                next_state = self.running_mean(next_state[0])
                nxt_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                nxt_denoised = self.apply_denoising_batch(nxt_tensor, is_source=True, u=None)
                next_state = nxt_denoised[0].cpu().detach().numpy()
                total_reward += reward
                state = next_state
            self.writer.add_scalar('Eval/Reward', total_reward, i)
            reward_all += total_reward
        avg_reward = reward_all / num_games
        self.writer.add_scalar('Eval/Avg Reward', avg_reward, num_games)
        print("Average Eval Reward:", avg_reward)
        return avg_reward

    def save_model(self, folder_name):
        import os
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
        from architectures.utils import polyak_update
        polyak_update(self.twin_q, self.target_twin_q, 1)
        polyak_update(self.twin_q, self.target_twin_q, 1)
        self.running_mean = pickle.load(open(path + '/running_mean', "rb"))

# -----------------------------------------------------------
# DARC_two: a variant using additional state indices (e.g., 100 and 226)
# -----------------------------------------------------------
class DARC_two(ContSAC):
    def __init__(self, policy_config, value_config, sa_config, sas_config,
                 source_env, target_env, device, running_mean,
                 log_dir="", memory_size=1e5, warmup_games=50, batch_size=64,
                 lr=0.0001, gamma=0.99, tau=0.003, alpha=0.2, ent_adj=False,
                 delta_r_scale=1.0, s_t_ratio=10, noise_scale=1.0,
                 target_update_interval=1, n_games_til_train=1, n_updates_per_train=1,
                 decay_rate=0.99, max_steps=200, if_normalize=False, print_on=False,
                 use_kf=False, use_fft=False, seed=42):
        if seed is not None:
            set_global_seed(seed)
        super(DARC_two, self).__init__(policy_config, value_config, source_env, device,
                                        log_dir, running_mean, noise_scale,
                                        memory_size, warmup_games, batch_size, lr, gamma,
                                        tau, alpha, ent_adj, target_update_interval,
                                        n_games_til_train, n_updates_per_train, max_steps)
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale

        self.source_env = source_env
        self.target_env = target_env
        self.warmup_games = warmup_games
        self.n_games_til_train = n_games_til_train

        self.sa_classifier = Model(sa_config).to(self.device)
        self.sas_adv_classifier = Model(sas_config).to(self.device)
        self.running_mean = running_mean
        self.max_steps = max_steps

        self.if_normalize = if_normalize    
        self.source_step = 0
        self.target_step = 0
        self.source_memory = ReplayBuffer(memory_size, batch_size)
        self.target_memory = ReplayBuffer(memory_size, batch_size)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.policy_opt, step_size=1, gamma=decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.twin_q_opt, step_size=1, gamma=decay_rate)
        self.scheduler_sa_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sa_classifier.parameters(), step_size=1, gamma=decay_rate)
        self.scheduler_sas_adv_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sas_adv_classifier.parameters(), step_size=1, gamma=decay_rate)
        self.print_on = print_on

        self.use_kf = use_kf
        self.use_fft = use_fft
        if self.use_kf:
            state_dim = source_env.observation_space.shape[0]
            self.kf = LearnableKalmanFilter(state_dim).to(device)
            self.kf_state = None
            self.kf_cov = None
        if self.use_fft:
            state_dim = source_env.observation_space.shape[0]
            self.fft_filter = LearnableFFTFilter(state_dim).to(device)

    def reset_kf(self, init_state):
        init_state_t = torch.tensor(init_state, dtype=torch.float32, device=self.device)
        self.kf_state = init_state_t
        self.kf_cov = torch.eye(init_state_t.shape[0], device=self.device)

    def apply_denoising(self, obs):
        noise = torch.tensor(np.random.normal(0, self.noise_scale, size=obs.shape),
                             dtype=obs.dtype, device=self.device)
        noisy_obs = obs + noise
        if self.use_fft:
            return self.fft_filter(noisy_obs)
        elif self.use_kf:
            if self.kf_state is None:
                self.reset_kf(noisy_obs[0])
            denoised = []
            for i in range(noisy_obs.shape[0]):
                ns = noisy_obs[i]
                self.kf_state, self.kf_cov = self.kf(self.kf_state, self.kf_cov, ns)
                denoised.append(self.kf_state)
            return torch.stack(denoised, dim=0)
        else:
            return noisy_obs

    def step_optim(self):
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_sa_classifier_opt.step()
        self.scheduler_sas_adv_classifier_opt.step()

    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        t_states, t_actions, _, t_next_states, _, game_count = args
        s_states = torch.as_tensor(s_states, dtype=torch.float32, device=self.device)
        s_actions = torch.as_tensor(s_actions, dtype=torch.float32, device=self.device)
        s_rewards = torch.as_tensor(s_rewards[:, np.newaxis], dtype=torch.float32, device=self.device)
        s_next_states = torch.as_tensor(s_next_states, dtype=torch.float32, device=self.device)
        s_done_masks = torch.as_tensor(s_done_masks[:, np.newaxis], dtype=torch.float32, device=self.device)
        t_states = torch.as_tensor(t_states, dtype=torch.float32, device=self.device)
        t_actions = torch.as_tensor(t_actions, dtype=torch.float32, device=self.device)
        t_next_states = torch.as_tensor(t_next_states, dtype=torch.float32, device=self.device)
        
        # ---- Classifier Branch (using additional state indices) ----
        s_states_denoised = self.apply_denoising(s_states)
        t_states_denoised = self.apply_denoising(t_states)
        
        s_sa_inputs = torch.cat([
            torch.cat([s_states_denoised[:, 100].unsqueeze(1),
                       s_states_denoised[:, 226].unsqueeze(1)], dim=1),
            s_actions
        ], dim=1)
        s_sas_inputs = torch.cat([
            torch.cat([s_states_denoised[:, 100].unsqueeze(1),
                       s_states_denoised[:, 226].unsqueeze(1)], dim=1),
            s_actions,
            torch.cat([s_next_states[:, 100].unsqueeze(1),
                       s_next_states[:, 226].unsqueeze(1)], dim=1)
        ], dim=1)
        t_sa_inputs = torch.cat([
            torch.cat([t_states_denoised[:, 100].unsqueeze(1),
                       t_states_denoised[:, 226].unsqueeze(1)], dim=1),
            t_actions
        ], dim=1)
        t_sas_inputs = torch.cat([
            torch.cat([t_states_denoised[:, 100].unsqueeze(1),
                       t_states_denoised[:, 226].unsqueeze(1)], dim=1),
            t_actions,
            torch.cat([t_next_states[:, 100].unsqueeze(1),
                       t_next_states[:, 226].unsqueeze(1)], dim=1)
        ], dim=1)
        
        s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale, s_sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(s_sas_inputs + gen_noise(self.noise_scale, s_sas_inputs, self.device))
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs, self.device))
        
        sa_log_probs = torch.log(torch.softmax(s_sa_logits, dim=1) + 1e-12)
        sas_log_probs = torch.log(torch.softmax(s_sas_logits + s_sa_logits, dim=1) + 1e-12)
        delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
        if game_count >= 2 * self.warmup_games:
            s_rewards = s_rewards + self.delta_r_scale * delta_r.unsqueeze(1)
        
        loss_function = nn.CrossEntropyLoss()
        label_zero = torch.zeros((s_sa_logits.shape[0],), dtype=torch.int64, device=self.device)
        label_one = torch.ones((t_sa_logits.shape[0],), dtype=torch.int64, device=self.device)
        classify_loss = (loss_function(s_sa_logits, label_zero) +
                         loss_function(t_sa_logits, label_one) +
                         loss_function(s_sas_logits, label_zero) +
                         loss_function(t_sas_logits, label_one))
        self.classifier_opt.zero_grad()
        classify_loss.backward()
        self.classifier_opt.step()
        
        train_info = {'Loss/Classify Loss': classify_loss,
                      'Stats/Avg Delta Reward': delta_r.mean(),
                      'Stats/Avg Source SA Acc': 1 - torch.argmax(s_sa_logits, dim=1).double().mean(),
                      'Stats/Avg Source SAS Acc': 1 - torch.argmax(s_sas_logits, dim=1).double().mean(),
                      'Stats/Avg Target SA Acc': torch.argmax(t_sa_logits, dim=1).double().mean(),
                      'Stats/Avg Target SAS Acc': torch.argmax(t_sas_logits, dim=1).double().mean()}
        
        s_states_rl = torch.as_tensor(s_states.cpu().detach().numpy(), dtype=torch.float32, device=self.device)
        s_actions_rl = s_actions
        s_rewards_rl = s_rewards
        s_next_states_rl = torch.as_tensor(s_next_states.cpu().detach().numpy(), dtype=torch.float32, device=self.device)
        s_done_masks_rl = s_done_masks
        rl_train_info = super(DARC_two, self).train_step(s_states_rl, s_actions_rl, s_rewards_rl, s_next_states_rl, s_done_masks_rl)
        train_info.update(rl_train_info)
        return train_info
    
    def train(self, num_games, deterministic=False):
        self.policy.train()
        self.twin_q.train()
        self.sa_classifier.train()
        self.sas_adv_classifier.train()
        for i in range(num_games):
            source_reward, source_step = self.simulate_env(i, "source", deterministic)

            if i < self.warmup_games or i % self.s_t_ratio == 0:
                target_reward, target_step = self.simulate_env(i, "target", deterministic)
                self.writer.add_scalar('Target Env/Rewards', target_reward, i)
                self.writer.add_scalar('Target Env/N_Steps', target_step, i)
                print("TARGET: index: {}, steps: {}, total_rewards: {}".format(i, target_step, target_reward))

            if i >= self.warmup_games:
                self.writer.add_scalar('Source Env/Rewards', source_reward, i)
                self.writer.add_scalar('Source Env/N_Steps', source_step, i)
                if i % self.n_games_til_train == 0:
                    for _ in range(source_step * self.n_updates_per_train):
                        self.total_train_steps += 1
                        s_s, s_a, s_r, s_s_, s_d = self.source_memory.sample()
                        t_s, t_a, t_r, t_s_, t_d = self.target_memory.sample()
                        train_info = self.train_step(s_s, s_a, s_r, s_s_, s_d, t_s, t_a, t_r, t_s_, t_d, i)
                        self.writer.add_train_step_info(train_info, i)
                    self.writer.write_train_step()
                if i % 100 == 0:
                    print('src', self.eval_src(10))
                    print('tgt', self.eval_tgt(10))
                    #self.save_model(str(i))

            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))

    def simulate_env(self, game_count, env_name, deterministic):
        if env_name == "source":
            env = self.source_env; memory = self.source_memory
        elif env_name == "target":
            env = self.target_env; memory = self.target_memory
        else:
            raise Exception("Env name not recognized")
        total_rewards = 0; n_steps = 0; done = False
        state = env.reset()[0]
        if self.if_normalize:
            state = self.running_mean(state[0])
        state_tensor = self.apply_denoising(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        state = state_tensor.cpu().detach().numpy()
        while not done:
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                action = self.get_action(state, deterministic)
            next_state, reward, done, _, _ = env.step(action)
            if self.if_normalize:
                next_state = self.running_mean(next_state)
            next_state_tensor = self.apply_denoising(torch.as_tensor(next_state, dtype=torch.float32, device=self.device))
            next_state = next_state_tensor.cpu().detach().numpy()
            done_mask = 1.0 if n_steps == env._max_episode_steps - 1 else 0.0
            if n_steps == self.max_steps:
                done = True
            memory.add(state, None, reward, next_state, done_mask)
            total_rewards += reward; n_steps += 1; state = next_state
        return total_rewards, n_steps

    def eval_src(self, num_games, render=False):
        self.policy.eval(); self.twin_q.eval(); reward_all = 0
        for i in range(num_games):
            state = self.source_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state[0])
            state_tensor = self.apply_denoising(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            state = state_tensor.cpu().detach().numpy()
            done = False; total_reward = 0; step = 0
            while not done:
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _, _ = self.source_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                next_state_tensor = self.apply_denoising(torch.as_tensor(next_state, dtype=torch.float32, device=self.device))
                next_state = next_state_tensor.cpu().detach().numpy()
                total_reward += reward; state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
            self.writer.add_scalar('Eval/Source Reward', total_reward, i)
            reward_all += total_reward
        avg_reward = reward_all / num_games
        self.writer.add_scalar('Eval/Source Avg Reward', avg_reward, num_games)
        return avg_reward

    def eval_tgt(self, num_games, render=False):
        self.policy.eval(); self.twin_q.eval(); reward_all = 0
        for i in range(num_games):
            state = self.target_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state[0])
            state_tensor = self.apply_denoising(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            state = state_tensor.cpu().detach().numpy()
            done = False; total_reward = 0; step = 0
            while not done:
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _, _ = self.target_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                next_state_tensor = self.apply_denoising(torch.as_tensor(next_state, dtype=torch.float32, device=self.device))
                next_state = next_state_tensor.cpu().detach().numpy()
                total_reward += reward; state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
            self.writer.add_scalar('Eval/Target Reward', total_reward, i)
            reward_all += total_reward
        avg_reward = reward_all / num_games
        self.writer.add_scalar('Eval/Target Avg Reward', avg_reward, num_games)
        return avg_reward

    def eval(self, num_games, render=True):
        self.policy.eval(); self.twin_q.eval(); reward_all = 0
        for i in range(num_games):
            state = self.env.reset()[0]
            state = self.running_mean(state)
            state_tensor = self.apply_denoising(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            state = state_tensor.cpu().detach().numpy()
            done = False; total_reward = 0
            while not done:
                action = self.get_action(state, deterministic=True)
                next_state, reward, done, _ , _ = self.env.step(action)
                next_state = self.running_mean(next_state[0])
                next_state_tensor = self.apply_denoising(torch.as_tensor(next_state, dtype=torch.float32, device=self.device))
                next_state = next_state_tensor.cpu().detach().numpy()
                total_reward += reward; state = next_state
            self.writer.add_scalar('Eval/Reward', total_reward, i)
            reward_all += total_reward
        avg_reward = reward_all / num_games
        self.writer.add_scalar('Eval/Avg Reward', avg_reward, num_games)
        print("Average Eval Reward:", avg_reward)
        return avg_reward

    def save_model(self, folder_name):
        import os
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
        from architectures.utils import polyak_update
        polyak_update(self.twin_q, self.target_twin_q, 1)
        polyak_update(self.twin_q, self.target_twin_q, 1)
        self.running_mean = pickle.load(open(path + '/running_mean', "rb"))
