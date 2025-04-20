import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import time
import pickle

from architectures.utils import Model, gen_noise
from replay_buffer import ReplayBuffer
from models.sac_refractored2 import ContSAC, set_global_seed

from online_denoising_AE import OnlineDenoisingAutoencoder, DenoisingDataset

# ============================================================
# BaseDARC: Contains common initialization and methods.
# ============================================================
class BaseDARC(ContSAC):
    def __init__(self, policy_config, value_config, sa_config, sas_config, source_env,
                 target_env, device, running_mean, log_dir="", memory_size=1e5, warmup_games=50,
                 batch_size=64, lr=0.0001, gamma=0.99, tau=0.003, alpha=0.2, ent_adj=False,
                 delta_r_scale=1.0, s_t_ratio=10, noise_scale=1.0, bias=0.5, target_update_interval=1,
                 n_games_til_train=1, n_updates_per_train=1, decay_rate=0.99, max_steps=200,
                 if_normalize=True, print_on=False, seed=42, use_denoiser=1, denoiser_dict = None,noise_cfrs=0.2, use_darc=True):
        # Set a fixed random seed if provided.
        if seed is not None:
            set_global_seed(seed)

        # Call parent (ContSAC) constructor
        super(BaseDARC, self).__init__(policy_config, value_config, source_env, device, log_dir,
                                       running_mean, noise_scale, bias, memory_size, warmup_games, batch_size,
                                       lr, gamma, tau, alpha, ent_adj, target_update_interval,
                                       n_games_til_train, n_updates_per_train, max_steps,noise_indices=None, use_denoiser=use_denoiser)

        # Save common hyperparameters
        self.use_darc = use_darc
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio

        self.source_env = source_env
        self.target_env = target_env

        self.sa_classifier = Model(sa_config).to(self.device)
        self.sa_classifier_opt = Adam(self.sa_classifier.parameters(), lr=lr)
        self.sas_adv_classifier = Model(sas_config).to(self.device)
        self.sas_adv_classifier_opt = Adam(self.sas_adv_classifier.parameters(), lr=lr)

        self.if_normalize = if_normalize

        self.source_step = 0
        self.target_step = 0
        self.source_memory = ReplayBuffer(memory_size, batch_size)
        self.target_memory = ReplayBuffer(memory_size, batch_size)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.policy_opt, step_size=1, gamma=decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.twin_q_opt, step_size=1, gamma=decay_rate)
        self.scheduler_sa_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sa_classifier_opt, step_size=1, gamma=decay_rate)
        self.scheduler_sas_adv_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sas_adv_classifier_opt, step_size=1, gamma=decay_rate)
        self.print_on = print_on

        self.noise_scale_cfrs = noise_cfrs

        # Load the pretrained online denoiser.
        self.use_denoiser = use_denoiser

        if self.use_denoiser:
            d_noise = denoiser_dict["noise"]
            d_bias = denoiser_dict["bias"]
            d_degree = denoiser_dict["degree"]
            self.denoiser = OnlineDenoisingAutoencoder(input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1).to(self.device)
            self.denoiser.load_state_dict(torch.load(f"Denoising_AE/best_online_denoising_autoencoder_Gaussian_Noise_{d_noise}_Bias_{d_bias}_Degree_{d_degree}.pth", map_location=self.device, weights_only=True))
            print("Denoiser loaded successfully.")
            self.denoiser.eval()
        else:
            self.denoiser = None

    # --- Methods to be specialized in child classes ---
    def add_obs_noise_percent(self, obs):
        """Abstract: add percentage noise to observation(s)."""
        raise NotImplementedError("Subclasses must implement add_obs_noise_percent")
        
    def add_obs_noise(self, obs):
        """Abstract: add noise (absolute) to observation(s)."""
        raise NotImplementedError("Subclasses must implement add_obs_noise")
        
    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        """Abstract: train step using replay data."""
        raise NotImplementedError("Subclasses must implement train_step")
        
    def simulate_env(self, game_count, env_name, deterministic):
        """Abstract: simulate an environment episode."""
        raise NotImplementedError("Subclasses must implement simulate_env")

    # --- Methods common to all versions ---
    def step_optim(self):
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_sa_classifier_opt.step()
        self.scheduler_sas_adv_classifier_opt.step()

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
                #if i % 50 == 0:
                if False:
                    print('src', self.eval_src(10))
                    print('tgt', self.eval_tgt(10))
            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))

    def eval_src(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            state = self.source_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state)
            state = self.add_obs_noise(state)
            done = False
            total_reward = 0
            step = 0
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _, _ = self.source_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                next_state = self.add_obs_noise(next_state)
                total_reward += reward
                state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
            reward_all += total_reward
        return reward_all / num_games

    def eval_tgt(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            step = 0
            state = self.target_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state)
            state = self.add_obs_noise(state)
            done = False
            total_reward = 0
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _, _ = self.target_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                next_state = self.add_obs_noise(next_state)
                total_reward += reward
                state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
            reward_all += total_reward
        return reward_all / num_games

    def save_model(self, folder_name):
        import os
        path = os.path.join('saved_weights/', folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), path + '/policy')
        torch.save(self.twin_q.state_dict(), path + '/twin_q_net')
        torch.save(self.sa_classifier.state_dict(), path + '/sa_classifier')
        torch.save(self.sas_adv_classifier.state_dict(), path + '/sas_adv_classifier')
        pickle.dump(self.running_mean, open(path + '/running_mean', 'wb'))

    def load_model(self, folder_name, device):
        super(BaseDARC, self).load_model(folder_name, device)
        path = 'saved_weights/' + folder_name
        self.sa_classifier.load_state_dict(torch.load(path + '/sa_classifier', map_location=torch.device(device)))
        self.sas_adv_classifier.load_state_dict(torch.load(path + '/sas_adv_classifier', map_location=torch.device(device)))
        self.running_mean = pickle.load(open(path + '/running_mean', "rb"))

# ============================================================
# DARC_one: Specializes BaseDARC for handling one observation index (e.g. index 100).
# ============================================================
class DARC_one(BaseDARC):
    # In this case, we treat index 100 as the only special element.
    def add_obs_noise_percent(self, obs):
        obs_noisy = np.copy(obs)
        if obs_noisy[100] != 0:
            noise = np.random.normal(0, self.noise_scale)
            obs_noisy[100] *= (1 + noise)
        else:
            obs_noisy[100] += np.random.normal(0, self.noise_scale)
        return obs_noisy

    def add_obs_noise(self, obs):
        obs_noisy = np.copy(obs)
        noise = np.random.normal(0, self.noise_scale) + self.bias
        obs_noisy[100] = obs_noisy[100] + noise
        return obs_noisy

    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        # The extra replay arguments: t_states, t_actions, _, t_next_states, _, game_count.
        t_states, t_actions, _, t_next_states, _, game_count = args
        # Ensure tensors are on device.
        if not torch.is_tensor(s_states):
            s_states = torch.as_tensor(s_states, dtype=torch.float32).to(self.device)
            s_actions = torch.as_tensor(s_actions, dtype=torch.float32).to(self.device)
            s_rewards = torch.as_tensor(s_rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
            s_next_states = torch.as_tensor(s_next_states, dtype=torch.float32).to(self.device)
            s_done_masks = torch.as_tensor(s_done_masks[:, np.newaxis], dtype=torch.float32).to(self.device)
            t_states = torch.as_tensor(t_states, dtype=torch.float32).to(self.device)
            t_actions = torch.as_tensor(t_actions, dtype=torch.float32).to(self.device)
            t_next_states = torch.as_tensor(t_next_states, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Extract raw values at index 100.
            s_recovered = s_states[:, 100].unsqueeze(1)
            s_next_recovered = s_next_states[:, 100].unsqueeze(1)
            t_recovered = t_states[:, 100].unsqueeze(1)
            
            # Build classifier inputs.
            sa_inputs = torch.cat([s_recovered, s_actions[:, 0].unsqueeze(1)], dim=1)
            sas_inputs = torch.cat([s_recovered, s_actions[:, 0].unsqueeze(1), s_next_recovered], dim=1)
            t_sa_inputs = torch.cat([t_recovered, t_actions[:, 0].unsqueeze(1)], dim=1)
            t_sas_inputs = torch.cat([t_recovered, t_actions[:, 0].unsqueeze(1), 
                                      t_next_states[:, 100].unsqueeze(1)], dim=1)
            
            # Inject noise.
            sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale_cfrs, sa_inputs, self.device))
            sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale_cfrs, sas_inputs, self.device))
            sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
            sas_log_probs = torch.log(torch.softmax(sas_logits, dim=1) + 1e-12)
            
            if self.use_darc:
                delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
            else:
                delta_r = torch.zeros_like(sas_log_probs[:, 1])
            if game_count >= 2 * self.warmup_games:
                s_rewards = s_rewards + self.delta_r_scale * delta_r.unsqueeze(1)
        
        train_info = ContSAC.train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks)
        
        # Rebuild classifier inputs.
        s_sa_inputs = torch.cat([s_recovered, s_actions[:, 0].unsqueeze(1)], dim=1)
        s_sas_inputs = torch.cat([s_recovered, s_actions[:, 0].unsqueeze(1), s_next_recovered], dim=1)
        t_sa_inputs = torch.cat([t_recovered, t_actions[:, 0].unsqueeze(1)], dim=1)
        t_sas_inputs = torch.cat([t_recovered, t_actions[:, 0].unsqueeze(1), t_next_states[:, 100].unsqueeze(1)], dim=1)
        
        s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale_cfrs, s_sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(s_sas_inputs + gen_noise(self.noise_scale_cfrs, s_sas_inputs, self.device))
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale_cfrs, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale_cfrs, t_sas_inputs, self.device))
        
        loss_function = nn.CrossEntropyLoss()
        label_zero = torch.zeros(s_sa_logits.shape[0], dtype=torch.int64).to(self.device)
        label_one = torch.ones(t_sa_logits.shape[0], dtype=torch.int64).to(self.device)
        
        classify_loss = loss_function(s_sa_logits, label_zero)
        classify_loss += loss_function(t_sa_logits, label_one)
        classify_loss += loss_function(s_sas_logits, label_zero)
        classify_loss += loss_function(t_sas_logits, label_one)
        
        self.sa_classifier_opt.zero_grad()
        self.sas_adv_classifier_opt.zero_grad()
        classify_loss.backward()
        self.sa_classifier_opt.step()
        self.sas_adv_classifier_opt.step()
        
        s_sa_acc = 1 - torch.argmax(s_sa_logits, dim=1).double().mean()
        s_sas_acc = 1 - torch.argmax(s_sas_logits, dim=1).double().mean()
        t_sa_acc = torch.argmax(t_sa_logits, dim=1).double().mean()
        t_sas_acc = torch.argmax(t_sas_logits, dim=1).double().mean()
        
        train_info['Loss/Classify Loss'] = classify_loss
        train_info['Stats/Avg Delta Reward'] = delta_r.mean()
        train_info['Stats/Avg Source SA Acc'] = s_sa_acc
        train_info['Stats/Avg Source SAS Acc'] = s_sas_acc
        train_info['Stats/Avg Target SA Acc'] = t_sa_acc
        train_info['Stats/Avg Target SAS Acc'] = t_sas_acc
        return train_info

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
            state = self.running_mean(state)
        
        # For target environment, optionally apply the denoiser.
        if env_name == "target":
            if self.use_denoiser:
                # For denoiser: process index 100.
                real_val = state[100]
                noisy_state = self.add_obs_noise(state)
                noisy_val = noisy_state[100]
                # Initialize accumulation buffer.
                accumulated_sequence = [noisy_val]
                acc_tensor = torch.tensor(accumulated_sequence, dtype=torch.float32)\
                                    .unsqueeze(0).unsqueeze(-1).to(self.device)
                with torch.no_grad():
                    denoised_seq, _ = self.denoiser.forward_online(acc_tensor)
                recovered_val = denoised_seq[0, -1, :].cpu().numpy()[0]
                state[100] = recovered_val

                # For possible plotting.
                real_values = [real_val]
                noisy_values = [noisy_val]
                recovered_values = [recovered_val]
            else:
                # If not using the denoiser, simply add noise.
                state = self.add_obs_noise(state)
        
        while not done:
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                action = self.get_action(state, deterministic)
            
            next_state, reward, done, _, _ = env.step(action)
            if self.if_normalize:
                next_state = self.running_mean(next_state)
            
            if env_name == "target":
                if self.use_denoiser:
                    real_val = next_state[100]
                    noisy_next_state = self.add_obs_noise(next_state)
                    noisy_val = noisy_next_state[100]
                    accumulated_sequence.append(noisy_val)
                    acc_tensor = torch.tensor(accumulated_sequence, dtype=torch.float32)\
                                    .unsqueeze(0).unsqueeze(-1).to(self.device)
                    with torch.no_grad():
                        denoised_seq, _ = self.denoiser.forward_online(acc_tensor)
                    recovered_val = denoised_seq[0, -1, :].cpu().numpy()[0]
                    next_state[100] = recovered_val

                    real_values.append(real_val)
                    noisy_values.append(noisy_val)
                    recovered_values.append(recovered_val)
                else:
                    next_state = self.add_obs_noise(next_state)
            
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
            
            #memory.add(state, action, reward, next_state, done_mask)
            memory.add(state, ver_actions_values, reward, next_state, done_mask)
            if env_name == "source":
                self.source_step += 1
            elif env_name == "target":
                self.target_step += 1
            
            n_steps += 1
            total_rewards += reward
            state = next_state
        
        # (Optional: plotting code can be added here)
        return total_rewards, n_steps


# ============================================================
# DARC_two: Specializes BaseDARC for handling two observation indices (e.g. indices 100 and 226).
# ============================================================
class DARC_two(BaseDARC):
    # In this version we handle a pair of indices.
    def add_obs_noise_percent(self, obs):
        obs_noisy = np.copy(obs)
        for idx in [100, 226]:
            if obs_noisy[idx] != 0:
                noise = np.random.normal(0, self.noise_scale)
                obs_noisy[idx] *= (1 + noise)
            else:
                obs_noisy[idx] += np.random.normal(0, self.noise_scale)
        return obs_noisy

    def add_obs_noise(self, obs):
        obs_noisy = np.copy(obs)
        for idx in [100, 226]:
            noise = np.random.normal(0, self.noise_scale) + self.bias
            obs_noisy[idx] = obs_noisy[idx] + noise
        return obs_noisy

    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        # Extra replay arguments.
        t_states, t_actions, _, t_next_states, _, game_count = args
        if not torch.is_tensor(s_states):
            s_states = torch.as_tensor(s_states, dtype=torch.float32).to(self.device)
            s_actions = torch.as_tensor(s_actions, dtype=torch.float32).to(self.device)
            s_rewards = torch.as_tensor(s_rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
            s_next_states = torch.as_tensor(s_next_states, dtype=torch.float32).to(self.device)
            s_done_masks = torch.as_tensor(s_done_masks[:, np.newaxis], dtype=torch.float32).to(self.device)
            t_states = torch.as_tensor(t_states, dtype=torch.float32).to(self.device)
            t_actions = torch.as_tensor(t_actions, dtype=torch.float32).to(self.device)
            t_next_states = torch.as_tensor(t_next_states, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Extract raw values at indices 100 and 226.
            s_recovered = s_states[:, [100, 226]]
            s_next_recovered = s_next_states[:, [100, 226]]
            t_recovered = t_states[:, [100, 226]]
            
            sa_inputs = torch.cat([s_recovered, s_actions], dim=1)
            sas_inputs = torch.cat([s_recovered, s_actions, s_next_recovered], dim=1)
            t_sa_inputs = torch.cat([t_recovered, t_actions], dim=1)
            t_sas_inputs = torch.cat([t_recovered, t_actions, t_next_states[:, [100, 226]]], dim=1)
            
            sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale_cfrs, sa_inputs, self.device))
            sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale_cfrs, sas_inputs, self.device))
            sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
            sas_log_probs = torch.log(torch.softmax(sas_logits, dim=1) + 1e-12)
            
            if self.use_darc:
                delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
            else:
                delta_r = torch.zeros_like(sas_log_probs[:, 1])
            if game_count >= 2 * self.warmup_games:
                s_rewards = s_rewards + self.delta_r_scale * delta_r.unsqueeze(1)
        
        train_info = ContSAC.train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks)
        
        s_sa_inputs = torch.cat([s_recovered, s_actions], dim=1)
        s_sas_inputs = torch.cat([s_recovered, s_actions, s_next_recovered], dim=1)
        t_sa_inputs = torch.cat([t_recovered, t_actions], dim=1)
        t_sas_inputs = torch.cat([t_recovered, t_actions, t_next_states[:, [100, 226]]], dim=1)
        
        s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale_cfrs, s_sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(s_sas_inputs + gen_noise(self.noise_scale_cfrs, s_sas_inputs, self.device))
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale_cfrs, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale_cfrs, t_sas_inputs, self.device))
        
        loss_function = nn.CrossEntropyLoss()
        label_zero = torch.zeros(t_sa_logits.shape[0], dtype=torch.int64).to(self.device)
        label_one = torch.ones(t_sa_logits.shape[0], dtype=torch.int64).to(self.device)
        
        classify_loss = loss_function(s_sa_logits, label_zero)
        classify_loss += loss_function(t_sa_logits, label_one)
        classify_loss += loss_function(s_sas_logits, label_zero)
        classify_loss += loss_function(t_sas_logits, label_one)
        
        self.sa_classifier_opt.zero_grad()
        self.sas_adv_classifier_opt.zero_grad()
        classify_loss.backward()
        self.sa_classifier_opt.step()
        self.sas_adv_classifier_opt.step()
        
        s_sa_acc = 1 - torch.argmax(s_sa_logits, dim=1).double().mean()
        s_sas_acc = 1 - torch.argmax(s_sas_logits, dim=1).double().mean()
        t_sa_acc = torch.argmax(t_sa_logits, dim=1).double().mean()
        t_sas_acc = torch.argmax(t_sas_logits, dim=1).double().mean()
        
        train_info['Loss/Classify Loss'] = classify_loss
        train_info['Stats/Avg Delta Reward'] = delta_r.mean()
        train_info['Stats/Avg Source SA Acc'] = s_sa_acc
        train_info['Stats/Avg Source SAS Acc'] = s_sas_acc
        train_info['Stats/Avg Target SA Acc'] = t_sa_acc
        train_info['Stats/Avg Target SAS Acc'] = t_sas_acc
        return train_info

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
            state = self.running_mean(state)
        
        if env_name == "target":
            # For indices 100 and 226, first apply noise to state.
            real_val_100 = state[100]
            real_val_226 = state[226]
            noisy_state = self.add_obs_noise(state)
            noisy_val_100 = noisy_state[100]
            noisy_val_226 = noisy_state[226]
            if self.use_denoiser:
                # Initialize per-index accumulation buffers.
                accumulated_sequence_100 = [noisy_val_100]
                accumulated_sequence_226 = [noisy_val_226]
                acc_tensor_100 = torch.tensor(accumulated_sequence_100, dtype=torch.float32)\
                                    .unsqueeze(0).unsqueeze(-1).to(self.device)
                with torch.no_grad():
                    denoised_seq_100, _ = self.denoiser.forward_online(acc_tensor_100)
                recovered_val_100 = denoised_seq_100[0, -1, :].cpu().numpy()[0]
                acc_tensor_226 = torch.tensor(accumulated_sequence_226, dtype=torch.float32)\
                                    .unsqueeze(0).unsqueeze(-1).to(self.device)
                with torch.no_grad():
                    denoised_seq_226, _ = self.denoiser.forward_online(acc_tensor_226)
                recovered_val_226 = denoised_seq_226[0, -1, :].cpu().numpy()[0]
                # Replace the initial state values.
                state[100] = recovered_val_100
                state[226] = recovered_val_226

                # Optional plotting variables.
                real_values_100 = [real_val_100]
                noisy_values_100 = [noisy_val_100]
                recovered_values_100 = [recovered_val_100]
                real_values_226 = [real_val_226]
                noisy_values_226 = [noisy_val_226]
                recovered_values_226 = [recovered_val_226]
            else:
                # If the denoiser is not used, simply update state with noise.
                state = noisy_state

        while not done:
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                action = self.get_action(state, deterministic)
            next_state, reward, done, _, _ = env.step(action)
            if self.if_normalize:
                next_state = self.running_mean(next_state)
            
            if env_name == "target":
                if self.use_denoiser:
                    # Process indices 100 and 226 using the denoiser.
                    real_val_100 = next_state[100]
                    real_val_226 = next_state[226]
                    noisy_next_state = self.add_obs_noise(next_state)
                    noisy_val_100 = noisy_next_state[100]
                    noisy_val_226 = noisy_next_state[226]
                    # Append new noisy measurements to their respective buffers.
                    accumulated_sequence_100.append(noisy_val_100)
                    accumulated_sequence_226.append(noisy_val_226)
                    acc_tensor_100 = torch.tensor(accumulated_sequence_100, dtype=torch.float32)\
                                        .unsqueeze(0).unsqueeze(-1).to(self.device)
                    with torch.no_grad():
                        denoised_seq_100, _ = self.denoiser.forward_online(acc_tensor_100)
                    recovered_val_100 = denoised_seq_100[0, -1, :].cpu().numpy()[0]
                    acc_tensor_226 = torch.tensor(accumulated_sequence_226, dtype=torch.float32)\
                                        .unsqueeze(0).unsqueeze(-1).to(self.device)
                    with torch.no_grad():
                        denoised_seq_226, _ = self.denoiser.forward_online(acc_tensor_226)
                    recovered_val_226 = denoised_seq_226[0, -1, :].cpu().numpy()[0]
                    next_state[100] = recovered_val_100
                    next_state[226] = recovered_val_226

                    real_values_100.append(real_val_100)
                    noisy_values_100.append(noisy_val_100)
                    recovered_values_100.append(recovered_val_100)
                    real_values_226.append(real_val_226)
                    noisy_values_226.append(noisy_val_226)
                    recovered_values_226.append(recovered_val_226)
                else:
                    # Without denoiser, simply add noise.
                    next_state = self.add_obs_noise(next_state)
            
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
            
            #memory.add(state, action, reward, next_state, done_mask)
            memory.add(state, ver_actions_values, reward, next_state, done_mask)

            if env_name == "source":
                self.source_step += 1
            elif env_name == "target":
                self.target_step += 1
            
            n_steps += 1
            total_rewards += reward
            state = next_state
        
        # (Optional: plotting code may be added here)
        return total_rewards, n_steps


    # (eval_src and eval_tgt are inherited from BaseDARC)
    # (save_model and load_model are also inherited)
