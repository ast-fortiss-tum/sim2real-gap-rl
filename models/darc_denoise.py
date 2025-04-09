import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import time
import pickle

from architectures.utils import Model, gen_noise
from replay_buffer import ReplayBuffer
from models.sac import ContSAC, set_global_seed

# -------------------------------------------
# Assume your already trained online autoencoder (denoiser)
# -------------------------------------------
class OnlineDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1):
        super(OnlineDenoisingAutoencoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, proj_dim)
        self.lstm = nn.LSTM(input_size=proj_dim, hidden_size=lstm_hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.output_linear = nn.Linear(lstm_hidden_dim, input_dim)
        
    def forward_online(self, x_seq, hidden=None):
        # x_seq: (batch, T, input_dim); T can vary (T>=1).
        batch, T, _ = x_seq.size()
        outputs = []
        for t in range(T):
            x_t = x_seq[:, t, :]             # (batch, input_dim)
            x_proj = self.input_linear(x_t)    # (batch, proj_dim)
            x_proj = x_proj.unsqueeze(1)       # (batch, 1, proj_dim)
            out, hidden = self.lstm(x_proj, hidden)  # (batch, 1, lstm_hidden_dim)
            out = self.output_linear(out.squeeze(1)) # (batch, input_dim)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, T, input_dim)
        return outputs, hidden

    def forward(self, x_seq):
        outputs, _ = self.forward_online(x_seq)
        return outputs

# -------------------------------------------
# DenoisingDataset (unchanged)
# -------------------------------------------
class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, clean_file, noisy_file):
        # Load the datasets saved as .npy files (lists of episodes)
        self.clean_data = np.load(clean_file, allow_pickle=True)
        self.noisy_data = np.load(noisy_file, allow_pickle=True)
        self.n = len(self.clean_data)
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Get episode from clean and noisy dataset.
        x_clean = self.clean_data[idx]
        x_noisy = self.noisy_data[idx]
        # Expand dimensions if needed.
        if len(x_clean.shape) == 1:
            x_clean = np.expand_dims(x_clean, -1)
        if len(x_noisy.shape) == 1:
            x_noisy = np.expand_dims(x_noisy, -1)
        # Convert to float32.
        x_clean = x_clean.astype(np.float32)
        x_noisy = x_noisy.astype(np.float32)
        # Here we assume each episode has a fixed length; e.g., 25 timesteps.
        # (Note: Although index 100 is used in your training code, this dataset is independent.)
        assert x_clean.shape[0] == 25, f"Expected episode length 25, got {x_clean.shape[0]}"
        assert x_noisy.shape[0] == 25, f"Expected episode length 25, got {x_noisy.shape[0]}"
        return torch.tensor(x_noisy), torch.tensor(x_clean)

# -------------------------------------------
# Modified DARC class using online denoiser with accumulating buffer.
# -------------------------------------------
class DARC(ContSAC):
    def __init__(self, policy_config, value_config, sa_config, sas_config, source_env,
                 target_env, device, running_mean, log_dir="", memory_size=1e5, warmup_games=50,
                 batch_size=64, lr=0.0001, gamma=0.99, tau=0.003, alpha=0.2, ent_adj=False,
                 delta_r_scale=1.0, s_t_ratio=10, noise_scale=1.0, target_update_interval=1,
                 n_games_til_train=1, n_updates_per_train=1, decay_rate=0.99, max_steps=200,
                 if_normalize=True, print_on=False, seed=42):
        if seed is not None:
            set_global_seed(seed)
        
        super(DARC, self).__init__(policy_config, value_config, source_env, device, log_dir,
                                     running_mean, noise_scale, memory_size, warmup_games, batch_size,
                                     lr, gamma, tau, alpha, ent_adj, target_update_interval,
                                     n_games_til_train, n_updates_per_train, max_steps)
        
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale

        self.source_env = source_env
        self.target_env = target_env
        self.warmup_games = warmup_games
        self.n_games_til_train = n_games_til_train

        self.sa_classifier = Model(sa_config).to(self.device)
        self.sa_classifier_opt = Adam(self.sa_classifier.parameters(), lr=lr)
        self.sas_adv_classifier = Model(sas_config).to(self.device)
        self.sas_adv_classifier_opt = Adam(self.sas_adv_classifier.parameters(), lr=lr)
        self.running_mean = running_mean
        self.max_steps = max_steps
        self.if_normalize = if_normalize    

        self.source_step = 0
        self.target_step = 0
        self.source_memory = ReplayBuffer(self.memory_size, self.batch_size)
        self.target_memory = ReplayBuffer(self.memory_size, self.batch_size)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.policy_opt, step_size=1, gamma=decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.twin_q_opt, step_size=1, gamma=decay_rate)
        self.scheduler_sa_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sa_classifier_opt, step_size=1, gamma=decay_rate)
        self.scheduler_sas_adv_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sas_adv_classifier_opt, step_size=1, gamma=decay_rate)

        self.print_on = print_on

        # Load the already trained online autoencoder as your denoiser.
        self.denoiser = OnlineDenoisingAutoencoder(input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1).to(self.device)
        self.denoiser.load_state_dict(torch.load("best_online_denoising_autoencoder.pth", map_location=self.device))
        self.denoiser.eval()

    def add_obs_noise(self, obs):
        """Adds Gaussian noise to the observation."""
        return obs + np.random.normal(0, self.noise_scale, size=obs.shape)

    def step_optim(self):
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_sa_classifier_opt.step()
        self.scheduler_sas_adv_classifier_opt.step()

    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        # The additional arguments from replay: t_states, t_actions, _, t_next_states, _, game_count.
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
            # For this example, the state vector is not time dependent (apart from the special feature at index 100).
            # We now accumulate state[100] over the episode.
            # In train_step, we process a batch of transitions where each transition’s state includes the
            # history of the special feature. For simplicity here, assume s_states[:, 100] already contains
            # the current buffered value.
            s_input = s_states[:, 100].unsqueeze(1)  # (batch, 1) the current buffered (accumulated) noisy value.
            s_seq = s_input.unsqueeze(1)             # (batch, 1, 1); note that in training you use one sample at a time.
            s_denoised, _ = self.denoiser.forward_online(s_seq)
            s_recovered = s_denoised[:, 0, :]          # (batch, 1)
            
            s_next_input = s_next_states[:, 100].unsqueeze(1)
            s_next_seq = s_next_input.unsqueeze(1)
            s_next_denoised, _ = self.denoiser.forward_online(s_next_seq)
            s_next_recovered = s_next_denoised[:, 0, :]
            
            t_input = t_states[:, 100].unsqueeze(1)
            t_seq = t_input.unsqueeze(1)
            t_denoised, _ = self.denoiser.forward_online(t_seq)
            t_recovered = t_denoised[:, 0, :]
            
            # Build classifier inputs using the recovered (denoised) values.
            sa_inputs = torch.cat([
                s_recovered,
                s_actions[:, 0].unsqueeze(1)
            ], dim=1)
            sas_inputs = torch.cat([
                s_recovered,
                s_actions[:, 0].unsqueeze(1),
                s_next_recovered
            ], dim=1)
            t_sa_inputs = torch.cat([
                t_recovered,
                t_actions[:, 0].unsqueeze(1)
            ], dim=1)
            t_sas_inputs = torch.cat([
                t_recovered,
                t_actions[:, 0].unsqueeze(1),
                t_next_states[:, 100].unsqueeze(1)  # raw value for next state (or denoise similarly if desired)
            ], dim=1)
            
            sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs, self.device))
            sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs, self.device))
            sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
            sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)
            delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
            if game_count >= 2 * self.warmup_games:
                s_rewards = s_rewards + self.delta_r_scale * delta_r.unsqueeze(1)

        train_info = super(DARC, self).train_step(s_states, s_actions, s_rewards, s_next_states, s_done_masks)
        # Rebuild classifier inputs (using the recovered values computed above)
        s_sa_inputs = torch.cat([s_recovered, s_actions[:, 0].unsqueeze(1)], dim=1)
        s_sas_inputs = torch.cat([s_recovered, s_actions[:, 0].unsqueeze(1), s_next_recovered], dim=1)
        t_sa_inputs = torch.cat([t_recovered, t_actions[:, 0].unsqueeze(1)], dim=1)
        t_sas_inputs = torch.cat([t_recovered, t_actions[:, 0].unsqueeze(1), t_next_states[:, 100].unsqueeze(1)], dim=1)
        
        s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale, s_sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(s_sas_inputs + gen_noise(self.noise_scale, s_sas_inputs, self.device))
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs, self.device))
        
        loss_function = nn.CrossEntropyLoss()
        label_zero = torch.zeros((s_sa_logits.shape[0],), dtype=torch.int64).to(self.device)
        label_one = torch.ones((t_sa_logits.shape[0],), dtype=torch.int64).to(self.device)
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
        # Get initial observation and process it.
        state = env.reset()[0]
        if self.if_normalize:
            state = self.running_mean(state[0])
        state = self.add_obs_noise(state)
        
        # Initialize an accumulating buffer for the special feature at index 100.
        accumulated_sequence = []
        accumulated_sequence.append(state[100])
        
        while not done:
            # Build the accumulated sequence tensor: shape (1, L, 1), where L is the current length.
            current_length = len(accumulated_sequence)
            acc_tensor = torch.tensor(accumulated_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                denoised_seq, _ = self.denoiser.forward_online(acc_tensor)
            # Extract the recovered value from the last element of the denoised sequence.
            recovered_value = denoised_seq[0, -1, :].cpu().numpy()[0]
            # Replace the noisy feature in the state with the recovered value.
            state[100] = recovered_value
            
            # Action selection.
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                action = self.get_action(state, deterministic)
            
            next_state, reward, done, _, _ = env.step(action)
            if self.if_normalize:
                next_state = self.running_mean(next_state)
            next_state = self.add_obs_noise(next_state)
            # Append the new noisy special feature from next_state.
            accumulated_sequence.append(next_state[100])
            
            done_mask = 1.0 if n_steps == env._max_episode_steps - 1 else 0.0
            if n_steps == self.max_steps:
                done = True
            
            memory.add(state, None, reward, next_state, done_mask)
            if env_name == "source":
                self.source_step += 1
            elif env_name == "target":
                self.target_step += 1
            n_steps += 1
            total_rewards += reward
            state = next_state
        return total_rewards, n_steps

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
                    # self.save_model(str(i))
            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))
            
    # You can include eval_src(), eval_tgt(), save_model(), load_model(), etc. as in your original class.

    def eval_src(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            state = self.source_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state[0])
            state = self.add_obs_noise(state)  # <-- Add noise
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
                next_state = self.add_obs_noise(next_state)  # <-- Add noise
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
                state = self.running_mean(state[0])
            state = self.add_obs_noise(state)  # <-- Add noise
            done = False
            total_reward = 0
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state, deterministic=False)
                next_state, reward, done, _, _ = self.target_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                next_state = self.add_obs_noise(next_state)  # <-- Add noise
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
        super(DARC, self).load_model(folder_name, device)
        path = 'saved_weights/' + folder_name
        self.sa_classifier.load_state_dict(torch.load(path + '/sa_classifier', map_location=torch.device(device)))
        self.sas_adv_classifier.load_state_dict(torch.load(path + '/sas_adv_classifier', map_location=torch.device(device)))
        self.running_mean = pickle.load(open(path + '/running_mean', "rb"))
