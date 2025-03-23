import os
import time
import pickle
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
from architectures.utils import Model, gen_noise  # your custom model and noise generator

class DARC_SB3(SAC):
    def __init__(self, policy, env, sa_config, sas_config,
                 source_env, target_env, device, savefolder, running_mean,
                 writer, buffer_size=int(1e5), batch_size=64, lr=0.0001, gamma=0.99, tau=0.003,
                 alpha=0.2, s_t_ratio=10, noise_scale=1.0, warmup_games=50, n_games_til_train=1,
                 n_updates_per_train=1, decay_rate=0.99, max_steps=200, if_normalize=False, print_on=False,
                 **kwargs):
        # Initialize the SB3 SAC agent
        super(DARC_SB3, self).__init__(policy, env, learning_rate=lr, gamma=gamma, tau=tau,
                                        batch_size=batch_size, ent_coef=alpha, **kwargs)
        self.device = device
        self.savefolder = savefolder
        self.running_mean = running_mean
        self.writer = writer  # Logging object (e.g., TensorBoard writer)

        # Environments & training settings
        self.source_env = source_env
        self.target_env = target_env
        self.warmup_games = warmup_games
        self.n_games_til_train = n_games_til_train
        self.n_updates_per_train = n_updates_per_train
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale
        self.max_steps = max_steps
        self.if_normalize = if_normalize
        self.print_on = print_on

        # Extra replay buffers (for source and target experience)
        self.source_buffer = SB3ReplayBuffer(buffer_size, self.observation_space, self.action_space, device=self.device)
        self.target_buffer = SB3ReplayBuffer(buffer_size, self.observation_space, self.action_space, device=self.device)

        # Extra classifier networks and their optimizers.
        self.sa_classifier = Model(sa_config).to(self.device)
        self.sa_classifier_opt = Adam(self.sa_classifier.parameters(), lr=lr)
        self.sas_adv_classifier = Model(sas_config).to(self.device)
        self.sas_adv_classifier_opt = Adam(self.sas_adv_classifier.parameters(), lr=lr)

        # Optionally add learning rate schedulers.
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.policy.optimizer, step_size=1, gamma=decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic.optimizer, step_size=1, gamma=decay_rate)
        self.scheduler_sa_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sa_classifier_opt, step_size=1, gamma=decay_rate)
        self.scheduler_sas_adv_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sas_adv_classifier_opt, step_size=1, gamma=decay_rate)

    def step_optim(self):
        """Step all learning rate schedulers."""
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_sa_classifier_opt.step()
        self.scheduler_sas_adv_classifier_opt.step()

    def custom_train_step(self, source_batch, target_batch, game_count):
        """
        Perform one training step that:
         1. Computes delta rewards (for logging)
         2. Updates the classifier networks
        """
        # Unpack source and target batches.
        s_states, s_actions, s_rewards, s_next_states, _ = source_batch
        t_states, t_actions, _, t_next_states, _ = target_batch

        # Ensure data is on the proper device.
        s_states = torch.as_tensor(s_states, dtype=torch.float32, device=self.device)
        s_actions = torch.as_tensor(s_actions, dtype=torch.float32, device=self.device)
        s_next_states = torch.as_tensor(s_next_states, dtype=torch.float32, device=self.device)
        t_states = torch.as_tensor(t_states, dtype=torch.float32, device=self.device)
        t_actions = torch.as_tensor(t_actions, dtype=torch.float32, device=self.device)
        t_next_states = torch.as_tensor(t_next_states, dtype=torch.float32, device=self.device)

        # Compute delta reward (for logging and potential reward modification)
        with torch.no_grad():
            sa_inputs = torch.cat([
                s_states[:, 100].unsqueeze(1),
                s_actions[:, 0].unsqueeze(1)
            ], dim=1)
            sas_inputs = torch.cat([
                s_states[:, 100].unsqueeze(1),
                s_actions[:, 0].unsqueeze(1),
                s_next_states[:, 100].unsqueeze(1)
            ], dim=1)
            sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs, self.device))
            sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs, self.device))
            sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
            sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)
            delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]

        # Classifier update: prepare inputs using the 100-th element.
        s_sa_inputs = torch.cat([
            s_states[:, 100].unsqueeze(1),
            s_actions[:, 0].unsqueeze(1)
        ], dim=1)
        s_sas_inputs = torch.cat([
            s_states[:, 100].unsqueeze(1),
            s_actions[:, 0].unsqueeze(1),
            s_next_states[:, 100].unsqueeze(1)
        ], dim=1)
        t_sa_inputs = torch.cat([
            t_states[:, 100].unsqueeze(1),
            t_actions[:, 0].unsqueeze(1)
        ], dim=1)
        t_sas_inputs = torch.cat([
            t_states[:, 100].unsqueeze(1),
            t_actions[:, 0].unsqueeze(1),
            t_next_states[:, 100].unsqueeze(1)
        ], dim=1)

        # Forward pass (with noise)
        s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale, s_sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(s_sas_inputs + gen_noise(self.noise_scale, s_sas_inputs, self.device))
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs, self.device))

        # Compute cross-entropy loss.
        loss_function = nn.CrossEntropyLoss()
        label_zero = torch.zeros(s_sa_logits.shape[0], dtype=torch.int64, device=self.device)
        label_one = torch.ones(t_sa_logits.shape[0], dtype=torch.int64, device=self.device)
        classify_loss = loss_function(s_sa_logits, label_zero)
        classify_loss += loss_function(t_sa_logits, label_one)
        classify_loss += loss_function(s_sas_logits, label_zero)
        classify_loss += loss_function(t_sas_logits, label_one)

        # Backpropagate classifier loss.
        self.sa_classifier_opt.zero_grad()
        self.sas_adv_classifier_opt.zero_grad()
        classify_loss.backward()
        self.sa_classifier_opt.step()
        self.sas_adv_classifier_opt.step()

        # Compute simple accuracies for logging.
        s_sa_acc = 1 - torch.argmax(s_sa_logits, dim=1).double().mean()
        s_sas_acc = 1 - torch.argmax(s_sas_logits, dim=1).double().mean()
        t_sa_acc = torch.argmax(t_sa_logits, dim=1).double().mean()
        t_sas_acc = torch.argmax(t_sas_logits, dim=1).double().mean()

        # Gather training info for logging.
        train_info = {
            'Loss/Classify Loss': classify_loss.item(),
            'Stats/Avg Delta Reward': delta_r.mean().item(),
            'Stats/Avg Source SA Acc': s_sa_acc.item(),
            'Stats/Avg Source SAS Acc': s_sas_acc.item(),
            'Stats/Avg Target SA Acc': t_sa_acc.item(),
            'Stats/Avg Target SAS Acc': t_sas_acc.item()
        }
        return train_info

    def custom_learn(self, total_timesteps):
        """
        Custom learning loop that alternates environment rollouts with SAC and classifier updates.
        Classifier updates only begin once both replay buffers have at least one batch of data.
        """
        game_count = 0
        while self.num_timesteps < total_timesteps:
            # --- Rollout in Source Environment ---
            source_reward, source_steps = self.simulate_env(self.source_env, self.source_buffer, game_count, env_name="source")
            # Log source environment info.
            self.writer.add_scalar('Source Env/Rewards', source_reward, game_count)
            self.writer.add_scalar('Source Env/N_Steps', source_steps, game_count)
            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(game_count, source_steps, source_reward))

            # --- Rollout in Target Environment (at a given ratio) ---
            if game_count % self.s_t_ratio == 0:
                target_reward, target_steps = self.simulate_env(self.target_env, self.target_buffer, game_count, env_name="target")
                self.writer.add_scalar('Target Env/Rewards', target_reward, game_count)
                self.writer.add_scalar('Target Env/N_Steps', target_steps, game_count)
                print("TARGET: index: {}, steps: {}, total_rewards: {}".format(game_count, target_steps, target_reward))

            # --- Training Updates ---
            if game_count >= self.warmup_games:
                # Only update if both buffers have at least one full batch.
                if self.source_buffer.pos >= self.batch_size and self.target_buffer.pos >= self.batch_size:
                    # For every step taken in the source env, perform some updates.
                    for _ in range(source_steps * self.n_updates_per_train):
                        self.num_timesteps += 1  # increment SB3 timesteps counter
                        source_batch = self.source_buffer.sample(self.batch_size)
                        target_batch = self.target_buffer.sample(self.batch_size)
                        train_info = self.custom_train_step(source_batch, target_batch, game_count)
                        self.writer.add_train_step_info(train_info, game_count)
                    self.writer.write_train_step()

            # --- Evaluation and Saving ---
            if game_count % 100 == 0:
                print('Evaluation on Source Env: ', self.eval_src(10))
                print('Evaluation on Target Env: ', self.eval_tgt(10))
                self.save_model(str(game_count))

            # Step the schedulers.
            self.step_optim()
            game_count += 1

    def simulate_env(self, env, buffer, game_count, env_name="source"):
        """
        Simulate one episode in the given environment.
        In each step, log verified actions/penalties and add the transition to the given replay buffer.
        """
        # Reset the environment.
        obs = env.reset()[0]  # assuming reset returns (obs, info)
        if self.if_normalize:
            obs = self.running_mean(obs[0])
        done = False
        total_reward = 0
        n_steps = 0

        while not done:
            if game_count < self.warmup_games:
                action = env.action_space.sample()
            else:
                action, _ = self.predict(obs, deterministic=False)
            next_obs, reward, done, _, _ = env.step(action)
            if self.if_normalize:
                next_obs = self.running_mean(next_obs)
            
            # --- Logging verified actions and penalties ---
            ver_actions = env.env.sys.verified_actions
            ver_penalties = env.env.sys.verification_penalties
            ver_actions_values = [node_data['p']
                                  for agent in ver_actions.values()
                                  for node_data in agent.values()
                                  if 'p' in node_data]
            ver_penalties_values = [ver_penalties.values()]
            ver_actions_values = [value for arr in ver_actions_values for value in arr]
            ver_penalties_values = [value for dv in ver_penalties_values for value in list(dv)]
            print("+++++++++++++++++++++++++++++++++")
            print("VERIFIED ACTIONS VALUES: ", ver_actions_values)
            print("VERIFIED PENALTIES VALUES: ", ver_penalties_values)
            print("SOC: ", obs[100])
            print("next_SOC: ", next_obs[100])
            print("UNVERIFIED ACTION: ", action)
            print("+++++++++++++++++++++++++++++++++")
            # ------------------------------

            # Add the experience to the replay buffer.
            buffer.add(obs, action, reward, next_obs, done)
            total_reward += reward
            n_steps += 1
            obs = next_obs

            if n_steps >= self.max_steps:
                done = True
        return total_reward, n_steps

    def save_model(self, folder_name):
        """Save model weights and running_mean."""
        path = os.path.join('saved_weights', self.savefolder, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), os.path.join(path, 'policy'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic'))
        torch.save(self.sa_classifier.state_dict(), os.path.join(path, 'sa_classifier'))
        torch.save(self.sas_adv_classifier.state_dict(), os.path.join(path, 'sas_adv_classifier'))
        pickle.dump(self.running_mean, open(os.path.join(path, 'running_mean'), 'wb'))

    def load_model(self, folder_name, device):
        """Load model parameters from disk."""
        path = os.path.join('saved_weights', folder_name)
        super(DARC_SB3, self).load_model(folder_name, device)
        self.sa_classifier.load_state_dict(torch.load(os.path.join(path, 'sa_classifier'), map_location=torch.device(device)))
        self.sas_adv_classifier.load_state_dict(torch.load(os.path.join(path, 'sas_adv_classifier'), map_location=torch.device(device)))
        self.running_mean = pickle.load(open(os.path.join(path, 'running_mean'), "rb"))

    def eval_src(self, num_games, render=False):
        """Evaluate performance in the source environment."""
        self.policy.eval()
        self.critic.eval()
        total_reward = 0
        for _ in range(num_games):
            state = self.source_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state[0])
            done = False
            game_reward = 0
            steps = 0
            while not done:
                if render:
                    self.source_env.render()
                action, _ = self.predict(state, deterministic=False)
                next_state, reward, done, _, _ = self.source_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                game_reward += reward
                state = next_state
                steps += 1
                if steps >= self.max_steps:
                    done = True
            total_reward += game_reward
        return total_reward / num_games

    def eval_tgt(self, num_games, render=False):
        """Evaluate performance in the target environment."""
        self.policy.eval()
        self.critic.eval()
        total_reward = 0
        for _ in range(num_games):
            state = self.target_env.reset()[0]
            if self.if_normalize:
                state = self.running_mean(state[0])
            done = False
            game_reward = 0
            steps = 0
            while not done:
                if render:
                    self.target_env.render()
                action, _ = self.predict(state, deterministic=False)
                next_state, reward, done, _, _ = self.target_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                game_reward += reward
                state = next_state
                steps += 1
                if steps >= self.max_steps:
                    done = True
            total_reward += game_reward
        return total_reward / num_games
