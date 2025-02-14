import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from architectures.utils import Model, gen_noise

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
# Import your custom models (e.g. Model, gen_noise) as needed
# from your_module import Model, gen_noise

class DARC(SAC):
    def __init__(self, policy, source_env, target_env, sa_config, sas_config, device, 
                 savefolder, running_mean,
                 memory_size=1e5, warmup_games=50, batch_size=64, lr=0.0001, gamma=0.99,
                 tau=0.003, alpha=0.2, delta_r_scale=1.0, s_t_ratio=10, noise_scale=1.0,
                 decay_rate=0.99, max_steps=200, if_normalize=False, **sac_kwargs):
        """
        Here:
          - `policy` is either a built-in SB3 policy (e.g. 'MlpPolicy') or a custom one.
          - `source_env` is used for SAC training (passed to super().__init__).
          - `target_env` is an additional environment for your DARC-specific logic.
          - `sa_config` and `sas_config` are configurations for your custom classifiers.
          - Additional arguments are used for your custom functionalities.
        """
        # Initialize the SAC agent (this sets up policy, critic, target networks, replay buffer, etc.)
        super(DARC, self).__init__(policy, source_env, learning_rate=lr, gamma=gamma,
                                    tau=tau, ent_coef=alpha, **sac_kwargs)
        
        self.device = device
        self.source_env = source_env
        self.target_env = target_env
        self.savefolder = savefolder
        self.running_mean = running_mean
        self.batch_size = batch_size
        self.warmup_games = warmup_games
        self.max_steps = max_steps
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale
        self.if_normalize = if_normalize
        
        # Replace or extend the default replay buffer if you need two separate ones.
        # Here, we keep the original replay buffer for the source environment (used by SAC)
        # and add a separate one for the target environment.
        self.source_memory = self.replay_buffer  # already created by SAC.__init__
        self.target_memory = ReplayBuffer(memory_size, 
                                          target_env.observation_space, 
                                          target_env.action_space, 
                                          device=device)
        
        # Create your custom classifiers and optimizers
        self.sa_classifier = Model(sa_config).to(self.device)
        self.sa_classifier_opt = Adam(self.sa_classifier.parameters(), lr=lr)
        self.sas_adv_classifier = Model(sas_config).to(self.device)
        self.sas_adv_classifier_opt = Adam(self.sas_adv_classifier.parameters(), lr=lr)
        
        # Setup schedulers (if needed)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.policy.optimizer, step_size=1, gamma=decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic.optimizer, step_size=1, gamma=decay_rate)
        self.scheduler_sa_classifier = torch.optim.lr_scheduler.StepLR(self.sa_classifier_opt, step_size=1, gamma=decay_rate)
        self.scheduler_sas_adv_classifier = torch.optim.lr_scheduler.StepLR(self.sas_adv_classifier_opt, step_size=1, gamma=decay_rate)
        
        self.total_train_steps = 0
        # (Optionally) set up your custom logger here if not using SB3’s built-in logger.

    def step_optim(self):
        # Step your learning rate schedulers.
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_sa_classifier.step()
        self.scheduler_sas_adv_classifier.step()

    def custom_train_step(self, s_batch, t_batch, game_count):
        """
        s_batch: (s_states, s_actions, s_rewards, s_next_states, s_done_masks)
        t_batch: (t_states, t_actions, t_rewards, t_next_states, t_done_masks)
        """
        s_states, s_actions, s_rewards, s_next_states, s_done_masks = s_batch
        t_states, t_actions, _, t_next_states, _ = t_batch
        
        # Convert to tensors if needed (SB3 replay buffer usually returns tensors already)

        # --- Compute custom reward adjustment using your classifiers ---
        with torch.no_grad():
            # Create inputs for the classifiers and add noise
            sa_inputs = torch.cat([s_states, s_actions], dim=1)
            sas_inputs = torch.cat([s_states, s_actions, s_next_states], dim=1)
            sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs, self.device))
            sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs, self.device))
            sa_log_probs = torch.log_softmax(sa_logits, dim=1)
            sas_log_probs = torch.log_softmax(sas_logits + sa_logits, dim=1)
            delta_r = (sas_log_probs[:, 1] - sas_log_probs[:, 0] -
                       sa_log_probs[:, 1] + sa_log_probs[:, 0])
            # Optionally adjust rewards after a warmup period
            if game_count >= 2 * self.warmup_games:
                s_rewards = s_rewards + self.delta_r_scale * delta_r.unsqueeze(1)
        
        # --- Run the SAC update step ---
        # Here, you can either call the SAC update routine directly (for example, using self._train())
        # or replicate part of it. In this example, we assume you have a helper that performs one SAC update.
        sac_train_info = self._train()  # Note: _train() is the internal update routine in SB3
        
        # --- Update your classifiers with a custom loss ---
        s_sa_inputs = torch.cat([s_states, s_actions], dim=1)
        s_sas_inputs = torch.cat([s_states, s_actions, s_next_states], dim=1)
        t_sa_inputs = torch.cat([t_states, t_actions], dim=1)
        t_sas_inputs = torch.cat([t_states, t_actions, t_next_states], dim=1)
        s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale, s_sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(s_sas_inputs + gen_noise(self.noise_scale, s_sas_inputs, self.device))
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs, self.device))
        
        loss_fn = nn.CrossEntropyLoss()
        label_zero = torch.zeros(s_sa_logits.shape[0], dtype=torch.long).to(self.device)
        label_one = torch.ones(t_sa_logits.shape[0], dtype=torch.long).to(self.device)
        classify_loss = loss_fn(s_sa_logits, label_zero)
        classify_loss += loss_fn(t_sa_logits, label_one)
        classify_loss += loss_fn(s_sas_logits, label_zero)
        classify_loss += loss_fn(t_sas_logits, label_one)
        
        self.sa_classifier_opt.zero_grad()
        self.sas_adv_classifier_opt.zero_grad()
        classify_loss.backward()
        self.sa_classifier_opt.step()
        self.sas_adv_classifier_opt.step()
        
        # (Optional) Compute additional metrics, update schedulers, etc.
        self.step_optim()
        
        # Combine SAC training info with your extra logging
        sac_train_info.update({
            'Loss/Classify Loss': classify_loss.item(),
            'Stats/Avg Delta Reward': delta_r.mean().item(),
            # Add any other stats you wish to log.
        })
        return sac_train_info

    def learn(self, total_timesteps, log_interval=100, deterministic=False, **kwargs):
        """
        Custom training loop that interacts with both source and target environments.
        """
        self.policy.train()
        self.critic.train()
        self.sa_classifier.train()
        self.sas_adv_classifier.train()
        
        game_count = 0
        timestep = 0
        while timestep < total_timesteps:
            # --- Simulate the source environment ---
            source_reward, source_steps = self.simulate_env(game_count, env_name="source", deterministic=deterministic)
            # Log source environment stats if desired
            
            # --- Simulate the target environment (at a given ratio) ---
            if game_count < self.warmup_games or game_count % self.s_t_ratio == 0:
                target_reward, target_steps = self.simulate_env(game_count, env_name="target", deterministic=deterministic)
                # Log target environment stats if desired
            
            # --- Once past warmup, perform training updates ---
            if game_count >= self.warmup_games:
                # For simplicity, assume you perform one training update per environment interaction.
                s_batch = self.source_memory.sample(self.batch_size)
                t_batch = self.target_memory.sample(self.batch_size)
                train_info = self.custom_train_step(s_batch, t_batch, game_count)
                # (Optionally) log training info here.
            
            game_count += 1
            timestep += 1  # or update based on steps collected
            
            # (Optional) Save models periodically
            if game_count % 100 == 0:
                print(f"Game {game_count} -- Source Reward: {source_reward}, Target Reward: {target_reward}")
                self.save_model(str(game_count))
        return self

    def simulate_env(self, game_count, env_name, deterministic):
        """
        Simulate one episode on either the source or target environment.
        """
        if env_name == "source":
            env = self.source_env
            memory = self.source_memory
        elif env_name == "target":
            env = self.target_env
            memory = self.target_memory
        else:
            raise ValueError("Unknown env_name")
        
        total_reward = 0
        n_steps = 0
        done = False
        state = env.reset()
        if self.if_normalize:
            state = self.running_mean(state[0])
        while not done and n_steps < self.max_steps:
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                # SB3 agents use `predict` to get an action
                action, _ = self.predict(state, deterministic=deterministic)
            next_state, reward, done, _, _ = env.step(action)
            if self.if_normalize:
                next_state = self.running_mean(next_state)
            done_mask = 1.0 if n_steps == env._max_episode_steps - 1 else float(not done)
            memory.add(state, action, reward, next_state, done_mask)
            total_reward += reward
            state = next_state
            n_steps += 1
        return total_reward, n_steps

    def save_model(self, folder_name):
        path = os.path.join('saved_weights', self.savefolder, folder_name)
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, 'policy.pt'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pt'))
        torch.save(self.sa_classifier.state_dict(), os.path.join(path, 'sa_classifier.pt'))
        torch.save(self.sas_adv_classifier.state_dict(), os.path.join(path, 'sas_adv_classifier.pt'))
        pickle.dump(self.running_mean, open(os.path.join(path, 'running_mean.pkl'), 'wb'))

    def load_model(self, folder_name, device):
        path = os.path.join('saved_weights', self.savefolder, folder_name)
        self.policy.load_state_dict(torch.load(os.path.join(path, 'policy.pt'), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pt'), map_location=device))
        self.sa_classifier.load_state_dict(torch.load(os.path.join(path, 'sa_classifier.pt'), map_location=device))
        self.sas_adv_classifier.load_state_dict(torch.load(os.path.join(path, 'sas_adv_classifier.pt'), map_location=device))
        self.running_mean = pickle.load(open(os.path.join(path, 'running_mean.pkl'), "rb"))
