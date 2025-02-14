import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.policies import ActorCriticPolicy

# === Example: a dummy Model class for the extra networks ===
# (Replace with your actual implementation from architectures.utils)
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # For example, a simple one-layer network:
        self.fc = nn.Linear(config['input_dim'], config['output_dim'])
    def forward(self, x):
        return self.fc(x)

# A helper to add noise to inputs (as in your original code)
def gen_noise(scale, inputs, device):
    return torch.randn_like(inputs) * scale

# === DARC_SAC: Inherits from SB3's SAC and adds domain-adaptation components ===
class DARC_SAC(SAC):
    def __init__(
        self,
        policy: ActorCriticPolicy,
        env: GymEnv,
        target_env: GymEnv,
        sa_classifier_config: dict,
        sas_classifier_config: dict,
        learning_rate: float = 3e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_update_interval: int = 1,
        delta_r_scale: float = 1.0,
        s_t_ratio: int = 10,
        noise_scale: float = 1.0,
        **kwargs,
    ):
        # Initialize the base SAC (SB3) with your chosen parameters.
        super(DARC_SAC, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            target_update_interval=target_update_interval,
            **kwargs,
        )
        self.target_env = target_env

        # Domain-adaptation hyperparameters:
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale

        # Initialize extra networks (classifiers) for domain adaptation.
        self.sa_classifier = Model(sa_classifier_config).to(self.device)
        self.sa_classifier_opt = Adam(self.sa_classifier.parameters(), lr=learning_rate)
        self.sas_adv_classifier = Model(sas_classifier_config).to(self.device)
        self.sas_adv_classifier_opt = Adam(self.sas_adv_classifier.parameters(), lr=learning_rate)

        # Create a separate replay buffer for the target environment.
        # (SB3’s SAC already creates self.replay_buffer for the source env.)
        self.target_replay_buffer = ReplayBuffer(
            buffer_size, self.observation_space, self.action_space, device=self.device
        )

    # --- Collecting rollouts from the target environment ---
    def collect_target_rollouts(self, n_steps: int) -> None:
        """
        Collect rollouts from the target environment and store them in the target replay buffer.
        You may need to adjust how you reset the environment and decide when an episode is done.
        """
        env = self.target_env
        state = env.reset()
        done = False
        step = 0
        while step < n_steps:
            # Use the current policy to get an action.
            # (Here we assume your policy has a method sample_action similar to your original code.)
            action, _, _ = self.policy.sample_action(
                torch.as_tensor(state).to(self.device).unsqueeze(0)
            )
            action = action.detach().cpu().numpy()[0]
            next_state, reward, done, info = env.step(action)
            # SB3 typically uses 0.0 for done=False and 1.0 for done=True
            done_mask = 0.0 if done else 1.0
            self.target_replay_buffer.add(state, next_state, action, reward, done_mask)
            state = next_state
            step += 1
            if done:
                state = env.reset()
                done = False

    # --- The core training step with domain adaptation ---
    def _train_darc(self, source_data, target_data):
        """
        Combine a SAC update (using source environment samples) with the extra losses
        computed from both source and target samples for domain adaptation.
        """
        # Unpack source data (from self.replay_buffer)
        src_obs = source_data.observations        # shape: [batch_size, obs_dim]
        src_next_obs = source_data.next_observations
        src_actions = source_data.actions
        src_rewards = source_data.rewards          # shape: [batch_size, 1]
        src_dones = source_data.dones              # shape: [batch_size, 1]

        # Unpack target data (from self.target_replay_buffer)
        tgt_obs = target_data.observations
        tgt_next_obs = target_data.next_observations
        tgt_actions = target_data.actions
        tgt_rewards = target_data.rewards
        tgt_dones = target_data.dones

        # --- Compute extra domain adaptation (DA) losses ---
        # For the classifiers, we form inputs by concatenating state and action,
        # and for the second classifier, also next_state.
        src_sa = torch.cat([src_obs, src_actions], dim=1)
        src_sas = torch.cat([src_obs, src_actions, src_next_obs], dim=1)
        tgt_sa = torch.cat([tgt_obs, tgt_actions], dim=1)
        tgt_sas = torch.cat([tgt_obs, tgt_actions, tgt_next_obs], dim=1)

        # Add noise as in your original code.
        src_sa_noisy = src_sa + gen_noise(self.noise_scale, src_sa, self.device)
        src_sas_noisy = src_sas + gen_noise(self.noise_scale, src_sas, self.device)
        tgt_sa_noisy = tgt_sa + gen_noise(self.noise_scale, tgt_sa, self.device)
        tgt_sas_noisy = tgt_sas + gen_noise(self.noise_scale, tgt_sas, self.device)

        # Get logits from the classifiers.
        sa_logits_src = self.sa_classifier(src_sa_noisy)
        sas_logits_src = self.sas_adv_classifier(src_sas_noisy)
        sa_logits_tgt = self.sa_classifier(tgt_sa_noisy)
        sas_logits_tgt = self.sas_adv_classifier(tgt_sas_noisy)

        # Compute log-softmax probabilities.
        sa_log_probs_src = torch.log_softmax(sa_logits_src, dim=1)
        sas_log_probs_src = torch.log_softmax(sas_logits_src, dim=1)
        sa_log_probs_tgt = torch.log_softmax(sa_logits_tgt, dim=1)
        sas_log_probs_tgt = torch.log_softmax(sas_logits_tgt, dim=1)

        # Compute a “delta reward” based on the difference in log probabilities.
        # (Here we assume a binary classification setting.)
        delta_r = ((sas_log_probs_src[:, 1] - sas_log_probs_src[:, 0]) -
                   (sa_log_probs_src[:, 1] - sa_log_probs_src[:, 0]))
        # Adjust source rewards:
        adjusted_src_rewards = src_rewards + self.delta_r_scale * delta_r.unsqueeze(1)

        # Compute classifier losses (using CrossEntropy).
        loss_fn = nn.CrossEntropyLoss()
        # For source samples, we label them as class 0; for target, as class 1.
        src_labels = torch.zeros(src_sa_noisy.shape[0], dtype=torch.long, device=self.device)
        tgt_labels = torch.ones(tgt_sa_noisy.shape[0], dtype=torch.long, device=self.device)

        loss_sa = loss_fn(sa_logits_src, src_labels) + loss_fn(sa_logits_tgt, tgt_labels)
        loss_sas = loss_fn(sas_logits_src, src_labels) + loss_fn(sas_logits_tgt, tgt_labels)
        classify_loss = loss_sa + loss_sas

        self.sa_classifier_opt.zero_grad()
        self.sas_adv_classifier_opt.zero_grad()
        classify_loss.backward()
        self.sa_classifier_opt.step()
        self.sas_adv_classifier_opt.step()

        # --- Now compute SAC losses using the adjusted source rewards ---
        # (You may wish to integrate your adjusted rewards into the SAC loss computation.
        #  One option is to re-run a SAC loss update using the source samples with adjusted_src_rewards.)
        #
        # For illustration, we assume a helper function _compute_sac_loss exists:
        sac_loss_info = self._compute_sac_loss(src_obs, src_actions, adjusted_src_rewards, src_next_obs, src_dones)
        #
        # Combine the losses for logging.
        loss_info = {'classify_loss': classify_loss.item(), **sac_loss_info}
        return loss_info

    def _compute_sac_loss(self, obs, actions, rewards, next_obs, dones):
        """
        This is a placeholder for computing the standard SAC losses (Q and policy losses).
        In a full integration you would mirror the loss computations from SB3’s SAC.
        """
        # For brevity, we return an empty dict here.
        # In practice, you’d compute the temporal-difference (TD) loss for Q-networks
        # and the policy loss (including the entropy term) exactly as in SAC.
        return {'sac_loss': 0.0}

    # --- Override the training loop ---
    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """
        Here we override the training loop so that we can sample from both the source and target replay buffers,
        apply the domain adaptation loss, and then update the SAC policy/Q-networks.
        """
        # (Optional) Every s_t_ratio timesteps, collect target rollouts.
        if self.num_timesteps % self.s_t_ratio == 0:
            self.collect_target_rollouts(n_steps=batch_size)

        # Ensure both buffers have enough samples.
        if self.replay_buffer.pos < batch_size or self.target_replay_buffer.pos < batch_size:
            return

        for gradient_step in range(gradient_steps):
            # Sample from source replay buffer.
            source_data = self.replay_buffer.sample(batch_size)
            # Sample from target replay buffer.
            target_data = self.target_replay_buffer.sample(batch_size)

            # Run the domain adaptation training step.
            loss_info = self._train_darc(source_data, target_data)

            # Now perform the standard SAC update.
            # (This example simply calls the base class train() for one step.
            #  In practice you might merge the two loss updates more tightly.)
            super().train(1, batch_size)

            # Optionally, log loss_info to your logger.

    # --- Saving and Loading Models ---
    def save_model(self, folder_name: str) -> None:
        path = os.path.join('saved_weights', folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), os.path.join(path, 'policy.pt'))
        torch.save(self.q_net.state_dict(), os.path.join(path, 'q_net.pt'))
        torch.save(self.sa_classifier.state_dict(), os.path.join(path, 'sa_classifier.pt'))
        torch.save(self.sas_adv_classifier.state_dict(), os.path.join(path, 'sas_adv_classifier.pt'))
        pickle.dump(self.running_mean, open(os.path.join(path, 'running_mean.pkl'), 'wb'))

    def load_model(self, folder_name: str, device) -> None:
        path = os.path.join('saved_weights', folder_name)
        self.policy.load_state_dict(torch.load(os.path.join(path, 'policy.pt'), map_location=device))
        self.q_net.load_state_dict(torch.load(os.path.join(path, 'q_net.pt'), map_location=device))
        self.sa_classifier.load_state_dict(torch.load(os.path.join(path, 'sa_classifier.pt'), map_location=device))
        self.sas_adv_classifier.load_state_dict(torch.load(os.path.join(path, 'sas_adv_classifier.pt'), map_location=device))
        self.running_mean = pickle.load(open(os.path.join(path, 'running_mean.pkl'), 'rb'))
