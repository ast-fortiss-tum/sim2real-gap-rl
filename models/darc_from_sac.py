import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
from torch.optim import Adam

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.buffers import ReplayBuffer  # SB3 standard replay buffer
from stable_baselines3.common.callbacks import EventCallback

# Import your custom modules:
#from architectures.gaussian_policy import ContGaussianPolicy
from architectures.utils import Model, gen_noise

class DARC_SB3(SAC):
    """
    DARCSAC extends Stable-Baselines3 SAC to incorporate domain adaptation components.
    It uses SB3's standard ReplayBuffer for both the source and target environments.
    """
    def __init__(self,
                 policy: str,  # e.g., a string specifying your custom policy if registered
                 env: GymEnv,
                 target_env: GymEnv,
                 sa_config: dict,
                 sas_config: dict,
                 delta_r_scale: float = 1.0,
                 s_t_ratio: int = 10,
                 noise_scale: float = 1.0,
                 n_steps_buffer: int = 1000,
                 **kwargs):
        """
        :param policy: Policy identifier (e.g. your custom policy, such as one using ContGaussianPolicy)
        :param env: Source environment used for training.
        :param target_env: Target environment for collecting additional rollouts.
        :param sa_config: Configuration dictionary for the state–action classifier.
        :param sas_config: Configuration dictionary for the state–action–next_state classifier.
        :param delta_r_scale: Scale factor for the reward correction term.
        :param s_t_ratio: Ratio controlling how often to collect target rollouts.
        :param noise_scale: Scale of the noise added to classifier inputs.
        :param kwargs: Additional keyword arguments for SAC (learning_rate, buffer_size, batch_size, etc.)
        """
        super(DARC_SB3, self).__init__(policy, env, **kwargs)

        # Initialize required buffers for episode info and success metrics:
        self.ep_info_buffer = []       # for episode info (e.g., rewards, lengths)
        self.ep_success_buffer = []    # for episode success info

        self.target_env = target_env
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale

        # Create a target replay buffer using SB3's standard ReplayBuffer.
        self.target_buffer = ReplayBuffer(
            buffer_size=int(kwargs.get("buffer_size", 1e5)),
            observation_space=target_env.observation_space,
            action_space=target_env.action_space,
            device=self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        # Initialize the classifiers using your Model class.
        self.sa_classifier = Model(sa_config).to(self.device)
        self.sa_classifier_opt = Adam(self.sa_classifier.parameters(), lr=self.learning_rate)
        self.sas_adv_classifier = Model(sas_config).to(self.device)
        self.sas_adv_classifier_opt = Adam(self.sas_adv_classifier.parameters(), lr=self.learning_rate)

        self.train_game_count = 0

    def update_classifiers(self, source_batch, target_batch):
        """
        Update the domain classifiers using samples from the source and target replay buffers.
        The SB3 ReplayBuffer sample returns a dictionary with keys like:
        "observations", "actions", "next_observations", etc.
        """
        # Convert source batch entries to tensors.
    
        s_states = torch.as_tensor(source_batch.observations, dtype=torch.float32).to(self.device)
        s_actions = torch.as_tensor(source_batch.actions, dtype=torch.float32).to(self.device)
        s_next_states = torch.as_tensor(source_batch.next_observations, dtype=torch.float32).to(self.device)

        # Convert target batch entries to tensors.
        t_states = torch.as_tensor(target_batch.observations, dtype=torch.float32).to(self.device)
        t_actions = torch.as_tensor(target_batch.actions, dtype=torch.float32).to(self.device)
        t_next_states = torch.as_tensor(target_batch.next_observations, dtype=torch.float32).to(self.device)

        # Build classifier inputs and add noise.
        sa_inputs = torch.cat([s_states, s_actions], dim=1)
        sas_inputs = torch.cat([s_states, s_actions, s_next_states], dim=1)
        sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs, self.device))
        sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs, self.device))
        sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
        sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)

        # Compute delta reward correction.
        delta_r = (sas_log_probs[:, 1] - sas_log_probs[:, 0] -
                   sa_log_probs[:, 1] + sa_log_probs[:, 0])

        # Prepare classifier training using source and target samples.
        s_sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs, self.device))
        t_sa_inputs = torch.cat([t_states, t_actions], dim=1)
        t_sas_inputs = torch.cat([t_states, t_actions, t_next_states], dim=1)
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs, self.device))

        loss_function = nn.CrossEntropyLoss()
        # Use label 0 for source and label 1 for target.
        label_source = torch.zeros(s_sa_logits.shape[0], dtype=torch.long, device=self.device)
        label_target = torch.ones(t_sa_logits.shape[0], dtype=torch.long, device=self.device)

        classify_loss = loss_function(s_sa_logits, label_source)
        classify_loss += loss_function(t_sa_logits, label_target)
        classify_loss += loss_function(s_sas_logits, label_source)
        classify_loss += loss_function(t_sas_logits, label_target)

        # Backpropagation.
        self.sa_classifier_opt.zero_grad()
        self.sas_adv_classifier_opt.zero_grad()
        classify_loss.backward()
        self.sa_classifier_opt.step()
        self.sas_adv_classifier_opt.step()

        return delta_r.mean(), classify_loss

    def collect_target_rollout(self):
        """
        Collect a single rollout (episode) from the target environment and add it to the target buffer.
        SB3's ReplayBuffer.add() expects: obs, next_obs, action, reward, done, infos.
        """
        obs, _ = self.target_env.reset()
        done = False
        while not done:
            action, _ = self.policy.predict(obs, deterministic=False)
            next_obs, reward, done, trunc ,info = self.target_env.step(action)
            self.target_buffer.add(obs, next_obs, action, reward, done, info)
            obs = next_obs

    def warmup(self, warmup_steps: int):
        """
        Run a few steps with a random policy to initialize self._last_obs and fill the replay buffer.
        """
        obs = self.env.reset()
        #print(obs)
        self._last_obs = obs  # Initialize _last_obs so that collect_rollouts has a starting point.
        for _ in range(warmup_steps):
            action = self.env.action_space.sample()
            action_1 = np.array([[action[0]]])
            #print(np.array([[action[0]]]))
            new_obs, reward, done ,info = self.env.step(action_1)
            # Add the transition to the replay buffer
            #print(action)
            self.replay_buffer.add(obs, new_obs, action, reward, done, info)
            if done:
                obs = self.env.reset()
            else:
                obs = new_obs
        self._last_obs = obs  # Ensure the last observation is set after warmup.

    def train(self, total_timesteps: int, log_interval: int = 100, **kwargs):
        """
        Overrides the SAC training loop to include target rollout collection and extra classifier updates.
        """
        # Warmup phase: collect a few transitions to initialize _last_obs and the replay buffer.
        #warmup_steps = 24  # or another suitable number
        #self.warmup(warmup_steps)

        train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)
        callback = EventCallback()
        callback.init_callback(self)

        n_rollout_steps = self.env.get_attr('_max_episode_steps')[0]
        self.n_steps = n_rollout_steps
        self._setup_learn(total_timesteps =total_timesteps)
        
    
        timesteps = 0
        while timesteps < total_timesteps:
            # Collect rollouts from the source environment using SAC's standard procedure.
            # or any other desired value

            super().collect_rollouts(self.env, callback=callback, train_freq= train_freq, replay_buffer=self.replay_buffer, action_noise=None)
          
            # Periodically collect a rollout from the target environment.
            if self.train_game_count % self.s_t_ratio == 0:
         
                self.collect_target_rollout()

            # Perform SAC updates.
            for _ in range(self.gradient_steps):
      
                source_batch = self.replay_buffer.sample(self.batch_size)
     
                sac_info = self._update(source_batch)  # Standard SAC update.
      

                # If the target buffer has enough samples, update the classifiers.
                if self.target_buffer.pos >= self.batch_size:
                    target_batch = self.target_buffer.sample(self.batch_size)
                    delta_r, classify_loss = self.update_classifiers(source_batch, target_batch)
                    self.logger.record("darc/delta_r", delta_r.item())
                    self.logger.record("darc/classify_loss", classify_loss.item())


            timesteps += self.n_steps
            self.train_game_count += 1
            if self.train_game_count % log_interval == 0:
                self.logger.dump(self.train_game_count)

            print("Timesteps: ", timesteps) 
            print("Train Game Count: ", self.train_game_count)
            print("n.steps: ", self.n_steps)

        return self

    def _update(self, sampler):
        """
        Optionally modify the sampled batch before the SAC update.
        For now, we simply call the parent SAC _update method.
        """
        #return super()._update(sampler)
        return super().train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
    
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["target_env", "target_buffer", "sa_classifier", "sas_adv_classifier"]

    def save(self, path: str):
        """
        Save both SAC components and the classifier networks.
        """
        super().save(path)
        torch.save(self.sa_classifier.state_dict(), path + "_sa_classifier.pt")
        torch.save(self.sas_adv_classifier.state_dict(), path + "_sas_adv_classifier.pt")

    def load(self, path: str):
        """
        Load SAC components and the classifier networks.
        """
        super().load(path)
        self.sa_classifier.load_state_dict(torch.load(path + "_sa_classifier.pt", map_location=self.device))
        self.sas_adv_classifier.load_state_dict(torch.load(path + "_sas_adv_classifier.pt", map_location=self.device))
