import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.buffers import ReplayBuffer  # SB3 standard replay buffer
from stable_baselines3.common.callbacks import BaseCallback  # For optional callbacks
from stable_baselines3.common.logger import Logger, configure  # For logging to TensorBoard

# Import your custom modules:
# from architectures.gaussian_policy import ContGaussianPolicy
from architectures.utils import Model, gen_noise


class DARC_SB3(SAC):
    """
    DARCSAC extends Stable-Baselines3 SAC to incorporate domain adaptation components.
    In this updated version, an episode is ended (i.e. done is set to True) once a fixed
    number of steps (n_steps) have been executed. Classifier updates are carried out at the end
    of each episode. Every n_eval episodes the policy is evaluated on both the source and target
    environments. Additionally, a warmup phase collects random transitions to fill the replay buffer,
    and every s_t_ratio warmup episodes a full target rollout is collected.
    
    The reward is updated (after a minimum number of episodes) using a delta computed by the classifiers.
    """
    def __init__(self,
                 policy: str,
                 env: GymEnv,
                 target_env: GymEnv,
                 sa_config: dict,
                 sas_config: dict,
                 delta_r_scale: float = 1.0,
                 s_t_ratio: int = 10,
                 noise_scale: float = 1.0,
                 n_steps_buffer: int = 1000,
                 warmup_steps: int = 1000,      # Number of warmup steps.
                 log_dir: str = "./logs",       # Base directory for TensorBoard logs.
                 n_eval: int = 100,             # Evaluate the policy every n_eval episodes.
                 save_checkpoint_freq: int = 10000,  # Save a model checkpoint every save_checkpoint_freq timesteps.
                 **kwargs):
        """
        :param policy: Policy identifier.
        :param env: Source environment used for training.
        :param target_env: Target environment for collecting additional rollouts.
        :param sa_config: Configuration for the state–action classifier.
        :param sas_config: Configuration for the state–action–next_state classifier.
        :param delta_r_scale: Scale factor for the reward correction term.
        :param s_t_ratio: Ratio controlling how often to collect target rollouts (in episodes).
        :param noise_scale: Scale of the noise added to classifier inputs.
        :param n_steps_buffer: Buffer length for episode steps.
        :param warmup_steps: Number of initial random steps to fill the replay buffer.
        :param log_dir: Base directory where TensorBoard logs will be saved.
        :param n_eval: Evaluate the policy every n_eval episodes.
        :param kwargs: Additional keyword arguments for SAC.
        """
        super(DARC_SB3, self).__init__(policy, env, **kwargs)

        # Create a unique log directory for this run (using a timestamp).
        unique_log_dir = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S"))
        self.tb_writer = SummaryWriter(unique_log_dir)
        print(f"Logging TensorBoard data to: {unique_log_dir}")

        # Create a directory for checkpoints inside the log folder.
        self.checkpoint_dir = os.path.join(unique_log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        # Buffers for episode info.
        self.ep_info_buffer = []
        self.ep_success_buffer = []

        self.target_env = target_env
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale
        self.save_checkpoint_freq = save_checkpoint_freq 

        # Create a target replay buffer.
        self.target_buffer = ReplayBuffer(
            buffer_size=int(kwargs.get("buffer_size", 1e3)),
            observation_space=target_env.observation_space,
            action_space=target_env.action_space,
            device=self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        # Initialize classifiers.
        self.sa_classifier = Model(sa_config).to(self.device)
        self.sa_classifier_opt = Adam(self.sa_classifier.parameters(), lr=self.learning_rate)
        self.sas_adv_classifier = Model(sas_config).to(self.device)
        self.sas_adv_classifier_opt = Adam(self.sas_adv_classifier.parameters(), lr=self.learning_rate)

        # Episode counters.
        self.num_episodes = 0
        self.current_episode_step = 0

        # For tracking total training steps and cumulative episode rewards.
        self.total_steps = 0
        self.episode_reward = 0.0

        # Evaluation frequency.
        self.n_eval = n_eval

        # Warmup steps before training begins.
        self.warmup_steps = warmup_steps
        # Number of episodes to wait before applying reward correction.
        self.warmup_games = kwargs.get("warmup_games", 10)

        # Fixed episode length. Try to fetch from env; otherwise use default.
        try:
            self.n_steps = self.env.get_attr('_max_episode_steps')[0]
        except Exception:
            self.n_steps = kwargs.get("n_steps", 1000)

        # Store the last observation.
        self._last_obs = None

    def update_classifiers(self, source_batch, target_batch, step):
        """
        Update the domain classifiers using samples from the source and target buffers.
        Logs classifier accuracy and loss to TensorBoard.
        """
        s_states = torch.as_tensor(source_batch.observations, dtype=torch.float32).to(self.device)
        s_actions = torch.as_tensor(source_batch.actions, dtype=torch.float32).to(self.device)
        s_next_states = torch.as_tensor(source_batch.next_observations, dtype=torch.float32).to(self.device)

        t_states = torch.as_tensor(target_batch.observations, dtype=torch.float32).to(self.device)
        t_actions = torch.as_tensor(target_batch.actions, dtype=torch.float32).to(self.device)
        t_next_states = torch.as_tensor(target_batch.next_observations, dtype=torch.float32).to(self.device)

        sa_inputs = torch.cat([s_states, s_actions], dim=1)
        sas_inputs = torch.cat([s_states, s_actions, s_next_states], dim=1)

        # Compute logits with added noise.
        sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs, self.device))
        sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs, self.device))
        sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
        sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)

        delta_r = (sas_log_probs[:, 1] - sas_log_probs[:, 0] -
                   sa_log_probs[:, 1] + sa_log_probs[:, 0])

        # For logging classifier accuracy, recompute logits with noise.
        s_sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs, self.device))
        s_sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs, self.device))
        t_sa_inputs = torch.cat([t_states, t_actions], dim=1)
        t_sas_inputs = torch.cat([t_states, t_actions, t_next_states], dim=1)
        t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs, self.device))
        t_sas_logits = self.sas_adv_classifier(t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs, self.device))

        loss_function = nn.CrossEntropyLoss()
        label_source = torch.zeros(s_sa_logits.shape[0], dtype=torch.long, device=self.device)
        label_target = torch.ones(t_sa_logits.shape[0], dtype=torch.long, device=self.device)

        classify_loss = loss_function(s_sa_logits, label_source)
        classify_loss += loss_function(t_sa_logits, label_target)
        classify_loss += loss_function(s_sas_logits, label_source)
        classify_loss += loss_function(t_sas_logits, label_target)
        
        # Compute classifier accuracies.
        with torch.no_grad():
            s_sa_pred = torch.argmax(s_sa_logits, dim=1)
            t_sa_pred = torch.argmax(t_sa_logits, dim=1)
            sa_source_acc = (s_sa_pred == label_source).float().mean()
            sa_target_acc = (t_sa_pred == label_target).float().mean()

            s_sas_pred = torch.argmax(s_sas_logits, dim=1)
            t_sas_pred = torch.argmax(t_sas_logits, dim=1)
            sas_source_acc = (s_sas_pred == label_source).float().mean()
            sas_target_acc = (t_sas_pred == label_target).float().mean()

        # Log metrics to TensorBoard.
        self.tb_writer.add_scalar("classifier/sa_source_accuracy", sa_source_acc.item(), step)
        self.tb_writer.add_scalar("classifier/sa_target_accuracy", sa_target_acc.item(), step)
        self.tb_writer.add_scalar("classifier/sas_source_accuracy", sas_source_acc.item(), step)
        self.tb_writer.add_scalar("classifier/sas_target_accuracy", sas_target_acc.item(), step)
        self.tb_writer.add_scalar("classifier/classify_loss", classify_loss.item(), step)
        self.tb_writer.add_scalar("classifier/delta_r", delta_r.mean().item(), step)

        # Backpropagate and update classifier parameters.
        self.sa_classifier_opt.zero_grad()
        self.sas_adv_classifier_opt.zero_grad()
        classify_loss.backward()
        self.sa_classifier_opt.step()
        self.sas_adv_classifier_opt.step()

        return delta_r.mean(), classify_loss

    def eval_src(self, num_games, render=False):
        """
        Evaluate the current policy on the source environment.
        Returns the average total reward over num_games episodes.
        """
        self.policy.eval()
        reward_all = 0.0
        for i in range(num_games):
            obs = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                if render:
                    self.env.render()
                action, _ = self.policy.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
                if steps >= self.n_steps:
                    done = True
            print("EVAL SOURCE: Game: {}, Steps: {}, Total Reward: {}".format(i, steps, total_reward))
            reward_all += total_reward
        return reward_all / num_games

    def eval_tgt(self, num_games, render=False):
        """
        Evaluate the current policy on the target environment.
        Returns the average total reward over num_games episodes.
        """
        self.policy.eval()
        reward_all = 0.0
        for i in range(num_games):
            obs, _ = self.target_env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                if render:
                    self.target_env.render()
                action, _ = self.policy.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = self.target_env.step(action)
                total_reward += reward
                steps += 1
                if steps >= self.n_steps:
                    done = True
            print("EVAL TARGET: Game: {}, Steps: {}, Total Reward: {}".format(i, steps, total_reward))
            reward_all += total_reward
        return reward_all / num_games

    def collect_target_rollout(self):
        """
        Collect a full rollout (episode) from the target environment.
        Logs the target domain episode reward to TensorBoard.
        """
        obs, _ = self.target_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = self.policy.predict(obs, deterministic=False)
            next_obs, reward, done, trunc, info = self.target_env.step(action)
            episode_reward += reward
            self.target_buffer.add(obs, next_obs, action, reward, done, info)
            obs = next_obs
        self.tb_writer.add_scalar("target/episode_reward", episode_reward, self.num_episodes)
        print("TARGET: index: {}, steps: {}, total_rewards: {}".format(self.num_episodes,
                                                                        self.current_episode_step,
                                                                        episode_reward))

    def warmup(self, warmup_steps: int):
        """
        Fill the replay buffer with a few random steps.
        This phase collects transitions using random actions.
        Additionally, every s_t_ratio episodes during warmup a full target rollout is collected.
        """
        obs = self.env.reset()
        self._last_obs = obs
        self.current_episode_step = 0
        for _ in range(warmup_steps):
            action = self.env.action_space.sample()
            # In case the action needs reshaping.
            action_1 = np.array([[action[0]]])
            new_obs, reward, done, info = self.env.step(action_1)
            self.replay_buffer.add(obs, new_obs, action, reward, done, info)
            self.episode_reward += reward
            self.current_episode_step += 1
            if done or self.current_episode_step >= self.n_steps:
                self.tb_writer.add_scalar("source/episode_reward", self.episode_reward, self.num_episodes)
                print("SOURCE (WARMUP): index: {}, steps: {}, total_rewards: {}".format(self.num_episodes,
                                                                                        self.n_steps,
                                                                                        self.episode_reward))
                obs = self.env.reset()
                self.current_episode_step = 0
                self.episode_reward = 0.0
                self.num_episodes += 1
                if self.num_episodes % self.s_t_ratio == 0:
                    self.collect_target_rollout()
            else:
                obs = new_obs
        self._last_obs = obs

    def train(self):
        """
        Performs a single training iteration corresponding to one environment step.
        If the fixed episode length is reached, the done flag is set to True.
        Also updates the reward using the classifier’s delta correction after warmup.
        
        Returns:
            Always 1 (one timestep processed).
        """
        if self._last_obs is None:
            self._last_obs = self.env.reset()
            self.current_episode_step = 0
            self.episode_reward = 0.0

        # Select an action.
        action, _ = self.policy.predict(self._last_obs, deterministic=False)
        # Execute the action.
        next_obs, reward, env_done, info = self.env.step(action)

        # Update reward using classifier delta_r after warmup_games episodes.
        if self.num_episodes >= 1*self.warmup_games:
            with torch.no_grad():
                s_state = torch.as_tensor(self._last_obs, dtype=torch.float32).to(self.device)
                s_action = torch.as_tensor(action, dtype=torch.float32).to(self.device)
                s_next_state = torch.as_tensor(next_obs, dtype=torch.float32).to(self.device)
                sa_input = torch.cat([s_state, s_action], dim=1)
                sas_input = torch.cat([s_state, s_action, s_next_state], dim=1)
                sa_logits = self.sa_classifier(sa_input + gen_noise(self.noise_scale, sa_input, self.device))
                sas_logits = self.sas_adv_classifier(sas_input + gen_noise(self.noise_scale, sas_input, self.device))
                sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
                sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)
                delta_r = (sas_log_probs[0, 1] - sas_log_probs[0, 0]) - (sa_log_probs[0, 1] - sa_log_probs[0, 0])
            reward = reward + self.delta_r_scale * delta_r.item()

        # Accumulate reward and step counters.
        self.episode_reward += reward
        self.current_episode_step += 1

        # Override done if the fixed episode length is reached.
        done = env_done or (self.current_episode_step >= self.n_steps)
        if self.current_episode_step >= self.n_steps:
            if isinstance(info, dict):
                info["timeout"] = True
            elif isinstance(info, list):
                for item in info:
                    if isinstance(item, dict):
                        item["timeout"] = True

        if self.replay_buffer.pos >= self.batch_size and self.target_buffer.pos >= self.batch_size:
            source_batch = self.replay_buffer.sample(self.batch_size)
            target_batch = self.target_buffer.sample(self.batch_size)
            delta_r, classify_loss = self.update_classifiers(source_batch, target_batch, self.total_steps)
            self.tb_writer.add_scalar("darc/delta_r", delta_r.item())
            self.tb_writer.add_scalar("darc/classify_loss", classify_loss.item())

        # Add the transition.
        self.replay_buffer.add(self._last_obs, next_obs, action, reward, done, info)
        self.total_steps += 1

        if done:
            self.tb_writer.add_scalar("source/episode_reward", self.episode_reward, self.num_episodes)
            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(self.num_episodes,
                                                                            self.current_episode_step,
                                                                            self.episode_reward))
            if self.num_episodes % self.n_eval == 0:
                src_avg_reward = self.eval_src(num_games=1, render=False)
                tgt_avg_reward = self.eval_tgt(num_games=1, render=False)
                print("EVALUATION at Episode {}: SOURCE Avg Reward: {}, TARGET Avg Reward: {}"
                      .format(self.num_episodes, src_avg_reward, tgt_avg_reward))
                self.tb_writer.add_scalar("evaluation/source_episode_reward", src_avg_reward, self.num_episodes)
                self.tb_writer.add_scalar("evaluation/target_episode_reward", tgt_avg_reward, self.num_episodes)
            self.num_episodes += 1
            self.current_episode_step = 0
            self.episode_reward = 0.0
            self._last_obs = self.env.reset()
            if self.num_episodes % self.s_t_ratio == 0:
                self.collect_target_rollout()
        else:
            self._last_obs = next_obs

        # Perform SAC gradient updates if enough samples are available.
        if self.replay_buffer.pos >= self.batch_size:
            for _ in range(self.gradient_steps):
                source_batch = self.replay_buffer.sample(self.batch_size)
                _ = self._update(source_batch)
        return 1

    def calculate_games(self):
        """
        Returns the number of completed episodes (games).
        """
        return self.num_episodes

    def learn(self, total_timesteps: int, log_interval: int = 100, **kwargs):
        """
        Main training loop. Each call to train() processes one environment step.
        Logs timesteps and completed episodes.
        Also saves a model checkpoint every 1self.save_checkpoint_freq timesteps.
        
        :param total_timesteps: Total number of steps to train.
        :param log_interval: Logging frequency (in timesteps).
        :return: self
        """
        self._setup_learn(total_timesteps=total_timesteps)
        self.warmup(self.warmup_steps)
        timesteps = 0
        while timesteps < total_timesteps:
            timesteps += self.train()  # Each train() call processes one timestep.
            if timesteps % self.save_checkpoint_freq == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}")
                self.save(checkpoint_path)
                print(f"Saved checkpoint at {checkpoint_path}")
            if timesteps % log_interval == 0:
                self.logger.dump(timesteps)
        return self

    def _update(self, sampler):
        # Call the parent class update and capture the loss information.
        loss_info = super().train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
        
        # Access the internal log buffer (note: this is a private attribute, so use with caution)
        log_dict = getattr(self.logger, "_log_buffer", None)
        
        if log_dict is not None:
            # Log every recorded statistic to TensorBoard and print it.
            for key, value in log_dict.items():
                self.tb_writer.add_scalar(f"train/{key}", value, self.total_steps)
                print(f"Statistic: {key} = {value}")
        
        return loss_info


    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "target_env", "target_buffer", "sa_classifier", "sas_adv_classifier", "tb_writer"
        ]

    def save(self, path: str):
        """
        Save SAC components and classifier networks.
        """
        super().save(path)
        torch.save(self.sa_classifier.state_dict(), path + "_sa_classifier.pt")
        torch.save(self.sas_adv_classifier.state_dict(), path + "_sas_adv_classifier.pt")

    def load(self, path: str):
        """
        Load SAC components and classifier networks.
        """
        super().load(path)
        self.sa_classifier.load_state_dict(torch.load(path + "_sa_classifier.pt", map_location=self.device))
        self.sas_adv_classifier.load_state_dict(torch.load(path + "_sas_adv_classifier.pt", map_location=self.device))


# === Optional: A custom callback for additional TensorBoard logging ===
class TensorBoardLoggingCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TensorBoardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if hasattr(self.model, 'tb_writer'):
            self.model.tb_writer.add_scalar("custom/num_timesteps", self.num_timesteps, self.num_timesteps)
        return True
