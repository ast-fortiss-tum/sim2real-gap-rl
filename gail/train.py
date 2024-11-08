import torch
import numpy as np
import time
from gail.running_mean_std import RunningMeanStd
from gail.test import evaluate_model
from torch.utils.tensorboard import SummaryWriter

import os

import numpy as np
import torch
from torch.nn import functional
from torch import nn
from torch.optim import Adam
from architectures.gaussian_policy import ContGaussianPolicy,DiscreteGaussianPolicy
from architectures.utils import Model, gen_noise
from replay_buffer import ReplayBuffer
from architectures.value_networks import ValueNet
import copy

from architectures.utils import polyak_update



class Train:
    def __init__(self,expert, env, test_env, env_name, n_iterations, agent, epochs, mini_batch_size, epsilon, horizon):
        self.expert = expert
        self.env = env
        self.env_name = env_name
        self.test_env = test_env
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        discriminator_config = {
            "input_dim": [state_dim  + action_dim],
            "architecture": [{"name": "linear1", "size": 64},
                            {"name": "linear2", "size": 1}],
            "hidden_activation": "tanh",
            "output_activation": "none"
        }

        self.device = 'cuda'
        self.discriminator = Model(discriminator_config).to(self.device)
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=0.001)

        self.start_time = 0
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))
        self.expert_memory = ReplayBuffer(1e-5,64)
        self.running_reward = 0
    
    def predict_reward(self, state, action):
        action = torch.as_tensor(action, dtype=torch.float).to(self.device)
        state = torch.as_tensor(state, dtype=torch.float).to(self.device)
        # print(torch.cat([state.unsqueeze(0).to(self.device), action.unsqueeze(0).to(self.device)],axis = 1))
        D = torch.sigmoid(self.discriminator(torch.cat([state.unsqueeze(0).to(self.device), action.unsqueeze(0).to(self.device)],axis = 1)))
        # h = torch.log(D + 1e-6) - torch.log1p(-D + 1e-6) # Add epsilon to improve numerical stability given limited floating point precision
        h = torch.log(D + 1e-6)
        return h

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, next_states,returns, advs, values, log_probs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices],next_states[indices], returns[indices], advs[indices], values[indices],\
                  log_probs[indices]

    def train(self, states, actions,next_states, advs, values, log_probs):

        values = np.vstack(values[:-1])
        log_probs = np.vstack(log_probs)
        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        actions = np.vstack(actions)
        for epoch in range(self.epochs):
            for state, action,next_state, return_, adv, old_value, old_log_prob in self.choose_mini_batch(self.mini_batch_size,
                                                                                               states, actions,next_states, returns,
                                                                                               advs, values, log_probs):
                state = torch.Tensor(state).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                next_state = torch.Tensor(next_state).to(self.agent.device)
                return_ = torch.Tensor(return_).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                old_value = torch.Tensor(old_value).to(self.agent.device)
                old_log_prob = torch.Tensor(old_log_prob).to(self.agent.device)

                w_t = self.cal_wt(state,action,next_state).reshape(64,1).cpu()
    

                value = self.agent.critic(state)
                # clipped_value = old_value + torch.clamp(value - old_value, -self.epsilon, self.epsilon)
                # clipped_v_loss = (clipped_value - return_).pow(2)
                # unclipped_v_loss = (value - return_).pow(2)
                # critic_loss = 0.5 * torch.max(clipped_v_loss, unclipped_v_loss).mean()
                clip_v = old_value + torch.clamp(value - old_value, -self.epsilon, self.epsilon)
                v_max = torch.max(((value - return_) ** 2), ((clip_v - return_) ** 2))
                critic_loss = v_max.mean()

                # critic_loss = self.agent.critic_loss(value, return_)

                new_log_prob = self.calculate_log_probs(self.agent.current_policy, state, action)
                
                ratio = (new_log_prob - old_log_prob).exp() * w_t
                actor_loss = self.compute_actor_loss(ratio, adv)

                self.agent.optimize(actor_loss, critic_loss)

        return actor_loss, critic_loss
    
    def cal_wt(self,s, a, s_):
        with torch.no_grad():
            s_states = torch.as_tensor(s, dtype=torch.float32).to(self.device)
            s_actions = torch.as_tensor(a, dtype=torch.float32).to(self.device)

            s_next_states = torch.as_tensor(s_, dtype=torch.float32).to(self.device)
            
            sa_inputs = torch.cat([s_states, s_actions], 1)

            sas_inputs = torch.cat([s_states, s_actions, s_next_states], 1)

            sa_logits = self.expert.sa_classifier(sa_inputs)
            sas_logits = self.expert.sas_adv_classifier(sas_inputs)
            
            sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
            sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)

            delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
            w_t = torch.exp(delta_r)
        
        return w_t

    def train_step_discriminator(self,s, a, s_, s_e, a_e):
        # s_ = np.clip((s_ - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)

        # train discriminator
        with torch.no_grad():
            s_states = torch.as_tensor(s, dtype=torch.float32).to(self.device)
            s_actions = torch.as_tensor(a, dtype=torch.float32).to(self.device)

            s_states_e = torch.as_tensor(s_e, dtype=torch.float32).to(self.device)
            s_actions_e = torch.as_tensor(a_e, dtype=torch.float32).to(self.device)

            s_next_states = torch.as_tensor(s_, dtype=torch.float32).to(self.device)
            
            sa_inputs = torch.cat([s_states, s_actions], 1)

            sas_inputs = torch.cat([s_states, s_actions, s_next_states], 1)

            sa_logits = self.expert.sa_classifier(sa_inputs)
            sas_logits = self.expert.sas_adv_classifier(sas_inputs)
            
            sa_log_probs = torch.log(torch.softmax(sa_logits, dim=1) + 1e-12)
            sas_log_probs = torch.log(torch.softmax(sas_logits + sa_logits, dim=1) + 1e-12)

            delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
            w_t = torch.exp(delta_r)
        
        
        self.discriminator_opt.zero_grad()
        output = torch.sigmoid(self.discriminator(torch.cat([s_states, s_actions],axis = 1)))

        output_darc = torch.sigmoid(self.discriminator(torch.cat([s_states_e,s_actions_e],axis = 1)))
        loss = -torch.mean((w_t * torch.log(output + 1e-8)) + torch.mean(torch.log(1-output_darc+ 1e-8)))
        loss.backward()
        self.discriminator_opt.step()
        return loss

    def get_expert_action(self, state, deterministic=True):
        with torch.no_grad():
            state = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            if deterministic:
                _, action_prob, action = self.expert.policy.sample(state)
            else:
                action, action_prob, _ = self.expert.policy.sample(state)
            return action.detach().cpu().numpy()[0], action_prob.detach().cpu().numpy()[0]
        

    
    def sample_expert_data(self):
        
        total_rewards = 0
        n_steps = 0
        done = False
        state = self.env.reset()[0]
        states = []
        actions = []
        while not done:
            action,_ = self.get_expert_action(state, deterministic = True)
            next_state, reward, done, _,_ = self.env.step(action)
            done_mask = 1.0 if n_steps == 200 else float(not done)
            if n_steps == 200:
                done = True
            self.expert_memory.add(state, action, reward, next_state, done_mask)
            n_steps += 1
            total_rewards += reward
            states.append(state)
            actions.append(action)
            state = next_state

        # states = np.clip((states - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
        return states, actions
        

    def step(self):
        
        for iteration in range(1, 1 + self.n_iterations):
            expert_state, expert_action = self.sample_expert_data()
            
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            next_states = []

            state = self.env.reset()[0]
            
            step_count = 0

            self.start_time = time.time()
            for t in range(self.horizon):
                # self.state_rms.update(state)
                # state = np.clip((state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                dist = self.agent.choose_dist(state)
                action = dist.sample().cpu().numpy()[0]
                # action = np.clip(action, self.agent.action_bounds[0], self.agent.action_bounds[1])
                log_prob = dist.log_prob(torch.Tensor(action))
                value = self.agent.get_value(state)
                next_state, reward, done, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                # rewards.append(self.predict_reward(torch.tensor(state),torch.tensor(action)).detach().cpu().numpy())
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)
                next_states.append(next_state)
                step_count += 1
                if step_count == 200:
                    done = True
                    break
                

                # if done:
                #     state = self.env.reset()[0]
                #     step_count = 0
                else:
                    state = next_state
            # self.state_rms.update(next_state)
            discriminator_loss = self.train_step_discriminator(states,actions,next_states,expert_state,expert_action)


            # next_state = np.clip((next_state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
            next_value = self.agent.get_value(next_state) * (1 - done)
            values.append(next_value)

            advs = self.get_gae(rewards, values, dones)
            states = np.vstack(states)
            next_states = np.vstack(next_states)
            actor_loss, critic_loss = self.train(states, actions,next_states, advs, values, log_probs)
            # self.agent.set_weights()
            self.agent.schedule_lr()
            eval_rewards = evaluate_model(self.agent, self.test_env, self.state_rms, self.agent.action_bounds)
            self.state_rms.update(states)
            print(iteration,step_count, actor_loss.item(), critic_loss.item(), eval_rewards)
            print('discriminator_loss',discriminator_loss.item())
            self.print_logs(iteration, actor_loss.item(), critic_loss.item(), eval_rewards)

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):

        advs = []
        gae = 0

        dones.append(0)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
        return np.vstack(advs)

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution = model(states)
        return policy_distribution.log_prob(actions)

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        return loss

    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards):
        if iteration == 1:
            self.running_reward = eval_rewards
        else:
            self.running_reward = self.running_reward * 0.99 + eval_rewards * 0.01

        if iteration % 100 == 0:
            print(f"Iter:{iteration}| "
                  f"Ep_Reward:{eval_rewards:.3f}| "
                  f"Running_reward:{self.running_reward:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
            self.agent.save_weights(iteration, self.state_rms)

        with SummaryWriter(self.env_name + "/logs") as writer:
            writer.add_scalar("Episode running reward", self.running_reward, iteration)
            writer.add_scalar("Episode reward", eval_rewards, iteration)
            writer.add_scalar("Actor loss", actor_loss, iteration)
            writer.add_scalar("Critic loss", critic_loss, iteration)
