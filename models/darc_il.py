import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from architectures.utils import Model, gen_noise
from replay_buffer import ReplayBuffer
from models.sac import ContSAC
import pickle

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from architectures.utils import Model, gen_noise
from replay_buffer import ReplayBuffer
from models.sac import ContSAC
import pickle
import os
import pickle

import numpy as np
import torch
from torch.nn import functional
from torch.optim import Adam

from architectures.gaussian_policy import ContGaussianPolicy
from architectures.value_networks import ContTwinQNet
from architectures.utils import polyak_update
from replay_buffer import ReplayBuffer
from tensor_writer import TensorWriter



from models.mlp_discriminator import Discriminator
from architectures.gaussian_policy import ContGaussianPolicy_transform, ContGaussianPolicy


class DARC(ContSAC):
    def __init__(self, policy_config, value_config, sa_config, sas_config, source_env, target_env, device,savefolder,running_mean,
                 log_dir="latest_runs", memory_size=1e5, warmup_games=50, batch_size=64, lr=0.0001, gamma=0.99,
                 tau=0.003, alpha=0.2, ent_adj=False, delta_r_scale=1.0, s_t_ratio=10, noise_scale=1.0,
                 target_update_interval=1, n_games_til_train=1, n_updates_per_train=1,decay_rate = 0.99,max_steps = 200,if_normalize = False):
        super(DARC, self).__init__(policy_config, value_config, source_env, device, log_dir,None,
                                   memory_size, None, batch_size, lr, gamma, tau,
                                   alpha, ent_adj, target_update_interval, None, n_updates_per_train)
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
        
        state_dim = source_env.observation_space.shape[0]
        action_dim = source_env.action_space.shape[0] 
        self.policy = ContGaussianPolicy(policy_config, self.action_range,action_dim).to(self.device)
        self.policy_opt = Adam(self.policy.parameters(), lr=lr)
        
        
        self.policy_il = ContGaussianPolicy(policy_config, self.action_range,action_dim).to(self.device)
        self.policy_il_opt = Adam(self.policy_il.parameters(), lr=lr)
        
        self.twin_q_il = ContTwinQNet(value_config).to(self.device)
        self.twin_q_il_opt = Adam(self.twin_q.parameters(), lr=lr)
        self.target_twin_q_il = ContTwinQNet(value_config).to(self.device)
        polyak_update(self.twin_q_il, self.target_twin_q_il, 1)
        
        
        self.discrim_net = Discriminator(state_dim + state_dim).to(self.device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=0.00001)
        
        self.running_mean = running_mean
        self.max_steps = max_steps
        self.savefolder = savefolder

        self.if_normalize = if_normalize 
        
        self.alpha_il = alpha
        self.log_alpha_il = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_il_opt = Adam([self.log_alpha_il], lr=lr)

        self.source_step = 0
        self.target_step = 0
        self.source_memory = self.memory
        self.target_memory = ReplayBuffer(self.memory_size, self.batch_size)
        self.IL_memory = ReplayBuffer(self.memory_size, self.batch_size)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.policy_opt,step_size=1, gamma=decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.twin_q_opt,step_size=1, gamma=decay_rate)
        self.scheduler_sa_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sa_classifier_opt,step_size=1, gamma=decay_rate)
        self.scheduler_sas_adv_classifier_opt = torch.optim.lr_scheduler.StepLR(self.sas_adv_classifier_opt,step_size=1, gamma=decay_rate)

    def step_optim(self):
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.scheduler_sa_classifier_opt.step()
        self.scheduler_sas_adv_classifier_opt.step()
    
    def get_action(self, state,policy, deterministic=False):
        with torch.no_grad():
            state = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            if deterministic:
                _, _, action = policy.sample(state)
            else:
                action, _, _ = policy.sample(state)
            return action.detach().cpu().numpy()[0]
    
    
    def update_discriminator(self, s_states, s_next_states, t_states, t_next_states):
        for _ in range(1):
            self.discrim_criterion = torch.nn.BCELoss()
            self.optimizer_discrim.zero_grad()
            s_state_next_state = torch.cat([torch.tensor(s_states,dtype=torch.float32), torch.tensor(s_next_states,dtype=torch.float32)], 1).to(self.device)
            t_state_next_state = torch.cat([torch.tensor(t_states,dtype=torch.float32), torch.tensor(t_next_states,dtype=torch.float32)], 1).to(self.device)
    
            g_o = self.discrim_net(s_state_next_state +  gen_noise(self.noise_scale, s_state_next_state, self.device))
            e_o = self.discrim_net(t_state_next_state +  gen_noise(self.noise_scale, t_state_next_state, self.device))      
            discrim_loss =  -torch.mean(torch.log(g_o))+ self.discrim_criterion(e_o, torch.zeros((e_o.shape[0], 1), device=self.device))
            discrim_loss.backward()
            self.optimizer_discrim.step()
    

    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
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

        train_info = super(DARC, self).train_step(s_states, s_actions, s_rewards, s_next_states, s_done_masks)
        self.update_discriminator( s_states, s_next_states, t_states, t_next_states)

        return train_info

    
    def train_step_policy(self,is_policy, states, actions, rewards, next_states, done_masks):
        
        if is_policy:
            policy = self.policy
            target_twin_q = self.target_twin_q
            alpha = self.alpha
            log_alpha = self.log_alpha
            twin_q = self.twin_q
            twin_q_opt = self.twin_q_opt
            alpha_opt = self.alpha_opt
            policy_opt = self.policy_opt
        else:
            policy = self.policy_il
            target_twin_q = self.target_twin_q_il
            alpha = self.alpha_il
            log_alpha = self.log_alpha_il
            twin_q = self.twin_q_il
            twin_q_opt = self.twin_q_il_opt
            alpha_opt = self.alpha_il_opt
            policy_opt = self.policy_il_opt
            
            
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
            rewards = torch.as_tensor(rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
            done_masks = torch.as_tensor(done_masks[:, np.newaxis], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = policy.sample(next_states)
            next_q = target_twin_q(next_states, next_action)[0]
            v = next_q - alpha * next_log_prob
 
            expected_q = rewards + done_masks * self.gamma * v

        # Q backprop
        q_val, pred_q1, pred_q2 = twin_q(states, actions)
        q_loss = functional.mse_loss(pred_q1, expected_q) + functional.mse_loss(pred_q2, expected_q)

        twin_q_opt.zero_grad()
        q_loss.backward()
        twin_q_opt.step()

        # Policy backprop
        s_action, s_log_prob, _ = policy.sample(states)
        policy_loss = alpha * s_log_prob - twin_q(states, s_action)[0]
        policy_loss = policy_loss.mean()

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        if self.ent_adj:
            alpha_loss = -(log_alpha * (s_log_prob + self.target_entropy).detach()).mean()

            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()
            
            if is_policy:

                self.alpha = self.log_alpha.exp()
            else:
                self.alpha_il = self.log_alpha_il.exp()
    
        if self.total_train_steps % self.n_until_target_update == 0:
            if is_policy:
                polyak_update(self.twin_q, self.target_twin_q, self.tau)
            else:
                polyak_update(self.twin_q_il, self.target_twin_q_il, self.tau)

        return {'Loss/Policy Loss': policy_loss,
                'Loss/Q Loss': q_loss,
                'Stats/Avg Q Val': q_val.mean(),
                'Stats/Avg Q Next Val': next_q.mean(),
                'Stats/Avg Alpha': self.alpha.item() if self.ent_adj else self.alpha}


    
    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
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

        train_info = self.train_step_policy(True,s_states, s_actions, s_rewards, s_next_states, s_done_masks)
        
        
        self.update_discriminator( s_states, s_next_states, t_states, t_next_states)
        state_pair = torch.hstack([s_states, s_next_states]).to(self.device)
        expert_reward = -torch.log(1e-8 + self.discrim_net(state_pair  + gen_noise(self.noise_scale, state_pair, self.device)))
        train_info = self.train_step_policy(False,s_states, s_actions, expert_reward, s_next_states, s_done_masks)
        
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
                if i %100 == 0:
                    print('src',self.eval_src(10))
                    print('tgt',self.eval_tgt(10))
                    self.save_model(str(i))
            print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))

    def simulate_env(self, game_count, env_name, deterministic):
        if env_name == "source":
            env = self.source_env
            memory = self.source_memory
            transform = True
            policy = self.policy_il
        elif env_name == "target":
            env = self.target_env
            memory = self.target_memory
            transform = False
            policy = self.policy
        else:
            raise Exception("Env name not recognized")

        total_rewards = 0
        n_steps = 0
        done = False
        state = env.reset()
        if self.if_normalize:
            state = self.running_mean(state)
        while not done:
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                action = self.get_action(state,policy, deterministic)
            next_state, reward, done, _ = env.step(action)
            if self.if_normalize:
                next_state = self.running_mean(next_state)
            done_mask = 1.0 if n_steps == env._max_episode_steps - 1 else float(not done)
            if n_steps == self.max_steps:
                done = True

            memory.add(state, action, reward, next_state, done_mask)

            if env_name == "source":
                self.source_step += 1
            elif env_name == "target":
                self.target_step += 1
            n_steps += 1
            total_rewards += reward
            state = next_state
        return total_rewards, n_steps

    def save_model(self, folder_name):
        import os
        # super(DARC, self).save_model(folder_name)
        path = os.path.join('saved_weights/'+self.savefolder, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), path + '/policy')
        torch.save(self.twin_q.state_dict(), path + '/twin_q_net')

        torch.save(self.sa_classifier.state_dict(), path + '/sa_classifier')
        torch.save(self.sas_adv_classifier.state_dict(), path + '/sas_adv_classifier')
        # torch.save(self.running_mean.state_dict(), path + '/running_mean')
        pickle.dump(self.running_mean,
                    open(path + '/running_mean', 'wb'))

    # Load model parameters
    def load_model(self, folder_name, device):
        super(DARC, self).load_model(folder_name, device)
        path = 'saved_weights/' + folder_name
        self.sa_classifier.load_state_dict(torch.load(path + '/sa_classifier', map_location=torch.device(device)))
        self.sas_adv_classifier.load_state_dict(
            torch.load(path + '/sas_adv_classifier', map_location=torch.device(device)))
        self.running_mean = pickle.load(open(path + '/running_mean', "rb"))

    def eval_src(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        
        for i in range(num_games):
            state = self.source_env.reset()
            if self.if_normalize:
                state = self.running_mean(state)
            done = False
            total_reward = 0
            step = 0
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state,self.policy_il, deterministic=False)
                next_state, reward, done, _ = self.source_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                total_reward += reward
                state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
                
            reward_all += total_reward
        return reward_all/num_games
    
    def eval_tgt(self, num_games, render=False):
        self.policy.eval()
        self.twin_q.eval()
        reward_all = 0
        for i in range(num_games):
            step = 0
            state = self.target_env.reset()
            if self.if_normalize:
                state = self.running_mean(state)
            done = False
            total_reward = 0
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state,self.policy, deterministic=False)
                next_state, reward, done, _ = self.target_env.step(action)
                if self.if_normalize:
                    next_state = self.running_mean(next_state)
                total_reward += reward
                state = next_state
                if step == self.max_steps:
                    done = True
                step += 1
            reward_all += total_reward
        return reward_all/num_games

