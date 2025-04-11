import torch
import torch.nn as nn
import torch.optim as optim
# TO BE EDITED
# -------------------------------------------------------------------------
# 1) Define two policies: one for the simulator, one for the real robot
#    Here, we'll assume they have the same architecture.
# -------------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        # We output parameters for a (Gaussian) distribution over actions
        # for simplicity. In practice, you might have separate layers for
        # mean and log_std, or use a more advanced distribution.
        return self.net(state)

    def get_action(self, state):
        # Sample an action from the distribution
        with torch.no_grad():
            logits = self.forward(state)
        # This is a toy example; a real code would handle continuous actions
        # with a normal distribution or do softmax for discrete actions, etc.
        action = torch.tanh(logits)
        return action

# -------------------------------------------------------------------------
# 2) Define a discriminator that takes in short sequences of states.
#    It outputs a scalar probability that the input comes from the simulator.
# -------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, state_dim, seq_len=2):
        super(Discriminator, self).__init__()
        # For simplicity, flatten seq_len states into a single vector:
        input_dim = seq_len * state_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state_sequence):
        # state_sequence shape: (batch_size, seq_len, state_dim)
        batch_size = state_sequence.shape[0]
        flat_input = state_sequence.view(batch_size, -1)
        return self.net(flat_input)

# -------------------------------------------------------------------------
# 3) Example training loop for MATL:
#    - We'll gather rollouts for simulator and real robot in parallel,
#      then train policy and discriminator.
# -------------------------------------------------------------------------
def matl_train_step(
    policy_sim, policy_real, discriminator, 
    sim_env, real_env, 
    sim_optimizer, real_optimizer, disc_optimizer,
    lambda_align=0.1, seq_len=2
):
    """
    One iteration of:
      1) Roll out simulation and real environment steps
      2) Compute environment returns
      3) Train the discriminator
      4) Train both policies with environment + alignment rewards
    """
    
    # --- 3.1) Collect trajectories from simulator and real robot
    sim_states = []
    sim_actions = []
    sim_rewards = []

    real_states = []
    real_actions = []
    real_rewards = []

    # (For demonstration, we collect just a few transitions)
    for _ in range(10):
        state_s = sim_env.get_state()
        action_s = policy_sim.get_action(torch.tensor(state_s, dtype=torch.float32))
        next_state_s, reward_s = sim_env.step(action_s.numpy())
        
        sim_states.append(state_s)
        sim_actions.append(action_s)
        sim_rewards.append(reward_s)

        state_r = real_env.get_state()
        action_r = policy_real.get_action(torch.tensor(state_r, dtype=torch.float32))
        next_state_r, reward_r = real_env.step(action_r.numpy())
        
        real_states.append(state_r)
        real_actions.append(action_r)
        real_rewards.append(reward_r)

    # Convert to tensors
    sim_states_tensor = torch.tensor(sim_states, dtype=torch.float32)
    real_states_tensor = torch.tensor(real_states, dtype=torch.float32)
    sim_rewards_tensor = torch.tensor(sim_rewards, dtype=torch.float32)
    real_rewards_tensor = torch.tensor(real_rewards, dtype=torch.float32)

    # --- 3.2) Prepare sequences for the discriminator
    # For simplicity, let's just pair consecutive states. In practice, you might pick non-consecutive frames.
    def make_sequences(states):
        # e.g. shape: (batch_size-1, seq_len, state_dim)
        # This is just a toy approach
        result = []
        for i in range(len(states) - seq_len + 1):
            chunk = states[i:i+seq_len]
            result.append(chunk)
        if not result:
            # if too few states, just dummy fill
            result.append(torch.zeros(seq_len, states.shape[1]))
        return torch.stack(result)

    sim_seq = make_sequences(sim_states_tensor)
    real_seq = make_sequences(real_states_tensor)

    # --- 3.3) Train the discriminator
    disc_optimizer.zero_grad()
    sim_labels = discriminator(sim_seq)          # Probability that it's "sim"
    real_labels = discriminator(real_seq)        # Probability that it's "sim" for real data
    
    # Loss: we want sim_labels -> 1, real_labels -> 0
    disc_loss = - torch.mean(torch.log(sim_labels + 1e-8)) \
                - torch.mean(torch.log(1.0 - real_labels + 1e-8))
    disc_loss.backward()
    disc_optimizer.step()

    # --- 3.4) Compute alignment rewards:
    # The policies are trained with confusion objectives:
    #    Robot reward   +=  +lambda_align * log(D( real_seq ))
    #    Sim   reward   +=  -lambda_align * log(D( sim_seq  ))
    
    # Evaluate the logs:
    with torch.no_grad():
        sim_disc_output = discriminator(sim_seq)
        real_disc_output = discriminator(real_seq)
    # For simplicity, let's just apply a single "average" alignment reward
    # to all transitions. A more detailed method would track them per step.
    sim_aux_reward = -lambda_align * torch.log(sim_disc_output + 1e-8)
    real_aux_reward =  lambda_align * torch.log(real_disc_output + 1e-8)

    # Suppose we sum up these alignment rewards and treat them as an "advantage"
    sim_alignment_value = sim_aux_reward.mean().item()
    real_alignment_value = real_aux_reward.mean().item()

    # --- 3.5) Policy gradient update with both environment reward and alignment
    # In practice, you’d run a more advanced RL algorithm (e.g. TRPO, PPO).
    # Here, we do a toy gradient-ascent step on the policy by building a simple objective.

    # Toy objective for simulator policy:
    sim_optimizer.zero_grad()
    # We'll interpret the environment reward as something we want to maximize,
    # plus the alignment reward. This is NOT a real PPO or TRPO update but a
    # simplistic demonstration to show the idea.
    sim_loss = - (sim_rewards_tensor.mean() + sim_alignment_value)
    sim_loss.backward()
    sim_optimizer.step()

    # Toy objective for real policy:
    real_optimizer.zero_grad()
    real_loss = - (real_rewards_tensor.mean() + real_alignment_value)
    real_loss.backward()
    real_optimizer.step()

    return {
        'disc_loss': disc_loss.item(),
        'sim_loss': sim_loss.item(),
        'real_loss': real_loss.item(),
        'avg_sim_reward': sim_rewards_tensor.mean().item(),
        'avg_real_reward': real_rewards_tensor.mean().item()
    }

# -------------------------------------------------------------------------
# Usage Example (Sketch)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose we have environments with identical state / action dimensions
    state_dim = 4
    action_dim = 2

    policy_sim = Policy(state_dim, action_dim)
    policy_real = Policy(state_dim, action_dim)
    discriminator = Discriminator(state_dim, seq_len=2)

    sim_optimizer = optim.Adam(policy_sim.parameters(), lr=1e-3)
    real_optimizer = optim.Adam(policy_real.parameters(), lr=1e-3)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

    # Dummy environment classes with stubs
    class DummyEnv:
        def __init__(self):
            self._state = torch.zeros(state_dim)
        def get_state(self):
            return self._state
        def step(self, action):
            # Move a tiny bit in the direction of the action for demonstration
            next_state = self._state + 0.05 * action
            reward = -torch.sum(next_state**2).item()  # negative distance from origin as a toy reward
            self._state = next_state
            return next_state, reward

    sim_env = DummyEnv()
    real_env = DummyEnv()

    # Training loop
    for iteration in range(50):
        metrics = matl_train_step(
            policy_sim, policy_real, discriminator,
            sim_env, real_env,
            sim_optimizer, real_optimizer, disc_optimizer,
            lambda_align=0.5, seq_len=2
        )
        print(f"Iteration {iteration} | Metrics: {metrics}")
