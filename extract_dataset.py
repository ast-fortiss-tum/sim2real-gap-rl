import gymnasium as gym
import numpy as np
from environments.get_customized_envs import (
    get_simple_linear_env, 
    get_new_soc_env, 
    get_new_charge_env, 
    get_new_discharge_env, 
    get_new_all_eff_env, 
    get_new_limited_capacity_env, 
    get_new_limited_plim_env, 
    get_twoHouses_env
)
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # <-- import tqdm for progress bar

# ------------------------------
def generate_action_schedule(env, max_steps):
    """
    Generate a random sequence of actions from the given environment's action space.
    
    Args:
        env: An environment with an action_space.
        max_steps (int): Number of actions to generate.
        
    Returns:
        List: A list of actions.
    """
    return [env.action_space.sample() for _ in range(max_steps)]

def collect_episode(env, action_schedule):
    """
    Roll out an episode on the given environment using the provided action schedule.
    
    Args:
        env: An environment that follows the gym API.
        action_schedule (list): A list of actions to take in sequence.
        
    Returns:
        np.ndarray: The array of observations collected over the episode.
    """
    obs_seq = []
    # For this customized env, we assume the env.reset() returns a tuple.
    obs = env.reset()
    # Notice: we select the observation at index 0 then index [100] as provided.
    obs_seq.append(obs[0][100])
    
    for action in action_schedule:
        obs, reward, trunc, done, info = env.step(action)
        obs_seq.append(obs[100])
        if done:
            break
    #print(obs_seq)
    return np.array(obs_seq)  # Convert to a numpy array for convenience

def collect_dataset(env_clean, env_noisy, num_episodes, max_steps, action_schedule_fn=None):
    """
    Collect paired datasets from the clean and noisy environments.
    
    For each episode, a schedule of actions is generated—either using a provided function 
    or randomly sampled from the environment's action_space. The same action schedule is 
    used to run both env_clean and env_noisy, returning a pair of observation sequences.
    
    Args:
        env_clean: The clean environment (returns clean observations).
        env_noisy: The noisy environment (returns noisy observations).
        num_episodes (int): Number of episodes to collect.
        max_steps (int): Maximum time steps per episode.
        action_schedule_fn (callable, optional): Function to generate a list of actions given env and max_steps.
    
    Returns:
        List of tuples: Each tuple is (episode_clean, episode_noisy).
    """
    dataset = []
    # Wrap the loop with tqdm for progress tracking.
    for ep in tqdm(range(num_episodes), desc="Collecting Episodes"):
        if action_schedule_fn is None:
            action_schedule = generate_action_schedule(env_clean, max_steps)
        else:
            action_schedule = action_schedule_fn(env_clean, max_steps)

        episode_clean = collect_episode(env_clean, action_schedule)
        episode_noisy = collect_episode(env_noisy, action_schedule)

        dataset.append((episode_clean, episode_noisy))
        # Uncomment the next line if you want detailed episode info printed.
        # print(f"Episode {ep+1} collected, length (clean): {episode_clean.shape[0]}, (noisy): {episode_noisy.shape[0]}")
    
    return dataset

# ------------------------------
# Example Usage:
# ------------------------------

# Create the clean environment.
env_clean = get_new_all_eff_env(degree=1.0, seed=42, rl=True).env  # Clean environment

# Create the noisy environment by wrapping the clean one.
class NoisyWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.2, bias=0.5):
        super(NoisyWrapper, self).__init__(env)
        self.noise_std = noise_std
        self.bias = bias
    
    def observation(self, obs):
        # Add Gaussian noise plus a bias to the observation
        return obs + np.random.normal(0, self.noise_std, size=obs.shape) + self.bias

env_noisy = NoisyWrapper(env_clean, noise_std=0.2)

# Set parameters for dataset collection.
num_episodes = 365  # Number of episodes to collect
max_steps = 24      # Maximum steps per episode

# Collect the paired dataset.
dataset = collect_dataset(env_clean, env_noisy, num_episodes, max_steps)

# ---------------------------------------------------
# PART 2: SAVE THE DATASET
# ---------------------------------------------------
# Separate clean and noisy episodes into two lists.
clean_dataset = [episode_clean for (episode_clean, _) in dataset]
noisy_dataset = [episode_noisy for (_, episode_noisy) in dataset]

# Save each dataset to a file.
np.save('clean_dataset.npy', clean_dataset)
np.save('noisy_dataset.npy', noisy_dataset)
print("Datasets saved successfully as 'clean_dataset.npy' and 'noisy_dataset.npy'.")

# ---------------------------------------------------
# PART 3: OPTIONAL PLOTTING OF THE OUTPUTS
# ---------------------------------------------------
if False:
    for i, (episode_clean, episode_noisy) in enumerate(dataset):
        plt.figure(figsize=(8, 4))
        plt.plot(episode_clean[:], '-o', label='Clean')
        plt.plot(episode_noisy[:], '-x', label='Noisy')
        plt.xlabel("Timestep")
        plt.ylabel("Observation")
        plt.title(f"Episode {i+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
