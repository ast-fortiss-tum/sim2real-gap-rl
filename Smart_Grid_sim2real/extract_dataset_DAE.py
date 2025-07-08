import argparse
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
from tqdm import tqdm

# ------------------------------
def generate_action_schedule(env, max_steps):
    return [env.action_space.sample() for _ in range(max_steps)]

def collect_episode(env, action_schedule):
    obs_seq = []
    obs = env.reset()
    obs_seq.append(obs[0][100])
    
    for action in action_schedule:
        obs, reward, trunc, done, info = env.step(action)
        obs_seq.append(obs[100])
        if done:
            break
    return np.array(obs_seq)

def collect_dataset(env_clean, env_noisy, num_episodes, max_steps, action_schedule_fn=None):
    dataset = []
    for ep in tqdm(range(num_episodes), desc="Collecting Episodes"):
        if action_schedule_fn is None:
            action_schedule = generate_action_schedule(env_clean, max_steps)
        else:
            action_schedule = action_schedule_fn(env_clean, max_steps)

        episode_clean = collect_episode(env_clean, action_schedule)
        episode_noisy = collect_episode(env_noisy, action_schedule)

        dataset.append((episode_clean, episode_noisy))
    
    return dataset

# ------------------------------
# Main Execution with Argparse
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect paired clean and noisy datasets.")
    parser.add_argument("--noise", type=float, default=0.2, help="Standard deviation of Gaussian noise")
    parser.add_argument("--bias", type=float, default=0.5, help="Bias added to observation")
    parser.add_argument("--degree", type=float, default=0.65, help="Degree of the environment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment")
    args = parser.parse_args()

    class NoisyWrapper(gym.ObservationWrapper):
        def __init__(self, env, noise_std=0.2, bias=0.5):
            super(NoisyWrapper, self).__init__(env)
            self.noise_std = noise_std
            self.bias = bias
        
        def observation(self, obs):
            return obs + np.random.normal(0, self.noise_std, size=obs.shape) + self.bias

    # Create environments
    DEGREE = args.degree
    SEED = args.seed
    NOISE = args.noise
    BIAS = args.bias

    #Select the environment you want to use:

    env_clean = get_new_all_eff_env(degree=DEGREE, seed=SEED, fixed_start = None, rl=True).env # Calibrated environment (with dynamical gap)
    #env_clean = get_simple_linear_env(seed=SEED,fixed_start=None, rl=True).env
    #env_noisy = get_new_all_eff_env(degree=DEGREE, seed=SEED, fixed_start = None, rl=True).env
    env_noisy = NoisyWrapper(env_clean, noise_std=NOISE, bias=BIAS) # Noisy environment (with dynamical gap)

    num_episodes = 365
    max_steps = 23

    dataset = collect_dataset(env_clean, env_noisy, num_episodes, max_steps)

    clean_dataset = [episode_clean for (episode_clean, _) in dataset]
    noisy_dataset = [episode_noisy for (_, episode_noisy) in dataset]

    print("Clean dataset: ", clean_dataset)
    print("Noisy dataset: ", noisy_dataset)

    base = "manual_control_dataset/"

    np.save(base + f'clean/clean_dataset_degree_{DEGREE}_Gaussian_noise_{NOISE}_bias_{BIAS}.npy', clean_dataset)
    np.save(base + f'noisy/noisy_dataset_degree_{DEGREE}_Gaussian_noise_{NOISE}_bias_{BIAS}.npy', noisy_dataset)
    print("Datasets saved successfully as 'clean_dataset_degree_{DEGREE}_Gaussian_noise_{NOISE}_bias_{BIAS}.npy' and 'noisy_dataset_degree_{DEGREE}_Gaussian_noise_{NOISE}_bias_{BIAS}.npy'.")

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
