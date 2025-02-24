import gymnasium as gym

# Create the HalfCheetah environment with render_mode set to "human"
env = gym.make("HalfCheetah-v2", render_mode="human")

# Reset the environment; newer Gym versions return (observation, info)
state, info = env.reset()

# Run a simulation loop for a fixed number of steps
for _ in range(1000):
    # Render the environment (opens a window)
    env.render()

    # Take a random action
    action = env.action_space.sample()

    # Step through the environment; Gym returns (state, reward, terminated, truncated, info)
    state, reward, terminated, truncated, info = env.step(action)

    # If the episode ends, reset the environment
    if terminated or truncated:
        state, info = env.reset()

# Close the environment window when done
env.close()
