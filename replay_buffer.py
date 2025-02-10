import random
import numpy as np

class ReplayBuffer:
    def __init__(self, memory_size, batch_size):
        """
        Initializes the replay buffer.
        
        Parameters:
        - memory_size: Maximum number of experiences to store.
        - batch_size: Number of experiences to sample in each batch.
        """
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done=0):
        """
        Adds a new experience tuple to the buffer. If the buffer is full,
        the oldest experience is removed.
        
        Parameters:
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The subsequent state.
        - done: Indicator if the episode ended (default is 0).
        """
        if len(self.buffer) >= self.memory_size:
            # Remove the oldest experience to maintain memory size
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly samples a batch of experiences from the buffer.
        Returns a tuple of (states, actions, rewards, next_states, dones),
        each converted to a NumPy array.
        """
        # Ensure we sample only the number of experiences we have (if less than batch_size)
        sample_size = min(self.batch_size, len(self.buffer))
        batch = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones))
    
    def __len__(self):
        """
        Returns the current number of experiences stored in the buffer.
        """
        return len(self.buffer)
