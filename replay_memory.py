"""Implements the replay buffer for agent training."""

from typing import Tuple

import random
import numpy as np

class ReplayMemory:
    """Replay Buffer for agent training.
    
    Attributes:
        caapcity: total buffer capacity.
        buffer: list of data samples.
        position: current position in the buffer.
    """
    def __init__(self, capacity: int) -> None:
        """Initializes the buffer object."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action:np.ndarray,
             reward: float, next_state: np.ndarray, done: bool) -> None:
        """Pushes data samples into the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray,
                                               np.ndarray]:
        """Samples a batch of data from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        """Returns the length of the buffer."""
        return len(self.buffer)
