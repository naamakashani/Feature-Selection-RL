import numpy as np
import random
from collections import namedtuple
from typing import List

Transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayMemory(object):
    def __init__(self, capacity: int, alpha=0.6, beta=0.4, beta_annealing_steps=10000) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.cursor = 0



    def push(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool, td_error: float) -> None:
        priority = (np.abs(td_error) + 1e-5) ** self.alpha
        if len(self) < self.capacity:
            self.memory.append(None)
        self.memory[self.cursor] = Transition(state, action, reward, next_state, done)
        self.priorities[self.cursor] = priority
        self.cursor = (self.cursor + 1) % self.capacity

    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        for i, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + 1e-5) ** self.alpha
            self.priorities[i] = priority

    def pop(self, batch_size: int) -> List[Transition]:
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        beta_increment = (1 - self.beta) / self.beta_annealing_steps
        self.beta = min(1.0, self.beta + beta_increment)

        return samples, indices, weights

    def __len__(self) -> int:
        return len(self.memory)
