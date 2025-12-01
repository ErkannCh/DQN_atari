from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: Tuple[int, ...]):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._pos
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self._pos = (self._pos + 1) % self.capacity
        if self._pos == 0:
            self._full = True

    def sample(self, batch_size: int):
        if len(self) < batch_size:
            raise ValueError(f"Not enough elements in the buffer to sample: {len(self)} < {batch_size}")
        max_idx = len(self)
        indices = np.random.randint(0, max_idx, size=batch_size)
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
