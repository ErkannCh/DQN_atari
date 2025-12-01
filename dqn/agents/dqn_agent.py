from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.memory.replay_buffer import ReplayBuffer


@dataclass
class EpsilonConfig:
    start: float
    end: float
    decay_frames: int


class DQNAgent:
    def __init__(
        self,
        model: nn.Module,
        num_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_config: EpsilonConfig,
        device: torch.device,
    ):
        self.model = model
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device

        self.epsilon_config = epsilon_config
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=learning_rate,
            alpha=0.95,
            eps=0.01,
        )
        self.loss_fn = nn.MSELoss()

    def epsilon_by_frame(self, frame_idx: int) -> float:
        if frame_idx >= self.epsilon_config.decay_frames:
            return self.epsilon_config.end
        ratio = frame_idx / self.epsilon_config.decay_frames
        return self.epsilon_config.start + ratio * (self.epsilon_config.end - self.epsilon_config.start)

    def select_action(self, state: np.ndarray, frame_idx: int) -> int:
        epsilon = self.epsilon_by_frame(frame_idx)
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action

    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> float | None:
        if len(replay_buffer) < batch_size:
            return None

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
            batch_size
        )

        states = torch.from_numpy(batch_states).to(self.device)
        actions = torch.from_numpy(batch_actions).long().to(self.device)
        rewards = torch.from_numpy(batch_rewards).to(self.device)
        next_states = torch.from_numpy(batch_next_states).to(self.device)
        dones = torch.from_numpy(batch_dones.astype(np.float32)).to(self.device)

        q_values = self.model(states)
        actions = actions.unsqueeze(1)
        q_values = q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values, _ = next_q_values.max(dim=1)
            targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
