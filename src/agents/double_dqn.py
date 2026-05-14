"""Custom Double DQN agent for the EURUSD intraday env.

Implements the DDQN target rule:

    y = r + gamma * Q_target(s_next, argmax_a Q_online(s_next, a)) * (1 - done)

with a simple FIFO replay buffer, epsilon-greedy exploration with linear decay
from ``eps_start`` to ``eps_end`` over ``eps_decay_steps`` *agent steps* (i.e.
calls to ``select_action``), MSE TD-loss, Adam, gradient clipping, and a hard
target-network sync every ``target_update_freq`` learner updates.

This module intentionally does not couple to the env or the training loop —
``train_dqn.py`` orchestrates the env interaction, logging, and checkpoints.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- network


class QNetwork(nn.Module):
    """MLP Q-network: state -> Q(s, .) for each discrete action."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- replay


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """FIFO replay buffer with uniform sampling.

    Stores transitions as five parallel ndarrays for fast batched sampling.
    """

    def __init__(self, capacity: int, state_dim: int, seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self._states[i] = np.asarray(state, dtype=np.float32)
        self._next_states[i] = np.asarray(next_state, dtype=np.float32)
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._dones[i] = 1.0 if done else 0.0
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer of size {self._size}."
            )
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        return {
            "states": self._states[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_states": self._next_states[idx],
            "dones": self._dones[idx],
        }


# --------------------------------------------------------------------------- agent


@dataclass
class DDQNConfig:
    state_dim: int = 15
    n_actions: int = 3
    hidden_sizes: tuple[int, ...] = (128, 128)
    lr: float = 1.0e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 100_000
    min_buffer_to_learn: int = 1_000
    target_update_freq: int = 1_000  # in learner updates
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    eps_decay_type: str = "linear"   # "linear" | "exponential"
    device: str = "cpu"
    seed: Optional[int] = None


class DoubleDQNAgent:
    """Custom Double DQN agent.

    Lifecycle:
        agent = DoubleDQNAgent(cfg)
        a = agent.select_action(s)              # uses current epsilon
        agent.buffer.push(s, a, r, s_next, done)
        loss = agent.learn()                    # one gradient step (None if buffer too small)
        agent.save(path); agent.load(path)
    """

    def __init__(self, cfg: DDQNConfig) -> None:
        self.cfg = cfg
        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))
            np.random.seed(int(cfg.seed))
            random.seed(int(cfg.seed))

        self.device = torch.device(cfg.device)
        self.online = QNetwork(cfg.state_dim, cfg.hidden_sizes, cfg.n_actions).to(self.device)
        self.target = QNetwork(cfg.state_dim, cfg.hidden_sizes, cfg.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_capacity, cfg.state_dim, seed=cfg.seed)

        self._rng = random.Random(cfg.seed)
        self._np_rng = np.random.default_rng(cfg.seed)
        self.action_steps = 0   # number of select_action calls (for epsilon decay)
        self.learn_steps = 0    # number of gradient updates done

    # ------------------------------------------------------------------ epsilon

    @property
    def epsilon(self) -> float:
        if self.cfg.eps_decay_steps <= 0:
            return float(self.cfg.eps_end)
        step = self.action_steps
        T = float(self.cfg.eps_decay_steps)
        e0 = float(self.cfg.eps_start)
        e1 = float(self.cfg.eps_end)
        if self.cfg.eps_decay_type == "exponential":
            # Geometric decay: ε(t) = ε_start × (ε_end/ε_start)^(t/T).
            # At t=0: ε_start; at t=T: ε_end. Clamp to ε_end for t > T.
            if step >= T:
                return e1
            ratio = e1 / e0 if e0 > 0 else 0.0
            return float(e0 * (ratio ** (step / T)))
        # linear (default)
        frac = min(1.0, step / T)
        return float(e0 + frac * (e1 - e0))

    # ------------------------------------------------------------------ act

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action; ``greedy=True`` always picks argmax (eval)."""
        eps = 0.0 if greedy else self.epsilon
        if not greedy:
            self.action_steps += 1
        if (not greedy) and self._rng.random() < eps:
            return int(self._np_rng.integers(0, self.cfg.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(torch.argmax(q, dim=1).item())

    # ------------------------------------------------------------------ learn

    def _compute_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """DDQN target: y = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)."""
        with torch.no_grad():
            next_q_online = self.online(next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)  # (B, 1)
            next_q_target = self.target(next_states).gather(1, next_actions).squeeze(1)
            y = rewards + self.cfg.gamma * next_q_target * (1.0 - dones)
        return y

    def learn(self) -> Optional[float]:
        """One gradient step. Returns the loss, or ``None`` if buffer too small."""
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.min_buffer_to_learn):
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_online_all = self.online(states)
        q_pred = q_online_all.gather(1, actions).squeeze(1)
        y = self._compute_target(rewards, next_states, dones)

        loss = F.mse_loss(q_pred, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.detach().cpu().item())

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "action_steps": self.action_steps,
                "learn_steps": self.learn_steps,
                "cfg": self.cfg.__dict__,
            },
            p,
        )
        return p

    def load(self, path: str | Path) -> None:
        p = Path(path)
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.action_steps = int(ckpt.get("action_steps", 0))
        self.learn_steps = int(ckpt.get("learn_steps", 0))
