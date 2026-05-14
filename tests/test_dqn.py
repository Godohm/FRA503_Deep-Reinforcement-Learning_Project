"""M6 tests: Double DQN components + short smoke training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.agents.double_dqn import (
    DDQNConfig,
    DoubleDQNAgent,
    QNetwork,
    ReplayBuffer,
)
from src.envs.eurusd_intraday_env import EURUSDIntradayTradingEnv
from src.features.state_builder import N_PRICE_FEATURES, STATE_DIM


# --------------------------------------------------------------------------- helpers


def _env_cfg(warmup_bars: int = 3) -> dict:
    return {
        "session": {
            "display_tz": "Asia/Bangkok",
            "start": "09:00",
            "end": "00:00",
            "warmup_bars": warmup_bars,
            "max_missing_bars_per_day": 30,
            "max_ffill_gap": 5,
        },
        "costs": {
            "point_size": 1.0e-5,
            "use_broker_spread": True,
            "commission_price": 0.0,
            "spread_cost_factor": 1.0,
        },
        "portfolio": {"initial_equity": 10000.0, "unit_size": 100000.0},
        "execution": "current_close",
        "reward": {"scaling": 1.0, "stop_equity_floor": 0.0},
        "split": {"train_months": 1, "test_months": 1},
        "seed": 42,
    }


def _make_df(day: str, n: int = 30) -> pd.DataFrame:
    start = pd.Timestamp(f"{day} 02:00:00", tz="UTC")
    idx = pd.date_range(start=start, periods=n, freq="1min")
    close = np.linspace(1.10000, 1.10000 + (n - 1) * 1e-5, n, dtype="float64")
    return pd.DataFrame({
        "open":  close,
        "high":  close + 1e-5,
        "low":   close - 1e-5,
        "close": close,
        "tick_volume": np.full(n, 10, dtype="int64"),
        "spread":      np.full(n, 10, dtype="int64"),
        "real_volume": np.zeros(n, dtype="int64"),
        "session_day": [pd.Timestamp(day).date()] * n,
        "bar_idx_in_day": np.arange(n, dtype="int64"),
    }, index=idx)


def _make_env(n: int = 30, mode: str = "sequential") -> EURUSDIntradayTradingEnv:
    df = _make_df("2024-03-04", n=n)
    pf = np.zeros((len(df), N_PRICE_FEATURES), dtype=np.float32)
    return EURUSDIntradayTradingEnv(df, pf, _env_cfg(warmup_bars=3), mode=mode)


# --------------------------------------------------------------------------- QNetwork


def test_qnetwork_output_shape():
    net = QNetwork(input_dim=STATE_DIM, hidden_sizes=(128, 128), output_dim=3)
    x = torch.zeros((4, STATE_DIM), dtype=torch.float32)
    out = net(x)
    assert out.shape == (4, 3)
    assert out.dtype == torch.float32


def test_qnetwork_layer_count():
    net = QNetwork(input_dim=STATE_DIM, hidden_sizes=(128, 128), output_dim=3)
    # Sequential: Linear, ReLU, Linear, ReLU, Linear = 5 modules.
    assert len(list(net.net.children())) == 5
    # Final layer maps to n_actions.
    final = list(net.net.children())[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.out_features == 3


# --------------------------------------------------------------------------- ReplayBuffer


def test_replay_buffer_push_grows_size():
    buf = ReplayBuffer(capacity=10, state_dim=STATE_DIM, seed=0)
    assert len(buf) == 0
    for _ in range(7):
        buf.push(np.zeros(STATE_DIM, dtype=np.float32), 1, 0.0,
                 np.zeros(STATE_DIM, dtype=np.float32), False)
    assert len(buf) == 7


def test_replay_buffer_wraps_at_capacity():
    cap = 5
    buf = ReplayBuffer(capacity=cap, state_dim=STATE_DIM, seed=0)
    for i in range(cap + 3):
        s = np.full(STATE_DIM, float(i), dtype=np.float32)
        buf.push(s, i % 3, float(i), s, i == cap + 2)
    assert len(buf) == cap
    # Oldest entries (i=0..2) overwritten; remaining must include the latest.
    # Verify ring-buffer index returned to 3 after wrap-around (cap+3 % cap).
    assert buf._idx == (cap + 3) % cap


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, seed=0)
    for i in range(64):
        s = np.random.default_rng(i).normal(size=STATE_DIM).astype(np.float32)
        buf.push(s, i % 3, float(i), s, False)
    batch = buf.sample(32)
    assert batch["states"].shape == (32, STATE_DIM)
    assert batch["next_states"].shape == (32, STATE_DIM)
    assert batch["actions"].shape == (32,)
    assert batch["rewards"].shape == (32,)
    assert batch["dones"].shape == (32,)
    assert batch["actions"].dtype == np.int64


def test_replay_buffer_sample_underflow_raises():
    buf = ReplayBuffer(capacity=10, state_dim=STATE_DIM, seed=0)
    buf.push(np.zeros(STATE_DIM, dtype=np.float32), 1, 0.0,
             np.zeros(STATE_DIM, dtype=np.float32), False)
    with pytest.raises(ValueError):
        buf.sample(32)


# --------------------------------------------------------------------------- epsilon-greedy / action


def test_epsilon_decays_linearly():
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3,
                     eps_start=1.0, eps_end=0.1, eps_decay_steps=100, seed=0)
    agent = DoubleDQNAgent(cfg)
    assert agent.epsilon == pytest.approx(1.0)
    # Bump action_steps directly to skip stochastic env interactions.
    agent.action_steps = 50
    assert agent.epsilon == pytest.approx(0.55, abs=1e-6)
    agent.action_steps = 100
    assert agent.epsilon == pytest.approx(0.1, abs=1e-6)
    agent.action_steps = 1000
    assert agent.epsilon == pytest.approx(0.1, abs=1e-6)


def test_select_action_returns_valid_action():
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, seed=0)
    agent = DoubleDQNAgent(cfg)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(100):
        a = agent.select_action(state)
        assert a in (0, 1, 2)


def test_greedy_select_action_does_not_decay_epsilon():
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3,
                     eps_start=1.0, eps_end=0.1, eps_decay_steps=10, seed=0)
    agent = DoubleDQNAgent(cfg)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    for _ in range(50):
        agent.select_action(state, greedy=True)
    # action_steps should NOT have advanced in greedy mode.
    assert agent.action_steps == 0
    assert agent.epsilon == pytest.approx(1.0)


# --------------------------------------------------------------------------- DDQN target


def test_double_dqn_target_shape_and_dtype():
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, gamma=0.9, seed=0)
    agent = DoubleDQNAgent(cfg)
    B = 8
    rewards = torch.zeros(B)
    next_states = torch.zeros((B, STATE_DIM))
    dones = torch.zeros(B)
    y = agent._compute_target(rewards, next_states, dones)
    assert y.shape == (B,)
    assert y.dtype == torch.float32


def test_double_dqn_target_done_zeros_bootstrap():
    # If done=1, target should reduce to reward (no bootstrap from Q_target).
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, gamma=0.9, seed=0)
    agent = DoubleDQNAgent(cfg)
    rewards = torch.tensor([1.0, -2.0, 0.5])
    next_states = torch.randn(3, STATE_DIM)
    dones = torch.tensor([1.0, 1.0, 1.0])
    y = agent._compute_target(rewards, next_states, dones)
    torch.testing.assert_close(y, rewards)


def test_double_dqn_target_uses_argmax_from_online():
    # Stuff distinguishable weights so argmax differs from target's argmax,
    # then verify the indices come from online.
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, gamma=1.0, seed=0)
    agent = DoubleDQNAgent(cfg)
    # Force online and target to pick predictable, different argmaxes.
    with torch.no_grad():
        agent.online.net[-1].weight.zero_()
        agent.online.net[-1].bias.copy_(torch.tensor([0.0, 0.0, 10.0]))  # argmax = 2
        agent.target.net[-1].weight.zero_()
        agent.target.net[-1].bias.copy_(torch.tensor([5.0, 7.0, -3.0]))  # argmax(target)=1, target[2]=-3
    rewards = torch.tensor([0.0])
    next_states = torch.zeros((1, STATE_DIM))
    dones = torch.tensor([0.0])
    y = agent._compute_target(rewards, next_states, dones)
    # DDQN uses target[argmax_online] = target[2] = -3.0  (NOT target argmax 1 -> 7.0)
    torch.testing.assert_close(y, torch.tensor([-3.0]))


# --------------------------------------------------------------------------- learn step


def test_learn_returns_none_when_buffer_below_threshold():
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, batch_size=8,
                     min_buffer_to_learn=64, buffer_capacity=128, seed=0)
    agent = DoubleDQNAgent(cfg)
    # Fewer transitions than min_buffer_to_learn.
    for _ in range(10):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        agent.buffer.push(s, 1, 0.0, s, False)
    assert agent.learn() is None


def test_learn_runs_when_buffer_ready():
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, batch_size=8,
                     min_buffer_to_learn=32, buffer_capacity=128, seed=0)
    agent = DoubleDQNAgent(cfg)
    for i in range(64):
        s = np.random.default_rng(i).normal(size=STATE_DIM).astype(np.float32)
        agent.buffer.push(s, i % 3, float(i) * 1e-4, s, False)
    loss = agent.learn()
    assert loss is not None
    assert np.isfinite(loss)
    assert agent.learn_steps == 1


def test_target_network_syncs_on_schedule():
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, batch_size=4,
                     min_buffer_to_learn=4, buffer_capacity=64,
                     target_update_freq=3, seed=0)
    agent = DoubleDQNAgent(cfg)
    for i in range(16):
        s = np.random.default_rng(i).normal(size=STATE_DIM).astype(np.float32)
        agent.buffer.push(s, i % 3, float(i) * 1e-4, s, False)
    # Run 3 learner steps; target should match online after the 3rd.
    for _ in range(3):
        agent.learn()
    # Compare a single parameter tensor.
    online_p = next(agent.online.parameters()).detach().clone()
    target_p = next(agent.target.parameters()).detach().clone()
    torch.testing.assert_close(online_p, target_p)


# --------------------------------------------------------------------------- save/load


def test_save_load_roundtrip(tmp_path: Path):
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, seed=0)
    agent = DoubleDQNAgent(cfg)
    # Run a couple of learn steps so weights are no longer init-random.
    for i in range(70):
        s = np.random.default_rng(i).normal(size=STATE_DIM).astype(np.float32)
        agent.buffer.push(s, i % 3, float(i) * 1e-4, s, False)
    for _ in range(5):
        agent.learn()
    state = np.random.default_rng(99).normal(size=STATE_DIM).astype(np.float32)
    with torch.no_grad():
        q_before = agent.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))

    ckpt = tmp_path / "ckpt.pt"
    agent.save(ckpt)
    assert ckpt.is_file()

    cfg2 = DDQNConfig(state_dim=STATE_DIM, n_actions=3, seed=999)  # different seed
    agent2 = DoubleDQNAgent(cfg2)
    agent2.load(ckpt)
    with torch.no_grad():
        q_after = agent2.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    torch.testing.assert_close(q_before, q_after)
    assert agent2.learn_steps == agent.learn_steps


# --------------------------------------------------------------------------- smoke training


def test_smoke_training_on_synthetic_env():
    """A few hundred steps on a synthetic env: loss is finite, no exceptions."""
    env = _make_env(n=30, mode="random")
    obs, _ = env.reset(seed=0)
    cfg = DDQNConfig(state_dim=STATE_DIM, n_actions=3, batch_size=16,
                     min_buffer_to_learn=32, buffer_capacity=512,
                     target_update_freq=25, eps_decay_steps=100,
                     seed=0)
    agent = DoubleDQNAgent(cfg)
    losses: list[float] = []
    for _ in range(300):
        action = agent.select_action(obs)
        next_obs, reward, term, trunc, _info = env.step(action)
        agent.buffer.push(obs, action, reward, next_obs, bool(term or trunc))
        obs = next_obs
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        if term or trunc:
            obs, _ = env.reset()
    assert len(losses) > 0
    assert all(np.isfinite(l) for l in losses)
