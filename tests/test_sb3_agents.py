"""M7 tests: A2C and PPO via Stable-Baselines3 on EURUSDIntradayTradingEnv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def _make_multi_day_df(n_days: int = 4, bars_per_day: int = 40) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base = 1.10000
    for d in range(n_days):
        day = f"2024-03-{4 + d:02d}"
        start = pd.Timestamp(f"{day} 02:00:00", tz="UTC")
        idx = pd.date_range(start=start, periods=bars_per_day, freq="1min")
        close = np.linspace(base, base + (bars_per_day - 1) * 1e-5, bars_per_day, dtype="float64")
        base = close[-1]
        frames.append(pd.DataFrame({
            "open":  close,
            "high":  close + 1e-5,
            "low":   close - 1e-5,
            "close": close,
            "tick_volume": np.full(bars_per_day, 10, dtype="int64"),
            "spread":      np.full(bars_per_day, 10, dtype="int64"),
            "real_volume": np.zeros(bars_per_day, dtype="int64"),
            "session_day": [pd.Timestamp(day).date()] * bars_per_day,
            "bar_idx_in_day": np.arange(bars_per_day, dtype="int64"),
        }, index=idx))
    return pd.concat(frames, axis=0)


def _make_env(mode: str = "random") -> EURUSDIntradayTradingEnv:
    df = _make_multi_day_df()
    pf = np.zeros((len(df), N_PRICE_FEATURES), dtype=np.float32)
    return EURUSDIntradayTradingEnv(df, pf, _env_cfg(warmup_bars=3), mode=mode)


# --------------------------------------------------------------------------- A2C


def test_a2c_model_creates_with_matched_arch():
    from stable_baselines3 import A2C

    env = _make_env(mode="random")
    model = A2C(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": [128, 128]},
        n_steps=5,
        verbose=0,
        seed=0,
        device="cpu",
    )
    # Predict returns an action in the env's discrete space.
    obs, _ = env.reset(seed=0)
    action, _state = model.predict(obs, deterministic=True)
    assert int(np.asarray(action).item()) in (0, 1, 2)


def test_a2c_predict_obs_batch_shape_compatible():
    """SB3's predict accepts (state_dim,) and produces a scalar action."""
    from stable_baselines3 import A2C

    env = _make_env(mode="random")
    model = A2C("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]},
                n_steps=5, seed=0, verbose=0, device="cpu")
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    action, _state = model.predict(obs, deterministic=True)
    assert np.asarray(action).shape == ()


def test_a2c_short_learn_does_not_crash():
    from stable_baselines3 import A2C

    env = _make_env(mode="random")
    model = A2C("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]},
                n_steps=5, seed=0, verbose=0, device="cpu")
    model.learn(total_timesteps=100, progress_bar=False)


# --------------------------------------------------------------------------- PPO


def test_ppo_model_creates_with_matched_arch():
    from stable_baselines3 import PPO

    env = _make_env(mode="random")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": [128, 128]},
        n_steps=64,
        batch_size=16,
        n_epochs=2,
        verbose=0,
        seed=0,
        device="cpu",
    )
    obs, _ = env.reset(seed=0)
    action, _state = model.predict(obs, deterministic=True)
    assert int(np.asarray(action).item()) in (0, 1, 2)


def test_ppo_short_learn_does_not_crash():
    from stable_baselines3 import PPO

    env = _make_env(mode="random")
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]},
                n_steps=64, batch_size=16, n_epochs=2,
                seed=0, verbose=0, device="cpu")
    model.learn(total_timesteps=128, progress_bar=False)


def test_ppo_save_and_load_roundtrip(tmp_path: Path):
    from stable_baselines3 import PPO

    env = _make_env(mode="random")
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]},
                n_steps=64, batch_size=16, n_epochs=1,
                seed=0, verbose=0, device="cpu")
    model.learn(total_timesteps=64, progress_bar=False)
    obs, _ = env.reset(seed=0)
    action_before, _ = model.predict(obs, deterministic=True)
    path = tmp_path / "ppo.zip"
    model.save(str(path))
    assert path.is_file()

    loaded = PPO.load(str(path), env=env, device="cpu")
    action_after, _ = loaded.predict(obs, deterministic=True)
    assert int(np.asarray(action_before).item()) == int(np.asarray(action_after).item())


# --------------------------------------------------------------------------- Monitor compatibility


def test_monitor_wraps_env_and_logs_episodes(tmp_path: Path):
    from stable_baselines3.common.monitor import Monitor

    env = _make_env(mode="sequential")
    # Monitor appends ".monitor.csv" to the filename you pass.
    monitor_base = tmp_path / "mon"
    wrapped = Monitor(env, filename=str(monitor_base))

    obs, _ = wrapped.reset(seed=0)
    done = False
    while not done:
        obs, _r, term, trunc, _info = wrapped.step(wrapped.action_space.sample())
        done = term or trunc
    # Monitor writes <filename>.monitor.csv with a header + per-episode rows.
    csv_path = Path(str(monitor_base) + ".monitor.csv")
    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8")
    assert "r,l,t" in content


# --------------------------------------------------------------------------- shared utility helpers


def test_sb3_greedy_policy_returns_python_int():
    """sb3_common._sb3_greedy_policy must return a plain int per Policy contract."""
    from stable_baselines3 import PPO

    from src.agents.sb3_common import _sb3_greedy_policy

    env = _make_env(mode="sequential")
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]},
                n_steps=64, batch_size=16, n_epochs=1, seed=0, verbose=0, device="cpu")
    pi = _sb3_greedy_policy(model)
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    a = pi(obs)
    assert isinstance(a, int)
    assert a in (0, 1, 2)
