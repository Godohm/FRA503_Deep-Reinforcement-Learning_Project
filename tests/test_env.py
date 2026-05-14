"""M4 environment tests: reset / step / obs shape / force-close / no-next-bar."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.envs.eurusd_intraday_env import (
    ACTION_TO_POSITION,
    EURUSDIntradayTradingEnv,
)
from src.features.state_builder import N_PRICE_FEATURES, STATE_DIM


# --------------------------------------------------------------------------- helpers


def _env_cfg(warmup_bars: int = 5,
             reward_scaling: float = 1.0,
             commission_price: float = 0.0,
             execution: str = "current_close") -> dict:
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
            "commission_price": commission_price,
        },
        "portfolio": {"initial_equity": 10000.0, "unit_size": 100000.0},
        "execution": execution,
        "reward": {"scaling": reward_scaling, "stop_equity_floor": 0.0},
        "split": {"train_months": 1, "test_months": 1},
        "seed": 42,
    }


def _make_day_df(day: str, n_bars: int = 20, prices: list[float] | None = None,
                 spread: int = 10) -> pd.DataFrame:
    """Single-session DataFrame in the shape preprocess.py emits."""
    start = pd.Timestamp(f"{day} 02:00:00", tz="UTC")
    idx = pd.date_range(start=start, periods=n_bars, freq="1min")
    if prices is None:
        prices = [1.10000 + 1e-5 * i for i in range(n_bars)]
    close = np.array(prices, dtype="float64")
    return pd.DataFrame(
        {
            "open":  close,
            "high":  close + 1e-5,
            "low":   close - 1e-5,
            "close": close,
            "tick_volume": np.full(n_bars, 10, dtype="int64"),
            "spread":      np.full(n_bars, spread, dtype="int64"),
            "real_volume": np.zeros(n_bars, dtype="int64"),
            "session_day": [pd.Timestamp(day).date()] * n_bars,
            "bar_idx_in_day": np.arange(n_bars, dtype="int64"),
        },
        index=idx,
    )


def _make_env(df: pd.DataFrame, cfg: dict | None = None,
              mode: str = "sequential") -> EURUSDIntradayTradingEnv:
    cfg = cfg or _env_cfg()
    # Use a deterministic zero-mean scaled feature matrix — the env only needs
    # the SHAPE to match. The reward/PnL logic does NOT depend on the price
    # features; it reads raw closes & spread from the df directly.
    price_features_scaled = np.zeros((len(df), N_PRICE_FEATURES), dtype=np.float32)
    return EURUSDIntradayTradingEnv(df, price_features_scaled, cfg, mode=mode)


# --------------------------------------------------------------------------- spaces / reset


def test_observation_and_action_space():
    env = _make_env(_make_day_df("2024-03-04", n_bars=20))
    assert env.observation_space.shape == (STATE_DIM,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 3


def test_reset_returns_correct_obs_shape_and_info():
    env = _make_env(_make_day_df("2024-03-04", n_bars=20))
    obs, info = env.reset(seed=0)
    assert obs.shape == (STATE_DIM,)
    assert obs.dtype == np.float32
    assert "session_day" in info
    assert info["trading_length"] == 20 - 5  # n_bars - warmup_bars
    # POS, PR, DR, HT all start at zero (TL starts at 1.0).
    assert obs[N_PRICE_FEATURES + 1] == 0.0   # POS
    assert obs[N_PRICE_FEATURES + 2] == 0.0   # PR
    assert obs[N_PRICE_FEATURES + 3] == 0.0   # DR
    assert obs[N_PRICE_FEATURES + 4] == 0.0   # HT
    # TL at first step = (last - warmup) / trading_length = (19-5)/15 = 14/15.
    np.testing.assert_allclose(obs[N_PRICE_FEATURES + 0], 14 / 15, atol=1e-6)


def test_reset_seeks_past_warmup():
    env = _make_env(_make_day_df("2024-03-04", n_bars=10), cfg=_env_cfg(warmup_bars=3))
    env.reset(seed=0)
    # Internal bar_idx must equal warmup_bars after reset.
    assert env._bar_idx == 3


def test_reset_requires_enough_bars():
    df = _make_day_df("2024-03-04", n_bars=3)  # < warmup_bars=5
    env = _make_env(df)
    with pytest.raises(ValueError, match="not enough"):
        env.reset(seed=0)


# --------------------------------------------------------------------------- step / obs shape


def test_step_returns_five_tuple_and_shapes():
    env = _make_env(_make_day_df("2024-03-04", n_bars=20))
    env.reset(seed=0)
    out = env.step(1)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == (STATE_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert truncated is False
    for k in ("raw_pnl", "transaction_cost", "net_pnl", "current_position",
              "action", "equity", "daily_return", "forced_close", "mtm_pnl",
              "bar_idx_in_day", "session_day", "time_utc", "time_local",
              "close", "spread"):
        assert k in info, f"missing info key {k}"
    # time_utc must be tz-aware UTC; time_local must be tz-aware in display_tz.
    assert str(info["time_utc"].tz) == "UTC"
    assert str(info["time_local"].tz) == "Asia/Bangkok"


def test_action_to_position_mapping_matches_constants():
    assert ACTION_TO_POSITION == (-1, 0, 1)


def test_step_updates_position_per_action_mapping():
    env = _make_env(_make_day_df("2024-03-04", n_bars=20))
    env.reset(seed=0)
    _, _, _, _, info = env.step(2)  # long
    assert info["current_position"] == 1
    _, _, _, _, info = env.step(0)  # short
    assert info["current_position"] == -1
    _, _, _, _, info = env.step(1)  # flat
    assert info["current_position"] == 0


# --------------------------------------------------------------------------- termination & force-close


def test_episode_terminates_at_session_end():
    env = _make_env(_make_day_df("2024-03-04", n_bars=10), cfg=_env_cfg(warmup_bars=3))
    env.reset(seed=0)
    terminated = False
    n_steps = 0
    while not terminated:
        _, _, terminated, _, info = env.step(1)
        n_steps += 1
        if n_steps > 20:
            pytest.fail("Episode did not terminate within expected bar count.")
    # trading_length = n_bars - warmup_bars = 7
    assert n_steps == 7
    assert info["forced_close"] is True


def test_force_close_at_last_bar_uses_no_next_bar():
    """If the env tried to access bar_idx+1 on the last step, we'd see an IndexError.
    Build a frame where the last bar is *truly* the last (no bar after it)."""
    df = _make_day_df("2024-03-04", n_bars=8, prices=[1.10000 + i * 1e-5 for i in range(8)])
    env = _make_env(df, cfg=_env_cfg(warmup_bars=3))
    env.reset(seed=0)
    # Hold long all the way; force-close happens at the last bar.
    last_info = None
    last_terminated = False
    while not last_terminated:
        _, _, last_terminated, _, last_info = env.step(2)
    assert last_info["forced_close"] is True
    assert last_info["current_position"] == 0, "Force-close must flatten position."


def test_force_close_overrides_target():
    """Even if the agent's chosen action is long, the last bar must force position=0."""
    df = _make_day_df("2024-03-04", n_bars=8)
    env = _make_env(df, cfg=_env_cfg(warmup_bars=3))
    env.reset(seed=0)
    # Step until last bar; on the final step, request action=2 (long) — should be flattened.
    for _ in range(env._trading_length - 1):
        env.step(2)
    _, _, terminated, _, info = env.step(2)
    assert terminated and info["forced_close"]
    assert info["current_position"] == 0


# --------------------------------------------------------------------------- sequential mode determinism


def test_sequential_mode_cycles_days_in_order():
    df1 = _make_day_df("2024-03-04", n_bars=10)
    df2 = _make_day_df("2024-03-05", n_bars=10)
    df = pd.concat([df1, df2], axis=0)
    env = _make_env(df, cfg=_env_cfg(warmup_bars=3), mode="sequential")
    _, info1 = env.reset(seed=0)
    while True:
        _, _, terminated, _, _ = env.step(1)
        if terminated:
            break
    _, info2 = env.reset(seed=0)
    assert info1["session_day"] != info2["session_day"]
    # And a third reset should wrap back to day 1.
    while True:
        _, _, terminated, _, _ = env.step(1)
        if terminated:
            break
    _, info3 = env.reset(seed=0)
    assert info3["session_day"] == info1["session_day"]


def test_invalid_action_raises():
    env = _make_env(_make_day_df("2024-03-04", n_bars=20))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_step_before_reset_raises():
    env = _make_env(_make_day_df("2024-03-04", n_bars=20))
    with pytest.raises(RuntimeError):
        env.step(1)


# --------------------------------------------------------------------------- env_checker (gymnasium)


def test_gymnasium_check_env_passes():
    """Run gymnasium's built-in conformance checker."""
    from gymnasium.utils.env_checker import check_env

    df = _make_day_df("2024-03-04", n_bars=30)
    env = _make_env(df, cfg=_env_cfg(warmup_bars=5))
    # check_env will call reset() and step() with random actions internally.
    check_env(env, skip_render_check=True)
