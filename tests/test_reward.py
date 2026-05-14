"""M4 reward tests: sign, unit consistency, force-close magnitude, no next-bar."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.envs.eurusd_intraday_env import EURUSDIntradayTradingEnv
from src.features.state_builder import N_PRICE_FEATURES


# Re-use helpers via copy (kept self-contained — tests files don't share helpers).
def _env_cfg(warmup_bars: int = 3,
             reward_scaling: float = 1.0,
             commission_price: float = 0.0,
             unit_size: float = 100000.0,
             initial_equity: float = 10000.0,
             execution: str = "current_close",
             spread_cost_factor: float = 1.0) -> dict:
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
            "spread_cost_factor": spread_cost_factor,
        },
        "portfolio": {"initial_equity": initial_equity, "unit_size": unit_size},
        "execution": execution,
        "reward": {"scaling": reward_scaling, "stop_equity_floor": 0.0},
        "split": {"train_months": 1, "test_months": 1},
        "seed": 42,
    }


def _make_df(day: str, closes: list[float], spreads: list[int] | int = 10) -> pd.DataFrame:
    n = len(closes)
    spreads_arr = (
        np.full(n, int(spreads), dtype="int64") if isinstance(spreads, int)
        else np.asarray(spreads, dtype="int64")
    )
    start = pd.Timestamp(f"{day} 02:00:00", tz="UTC")
    idx = pd.date_range(start=start, periods=n, freq="1min")
    close = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {
            "open":  close,
            "high":  close + 1e-5,
            "low":   close - 1e-5,
            "close": close,
            "tick_volume": np.full(n, 10, dtype="int64"),
            "spread":      spreads_arr,
            "real_volume": np.zeros(n, dtype="int64"),
            "session_day": [pd.Timestamp(day).date()] * n,
            "bar_idx_in_day": np.arange(n, dtype="int64"),
        },
        index=idx,
    )


def _make_env(df: pd.DataFrame, cfg: dict) -> EURUSDIntradayTradingEnv:
    pf = np.zeros((len(df), N_PRICE_FEATURES), dtype=np.float32)
    return EURUSDIntradayTradingEnv(df, pf, cfg, mode="sequential")


# --------------------------------------------------------------------------- baselines: zero PnL


def test_flat_on_constant_price_gives_zero_reward():
    """Position never opens → mtm = 0, txn_cost = 0, reward = 0 throughout."""
    df = _make_df("2024-03-04", closes=[1.10000] * 10, spreads=10)
    env = _make_env(df, _env_cfg(warmup_bars=3))
    env.reset(seed=0)
    total_reward = 0.0
    while True:
        _, r, terminated, _, info = env.step(1)  # flat
        total_reward += r
        assert info["raw_pnl"] == 0.0
        assert info["transaction_cost"] == 0.0
        assert info["net_pnl"] == 0.0
        if terminated:
            break
    assert total_reward == 0.0


# --------------------------------------------------------------------------- sign / magnitude


def test_long_when_price_rises_gives_positive_pnl():
    # Prices: warmup(3) + 4 trading bars, monotone up.
    closes = [1.10000, 1.10001, 1.10002, 1.10003, 1.10004, 1.10005, 1.10006]
    df = _make_df("2024-03-04", closes=closes, spreads=0)  # zero spread to isolate sign
    env = _make_env(df, _env_cfg(warmup_bars=3, commission_price=0.0))
    env.reset(seed=0)
    # First action: go long. mtm = prev_pos(0) * ... = 0 on this bar; cost applied.
    _, _, _, _, info0 = env.step(2)
    assert info0["raw_pnl"] == 0.0  # prev_pos was 0
    # Subsequent steps while holding long should be positive (price rises).
    rewards = []
    while True:
        _, r, terminated, _, info = env.step(2)  # stay long
        rewards.append((r, info["net_pnl"]))
        if terminated:
            # Force-close at the last bar; even then net_pnl on that bar should
            # be the carry minus zero cost (spread=0, position already long, but
            # forced flat: cost = 1 * 0 * unit_size = 0; mtm = +ve).
            assert info["forced_close"]
            break
    # Sum of net_pnls > 0 on a rising series with zero costs.
    assert sum(p for _, p in rewards) > 0


def test_short_when_price_falls_gives_positive_pnl():
    closes = [1.10000, 1.09999, 1.09998, 1.09997, 1.09996, 1.09995]
    df = _make_df("2024-03-04", closes=closes, spreads=0)
    env = _make_env(df, _env_cfg(warmup_bars=3))
    env.reset(seed=0)
    env.step(0)  # short
    total = 0.0
    while True:
        _, _, terminated, _, info = env.step(0)
        total += info["net_pnl"]
        if terminated:
            break
    assert total > 0


def test_long_when_price_falls_gives_negative_pnl():
    closes = [1.10000, 1.09999, 1.09998, 1.09997, 1.09996, 1.09995]
    df = _make_df("2024-03-04", closes=closes, spreads=0)
    env = _make_env(df, _env_cfg(warmup_bars=3))
    env.reset(seed=0)
    env.step(2)  # long
    total = 0.0
    while True:
        _, _, terminated, _, info = env.step(2)
        total += info["net_pnl"]
        if terminated:
            break
    assert total < 0


# --------------------------------------------------------------------------- transaction-cost


def test_transaction_cost_zero_when_no_position_change():
    # 7 bars so the "hold" step is NOT the force-close step (force-close is a
    # position change long→flat and would incur cost).
    closes = [1.10000 + i * 1e-5 for i in range(7)]
    df = _make_df("2024-03-04", closes=closes, spreads=20)
    env = _make_env(df, _env_cfg(warmup_bars=3))
    env.reset(seed=0)
    env.step(2)  # open long -> cost applied (bar_idx 3)
    # Subsequent step with same action and NOT the last bar -> no cost.
    _, _, _, _, info = env.step(2)  # bar_idx 4, not force-close
    assert info["transaction_cost"] == 0.0


def test_transaction_cost_scales_with_delta_position():
    closes = [1.10000] * 10
    df = _make_df("2024-03-04", closes=closes, spreads=20)
    env = _make_env(df, _env_cfg(warmup_bars=3))
    env.reset(seed=0)

    # 0 -> long: |Δ| = 1.
    _, _, _, _, info_open_long = env.step(2)
    cost_open = info_open_long["transaction_cost"]
    # long -> short: |Δ| = 2.
    _, _, _, _, info_flip = env.step(0)
    cost_flip = info_flip["transaction_cost"]

    assert cost_open > 0
    np.testing.assert_allclose(cost_flip, 2 * cost_open, rtol=0, atol=1e-9)


def test_transaction_cost_unit_consistency_account_currency():
    """txn_cost and mtm_pnl must BOTH be ∝ unit_size, so account-currency PnL is consistent."""
    # 7 bars: warmup=3 → trading bars at idx 3..6. We open on bar 3 and
    # measure the carry on bar 4 (NOT force-close).
    closes = [1.10000 + i * 1e-4 for i in range(7)]  # +10 pips/bar
    df = _make_df("2024-03-04", closes=closes, spreads=10)
    point_size = 1e-5
    unit_size = 100000.0

    env = _make_env(df, _env_cfg(warmup_bars=3, unit_size=unit_size))
    env.reset(seed=0)
    # Open long at bar 3; cost = |1 - 0| * (10 * 1e-5 + 0) * 100000 = 10.0 USD
    _, _, _, _, info_open = env.step(2)
    expected_cost = 1.0 * (10 * point_size + 0.0) * unit_size
    np.testing.assert_allclose(info_open["transaction_cost"], expected_cost, atol=1e-9)
    # mtm_pnl on the open bar is 0 (prev_pos = 0).
    assert info_open["mtm_pnl"] == 0.0
    # Bar 4 (NOT force-close): hold long, price rose by 0.0001 = 10 pips.
    # mtm = 1 * (close[4] - close[3]) * 100000 = 0.0001 * 100000 = 10 USD.
    _, _, _, _, info_carry = env.step(2)
    expected_mtm = 1.0 * (closes[4] - closes[3]) * unit_size
    np.testing.assert_allclose(info_carry["mtm_pnl"], expected_mtm, atol=1e-9)
    # No position change → txn_cost = 0; net_pnl = mtm.
    assert info_carry["transaction_cost"] == 0.0
    np.testing.assert_allclose(info_carry["net_pnl"], expected_mtm, atol=1e-9)


# --------------------------------------------------------------------------- reward magnitude / scaling


def test_reward_equals_net_pnl_over_initial_equity():
    closes = [1.10000, 1.10005, 1.10010, 1.10015, 1.10020]
    df = _make_df("2024-03-04", closes=closes, spreads=0)
    initial_equity = 10000.0
    env = _make_env(df, _env_cfg(warmup_bars=3, reward_scaling=1.0,
                                 initial_equity=initial_equity))
    env.reset(seed=0)
    _, reward, _, _, info = env.step(2)  # open long; mtm=0 (prev_pos=0)
    np.testing.assert_allclose(reward, info["net_pnl"] / initial_equity, atol=1e-12)
    _, reward, _, _, info = env.step(2)  # hold long
    np.testing.assert_allclose(reward, info["net_pnl"] / initial_equity, atol=1e-12)


def test_reward_scaling_applies_multiplicatively():
    closes = [1.10000, 1.10001, 1.10002, 1.10003, 1.10004]
    df = _make_df("2024-03-04", closes=closes, spreads=0)
    env1 = _make_env(df, _env_cfg(warmup_bars=3, reward_scaling=1.0))
    env100 = _make_env(df, _env_cfg(warmup_bars=3, reward_scaling=100.0))
    env1.reset(seed=0); env100.reset(seed=0)
    r1 = 0.0; r100 = 0.0
    while True:
        _, r, terminated, _, _ = env1.step(2)
        r1 += r
        _, rr, term100, _, _ = env100.step(2)
        r100 += rr
        if terminated or term100:
            break
    np.testing.assert_allclose(r100, 100.0 * r1, atol=1e-9)


# --------------------------------------------------------------------------- force-close: no next bar


def test_force_close_does_not_index_past_end():
    """The env must compute the last-bar PnL without reading any bar beyond _last_bar_idx.

    Construction: a frame with EXACTLY enough bars so the trading window includes the
    last index. Running to termination must not raise IndexError and must produce a
    forced_close info flag.
    """
    closes = [1.10000, 1.10005, 1.10010, 1.10015, 1.10020]  # warmup=3, last bar = idx 4
    df = _make_df("2024-03-04", closes=closes, spreads=10)
    env = _make_env(df, _env_cfg(warmup_bars=3))
    env.reset(seed=0)
    # Hold long: open on bar 3, carry on bar 4 (last bar -> force-close).
    env.step(2)
    _, _, terminated, _, info = env.step(2)
    assert terminated
    assert info["forced_close"]
    # mtm = 1 * (1.10020 - 1.10015) * 100000 = 0.5 USD
    np.testing.assert_allclose(info["mtm_pnl"], 0.00005 * 100000.0, atol=1e-9)
    # Cost = |0 - 1| * (10 * 1e-5) * 100000 = 10 USD (flattening)
    np.testing.assert_allclose(info["transaction_cost"], 1.0 * (10 * 1e-5) * 100000.0, atol=1e-9)


def test_spread_cost_factor_halves_cost():
    """spread_cost_factor=0.5 must produce exactly half the per-side cost vs 1.0."""
    closes = [1.10000] * 8  # flat prices to isolate cost
    df = _make_df("2024-03-04", closes=closes, spreads=20)

    env_full = _make_env(df, _env_cfg(warmup_bars=3, spread_cost_factor=1.0))
    env_half = _make_env(df, _env_cfg(warmup_bars=3, spread_cost_factor=0.5))
    env_full.reset(seed=0)
    env_half.reset(seed=0)
    _, _, _, _, info_full = env_full.step(2)  # open long, single-side cost
    _, _, _, _, info_half = env_half.step(2)
    np.testing.assert_allclose(info_half["transaction_cost"],
                               0.5 * info_full["transaction_cost"],
                               rtol=0, atol=1e-9)
    # Sanity: cost > 0 in both cases.
    assert info_full["transaction_cost"] > 0
    assert info_half["transaction_cost"] > 0


def test_two_day_episode_independence():
    """Equity and position reset to initial values at every reset (new session_day)."""
    df1 = _make_df("2024-03-04", closes=[1.10000] * 7, spreads=10)
    df2 = _make_df("2024-03-05", closes=[1.20000] * 7, spreads=10)
    df = pd.concat([df1, df2], axis=0)
    env = _make_env(df, _env_cfg(warmup_bars=3))
    env.reset(seed=0)
    # Burn through day 1 holding long; equity will be negative due to costs.
    while True:
        _, _, terminated, _, _ = env.step(2)
        if terminated:
            break
    eq_after_day1 = env.equity
    assert eq_after_day1 < 10000.0  # incurred costs
    # Day 2: reset → equity should be back to initial.
    env.reset(seed=0)
    assert env.equity == 10000.0
    assert env.position == 0
    assert env.daily_pnl == 0.0
