"""Deterministic rollout through every session_day in an env, recording one
row of detailed state per step.

The output is the canonical trade-log DataFrame consumed by ``metrics.py``
and ``plots.py``.

Columns
-------
    session_day, time_utc, time_local, bar_idx_in_day, action,
    current_position, close, spread, raw_pnl, transaction_cost, net_pnl,
    mtm_pnl, equity, forced_close

``time_utc`` and ``time_local`` come from the env's per-day timestamp cache;
``time_local`` is ``time_utc`` converted to ``session.display_tz`` from the
env config (Asia/Bangkok by default).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.envs.eurusd_intraday_env import EURUSDIntradayTradingEnv

Policy = Callable[[np.ndarray], int]


def run_backtest(
    env: EURUSDIntradayTradingEnv,
    policy: Policy,
    n_days: int | None = None,
) -> pd.DataFrame:
    """Run the given policy across ``n_days`` session_days (default: all).

    Parameters
    ----------
    env : EURUSDIntradayTradingEnv
        Must be in ``sequential`` mode for reproducibility.
    policy : callable
        ``policy(obs) -> int`` returning a discrete action.
    n_days : int, optional
        Number of session_days to roll through. ``None`` runs through all
        days in the env (one episode each).
    """
    if env.mode != "sequential":
        raise ValueError("Backtest requires env.mode='sequential'; got 'random'.")

    total_days = env.n_days if n_days is None else min(int(n_days), env.n_days)

    rows: list[dict] = []
    for day_pos in range(total_days):
        obs, _info_reset = env.reset(options={"day_index": day_pos})
        terminated = False
        while not terminated:
            action = int(policy(obs))
            obs, _reward, terminated, _truncated, info = env.step(action)
            rows.append({
                "session_day": info["session_day"],
                "time_utc": info["time_utc"],
                "time_local": info["time_local"],
                "bar_idx_in_day": int(info["bar_idx_in_day"]),
                "action": int(info["action"]),
                "current_position": int(info["current_position"]),
                "close": float(info["close"]),
                "spread": int(info["spread"]),
                "raw_pnl": float(info["raw_pnl"]),
                "transaction_cost": float(info["transaction_cost"]),
                "net_pnl": float(info["net_pnl"]),
                "mtm_pnl": float(info["mtm_pnl"]),
                "equity": float(info["equity"]),
                "forced_close": bool(info["forced_close"]),
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "session_day", "time_utc", "time_local", "bar_idx_in_day",
                "action", "current_position", "close", "spread",
                "raw_pnl", "transaction_cost", "net_pnl", "mtm_pnl",
                "equity", "forced_close",
            ]
        )
    return pd.DataFrame(rows)
