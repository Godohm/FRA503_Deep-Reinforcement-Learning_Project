"""Gymnasium environment for EURUSD intraday RL training.

One episode = one session_day; one step = one 1-minute bar. Execution model is
``current_close`` (force-close at the last bar always uses current_close so
that no non-existent next-bar field is ever read).

State (15-D) layout is defined in ``src.features.state_builder``:
  - indices 0..9: scaled price features (precomputed, see ``PriceFeatureScaler``)
  - indices 10..14: positional features (TL, POS, PR, DR, HT), updated per step

Reward at step t:
    reward_t = (net_pnl_t / initial_equity) * reward_scaling

with
    mtm_pnl_t   = prev_pos * (close_t - close_{t-1}) * unit_size
    per_side    = spread_t * point_size * spread_cost_factor + commission_price
    txn_cost_t  = |target - prev_pos| * per_side * unit_size
    net_pnl_t   = mtm_pnl_t - txn_cost_t

Both ``mtm_pnl`` and ``txn_cost`` are in *account currency* (each multiplied
by ``unit_size``). The reward is therefore a dimensionless fractional return.

**Transaction-cost convention.** The CSV ``spread`` column is the FULL broker
spread in points. We multiply by ``spread_cost_factor`` (config, default
``1.0``) when computing per-side cost:

  - ``spread_cost_factor = 1.0`` (default): charge the full spread on every
    position-change side. A ``0 → long → flat`` round-trip pays
    ``2 * spread * point_size * unit_size``. Conservative; assumes the agent
    crosses the spread on every order it sends.
  - ``spread_cost_factor = 0.5``: half-spread per side. A round-trip pays
    exactly one full spread (closer to mid-priced execution accounting).

A pure ``0 → ±1`` open or ``±1 → 0`` close has ``|Δpos| = 1`` and pays one
per-side cost. A ``+1 → −1`` flip has ``|Δpos| = 2`` and pays two per-side
costs in a single bar — this matches reality (a flip is two separate fills).

The env intentionally does not load data on its own — pass in a preprocessed
DataFrame + the precomputed (scaled) price features + the env config. This
keeps the env testable and keeps scaler-leakage prevention at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.features.state_builder import (
    N_POSITIONAL_FEATURES,
    N_PRICE_FEATURES,
    STATE_DIM,
    assemble_state,
)

# Action <-> position mapping is fixed and tested.
# 0 -> short (-1), 1 -> flat (0), 2 -> long (+1)
ACTION_TO_POSITION: tuple[int, int, int] = (-1, 0, 1)
POSITION_TO_ACTION: dict[int, int] = {-1: 0, 0: 1, 1: 2}


@dataclass
class _DaySlice:
    """Per-session_day data cached at env init for fast step()."""

    day: Any  # session_day label (date or string)
    closes: np.ndarray          # float64, shape (N,)
    spreads: np.ndarray         # int64, shape (N,) — raw points
    price_features_scaled: np.ndarray  # float32, shape (N, 10) — NaN before warmup
    bar_idx_in_day: np.ndarray  # int64, shape (N,)
    times_utc: pd.DatetimeIndex  # tz-aware UTC, shape (N,) — preserved for trade logs


class EURUSDIntradayTradingEnv(gym.Env):
    """Single-instrument 1-min intraday env for EURUSD CFD, per ``configs/env.yaml``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``preprocess()`` for a single split (train or test). Must
        contain ``session_day``, ``bar_idx_in_day``, and OHLC + spread columns.
    price_features_scaled : np.ndarray
        Shape ``(len(df), 10)``. Already scaled by the train-fit
        ``PriceFeatureScaler``. NaN rows are expected in the warmup region.
    env_cfg : dict
        Parsed ``configs/env.yaml``.
    mode : str, optional
        ``"sequential"`` (default) cycles through session_days in chronological
        order — used for deterministic eval/backtest. ``"random"`` shuffles
        days each pass and is intended for training.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        price_features_scaled: np.ndarray,
        env_cfg: dict[str, Any],
        mode: str = "sequential",
    ) -> None:
        super().__init__()

        if mode not in ("sequential", "random"):
            raise ValueError(f"mode must be 'sequential' or 'random', got {mode!r}")
        if "session_day" not in df.columns:
            raise ValueError("df must contain a 'session_day' column.")
        if price_features_scaled.shape[0] != len(df):
            raise ValueError(
                f"price_features_scaled rows ({price_features_scaled.shape[0]}) "
                f"must equal len(df) ({len(df)})."
            )
        if price_features_scaled.shape[1] != N_PRICE_FEATURES:
            raise ValueError(
                f"price_features_scaled must have {N_PRICE_FEATURES} columns, "
                f"got {price_features_scaled.shape[1]}."
            )

        self.mode = mode
        self.env_cfg = env_cfg

        sess_cfg = env_cfg["session"]
        cost_cfg = env_cfg["costs"]
        port_cfg = env_cfg["portfolio"]
        reward_cfg = env_cfg.get("reward", {})

        self.warmup_bars: int = int(sess_cfg.get("warmup_bars", 60))
        self.display_tz: str = str(sess_cfg.get("display_tz", "Asia/Bangkok"))
        self.point_size: float = float(cost_cfg["point_size"])
        self.commission_price: float = float(cost_cfg.get("commission_price", 0.0))
        # Transaction cost convention (see docstring at module top): the spread
        # column is the FULL broker spread in points. We multiply it by
        # spread_cost_factor when computing per-side cost. Default 1.0 means
        # "charge full spread on every position change side" (conservative).
        # Set to 0.5 for the half-spread-per-side convention (a round trip
        # then pays one full spread).
        self.spread_cost_factor: float = float(cost_cfg.get("spread_cost_factor", 1.0))
        self.initial_equity: float = float(port_cfg["initial_equity"])
        self.unit_size: float = float(port_cfg["unit_size"])
        self.reward_scaling: float = float(reward_cfg.get("scaling", 1.0))
        self.stop_equity_floor: float = float(reward_cfg.get("stop_equity_floor", 0.0))
        self.execution: str = str(env_cfg.get("execution", "current_close"))
        if self.execution not in ("current_close", "next_open"):
            raise ValueError(f"execution must be 'current_close' or 'next_open'; got {self.execution!r}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Cache per-day arrays once; step() never touches pandas.
        self._days: list[_DaySlice] = self._slice_by_day(df, price_features_scaled)
        if not self._days:
            raise ValueError("df produced no session_days.")
        self._day_order: list[int] = list(range(len(self._days)))
        self._day_cursor: int = 0  # index into _day_order
        self._np_random_seed_buffer: list[int] = []  # for reproducible random mode

        # Episode state (initialised in reset).
        self._current_day: Optional[_DaySlice] = None
        self._bar_idx: int = 0
        self._last_bar_idx: int = 0
        self._trading_length: int = 0
        self.position: int = 0
        self.equity: float = self.initial_equity
        self.daily_pnl: float = 0.0
        self.entry_price: float = 0.0
        self.holding_time: int = 0

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _slice_by_day(
        df: pd.DataFrame, price_features_scaled: np.ndarray
    ) -> list[_DaySlice]:
        slices: list[_DaySlice] = []
        # Stable groupby preserves the chronological order produced by preprocess.
        for day, day_df in df.groupby("session_day", sort=False):
            row_positions = df.index.get_indexer(day_df.index)
            slices.append(
                _DaySlice(
                    day=day,
                    closes=day_df["close"].to_numpy(dtype="float64"),
                    spreads=day_df["spread"].to_numpy(dtype="int64"),
                    price_features_scaled=price_features_scaled[row_positions].astype(
                        np.float32, copy=False
                    ),
                    bar_idx_in_day=day_df["bar_idx_in_day"].to_numpy(dtype="int64"),
                    times_utc=day_df.index,
                )
            )
        return slices

    # ------------------------------------------------------------------ helpers

    @property
    def n_days(self) -> int:
        return len(self._days)

    @property
    def current_day_label(self) -> Any:
        if self._current_day is None:
            return None
        return self._current_day.day

    def _build_observation(self) -> np.ndarray:
        """Build the 15-D state at the current bar.

        Uses the (already scaled) price features for ``self._bar_idx`` and the
        agent's running positional state.
        """
        day = self._current_day
        assert day is not None
        price = day.price_features_scaled[self._bar_idx]
        # NaN price features only occur in warmup, which we never expose at step time.
        # Defensive: replace any leftover NaN with 0 to avoid poisoning downstream nets.
        if np.isnan(price).any():
            price = np.where(np.isnan(price), 0.0, price)

        # Positional features (TL, POS, PR, DR, HT)
        bars_remaining = self._last_bar_idx - self._bar_idx
        tl = float(bars_remaining) / float(self._trading_length) if self._trading_length else 0.0

        close_t = day.closes[self._bar_idx]
        if self.position != 0 and self.entry_price > 0.0:
            pr = float(self.position) * (close_t - self.entry_price) / self.entry_price
        else:
            pr = 0.0
        dr = self.daily_pnl / self.initial_equity
        ht = float(self.holding_time) / float(self._trading_length) if self._trading_length else 0.0

        return assemble_state(price, (tl, float(self.position), pr, dr, ht))

    def _next_day_index(self) -> int:
        if self.mode == "random":
            return int(self.np_random.integers(0, len(self._days)))
        # sequential
        idx = self._day_order[self._day_cursor]
        self._day_cursor = (self._day_cursor + 1) % len(self._day_order)
        return idx

    # ------------------------------------------------------------------ API

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Allow callers to pin a specific day for tests/eval.
        day_idx: Optional[int] = None
        if options is not None and "day_index" in options:
            day_idx = int(options["day_index"])

        if day_idx is None:
            day_idx = self._next_day_index()
        if not (0 <= day_idx < len(self._days)):
            raise IndexError(f"day_index={day_idx} out of range [0, {len(self._days)}).")

        self._current_day = self._days[day_idx]
        n_bars = len(self._current_day.closes)
        if n_bars <= self.warmup_bars:
            raise ValueError(
                f"Session {self._current_day.day} has {n_bars} bars but warmup_bars="
                f"{self.warmup_bars}; not enough for a single trading step."
            )

        self._bar_idx = self.warmup_bars
        self._last_bar_idx = n_bars - 1
        self._trading_length = self._last_bar_idx - self.warmup_bars + 1

        self.position = 0
        self.equity = self.initial_equity
        self.daily_pnl = 0.0
        self.entry_price = 0.0
        self.holding_time = 0

        obs = self._build_observation()
        info = {
            "session_day": str(self._current_day.day),
            "n_bars": int(n_bars),
            "trading_length": int(self._trading_length),
            "warmup_bars": int(self.warmup_bars),
            "day_index": int(day_idx),
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._current_day is None:
            raise RuntimeError("step() called before reset().")
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action}; must be 0, 1, or 2.")

        day = self._current_day
        prev_pos = self.position
        target = ACTION_TO_POSITION[int(action)]
        forced_close = self._bar_idx == self._last_bar_idx
        if forced_close:
            target = 0

        # --- prices & cost -------------------------------------------------
        close_t = day.closes[self._bar_idx]
        close_prev = (
            day.closes[self._bar_idx - 1] if self._bar_idx > 0 else close_t
        )
        spread_t = day.spreads[self._bar_idx]

        # All quantities below are in ACCOUNT CURRENCY (both ∝ unit_size).
        # Cost convention: per position-change side we charge
        #     (spread_t * point_size * spread_cost_factor + commission_price) * unit_size
        # The default spread_cost_factor=1.0 means "full broker spread per side"
        # (a 0→long→flat round-trip then pays 2 × spread × point_size × unit_size).
        # Set spread_cost_factor=0.5 in env.yaml for the half-spread-per-side
        # convention where a round-trip costs exactly one full spread.
        mtm_pnl = float(prev_pos) * (close_t - close_prev) * self.unit_size
        delta_pos = abs(target - prev_pos)
        per_side_price_cost = (
            float(spread_t) * self.point_size * self.spread_cost_factor
            + self.commission_price
        )
        txn_cost = float(delta_pos) * per_side_price_cost * self.unit_size
        net_pnl = mtm_pnl - txn_cost

        # --- update internal state ----------------------------------------
        self.position = target
        self.equity += net_pnl
        self.daily_pnl += net_pnl
        if target == 0:
            self.entry_price = 0.0
        elif target != prev_pos:
            self.entry_price = float(close_t)
        # Holding-time: reset on position change, else increment.
        if target != prev_pos:
            self.holding_time = 0
        else:
            self.holding_time += 1

        reward = (net_pnl / self.initial_equity) * self.reward_scaling
        terminated = forced_close or (self.equity <= self.stop_equity_floor)
        truncated = False

        # --- info dict ----------------------------------------------------
        time_utc_t = day.times_utc[self._bar_idx]
        info: dict[str, Any] = {
            "raw_pnl": float(mtm_pnl),
            "transaction_cost": float(txn_cost),
            "net_pnl": float(net_pnl),
            "current_position": int(self.position),
            "action": int(action),
            "equity": float(self.equity),
            "daily_return": float(self.daily_pnl / self.initial_equity),
            "forced_close": bool(forced_close),
            "mtm_pnl": float(mtm_pnl),
            "bar_idx": int(self._bar_idx),
            "bar_idx_in_day": int(day.bar_idx_in_day[self._bar_idx]),
            "session_day": str(day.day),
            "time_utc": time_utc_t,
            "time_local": time_utc_t.tz_convert(self.display_tz),
            "close": float(close_t),
            "spread": int(spread_t),
        }

        # --- next observation ---------------------------------------------
        if terminated:
            obs_next = np.zeros(STATE_DIM, dtype=np.float32)
        else:
            self._bar_idx += 1
            obs_next = self._build_observation()

        return obs_next, float(reward), bool(terminated), truncated, info
