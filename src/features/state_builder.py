"""Build the 15-dimensional state vector for the trading environment.

Layout
------
Index   Feature        Source
----- | ------------ | -------------------------------------------------
 0    | ret_1m       | log(close_t / close_{t-1})            (price)
 1    | ret_5m       |                                       (price)
 2    | ret_15m      |                                       (price)
 3    | ret_30m      |                                       (price)
 4    | ret_60m      |                                       (price)
 5    | spread       | spread_t * point_size  (price units)  (price)
 6    | macd_hist    | macd_line - signal_line               (price)
 7    | stoch_k      | %K, range [0, 100]                    (price)
 8    | rsi_norm     | RSI / 100, range [0, 1]               (price)
 9    | atr          | Wilder ATR (price units)              (price)
10    | TL           | bars_remaining / trading_length       (positional)
11    | POS          | current position in {-1, 0, +1}       (positional)
12    | PR           | unrealized return of open position    (positional)
13    | DR           | daily_pnl / initial_equity            (positional)
14    | HT           | holding_time / trading_length         (positional)

Design
------
- The 10 price features (0..9) only depend on the price/spread history and are
  computed once at dataset-prep time. They are then scaled by a train-fit
  ``PriceFeatureScaler`` (see ``normalization.py``).
- The 5 positional features (10..14) depend on the agent's running state and
  are computed inside the env at every step. They are *not* scaled — each one
  is bounded by design.
- Indicators are computed per ``session_day`` (groupby) to avoid contamination
  across the ~9 h overnight gap between sessions. With ``warmup_bars = 60`` in
  the env config, every state at ``bar_idx >= warmup_bars`` has fully-formed
  features.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .indicators import atr, log_returns, macd, rsi, stochastic_k

# Public ordering — DO NOT REORDER without updating the env, tests, and reports.
PRICE_FEATURE_NAMES: tuple[str, ...] = (
    "ret_1m",       # 0
    "ret_5m",       # 1
    "ret_15m",      # 2
    "ret_30m",      # 3
    "ret_60m",      # 4
    "spread",       # 5 -- price units
    "macd_hist",    # 6
    "stoch_k",      # 7
    "rsi_norm",     # 8 -- normalized to [0, 1]
    "atr",          # 9
)
POSITIONAL_FEATURE_NAMES: tuple[str, ...] = (
    "TL",   # 10
    "POS",  # 11
    "PR",   # 12
    "DR",   # 13
    "HT",   # 14
)
STATE_DIM: int = 15
N_PRICE_FEATURES: int = 10
N_POSITIONAL_FEATURES: int = 5
RETURN_PERIODS: tuple[int, ...] = (1, 5, 15, 30, 60)


def _compute_one_day(day_df: pd.DataFrame, point_size: float) -> pd.DataFrame:
    """Compute the 10 price-derived features for one session_day's bars.

    ``day_df`` must be a *single* contiguous session, already sorted by time
    and containing at least the columns ``open, high, low, close, spread``.
    """
    out = pd.DataFrame(index=day_df.index)
    close = day_df["close"].astype("float64")
    for period in RETURN_PERIODS:
        out[f"ret_{period}m"] = log_returns(close, period)
    out["spread"] = day_df["spread"].astype("float64") * float(point_size)
    _, _, hist = macd(close)
    out["macd_hist"] = hist
    out["stoch_k"] = stochastic_k(
        day_df["high"].astype("float64"),
        day_df["low"].astype("float64"),
        close,
    )
    out["rsi_norm"] = rsi(close) / 100.0
    out["atr"] = atr(
        day_df["high"].astype("float64"),
        day_df["low"].astype("float64"),
        close,
    )
    return out[list(PRICE_FEATURE_NAMES)]


def compute_price_features(df: pd.DataFrame, point_size: float) -> pd.DataFrame:
    """Compute the 10 price-derived state features for the whole dataset.

    Indicators are reset at every ``session_day`` boundary to prevent the
    overnight gap from leaking into the next day's first values.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``preprocess()``. Must contain ``session_day`` and the
        OHLC + spread columns. UTC-aware index.
    point_size : float
        Multiplier converting raw spread (in points) to price units, e.g.
        ``1e-5`` for a 5-digit EURUSD broker.

    Returns
    -------
    pd.DataFrame
        Same index as ``df``, columns in ``PRICE_FEATURE_NAMES`` order.
    """
    if "session_day" not in df.columns:
        raise ValueError("compute_price_features requires a 'session_day' column.")
    pieces: list[pd.DataFrame] = []
    # Preserve chronological order; sort by session_day for groupby stability.
    for _, day_df in df.groupby("session_day", sort=False):
        pieces.append(_compute_one_day(day_df, point_size=point_size))
    return pd.concat(pieces, axis=0).reindex(df.index)


def assemble_state(
    price_features_scaled: np.ndarray,
    positional: Iterable[float],
) -> np.ndarray:
    """Concatenate the 10 scaled price features and the 5 positional features.

    Parameters
    ----------
    price_features_scaled : np.ndarray
        Shape ``(10,)``. Already scaled by the train-fit ``PriceFeatureScaler``.
    positional : Iterable[float]
        Length 5, in the order ``(TL, POS, PR, DR, HT)``.

    Returns
    -------
    np.ndarray
        Shape ``(15,)``, dtype ``float32``.
    """
    price = np.asarray(price_features_scaled, dtype=np.float32).reshape(-1)
    if price.shape[0] != N_PRICE_FEATURES:
        raise ValueError(f"price_features_scaled must have {N_PRICE_FEATURES} entries, "
                         f"got {price.shape[0]}.")
    pos = np.asarray(list(positional), dtype=np.float32).reshape(-1)
    if pos.shape[0] != N_POSITIONAL_FEATURES:
        raise ValueError(f"positional must have {N_POSITIONAL_FEATURES} entries, "
                         f"got {pos.shape[0]}.")
    state = np.concatenate([price, pos], axis=0).astype(np.float32)
    return state
