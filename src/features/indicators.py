"""Technical-indicator primitives used by the state builder.

All functions are pure: they consume pandas Series and return pandas Series of
the same length and index. No function reads any value beyond the *current*
position when computing index ``t``, so the no-look-ahead invariant is
enforced by construction.

Conventions
-----------
- "Wilder smoothing" uses an exponential moving average with ``alpha = 1/n``
  (equivalent to ``ewm(alpha=1/n, adjust=False)``) — the standard for RSI/ATR.
- The first ``n-1`` values of any rolling-window calculation are NaN; callers
  are expected to skip them via the env's ``warmup_bars`` setting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series, period: int) -> pd.Series:
    """Log return over ``period`` bars: ``log(close_t / close_{t-period})``.

    The first ``period`` values are NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    shifted = close.shift(period)
    return np.log(close / shifted)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, range [0, 100]."""
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # If avg_loss == 0 (all gains), RSI = 100. If avg_gain == 0 (all losses), RSI = 0.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(macd_line, signal_line, hist)`` where:

    - ``macd_line = EMA(close, fast) - EMA(close, slow)``
    - ``signal_line = EMA(macd_line, signal)``
    - ``hist = macd_line - signal_line``

    EMAs use ``adjust=False`` (recursive form). The histogram is the
    canonical "MACD histogram" feature used in the state vector.
    """
    if not (fast < slow):
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist


def stochastic_k(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Stochastic %K = 100 * (close - lowest_low_period) / (highest_high_period - lowest_low_period).

    Range [0, 100]. NaN when the window is incomplete. When the period-high
    equals the period-low (flat market), the ratio is undefined and we set %K = 50.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    ll = low.rolling(period, min_periods=period).min()
    hh = high.rolling(period, min_periods=period).max()
    denom = hh - ll
    k = 100.0 * (close - ll) / denom.replace(0.0, np.nan)
    k = k.where(denom != 0.0, 50.0)
    return k


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average True Range over ``period`` bars (price units)."""
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
