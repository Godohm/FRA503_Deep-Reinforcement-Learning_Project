"""Performance & risk metrics computed from a backtest trade log.

The canonical input shape is the trade-log DataFrame produced by
``backtest.run_backtest``: one row per step, with columns

    session_day, time_utc, bar_idx, action, current_position,
    raw_pnl, transaction_cost, net_pnl, mtm_pnl, equity, forced_close

All money quantities are in **account currency**. Returns are computed as
``net_pnl / initial_equity`` so they are dimensionless. The Sharpe and Sortino
ratios are annualised assuming **252 trading days per year**.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class BacktestMetrics:
    total_return: float
    final_equity: float
    mean_daily_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    n_trades: int
    win_rate: float
    avg_pnl_per_trade: float
    total_transaction_cost: float
    exposure_time: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- primitives


def total_return(initial_equity: float, final_equity: float) -> float:
    """Return as a fraction of the starting equity."""
    return (final_equity - initial_equity) / initial_equity


def max_drawdown_pct(equity_curve: np.ndarray | pd.Series) -> float:
    """Worst peak-to-trough drawdown on the equity curve, expressed as a fraction.

    Returns a non-positive number (e.g. ``-0.05`` for a 5% drawdown). If the
    curve is non-decreasing, returns ``0.0``.
    """
    eq = np.asarray(equity_curve, dtype="float64")
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    # Guard against zero (or near-zero) running_max from a starting equity of 0.
    safe = np.where(running_max == 0, 1.0, running_max)
    drawdowns = (eq - running_max) / safe
    return float(drawdowns.min())


def sharpe_ratio(
    daily_returns: np.ndarray | pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(periods).

    Risk-free rate is assumed zero. Returns ``0.0`` if ``std == 0`` (no signal).
    """
    r = np.asarray(daily_returns, dtype="float64")
    if r.size == 0:
        return 0.0
    sd = r.std(ddof=0)
    if sd == 0.0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    daily_returns: np.ndarray | pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    target: float = 0.0,
) -> float:
    """Annualised Sortino. Downside denominator = sqrt(mean(min(r-target, 0)^2)).

    If no downside observations exist, returns 0.0.
    """
    r = np.asarray(daily_returns, dtype="float64")
    if r.size == 0:
        return 0.0
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0.0:
        return 0.0
    return float((r.mean() - target) / dd * np.sqrt(periods_per_year))


def count_trades(positions: np.ndarray | pd.Series) -> int:
    """A 'trade' is a bar on which the position changed from the previous bar.

    The first bar's transition (from the implicit 0 at episode start) counts.
    """
    p = np.asarray(positions, dtype="int64")
    if p.size == 0:
        return 0
    prev = np.concatenate([[0], p[:-1]])  # implicit starting position 0
    return int((p != prev).sum())


def win_rate(per_trade_pnl: np.ndarray | pd.Series) -> float:
    """Fraction of trades with positive realised PnL.

    A 'trade' here is a position-holding episode that ended either by a flip
    or by force-close; see ``trade_pnls`` for the canonical extractor.
    """
    p = np.asarray(per_trade_pnl, dtype="float64")
    if p.size == 0:
        return 0.0
    return float((p > 0).mean())


def avg_pnl_per_trade(per_trade_pnl: np.ndarray | pd.Series) -> float:
    p = np.asarray(per_trade_pnl, dtype="float64")
    if p.size == 0:
        return 0.0
    return float(p.mean())


def exposure_time(positions: np.ndarray | pd.Series) -> float:
    """Fraction of bars where |position| == 1."""
    p = np.asarray(positions, dtype="int64")
    if p.size == 0:
        return 0.0
    return float((np.abs(p) == 1).mean())


# --------------------------------------------------------------------------- trade extraction


def trade_pnls(trade_log: pd.DataFrame) -> np.ndarray:
    """Aggregate the per-bar ``net_pnl`` into per-position-holding sums.

    A new "trade" starts whenever the position transitions from 0 → ±1 or
    flips sign. Bars while the position is 0 still contribute to the *open*
    trade's running PnL if a trade is open at that point (it shouldn't be —
    position 0 means flat — but we are defensive).

    The PnL attributed to a trade equals the sum of ``net_pnl`` across every
    bar from the bar where the position was opened, up to and including the
    bar where it closed (long → flat, short → flat, or flip).
    """
    if trade_log.empty:
        return np.empty(0, dtype="float64")
    df = trade_log[["current_position", "net_pnl"]].copy()
    pos = df["current_position"].to_numpy(dtype="int64")
    pnl = df["net_pnl"].to_numpy(dtype="float64")
    prev = np.concatenate([[0], pos[:-1]])
    starts = (prev == 0) & (pos != 0)
    flips = (prev != 0) & (pos != 0) & (prev != pos)
    closes = (prev != 0) & (pos == 0)

    trade_ids = np.zeros(len(pos), dtype="int64")
    current_id = 0
    in_trade = False
    for i in range(len(pos)):
        if starts[i] or flips[i]:
            current_id += 1
            in_trade = True
        trade_ids[i] = current_id if in_trade else 0
        if closes[i]:
            in_trade = False
    if current_id == 0:
        return np.empty(0, dtype="float64")
    df["_trade_id"] = trade_ids
    grouped = df.loc[df["_trade_id"] > 0].groupby("_trade_id")["net_pnl"].sum()
    return grouped.to_numpy(dtype="float64")


def daily_returns_from_log(trade_log: pd.DataFrame, initial_equity: float) -> pd.Series:
    """Per-session_day fractional return = sum(net_pnl_in_day) / initial_equity."""
    if trade_log.empty:
        return pd.Series(dtype="float64")
    grouped = trade_log.groupby("session_day", sort=True)["net_pnl"].sum()
    return grouped / float(initial_equity)


def cumulative_return_curve(
    trade_log: pd.DataFrame, initial_equity: float
) -> pd.Series:
    """Non-compounded cumulative daily-return series, dimensionless.

    The env resets equity to ``initial_equity`` at the start of every session,
    so each day's return is realised on a fresh bankroll — daily returns are
    additive, not multiplicative. This function therefore uses ``cumsum``:

        cumret_d = 1 + sum_{i<=d} (daily_pnl_i / initial_equity)

    Starts at slightly above 1 on day 1 (= 1 + day_1_return). The series is
    dimensionless: callers wanting a USD view should multiply by
    ``initial_equity`` (see ``equity_curve_account_ccy``). Max-drawdown computed
    on either series is identical because MDD is normalised by the running max.
    """
    dr = daily_returns_from_log(trade_log, initial_equity)
    if dr.empty:
        return pd.Series(dtype="float64")
    return 1.0 + dr.cumsum()


def equity_curve_account_ccy(
    trade_log: pd.DataFrame, initial_equity: float
) -> pd.Series:
    """Non-compounded daily equity proxy in account currency = init * cumret.

    Same semantics as ``cumulative_return_curve``: additive across days because
    the env resets the bankroll every session. NOT compounded — for a truly
    compounded view use ``cumprod(1 + daily_returns) * initial_equity`` and
    rename accordingly; we don't do that here because it would misrepresent
    the intraday-only strategy where day n's position size does not scale with
    day n-1's PnL.
    """
    cumret = cumulative_return_curve(trade_log, initial_equity)
    if cumret.empty:
        return cumret
    return cumret * initial_equity


# Legacy alias retained for callers still using the old name; emits no warning
# because the formula is unchanged — only the semantics/labels were sharpened.
equity_curve_from_log = equity_curve_account_ccy


# --------------------------------------------------------------------------- top-level entry


def compute_all_metrics(
    trade_log: pd.DataFrame,
    initial_equity: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestMetrics:
    """Roll a trade log up into a ``BacktestMetrics`` summary.

    Parameters
    ----------
    trade_log : pd.DataFrame
        Output of ``backtest.run_backtest``.
    initial_equity : float
        Used to normalise PnL into fractional returns. Must match the env's
        ``portfolio.initial_equity``.
    """
    if trade_log.empty:
        return BacktestMetrics(
            total_return=0.0,
            final_equity=initial_equity,
            mean_daily_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            n_trades=0,
            win_rate=0.0,
            avg_pnl_per_trade=0.0,
            total_transaction_cost=0.0,
            exposure_time=0.0,
        )

    dr = daily_returns_from_log(trade_log, initial_equity)
    eq = equity_curve_account_ccy(trade_log, initial_equity)
    pos = trade_log["current_position"].to_numpy(dtype="int64")
    t_pnl = trade_pnls(trade_log)

    final_equity = float(eq.iloc[-1]) if not eq.empty else initial_equity

    return BacktestMetrics(
        total_return=total_return(initial_equity, final_equity),
        final_equity=final_equity,
        mean_daily_return=float(dr.mean()) if not dr.empty else 0.0,
        sharpe_ratio=sharpe_ratio(dr, periods_per_year),
        sortino_ratio=sortino_ratio(dr, periods_per_year),
        max_drawdown_pct=max_drawdown_pct(eq),
        n_trades=count_trades(pos),
        win_rate=win_rate(t_pnl),
        avg_pnl_per_trade=avg_pnl_per_trade(t_pnl),
        total_transaction_cost=float(trade_log["transaction_cost"].sum()),
        exposure_time=exposure_time(pos),
    )
