"""Lightweight matplotlib plotting for backtest results.

All functions take pre-computed series (no business logic), write a single
PNG, and close the figure to avoid leaks in long sweeps.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; safe for scripts/CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_equity_curve(
    equity: pd.Series,
    save_path: str | Path,
    title: str = "Non-compounded equity (additive across daily resets)",
) -> Path:
    """Plot the per-day equity proxy in account currency.

    NOTE on naming/semantics: the env resets bankroll each session, so daily
    returns are additive. The series passed here is
    ``initial_equity * (1 + cumsum(daily_returns))`` — *not* a compounded
    equity curve. Title reflects that.
    """
    p = _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(equity.index.astype(str), equity.values, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("session_day")
    ax.set_ylabel("equity proxy (account ccy, non-compounded)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_drawdown(equity: pd.Series, save_path: str | Path, title: str = "Drawdown") -> Path:
    p = _ensure_dir(save_path)
    eq = equity.to_numpy(dtype="float64")
    running_max = np.maximum.accumulate(eq) if eq.size else eq
    safe = np.where(running_max == 0, 1.0, running_max)
    dd = (eq - running_max) / safe
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(equity.index.astype(str), dd, 0.0, color="C3", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("session_day")
    ax.set_ylabel("drawdown (fraction)")
    ax.set_ylim(min(dd.min(), -0.001) if eq.size else -0.001, 0.001)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_daily_returns_hist(
    daily_returns: pd.Series,
    save_path: str | Path,
    title: str = "Daily return distribution",
    bins: int = 30,
) -> Path:
    p = _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not daily_returns.empty:
        ax.hist(daily_returns.values, bins=bins, edgecolor="black", linewidth=0.5)
        ax.axvline(float(daily_returns.mean()), color="C1", linestyle="--",
                   label=f"mean={daily_returns.mean():.4f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("daily return (fraction)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
