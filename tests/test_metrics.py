"""M5 metric tests: Sharpe, Sortino, MDD, trade counting, win rate.

All tests use small synthetic series whose ground truth is computable by hand
or by a one-line numpy expression so the test itself documents the formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    BacktestMetrics,
    avg_pnl_per_trade,
    compute_all_metrics,
    count_trades,
    cumulative_return_curve,
    daily_returns_from_log,
    equity_curve_account_ccy,
    equity_curve_from_log,  # legacy alias — verify it still resolves
    exposure_time,
    max_drawdown_pct,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    trade_pnls,
    win_rate,
)


# --------------------------------------------------------------------------- primitives


def test_total_return_basic():
    assert total_return(100.0, 110.0) == pytest.approx(0.1)
    assert total_return(100.0, 90.0) == pytest.approx(-0.1)
    assert total_return(100.0, 100.0) == 0.0


def test_max_drawdown_known_curve():
    # 100 -> 110 -> 80 -> 90 : peak 110, trough 80 -> dd = (80-110)/110 ≈ -0.2727
    eq = np.array([100.0, 110.0, 80.0, 90.0])
    assert max_drawdown_pct(eq) == pytest.approx((80 - 110) / 110, rel=1e-9)


def test_max_drawdown_monotone_curve_is_zero():
    eq = np.array([100.0, 101.0, 102.0, 103.0])
    assert max_drawdown_pct(eq) == 0.0


def test_max_drawdown_empty_is_zero():
    assert max_drawdown_pct(np.array([])) == 0.0


def test_sharpe_matches_numpy():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.005, 252)
    expected = r.mean() / r.std(ddof=0) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-9)


def test_sharpe_zero_std_returns_zero():
    r = np.array([0.01, 0.01, 0.01])
    assert sharpe_ratio(r) == 0.0


def test_sharpe_empty_returns_zero():
    assert sharpe_ratio(np.array([])) == 0.0


def test_sortino_only_downside_in_denominator():
    # With r = [+0.05, +0.05, -0.02], downside = [0, 0, -0.02], dd = sqrt(0.02^2/3).
    r = np.array([0.05, 0.05, -0.02])
    downside = np.minimum(r, 0.0)
    dd = float(np.sqrt((downside ** 2).mean()))
    expected = r.mean() / dd * np.sqrt(252)
    assert sortino_ratio(r) == pytest.approx(expected, rel=1e-9)


def test_sortino_all_positive_returns_zero():
    # No downside observations → dd = 0 → return 0.
    r = np.array([0.01, 0.02, 0.03])
    assert sortino_ratio(r) == 0.0


# --------------------------------------------------------------------------- trade counting / win-rate


def test_count_trades_from_position_changes():
    # Positions:  [+1, +1, -1, 0, +1]  (implicit prior 0)
    # transitions: 0→+1 (Y), +1→+1 (N), +1→-1 (Y), -1→0 (Y), 0→+1 (Y) = 4
    pos = np.array([1, 1, -1, 0, 1])
    assert count_trades(pos) == 4


def test_count_trades_empty():
    assert count_trades(np.array([], dtype="int64")) == 0


def test_count_trades_all_flat():
    assert count_trades(np.array([0, 0, 0, 0])) == 0


def test_win_rate_edge_cases():
    assert win_rate(np.array([])) == 0.0
    assert win_rate(np.array([0.0, 0.0])) == 0.0  # zero is not a win
    assert win_rate(np.array([1.0, -1.0, 2.0])) == pytest.approx(2 / 3)
    assert win_rate(np.array([1.0, 1.0])) == 1.0


def test_avg_pnl_per_trade_empty():
    assert avg_pnl_per_trade(np.array([])) == 0.0
    assert avg_pnl_per_trade(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_exposure_time():
    # Two of four bars hold a position.
    assert exposure_time(np.array([0, 1, -1, 0])) == 0.5
    assert exposure_time(np.array([0, 0, 0])) == 0.0
    assert exposure_time(np.array([1, 1, 1])) == 1.0


# --------------------------------------------------------------------------- trade-log aggregation


def _make_log(positions: list[int], net_pnls: list[float],
              session_days: list[str] | None = None,
              transaction_costs: list[float] | None = None) -> pd.DataFrame:
    n = len(positions)
    return pd.DataFrame({
        "session_day": session_days or [pd.Timestamp("2024-12-02").date()] * n,
        "bar_idx": list(range(n)),
        "action": [1] * n,
        "current_position": positions,
        "raw_pnl": net_pnls,
        "transaction_cost": transaction_costs or [0.0] * n,
        "net_pnl": net_pnls,
        "mtm_pnl": net_pnls,
        "equity": [10000.0 + sum(net_pnls[: i + 1]) for i in range(n)],
        "forced_close": [False] * (n - 1) + [True],
    })


def test_trade_pnls_aggregates_per_position_holding():
    # Open long at bar 0, hold for 3 bars (+1, +2, +3), close to flat at bar 3
    # (the close bar carries the final mtm + the close-out txn cost: -0.5).
    # Then open short at bar 4 (-2), close to flat at bar 5 (close-out: -1).
    log = _make_log(
        positions=[1, 1, 1, 0, -1, 0],
        net_pnls=[1.0, 2.0, 3.0, -0.5, -2.0, -1.0],
    )
    t = trade_pnls(log)
    # The close-out bar's net_pnl is attributed to the trade that just closed
    # because that's where the realised PnL settles. Trade 1: 1+2+3-0.5 = 5.5;
    # Trade 2: -2-1 = -3.0.
    np.testing.assert_allclose(t, [5.5, -3.0])


def test_daily_returns_from_log():
    log = _make_log(
        positions=[1, 1, 0, 1, 0],
        net_pnls=[10.0, 20.0, -5.0, 15.0, -3.0],
        session_days=[pd.Timestamp("2024-12-02").date()] * 3
                     + [pd.Timestamp("2024-12-03").date()] * 2,
    )
    dr = daily_returns_from_log(log, initial_equity=10000.0)
    np.testing.assert_allclose(dr.values, [25.0 / 10000.0, 12.0 / 10000.0])


def test_equity_curve_account_ccy_non_compounded():
    """Per-day equity proxy uses cumsum(daily_returns), not cumprod.

    Daily returns dr = [+0.01, -0.005]:
        cumret_curve = [1.01, 1.005]                 (dimensionless)
        equity_ccy   = init * cumret_curve            (account ccy)
    A truly compounded curve would be init * cumprod(1 + dr) = [10100, 10049.5].
    We use cumsum because the env resets bankroll every session — daily PnL
    is additive, not multiplicative.
    """
    log = _make_log(
        positions=[1, 0],
        net_pnls=[100.0, -50.0],
        session_days=[pd.Timestamp("2024-12-02").date(),
                      pd.Timestamp("2024-12-03").date()],
    )
    eq = equity_curve_account_ccy(log, initial_equity=10000.0)
    np.testing.assert_allclose(eq.values, [10000.0 * 1.01, 10000.0 * (1 + 0.01 - 0.005)])
    # Legacy alias must give the same result.
    np.testing.assert_allclose(equity_curve_from_log(log, initial_equity=10000.0).values,
                               eq.values)


def test_cumulative_return_curve_is_dimensionless_and_starts_above_one():
    log = _make_log(
        positions=[1, 0],
        net_pnls=[100.0, -50.0],
        session_days=[pd.Timestamp("2024-12-02").date(),
                      pd.Timestamp("2024-12-03").date()],
    )
    cumret = cumulative_return_curve(log, initial_equity=10000.0)
    # cumret = 1 + cumsum(dr) ; dr = [0.01, -0.005] → [1.01, 1.005]
    np.testing.assert_allclose(cumret.values, [1.01, 1.005])
    # And initial_equity * cumret == equity_curve_account_ccy.
    eq = equity_curve_account_ccy(log, initial_equity=10000.0)
    np.testing.assert_allclose(eq.values, 10000.0 * cumret.values)


def test_compute_all_metrics_smoke():
    log = _make_log(
        positions=[1, 1, 0, -1, 0],
        net_pnls=[5.0, 5.0, -2.0, -3.0, 1.0],
        transaction_costs=[2.0, 0.0, 1.0, 2.0, 1.0],
    )
    m = compute_all_metrics(log, initial_equity=10000.0)
    assert isinstance(m, BacktestMetrics)
    assert m.total_transaction_cost == pytest.approx(6.0)
    assert m.n_trades >= 2  # 0->+1 and +1->-1 at minimum
    # Returns are tiny on a 10k account — Sharpe should be finite.
    assert np.isfinite(m.sharpe_ratio)


def test_compute_all_metrics_empty_log():
    log = pd.DataFrame(columns=[
        "session_day", "bar_idx", "action", "current_position",
        "raw_pnl", "transaction_cost", "net_pnl", "mtm_pnl", "equity",
        "forced_close",
    ])
    m = compute_all_metrics(log, initial_equity=10000.0)
    assert m.total_return == 0.0
    assert m.n_trades == 0
    assert m.final_equity == 10000.0
