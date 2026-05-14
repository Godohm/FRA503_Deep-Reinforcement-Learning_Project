"""Run long-only / short-only / flat baselines on the held-out test set.

Outputs (per baseline):
    results/baselines/<name>/trade_log.csv
    results/baselines/<name>/metrics.json
    results/baselines/<name>/equity_curve.png
    results/baselines/<name>/drawdown.png
    results/baselines/<name>/daily_returns_hist.png

Also writes a comparison summary CSV at results/baselines/summary.csv.

Usage
-----
    python scripts/run_baselines.py --config configs/env.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.envs.eurusd_intraday_env import EURUSDIntradayTradingEnv
from src.evaluation.backtest import run_backtest
from src.evaluation.baselines import BASELINES
from src.evaluation.metrics import (
    compute_all_metrics,
    daily_returns_from_log,
    equity_curve_account_ccy,
)
from src.evaluation.plots import (
    plot_daily_returns_hist,
    plot_drawdown,
    plot_equity_curve,
)
from src.features.normalization import PriceFeatureScaler
from src.features.state_builder import compute_price_features
from src.utils.config import load_config


def _restore_session_day_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """parquet round-trips session_day as 'string'; convert back to date for the env."""
    if df["session_day"].dtype.name == "string" or df["session_day"].dtype == object:
        df = df.copy()
        df["session_day"] = pd.to_datetime(df["session_day"]).dt.date
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest baseline policies on the test split.")
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--results-dir", default="results/baselines",
                        help="Output directory for baseline artefacts.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    logger = logging.getLogger("run_baselines")

    cfg = load_config(args.config)
    processed_dir = Path(cfg["data"]["processed_dir"])
    initial_equity = float(cfg["portfolio"]["initial_equity"])
    point_size = float(cfg["costs"]["point_size"])

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    for p in (train_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Run `python scripts/prepare_data.py` first."
            )

    logger.info("Loading splits: %s, %s", train_path, test_path)
    train_df = _restore_session_day_dtype(pd.read_parquet(train_path))
    test_df = _restore_session_day_dtype(pd.read_parquet(test_path))

    logger.info("Computing price features per session_day for train + test...")
    train_feats = compute_price_features(train_df, point_size=point_size)
    test_feats = compute_price_features(test_df, point_size=point_size)

    logger.info("Fitting PriceFeatureScaler on TRAIN ONLY (%d rows after dropna).",
                int(train_feats.dropna().shape[0]))
    scaler = PriceFeatureScaler().fit(train_feats)
    test_scaled = scaler.transform(test_feats)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for name, policy in BASELINES.items():
        logger.info("Running baseline: %s", name)
        env = EURUSDIntradayTradingEnv(test_df, test_scaled, cfg, mode="sequential")
        trade_log = run_backtest(env, policy)
        metrics = compute_all_metrics(trade_log, initial_equity=initial_equity)

        out_dir = results_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        trade_log.to_csv(out_dir / "trade_log.csv", index=False)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics.to_dict(), indent=2, default=str)
        )

        dr = daily_returns_from_log(trade_log, initial_equity)
        eq = equity_curve_account_ccy(trade_log, initial_equity)
        plot_equity_curve(eq, out_dir / "equity_curve.png",
                          title=f"Equity curve — {name}")
        plot_drawdown(eq, out_dir / "drawdown.png",
                      title=f"Drawdown — {name}")
        plot_daily_returns_hist(dr, out_dir / "daily_returns_hist.png",
                                title=f"Daily returns — {name}")

        row = {"baseline": name, **metrics.to_dict()}
        summary_rows.append(row)
        logger.info("  total_return=%.4f sharpe=%.3f MDD=%.4f trades=%d win_rate=%.3f txn_cost=%.2f",
                    metrics.total_return, metrics.sharpe_ratio,
                    metrics.max_drawdown_pct, metrics.n_trades,
                    metrics.win_rate, metrics.total_transaction_cost)

    summary = pd.DataFrame(summary_rows).set_index("baseline")
    summary_path = results_dir / "summary.csv"
    summary.to_csv(summary_path)

    print()
    print("=" * 80)
    print("BASELINE SUMMARY (test split)")
    print("=" * 80)
    print(summary.round(6).to_string())
    print()
    print(f"Saved per-baseline artefacts under: {results_dir}/")
    print(f"Saved summary table at:            {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
