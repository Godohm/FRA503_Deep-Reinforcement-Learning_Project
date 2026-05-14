"""Aggregate all M8 runs + baselines into a single comparison table.

Walks every ``results/<algo>/<run_id>/summary.yaml`` (DDQN/A2C/PPO) plus
``results/baselines/<name>/metrics.json`` (when available), normalises into a
single row per run, and emits:

    results/comparison_table.csv
    results/m8_summary.md      (markdown with three ranking tables)

Columns
-------
    algorithm, seed, run_id, total_return, total_return_pct, sharpe,
    sortino, max_drawdown, max_drawdown_pct, num_trades, win_rate,
    transaction_cost, exposure_time, final_equity, model_path, result_path

Notes
-----
* Baselines have no seed — the seed column is left blank for those rows.
* For RL runs ``best_metrics`` is preferred over ``final_metrics`` (it is the
  checkpoint the orchestrator considers "the model"). If neither is present,
  the run is skipped with a warning.
* ``max_drawdown`` is the metric's stored fraction (e.g. −0.165); the
  ``_pct`` variant is the same value × 100, for human-readable tables.

Usage
-----
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --run-prefix m8_full
    python scripts/evaluate_all.py --results-dir results --out-dir results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

RL_ALGOS = ("ddqn", "a2c", "ppo")
BASELINE_DIRNAME = "baselines"

# Run-id like "m8_full_ddqn_seed42_steps200000" -> seed=42.
_SEED_RE = re.compile(r"_seed(\d+)_")


# --------------------------------------------------------------------------- helpers


def _pick_metrics(summary: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Prefer best_metrics; fall back to final_metrics."""
    if isinstance(summary.get("best_metrics"), dict):
        return summary["best_metrics"]
    if isinstance(summary.get("final_metrics"), dict):
        return summary["final_metrics"]
    return None


def _seed_from_run_id(run_id: str) -> Optional[int]:
    m = _SEED_RE.search(run_id)
    return int(m.group(1)) if m else None


def _row_from_metrics(metrics: dict[str, Any], *, algorithm: str, seed: Optional[int],
                      run_id: str, model_path: str, result_path: str) -> dict[str, Any]:
    total_return = float(metrics.get("total_return", 0.0))
    max_dd = float(metrics.get("max_drawdown_pct", 0.0))  # historic name: this IS the fraction
    return {
        "algorithm": algorithm,
        "seed": seed,
        "run_id": run_id,
        "total_return": total_return,
        "total_return_pct": total_return * 100.0,
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino": float(metrics.get("sortino_ratio", 0.0)),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100.0,
        "num_trades": int(metrics.get("n_trades", 0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "transaction_cost": float(metrics.get("total_transaction_cost", 0.0)),
        "exposure_time": float(metrics.get("exposure_time", 0.0)),
        "final_equity": float(metrics.get("final_equity", 0.0)),
        "model_path": model_path,
        "result_path": result_path,
    }


# --------------------------------------------------------------------------- discovery


def _collect_rl_runs(results_dir: Path, run_prefix: Optional[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algo in RL_ALGOS:
        algo_dir = results_dir / algo
        if not algo_dir.is_dir():
            continue
        for run_dir in sorted(algo_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            if run_prefix is not None and not run_id.startswith(run_prefix):
                continue
            summary_path = run_dir / "summary.yaml"
            if not summary_path.is_file():
                print(f"[warn] {run_id}: no summary.yaml, skipping.")
                continue
            with summary_path.open("r", encoding="utf-8") as f:
                summary = yaml.safe_load(f) or {}
            metrics = _pick_metrics(summary)
            if metrics is None:
                print(f"[warn] {run_id}: no metrics in summary.yaml, skipping.")
                continue
            model_path = str(Path("models") / algo / run_id)
            rows.append(_row_from_metrics(
                metrics,
                algorithm=algo,
                seed=_seed_from_run_id(run_id),
                run_id=run_id,
                model_path=model_path,
                result_path=str(run_dir),
            ))
    return rows


def _collect_baselines(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bdir = results_dir / BASELINE_DIRNAME
    if not bdir.is_dir():
        return rows
    for sub in sorted(bdir.iterdir()):
        if not sub.is_dir():
            continue
        metrics_path = sub / "metrics.json"
        if not metrics_path.is_file():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(_row_from_metrics(
            metrics,
            algorithm=f"baseline:{sub.name}",
            seed=None,
            run_id=sub.name,
            model_path="",
            result_path=str(sub),
        ))
    return rows


# --------------------------------------------------------------------------- formatting


def _df_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    """Compact markdown table; avoids requiring pandas's optional tabulate dep."""
    if df.empty:
        return "(no rows)\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        cells: list[str] = []
        for c in columns:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            elif v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown_summary(out_path: Path, df: pd.DataFrame, run_prefix: Optional[str]) -> None:
    cols = ["algorithm", "seed", "run_id", "total_return_pct", "sharpe",
            "sortino", "max_drawdown_pct", "num_trades", "win_rate", "exposure_time"]
    by_sharpe = df.sort_values("sharpe", ascending=False)
    by_return = df.sort_values("total_return", ascending=False)
    by_dd = df.sort_values("max_drawdown", ascending=False)  # closer to 0 first

    pieces = [
        "# M8 — Experiment 1 Comparison",
        "",
        f"Run prefix filter: `{run_prefix or '(none — all runs)'}`",
        "",
        f"Rows: **{len(df)}** "
        f"({(df['algorithm'].isin(RL_ALGOS)).sum()} RL, "
        f"{(df['algorithm'].str.startswith('baseline:')).sum()} baseline)",
        "",
        "## Ranked by Sharpe (higher is better)",
        "",
        _df_to_markdown(by_sharpe, cols),
        "## Ranked by total return (higher is better)",
        "",
        _df_to_markdown(by_return, cols),
        "## Ranked by max drawdown (closer to 0 is better)",
        "",
        _df_to_markdown(by_dd, cols),
    ]
    out_path.write_text("\n".join(pieces), encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate RL runs + baselines into a comparison table."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Root directory containing <algo>/<run_id>/ folders.")
    parser.add_argument("--out-dir", default="results",
                        help="Where to write comparison_table.csv and m8_summary.md.")
    parser.add_argument("--run-prefix", default=None,
                        help="If set, only include RL runs whose run_id starts with this prefix.")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip the baselines/ collection.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_rows = _collect_rl_runs(results_dir, args.run_prefix)
    baseline_rows = [] if args.no_baselines else _collect_baselines(results_dir)
    all_rows = rl_rows + baseline_rows

    if not all_rows:
        print("[warn] no rows discovered — nothing to write.")
        return 1

    df = pd.DataFrame(all_rows, columns=[
        "algorithm", "seed", "run_id",
        "total_return", "total_return_pct",
        "sharpe", "sortino",
        "max_drawdown", "max_drawdown_pct",
        "num_trades", "win_rate", "transaction_cost",
        "exposure_time", "final_equity",
        "model_path", "result_path",
    ])

    csv_path = out_dir / "comparison_table.csv"
    md_path = out_dir / "m8_summary.md"
    df.to_csv(csv_path, index=False)
    _write_markdown_summary(md_path, df, args.run_prefix)

    print(f"[done] wrote {len(df)} rows ({len(rl_rows)} RL, {len(baseline_rows)} baseline)")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
