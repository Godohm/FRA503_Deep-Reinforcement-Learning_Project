"""Hyperparameter optimisation via short-sweep grid search.

Strategy: one-at-a-time (OAT) — flip a single parameter from baseline per run.
The change in test-set metrics is then directly attributable to that one
parameter, which makes the write-up readable.

Each run is a small training (default 20k env steps, seed=42, eval every 5k)
saved under deterministic paths so re-running with --skip-existing resumes a
partial sweep cleanly.

Outputs
-------
    models/<algo>/hpo_<algo>_<variant>_steps<N>/{best.*, final.*, env.yaml, ...}
    results/<algo>/hpo_<algo>_<variant>_steps<N>/{summary.yaml, train_log.csv, ...}
    results/hpo/hpo_results.csv   (consolidated metrics + which param changed)
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.utils.config import load_config  # noqa: E402

ALGO_CONFIG_FILE = {"ddqn": "dqn.yaml", "a2c": "a2c.yaml", "ppo": "ppo.yaml"}


# --------------------------------------------------------------------------- helpers


def _train_dispatch(algo: str):
    if algo == "ddqn":
        from src.agents.train_dqn import train_dqn
        return train_dqn
    if algo == "a2c":
        from src.agents.train_a2c import train_a2c
        return train_a2c
    if algo == "ppo":
        from src.agents.train_ppo import train_ppo
        return train_ppo
    raise ValueError(f"Unknown algo: {algo!r}")


def _apply_overrides(algo_cfg: dict[str, Any], section: str,
                     params: dict[str, Any]) -> dict[str, Any]:
    new_cfg = copy.deepcopy(algo_cfg)
    new_cfg.setdefault(section, {})
    new_cfg[section].update(params)
    return new_cfg


def _run_id(prefix: str, algo: str, variant: str, steps: int) -> str:
    return f"{prefix}_{algo}_{variant}_steps{steps}"


def _summarise_overrides(params: dict[str, Any]) -> tuple[str, str]:
    """For the CSV row: which param + what value (empty for baseline)."""
    if not params:
        return ("(baseline)", "")
    if len(params) == 1:
        k, v = next(iter(params.items()))
        return (k, str(v))
    # Multi-param variation (not used in current grid, kept for flexibility)
    keys = "+".join(params.keys())
    vals = "+".join(str(v) for v in params.values())
    return (keys, vals)


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="HPO short-sweep grid search.")
    parser.add_argument("--hpo-config", default="configs/hpo.yaml")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="If set, restrict the sweep to these algos.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    hpo_cfg = load_config(args.hpo_config)
    settings = hpo_cfg["settings"]
    steps = int(settings["steps_per_config"])
    eval_every = int(settings["eval_every"])
    seed = int(settings["seed"])
    prefix = str(settings["run_prefix"])

    env_cfg = load_config(args.env_config)
    config_dir = Path(args.config_dir)

    algos_to_run = list(args.algos) if args.algos else list(hpo_cfg["algos"].keys())
    plan: list[dict[str, Any]] = []
    for algo in algos_to_run:
        spec = hpo_cfg["algos"][algo]
        section = spec["section"]
        for variant in spec["variations"]:
            plan.append({
                "algo": algo,
                "section": section,
                "variant_name": variant["name"],
                "params": variant.get("params", {}) or {},
            })

    print(f"[plan] {len(plan)} runs across {len(algos_to_run)} algos "
          f"(steps={steps}, eval_every={eval_every}, seed={seed})")
    for p in plan:
        param_key, param_val = _summarise_overrides(p["params"])
        run_id = _run_id(prefix, p["algo"], p["variant_name"], steps)
        print(f"  - {p['algo']:<5} {p['variant_name']:<18} {param_key}={param_val:<10} run_id={run_id}")
    if args.dry_run:
        print("[dry-run] no training started.")
        return 0

    # Consolidated CSV will be appended-to as runs complete (resumable).
    csv_dir = Path("results/hpo")
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "hpo_results.csv"
    csv_columns = [
        "algo", "variant", "param_changed", "param_value",
        "total_return", "sharpe", "sortino", "max_drawdown_pct",
        "n_trades", "win_rate", "exposure_time", "total_transaction_cost",
        "final_equity", "wall_time_s", "run_id", "model_dir", "result_dir",
    ]
    if not csv_path.is_file():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(csv_columns)

    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    t_total = time.time()

    for run in plan:
        algo = run["algo"]
        variant = run["variant_name"]
        section = run["section"]
        params = run["params"]
        run_id = _run_id(prefix, algo, variant, steps)
        model_dir = Path("models") / algo / run_id
        result_dir = Path("results") / algo / run_id
        summary_yaml = result_dir / "summary.yaml"

        if args.skip_existing and summary_yaml.is_file():
            print(f"[skip] {run_id}")
            skipped.append({"run_id": run_id})
            continue

        # Build the per-run algo_cfg (deepcopy + override + force eval cadence).
        algo_cfg = load_config(config_dir / ALGO_CONFIG_FILE[algo])
        algo_cfg = _apply_overrides(algo_cfg, section, params)
        algo_cfg.setdefault("training", {})
        algo_cfg["training"]["eval_every_steps"] = eval_every
        algo_cfg["training"]["total_steps"] = steps

        param_key, param_val = _summarise_overrides(params)
        t0 = time.time()
        print(f"[run ] {run_id}  ({param_key}={param_val})")
        try:
            train_fn = _train_dispatch(algo)
            summary = train_fn(env_cfg, algo_cfg, total_steps=steps,
                               run_id=run_id, seed=seed)
            metrics = summary.get("best_metrics") or summary.get("final_metrics") or {}
            row = [
                algo, variant, param_key, param_val,
                metrics.get("total_return", ""),
                metrics.get("sharpe_ratio", ""),
                metrics.get("sortino_ratio", ""),
                metrics.get("max_drawdown_pct", ""),
                metrics.get("n_trades", ""),
                metrics.get("win_rate", ""),
                metrics.get("exposure_time", ""),
                metrics.get("total_transaction_cost", ""),
                metrics.get("final_equity", ""),
                f"{time.time() - t0:.1f}",
                run_id, str(model_dir), str(result_dir),
            ]
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            completed.append({"run_id": run_id, "metrics": metrics})
            sharpe = metrics.get("sharpe_ratio", float("nan"))
            tr = metrics.get("total_return", float("nan"))
            print(f"[done] {run_id}  Sharpe={sharpe:.2f} total_return={tr:.4f}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            failed.append({"run_id": run_id, "error": str(e)})
            print(f"[fail] {run_id}: {type(e).__name__}: {e}")

    print(f"[summary] completed={len(completed)} skipped={len(skipped)} "
          f"failed={len(failed)} elapsed={time.time()-t_total:.1f}s")
    print(f"  CSV: {csv_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
