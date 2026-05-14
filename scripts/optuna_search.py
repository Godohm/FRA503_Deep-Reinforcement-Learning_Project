"""Bayesian hyperparameter optimisation via Optuna (TPE sampler).

Per-algo search spaces are defined in this file (kept here rather than YAML so
the type information — log-uniform vs uniform vs categorical — is co-located
with the sampler code that needs it).

Each trial runs a short training with the proposed hyperparameters and
reports back ``best_metrics["sharpe_ratio"]`` (on the test split) as the
objective. TPE then proposes the next configuration to evaluate.

Outputs
-------
    results/optuna/<algo>_study.db   (Optuna SQLite — resumable)
    results/optuna/<algo>_trials.csv (flat per-trial metrics)
    models/<algo>/optuna_<algo>_t<NNN>/   (per-trial artifacts)
    results/<algo>/optuna_<algo>_t<NNN>/  (per-trial logs)

Usage
-----
    # Full research-grade run (50 trials × 50k steps each algo, ~6-9 h CPU):
    python scripts/optuna_search.py --algos ddqn a2c ppo --trials 50 --steps 50000

    # Reduced demo (faster, fewer trials):
    python scripts/optuna_search.py --algos ddqn a2c ppo --trials 15 --steps 15000

    # Resume a partial study (storage is SQLite-backed; just re-run):
    python scripts/optuna_search.py --algos ddqn --trials 50 --steps 50000

    # Pick up where you left off by adding more trials:
    python scripts/optuna_search.py --algos ddqn --trials 100 --steps 50000
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402

from src.utils.config import load_config  # noqa: E402


# --------------------------------------------------------------------------- search spaces


def _sample_ddqn(trial: optuna.Trial) -> dict[str, Any]:
    """DDQN search space.

    Six knobs: optimiser (lr), credit (gamma), exploration schedule
    (eps_decay_steps), target-network lag (target_update_freq), sample size
    (batch_size), and learning-start threshold (min_buffer_to_learn).
    """
    return {
        "dqn": {
            "lr":                  trial.suggest_float("lr", 1.0e-5, 5.0e-3, log=True),
            "gamma":               trial.suggest_float("gamma", 0.90, 0.999),
            "eps_decay_steps":     trial.suggest_int("eps_decay_steps", 10_000, 200_000, step=5_000),
            "target_update_freq":  trial.suggest_int("target_update_freq", 100, 5_000, step=100),
            "batch_size":          trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "min_buffer_to_learn": trial.suggest_int("min_buffer_to_learn", 500, 20_000, step=500),
        },
    }


def _sample_a2c(trial: optuna.Trial) -> dict[str, Any]:
    """A2C search space.

    Optimiser (lr), rollout-window length (n_steps), credit (gamma), GAE
    smoothing (gae_lambda), exploration via entropy (ent_coef), value-loss
    weight (vf_coef).
    """
    return {
        "a2c": {
            "learning_rate": trial.suggest_float("learning_rate", 1.0e-5, 1.0e-2, log=True),
            "n_steps":       trial.suggest_int("n_steps", 5, 100),
            "gamma":         trial.suggest_float("gamma", 0.90, 0.999),
            "ent_coef":      trial.suggest_float("ent_coef", 1.0e-6, 1.0e-1, log=True),
            "vf_coef":       trial.suggest_float("vf_coef", 0.1, 1.0),
            "gae_lambda":    trial.suggest_float("gae_lambda", 0.85, 1.0),
        },
    }


def _sample_ppo(trial: optuna.Trial) -> dict[str, Any]:
    """PPO search space.

    Optimiser (lr), rollout buffer size (n_steps), minibatch (batch_size),
    epoch count (n_epochs), credit (gamma), clip range, GAE (gae_lambda),
    entropy/value coefficients.
    """
    return {
        "ppo": {
            "learning_rate": trial.suggest_float("learning_rate", 1.0e-5, 1.0e-2, log=True),
            "n_steps":       trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
            "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "n_epochs":      trial.suggest_int("n_epochs", 3, 20),
            "gamma":         trial.suggest_float("gamma", 0.90, 0.999),
            "clip_range":    trial.suggest_float("clip_range", 0.05, 0.4),
            "ent_coef":      trial.suggest_float("ent_coef", 1.0e-6, 1.0e-1, log=True),
            "vf_coef":       trial.suggest_float("vf_coef", 0.1, 1.0),
            "gae_lambda":    trial.suggest_float("gae_lambda", 0.80, 1.0),
        },
    }


SAMPLERS = {"ddqn": _sample_ddqn, "a2c": _sample_a2c, "ppo": _sample_ppo}
CFG_FILES = {"ddqn": "dqn.yaml", "a2c": "a2c.yaml", "ppo": "ppo.yaml"}


def _train_fn(algo: str):
    if algo == "ddqn":
        from src.agents.train_dqn import train_dqn
        return train_dqn
    if algo == "a2c":
        from src.agents.train_a2c import train_a2c
        return train_a2c
    if algo == "ppo":
        from src.agents.train_ppo import train_ppo
        return train_ppo
    raise ValueError(algo)


# --------------------------------------------------------------------------- core


def _make_objective(algo: str, base_cfg: dict[str, Any], env_cfg: dict[str, Any],
                    steps: int, eval_every: int, seed: int,
                    static_overrides: dict[str, dict[str, Any]] | None = None,
                    run_id_prefix: str = "optuna"):
    train_fn = _train_fn(algo)
    static_overrides = static_overrides or {}

    def objective(trial: optuna.Trial) -> float:
        # Build the per-trial algo_cfg by overriding the section returned by sampler.
        overrides = SAMPLERS[algo](trial)
        algo_cfg = copy.deepcopy(base_cfg)
        for section, params in overrides.items():
            algo_cfg.setdefault(section, {}).update(params)
        # Apply any static (non-sampled) overrides last so they always win.
        for section, params in static_overrides.items():
            algo_cfg.setdefault(section, {}).update(params)
        algo_cfg.setdefault("training", {})
        algo_cfg["training"]["eval_every_steps"] = int(eval_every)
        algo_cfg["training"]["total_steps"] = int(steps)

        run_id = f"{run_id_prefix}_{algo}_t{trial.number:03d}"
        try:
            summary = train_fn(env_cfg, algo_cfg, total_steps=steps, run_id=run_id, seed=seed)
        except Exception as e:
            print(f"  [trial {trial.number}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Pruner-friendly: return a very bad value rather than raising.
            return -1.0e6

        metrics = summary.get("best_metrics") or summary.get("final_metrics") or {}
        sharpe = float(metrics.get("sharpe_ratio", -1.0e6))
        # Stash extras for downstream analysis (not affecting Optuna direction).
        # Guard against UpdateFinishedTrialError from concurrent DB writes.
        try:
            trial.set_user_attr("total_return",        float(metrics.get("total_return", 0.0)))
            trial.set_user_attr("max_drawdown_pct",    float(metrics.get("max_drawdown_pct", 0.0)))
            trial.set_user_attr("n_trades",            int(metrics.get("n_trades", 0)))
            trial.set_user_attr("win_rate",            float(metrics.get("win_rate", 0.0)))
            trial.set_user_attr("exposure_time",       float(metrics.get("exposure_time", 0.0)))
            trial.set_user_attr("total_transaction_cost",
                                float(metrics.get("total_transaction_cost", 0.0)))
            trial.set_user_attr("final_equity",        float(metrics.get("final_equity", 0.0)))
            trial.set_user_attr("run_id", run_id)
        except Exception as e:
            print(f"  [trial {trial.number}] WARNING: could not set user attrs: {e}")
        return sharpe

    return objective


def run_search(algo: str, n_trials: int, steps: int, env_cfg: dict[str, Any],
               seed: int, config_dir: Path, storage_dir: Path,
               n_startup_trials: int,
               study_suffix: str = "",
               static_overrides: dict[str, dict[str, Any]] | None = None) -> optuna.Study:
    base_cfg = load_config(config_dir / CFG_FILES[algo])
    eval_every = max(5_000, steps // 4)

    sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    storage_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{algo}{study_suffix}"
    storage_uri = f"sqlite:///{storage_dir / (tag + '_study.db')}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"hpo_{tag}",
        storage=storage_uri,
        load_if_exists=True,
    )

    run_id_prefix = f"optuna{study_suffix}"
    objective = _make_objective(algo, base_cfg, env_cfg, steps, eval_every, seed,
                                static_overrides=static_overrides,
                                run_id_prefix=run_id_prefix)

    # Disable Optuna's default INFO logging — too noisy for ~50 trials.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"[{tag}] starting search: trials={n_trials} steps={steps} "
          f"eval_every={eval_every} startup={n_startup_trials}", flush=True)
    t0 = time.time()
    # Custom callback for progress lines (Optuna's default progress bar requires tqdm).
    def _cb(study_, trial_):
        attrs = trial_.user_attrs
        tr = attrs.get("total_return", float("nan"))
        ntr = attrs.get("n_trades", 0)
        print(f"  [{tag} t{trial_.number:03d}] "
              f"Sharpe={trial_.value:.2f} total_return={tr:.4f} trades={ntr}", flush=True)

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
    except Exception as exc:
        print(f"[{tag}] FATAL in study.optimize: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    elapsed = time.time() - t0

    # Persist a flat CSV of trials for quick downstream consumption.
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs",
                                       "state", "datetime_start", "datetime_complete"))
    df.to_csv(storage_dir / f"{tag}_trials.csv", index=False)

    best = study.best_trial
    print(f"[{tag}] done in {elapsed:.0f}s | best Sharpe={best.value:.3f} "
          f"at trial #{best.number}")
    print(f"[{tag}] best params: {best.params}")
    return study


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="Bayesian HPO via Optuna TPE.")
    parser.add_argument("--algos", nargs="+", default=["ddqn", "a2c", "ppo"],
                        choices=["ddqn", "a2c", "ppo"])
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of trials per algorithm.")
    parser.add_argument("--steps", type=int, default=50_000,
                        help="Env steps per trial.")
    parser.add_argument("--startup-trials", type=int, default=10,
                        help="Random sampling trials before TPE kicks in.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for TPE sampler AND each training run.")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--storage-dir", default="results/optuna",
                        help="Where SQLite study DB and trials.csv go.")
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--study-suffix", default="",
                        help="Suffix appended to study/DB/CSV names "
                             "(e.g. '_expdecay' → ddqn_expdecay_study.db). "
                             "Lets you run ablations without clobbering existing studies.")
    parser.add_argument("--ddqn-decay-type", choices=["linear", "exponential"],
                        default="linear",
                        help="ε-greedy decay schedule for DDQN. "
                             "Static override — NOT a sampled hyperparameter.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    env_cfg = load_config(args.env_config)
    config_dir = Path(args.config_dir)
    storage_dir = Path(args.storage_dir)

    print(f"[start] algos={args.algos} trials={args.trials} steps={args.steps} "
          f"startup={args.startup_trials} seed={args.seed} "
          f"suffix='{args.study_suffix}' ddqn_decay={args.ddqn_decay_type}", flush=True)
    t_total = time.time()
    for algo in args.algos:
        static: dict[str, dict[str, Any]] = {}
        if algo == "ddqn" and args.ddqn_decay_type != "linear":
            static = {"dqn": {"eps_decay_type": args.ddqn_decay_type}}
        run_search(algo, args.trials, args.steps, env_cfg,
                   seed=args.seed, config_dir=config_dir,
                   storage_dir=storage_dir,
                   n_startup_trials=args.startup_trials,
                   study_suffix=args.study_suffix,
                   static_overrides=static)
    print(f"[summary] all studies done in {time.time()-t_total:.0f}s")
    print(f"  SQLite + CSVs under: {storage_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
