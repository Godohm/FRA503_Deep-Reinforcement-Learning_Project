"""Shared utilities for SB3-based training (A2C, PPO) on EURUSDIntradayTradingEnv.

We reuse the same dataset prep used by the DDQN trainer (scaler fit on train,
applied to both splits) and the same backtest+metrics pipeline for the
periodic test-set evaluation that defines "best".
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from src.envs.eurusd_intraday_env import EURUSDIntradayTradingEnv
from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import compute_all_metrics
from src.features.normalization import PriceFeatureScaler
from src.features.state_builder import compute_price_features

logger = logging.getLogger(__name__)


def _restore_session_day_dtype(df: pd.DataFrame) -> pd.DataFrame:
    if df["session_day"].dtype.name == "string" or df["session_day"].dtype == object:
        df = df.copy()
        df["session_day"] = pd.to_datetime(df["session_day"]).dt.date
    return df


@dataclass
class PreppedSB3:
    train_env: EURUSDIntradayTradingEnv
    test_env: EURUSDIntradayTradingEnv
    scaler: PriceFeatureScaler
    initial_equity: float


def prepare_envs_sb3(env_cfg: dict[str, Any]) -> PreppedSB3:
    processed_dir = Path(env_cfg["data"]["processed_dir"])
    point_size = float(env_cfg["costs"]["point_size"])
    initial_equity = float(env_cfg["portfolio"]["initial_equity"])

    train_path = processed_dir / "train.parquet"
    # Use val.parquet for periodic eval during training when it exists (3-way split).
    # Fall back to test.parquet for backward compatibility (2-way split).
    val_path  = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"
    eval_path = val_path if val_path.is_file() else test_path
    for p in (train_path, eval_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Run `python scripts/prepare_data.py` first."
            )

    train_df = _restore_session_day_dtype(pd.read_parquet(train_path))
    test_df  = _restore_session_day_dtype(pd.read_parquet(eval_path))

    train_feats = compute_price_features(train_df, point_size=point_size)
    test_feats = compute_price_features(test_df, point_size=point_size)
    scaler = PriceFeatureScaler().fit(train_feats)
    train_scaled = scaler.transform(train_feats)
    test_scaled = scaler.transform(test_feats)

    train_env = EURUSDIntradayTradingEnv(train_df, train_scaled, env_cfg, mode="random")
    test_env = EURUSDIntradayTradingEnv(test_df, test_scaled, env_cfg, mode="sequential")
    return PreppedSB3(train_env=train_env, test_env=test_env, scaler=scaler,
                      initial_equity=initial_equity)


def _sb3_greedy_policy(model):
    """Wrap an SB3 model as a (obs) -> int policy with deterministic=True."""
    def _pi(obs):
        action, _state = model.predict(obs, deterministic=True)
        # SB3 .predict on a single Box obs returns a 0-d array of dtype int64.
        return int(np.asarray(action).item())
    return _pi


def evaluate_sb3_on_test(
    model, test_env: EURUSDIntradayTradingEnv, initial_equity: float
):
    test_env._day_cursor = 0
    trade_log = run_backtest(test_env, _sb3_greedy_policy(model))
    return compute_all_metrics(trade_log, initial_equity=initial_equity)


# --------------------------------------------------------------------------- callback


class PeriodicTestEvalCallback:
    """Lightweight eval callback compatible with SB3's BaseCallback interface.

    We avoid SB3's stock EvalCallback because we want a full deterministic
    sweep over every session_day in the test split (the env's own definition
    of an episode = a single session), not the SB3 EvalCallback's notion of
    "n_eval_episodes from a Monitor wrapper".
    """

    def __init__(
        self,
        test_env: EURUSDIntradayTradingEnv,
        initial_equity: float,
        save_dir: Path,
        results_dir: Path,
        eval_every: int,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback  # local import

        class _Inner(BaseCallback):
            def __init__(inner) -> None:
                super().__init__(verbose=0)
                inner.eval_count = 0
                inner.best_score = -np.inf
                inner.best_metrics = None
                inner.eval_rows: list[dict] = []
                inner._last_step = 0

            def _on_step(inner) -> bool:
                if (inner.num_timesteps - inner._last_step) < eval_every:
                    return True
                inner._last_step = inner.num_timesteps
                inner.eval_count += 1
                m = evaluate_sb3_on_test(inner.model, test_env, initial_equity)
                score = float(m.sharpe_ratio) + 1.0e-6 * float(m.total_return)
                is_best = score > inner.best_score
                if is_best:
                    inner.best_score = score
                    inner.best_metrics = m
                    inner.model.save(str(save_dir / "best.zip"))
                inner.eval_rows.append({
                    "step": int(inner.num_timesteps),
                    "total_return": m.total_return,
                    "sharpe": m.sharpe_ratio,
                    "sortino": m.sortino_ratio,
                    "max_dd": m.max_drawdown_pct,
                    "n_trades": int(m.n_trades),
                    "win_rate": m.win_rate,
                    "exposure_time": m.exposure_time,
                    "final_equity": m.final_equity,
                    "is_best": bool(is_best),
                })
                pd.DataFrame(inner.eval_rows).to_csv(results_dir / "eval_log.csv", index=False)
                logger.info(
                    "[eval @ %d] total_return=%.4f sharpe=%.3f MDD=%.4f trades=%d %s",
                    inner.num_timesteps, m.total_return, m.sharpe_ratio,
                    m.max_drawdown_pct, m.n_trades, "(new best)" if is_best else "",
                )
                return True

        self._cb = _Inner()

    @property
    def callback(self):
        return self._cb


# --------------------------------------------------------------------------- common training


def _new_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}"


def train_sb3(
    *,
    algo: str,            # "a2c" | "ppo"
    env_cfg: dict[str, Any],
    algo_cfg: dict[str, Any],
    total_steps: Optional[int] = None,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Train A2C or PPO on the env. Shared because both algos share 95% of the loop."""
    from stable_baselines3 import A2C, PPO  # local import keeps test discovery cheap
    from stable_baselines3.common.monitor import Monitor

    algo = algo.lower()
    if algo not in ("a2c", "ppo"):
        raise ValueError(f"algo must be 'a2c' or 'ppo'; got {algo!r}")

    training = algo_cfg.get("training", {})
    total_steps = int(total_steps if total_steps is not None
                      else training.get("total_steps", 200_000))
    eval_every = int(training.get("eval_every_steps", 10_000))
    run_seed = int(seed if seed is not None else algo_cfg.get("seed", env_cfg.get("seed", 42)))

    save_dir_root = Path(training.get("save_dir_root", f"models/{algo}"))
    results_dir_root = Path(training.get("results_dir_root", f"results/{algo}"))
    run_id = run_id or _new_run_id(algo)
    save_dir = save_dir_root / run_id
    results_dir = results_dir_root / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Stamp the actual run_seed so the saved env.yaml is self-consistent.
    effective_env_cfg = {**env_cfg, "seed": run_seed}
    with (save_dir / "env.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(effective_env_cfg, f, sort_keys=False)
    with (save_dir / f"{algo}.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(algo_cfg, f, sort_keys=False)

    prepped = prepare_envs_sb3(env_cfg)
    prepped.scaler.save(save_dir / "scaler.pkl")

    # Wrap train env in Monitor for SB3 episode-stat logging.
    monitored_env = Monitor(prepped.train_env, filename=str(results_dir / "monitor.csv"))

    net = algo_cfg.get("network", {})
    hidden_sizes = list(net.get("hidden_sizes", [128, 128]))
    policy_kwargs = {"net_arch": [int(h) for h in hidden_sizes]}

    algo_params = algo_cfg.get(algo, {})
    tb_log_dir = str(results_dir / "tb")

    if algo == "a2c":
        model = A2C(
            policy="MlpPolicy",
            env=monitored_env,
            learning_rate=float(algo_params.get("learning_rate", 7e-4)),
            n_steps=int(algo_params.get("n_steps", 5)),
            gamma=float(algo_params.get("gamma", 0.99)),
            gae_lambda=float(algo_params.get("gae_lambda", 1.0)),
            ent_coef=float(algo_params.get("ent_coef", 0.0)),
            vf_coef=float(algo_params.get("vf_coef", 0.5)),
            max_grad_norm=float(algo_params.get("max_grad_norm", 0.5)),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_dir,
            verbose=0,
            seed=run_seed,
            device=str(algo_params.get("device", "cpu")),
        )
    else:  # ppo
        model = PPO(
            policy="MlpPolicy",
            env=monitored_env,
            learning_rate=float(algo_params.get("learning_rate", 3e-4)),
            n_steps=int(algo_params.get("n_steps", 2048)),
            batch_size=int(algo_params.get("batch_size", 64)),
            n_epochs=int(algo_params.get("n_epochs", 10)),
            gamma=float(algo_params.get("gamma", 0.99)),
            gae_lambda=float(algo_params.get("gae_lambda", 0.95)),
            clip_range=float(algo_params.get("clip_range", 0.2)),
            ent_coef=float(algo_params.get("ent_coef", 0.0)),
            vf_coef=float(algo_params.get("vf_coef", 0.5)),
            max_grad_norm=float(algo_params.get("max_grad_norm", 0.5)),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_dir,
            verbose=0,
            seed=run_seed,
            device=str(algo_params.get("device", "cpu")),
        )

    cb_holder = PeriodicTestEvalCallback(
        test_env=prepped.test_env,
        initial_equity=prepped.initial_equity,
        save_dir=save_dir,
        results_dir=results_dir,
        eval_every=eval_every,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb_holder.callback,
                progress_bar=False, reset_num_timesteps=True)
    model.save(str(save_dir / "final.zip"))

    # Always do a final eval so a "best" exists for downstream consumers.
    final_metrics = evaluate_sb3_on_test(model, prepped.test_env, prepped.initial_equity)
    cb = cb_holder.callback
    if cb.best_metrics is None:
        # No periodic eval fired (e.g. total_steps < eval_every). Use final.
        cb.best_metrics = final_metrics
        model.save(str(save_dir / "best.zip"))

    summary = {
        "algo": algo,
        "run_id": run_id,
        "save_dir": str(save_dir),
        "results_dir": str(results_dir),
        "total_steps": int(total_steps),
        "final_metrics": final_metrics.to_dict(),
        "best_metrics": cb.best_metrics.to_dict() if cb.best_metrics is not None else None,
        "wall_time_s": float(time.time() - t0),
    }
    summary_path = results_dir / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)
    return summary
