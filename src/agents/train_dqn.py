"""Training loop for the custom Double DQN agent on EURUSDIntradayTradingEnv.

This module is engine-agnostic of the CLI — ``scripts/train_dqn.py`` is the
thin wrapper that parses args, loads YAML configs, and calls ``train_dqn``.

Layout per run::

    models/ddqn/<run_id>/{best.pt, final.pt, config.yaml, scaler.pkl}
    results/ddqn/<run_id>/{train_log.csv, eval_log.csv, tb/<events>, ...}

Eval is a deterministic full-pass over the test parquet using ``run_backtest``
+ ``compute_all_metrics``; we track ``best.pt`` by Sharpe (with total_return as
a tie-break against zero-volatility eval runs).
"""
from __future__ import annotations

import csv
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from src.agents.double_dqn import DDQNConfig, DoubleDQNAgent
from src.envs.eurusd_intraday_env import EURUSDIntradayTradingEnv
from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import BacktestMetrics, compute_all_metrics
from src.features.normalization import PriceFeatureScaler
from src.features.state_builder import compute_price_features

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- helpers


def _restore_session_day_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """parquet round-trips session_day as 'string' / object; convert back to date."""
    if df["session_day"].dtype.name == "string" or df["session_day"].dtype == object:
        df = df.copy()
        df["session_day"] = pd.to_datetime(df["session_day"]).dt.date
    return df


def _new_run_id(prefix: str = "ddqn") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}"


def _try_import_tb_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        return SummaryWriter
    except Exception as e:  # pragma: no cover - TB optional
        logger.warning("TensorBoard unavailable (%s); CSV logging only.", e)
        return None


# --------------------------------------------------------------------------- prep


@dataclass
class _Prepped:
    train_env: EURUSDIntradayTradingEnv
    test_env: EURUSDIntradayTradingEnv
    scaler: PriceFeatureScaler
    initial_equity: float


def _prepare_envs(env_cfg: dict[str, Any]) -> _Prepped:
    """Load splits, fit scaler on TRAIN ONLY, build train (random) and test (seq) envs."""
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
    return _Prepped(train_env=train_env, test_env=test_env, scaler=scaler,
                    initial_equity=initial_equity)


def _agent_from_cfg(dqn_cfg: dict[str, Any], state_dim: int, n_actions: int,
                    seed: int) -> DoubleDQNAgent:
    net = dqn_cfg.get("network", {})
    rl = dqn_cfg.get("dqn", {})
    cfg = DDQNConfig(
        state_dim=int(state_dim),
        n_actions=int(n_actions),
        hidden_sizes=tuple(int(h) for h in net.get("hidden_sizes", [128, 128])),
        lr=float(rl.get("lr", 1.0e-3)),
        gamma=float(rl.get("gamma", 0.99)),
        batch_size=int(rl.get("batch_size", 64)),
        buffer_capacity=int(rl.get("buffer_capacity", 100_000)),
        min_buffer_to_learn=int(rl.get("min_buffer_to_learn", 1_000)),
        target_update_freq=int(rl.get("target_update_freq", 1_000)),
        grad_clip=float(rl.get("grad_clip", 10.0)),
        eps_start=float(rl.get("eps_start", 1.0)),
        eps_end=float(rl.get("eps_end", 0.05)),
        eps_decay_steps=int(rl.get("eps_decay_steps", 50_000)),
        eps_decay_type=str(rl.get("eps_decay_type", "linear")),
        device=str(rl.get("device", "cpu")),
        seed=int(seed),
    )
    return DoubleDQNAgent(cfg)


# --------------------------------------------------------------------------- eval


def _greedy_policy(agent: DoubleDQNAgent):
    def _pi(obs: np.ndarray) -> int:
        return int(agent.select_action(obs, greedy=True))
    return _pi


def evaluate_on_test(
    agent: DoubleDQNAgent, test_env: EURUSDIntradayTradingEnv, initial_equity: float
) -> BacktestMetrics:
    """Deterministic full-pass over the test split with greedy actions."""
    test_env._day_cursor = 0  # rewind day pointer (sequential mode)
    trade_log = run_backtest(test_env, _greedy_policy(agent))
    return compute_all_metrics(trade_log, initial_equity=initial_equity)


# --------------------------------------------------------------------------- training


def train_dqn(
    env_cfg: dict[str, Any],
    dqn_cfg: dict[str, Any],
    *,
    total_steps: Optional[int] = None,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run a full (or smoke) DDQN training loop. Returns a summary dict."""
    training = dqn_cfg.get("training", {})
    total_steps = int(total_steps if total_steps is not None
                      else training.get("total_steps", 200_000))
    eval_every = int(training.get("eval_every_steps", 10_000))
    log_every = int(training.get("log_every_steps", 1_000))
    learn_every = int(dqn_cfg.get("dqn", {}).get("learn_every", 1))
    run_seed = int(seed if seed is not None else dqn_cfg.get("seed", env_cfg.get("seed", 42)))

    save_dir_root = Path(training.get("save_dir_root", "models/ddqn"))
    results_dir_root = Path(training.get("results_dir_root", "results/ddqn"))
    run_id = run_id or _new_run_id("ddqn")
    save_dir = save_dir_root / run_id
    results_dir = results_dir_root / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Persist effective configs alongside checkpoints.
    # Stamp the actual run_seed so the saved env.yaml is self-consistent.
    effective_env_cfg = {**env_cfg, "seed": run_seed}
    with (save_dir / "env.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(effective_env_cfg, f, sort_keys=False)
    with (save_dir / "dqn.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dqn_cfg, f, sort_keys=False)

    prepped = _prepare_envs(env_cfg)
    prepped.scaler.save(save_dir / "scaler.pkl")

    agent = _agent_from_cfg(dqn_cfg, state_dim=prepped.train_env.observation_space.shape[0],
                            n_actions=prepped.train_env.action_space.n, seed=run_seed)

    SummaryWriter = _try_import_tb_writer()
    tb_writer = SummaryWriter(str(results_dir / "tb")) if SummaryWriter is not None else None

    train_log_path = results_dir / "train_log.csv"
    eval_log_path = results_dir / "eval_log.csv"
    train_log_f = open(train_log_path, "w", newline="", encoding="utf-8")
    train_writer = csv.writer(train_log_f)
    train_writer.writerow([
        "step", "episode", "ep_return", "ep_length", "epsilon",
        "avg_loss", "buffer_size", "learn_steps", "wall_time_s",
    ])
    eval_log_f = open(eval_log_path, "w", newline="", encoding="utf-8")
    eval_writer = csv.writer(eval_log_f)
    eval_writer.writerow([
        "step", "total_return", "sharpe", "sortino", "max_dd",
        "n_trades", "win_rate", "exposure_time", "final_equity", "is_best",
    ])

    # Training loop -------------------------------------------------------
    obs, _info = prepped.train_env.reset(seed=run_seed)
    ep_return = 0.0
    ep_length = 0
    episode = 0
    loss_buf: list[float] = []
    best_score = -np.inf
    best_metrics: Optional[BacktestMetrics] = None
    t0 = time.time()

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs, greedy=False)
        next_obs, reward, terminated, truncated, _info = prepped.train_env.step(action)
        done = bool(terminated or truncated)
        agent.buffer.push(obs, int(action), float(reward), next_obs, done)
        obs = next_obs
        ep_return += float(reward)
        ep_length += 1

        if step % learn_every == 0:
            loss = agent.learn()
            if loss is not None:
                loss_buf.append(loss)

        if done:
            episode += 1
            # Per-episode TB scalars (cheap, useful curve).
            if tb_writer is not None:
                tb_writer.add_scalar("train/episode_return", ep_return, step)
                tb_writer.add_scalar("train/episode_length", ep_length, step)
                tb_writer.add_scalar("train/epsilon", agent.epsilon, step)
            obs, _info = prepped.train_env.reset()
            ep_return = 0.0
            ep_length = 0

        if step % log_every == 0:
            avg_loss = float(np.mean(loss_buf)) if loss_buf else float("nan")
            row = [
                int(step), int(episode), float(ep_return), int(ep_length),
                float(agent.epsilon), avg_loss, int(len(agent.buffer)),
                int(agent.learn_steps), float(time.time() - t0),
            ]
            train_writer.writerow(row); train_log_f.flush()
            if tb_writer is not None:
                tb_writer.add_scalar("train/avg_loss", avg_loss, step)
                tb_writer.add_scalar("train/buffer_size", len(agent.buffer), step)
            logger.info(
                "step=%d ep=%d eps=%.3f avg_loss=%.4g buffer=%d",
                step, episode, agent.epsilon, avg_loss, len(agent.buffer),
            )
            loss_buf.clear()

        if step % eval_every == 0 or step == total_steps:
            m = evaluate_on_test(agent, prepped.test_env, prepped.initial_equity)
            score = float(m.sharpe_ratio) + 1.0e-6 * float(m.total_return)  # tie-break
            is_best = score > best_score
            if is_best:
                best_score = score
                best_metrics = m
                agent.save(save_dir / "best.pt")
            eval_writer.writerow([
                int(step), m.total_return, m.sharpe_ratio, m.sortino_ratio,
                m.max_drawdown_pct, m.n_trades, m.win_rate, m.exposure_time,
                m.final_equity, bool(is_best),
            ])
            eval_log_f.flush()
            if tb_writer is not None:
                tb_writer.add_scalar("eval/total_return", m.total_return, step)
                tb_writer.add_scalar("eval/sharpe", m.sharpe_ratio, step)
                tb_writer.add_scalar("eval/max_dd", m.max_drawdown_pct, step)
                tb_writer.add_scalar("eval/n_trades", m.n_trades, step)
            logger.info(
                "[eval @ step=%d] total_return=%.4f sharpe=%.3f MDD=%.4f trades=%d %s",
                step, m.total_return, m.sharpe_ratio, m.max_drawdown_pct, m.n_trades,
                "(new best)" if is_best else "",
            )

    agent.save(save_dir / "final.pt")
    # If for some reason no eval happened (total_steps < eval_every and != total_steps),
    # still produce a best.pt for downstream consumers — copy final.
    if not (save_dir / "best.pt").exists():
        shutil.copy2(save_dir / "final.pt", save_dir / "best.pt")

    train_log_f.close()
    eval_log_f.close()
    if tb_writer is not None:
        tb_writer.close()

    summary = {
        "run_id": run_id,
        "save_dir": str(save_dir),
        "results_dir": str(results_dir),
        "total_steps": int(total_steps),
        "episodes": int(episode),
        "best_score": float(best_score) if best_metrics is not None else None,
        "best_metrics": best_metrics.to_dict() if best_metrics is not None else None,
        "wall_time_s": float(time.time() - t0),
    }
    summary_path = results_dir / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)
    return summary
