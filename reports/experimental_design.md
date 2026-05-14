# Experimental Design

This document defines the experiments, controlled variables, evaluation protocol, and reporting standards used to compare DRL agents on the EURUSD intraday environment.

## 1. Research Questions

1. **RQ1 — Algorithm:** Does Double DQN, A2C, or PPO produce the best risk-adjusted intraday performance on EURUSD M1 under identical state/action/reward/cost settings?
2. **RQ2 — Architecture:** For the best algorithm from RQ1, does an LSTM policy meaningfully improve performance over an MLP of comparable parameter count? *(deferred — see Experiment 2)*
3. **RQ3 — Baseline gap:** Does the best DRL model outperform trivial baselines (long-only, short-only, flat) after transaction costs?

## 2. Controlled Variables (identical across all runs)

| Variable | Value |
|---|---|
| Instrument | EURUSD CFD (5-digit) |
| Bar interval | 1 minute |
| Session window | 09:00–00:00 UTC+7 |
| State | 15-D vector (see `data_schema.md` & `implementation_plan.md` §3) |
| Action | `Discrete(3)` → target position ∈ {−1, 0, +1} |
| Reward | `net_pnl / initial_equity × reward_scaling`, `reward_scaling = 1.0` |
| `initial_equity` | 10 000 USD |
| `unit_size` | 100 000 EUR (1 std lot) |
| Spread source | broker `spread` column |
| Spread cap | 0.99 quantile of train spread (configurable) |
| Commission | 0.0 (configurable) |
| Execution | `current_close` (force-close always `current_close`) |
| Force-close | last bar of session, target = 0 |
| Train / test split | 11 months train (Jan–Nov 2024) / 1 month test (Dec 2024), chronological |
| Scaler | `StandardScaler` on features 0–9, fit **on train only** |
| Network shape | MLP 15 → 128 → 128 → 3 (ReLU) — same hidden sizes for parity across DDQN / A2C / PPO |
| Seeds per algo (Exp 1) | `{42, 123, 2024}` |
| Total training steps per run | 1 000 000 (or convergence, whichever first) |

## 3. Experiment 1 — Algorithm Comparison

**Goal:** RQ1. **Setup:** Run each algorithm with three seeds. Report mean ± std of each metric on the test month.

### 3.1 Configurations

| Algo | Source | Key hparams |
|---|---|---|
| **Double DQN** | custom PyTorch (`src/agents/double_dqn.py`) | lr 1e-4, γ 0.99, ε 1.0→0.05 linear over 200k steps, replay 100k, batch 64, target sync every 1k steps, MSE loss, Adam, grad-clip 10 |
| **A2C** | `stable_baselines3.A2C` | lr 7e-4, γ 0.99, n_steps 5, vf_coef 0.5, ent_coef 0.01, max_grad_norm 0.5 |
| **PPO** | `stable_baselines3.PPO` | lr 3e-4, γ 0.99, n_steps 2048, batch 64, n_epochs 10, clip 0.2, gae_lambda 0.95, ent_coef 0.0 |

Configs are committed to `configs/{dqn,a2c,ppo}.yaml` and loaded by `src/utils/config.py`. Any tuning is recorded as a new YAML, never an edit-in-place.

### 3.2 Evaluation Protocol

- During training: `EvalCallback` (or DDQN equivalent) every 10 000 steps, snapshot the best model by **mean episode reward on the test month**.
- After training: re-run the best snapshot deterministically (`deterministic=True` for SB3, `argmax` Q for DDQN) over the full test month → `trade_log.csv` per run.
- All metrics computed from the `trade_log` only (no in-training metric snooping).

### 3.3 Metrics (per run)

| Metric | Definition |
|---|---|
| Total return (%) | `(equity_final − initial_equity) / initial_equity` |
| Mean daily return (%) | mean of per-session_day return |
| Sharpe (annualised) | `mean_daily / std_daily × √252` |
| Sortino (annualised) | downside-only denominator |
| Max Drawdown (%) | worst peak-to-trough on equity curve |
| # of trades | count of bars where `position` changes |
| Win rate (%) | fraction of position-holding episodes with positive net_pnl |
| Avg PnL / trade | account ccy |
| Total transaction cost | account ccy |
| Exposure time (%) | fraction of bars with `|position| = 1` |
| Final equity | account ccy |

### 3.4 Reporting

- `results/<algo>/<run-id>/` per seed: model, scaler, config, trade_log, plots.
- `results/comparison_table.csv`: rows = algos, cols = mean ± std for each metric, populated by `scripts/evaluate_all.py`.
- `reports/results_summary.md`: narrative + plots.

## 4. Experiment 2 — Architecture Comparison *(deferred)*

**Goal:** RQ2. **Trigger:** only after Experiment 1 is fully reported.

- Take the winning algorithm from Exp 1.
- Compare MLP (128, 128) vs LSTM (hidden 128, 1 layer) under matched parameter counts (adjust LSTM hidden if needed for parity).
- Same protocol as Exp 1, same three seeds.

If the winner is PPO or A2C, SB3 provides `MlpLstmPolicy` via `sb3-contrib` (RecurrentPPO). For DDQN we add a small recurrent variant only if the winner is DDQN.

## 5. Experiment 3 — Baseline Comparison

**Goal:** RQ3.

Baselines run in the same env (identical costs / forced-close / etc.) with hard-coded policies:

| Baseline | Policy |
|---|---|
| Long-only | Set `target = +1` at every step until force-close |
| Short-only | Set `target = −1` at every step until force-close |
| Flat | Set `target = 0` at every step (sanity check; should yield 0 PnL, 0 cost) |

Reported in the same `comparison_table.csv` alongside DRL models. Sharpe of long-only on a year with a directional trend may look strong; the point of the comparison is to show that the DRL agent provides **net-of-cost** improvement, not raw bullish exposure.

## 6. Reproducibility Checklist

- [ ] All hyperparameters in YAML, hashed and stored alongside the run.
- [ ] Seeds set for Python `random`, `numpy`, `torch` (CPU + CUDA), and `gymnasium` action space.
- [ ] Scaler fit on train-only, dumped alongside the model.
- [ ] Smoke test (`pytest tests/ -q` + 1 000-step micro-train per algo) passes before any full run.
- [ ] `metadata.json` for processed data contains row counts and date ranges in UTC and UTC+7.
- [ ] Git commit hash recorded in each `train_log.csv` row.

## 7. Pitfalls to Avoid

| Pitfall | Guard |
|---|---|
| Look-ahead via global scaler | Scaler fit on train only, frozen for test |
| Look-ahead via feature using `close_{t+1}` | `build_state(t)` reads bars ≤ `t` only; unit test |
| Force-close needing a missing next bar | MtM PnL uses `close_t − close_{t-1}` only; force-close always `current_close` |
| Reward magnitude tuned via opaque `1e4` | Default `net_pnl / initial_equity`, `reward_scaling = 1.0` |
| Unit mismatch (price vs account ccy) | Both `mtm_pnl` and `txn_cost` × `unit_size` before subtraction |
| Wrong timezone | MT5 returns UTC; UTC+7 is only used for session masking; timestamps logged |
| Reporting in-training metrics | Final metrics from deterministic post-train rollout only |
| Selecting on test by accident | Best-by-test-reward chosen via `EvalCallback`; *final* metrics still reported on the same test split — **explicitly disclosed** as model selection, not held-out evaluation. With only 1 test month, true held-out evaluation would require additional data (see §8) |

## 8. Limitations

- **Single test month** — Exp 1 results have one observation per algo-seed. Use mean ± std across seeds; do not over-claim significance.
- **No external validation period** — Best model is selected by the same month it is evaluated on. To do this cleanly later, either acquire more data and add a validation window, or use rolling-window evaluation.
- **No slippage model** — v1 charges spread + commission only. Real fill prices on news bars would be worse. Recorded as an assumption.
- **No funding / overnight handling** — Intraday only by design; force-close removes overnight risk.
- **Single instrument** — Conclusions about EURUSD do not transfer to other CFDs without re-running.
