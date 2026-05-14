# TODO — EURUSD DRL Trading Agent

> Note: M3–M5 do not implement MLP or LSTM. They only prepare features, environment, metrics, and
> baselines. MLP starts in M6/M7 as the agent network architecture. LSTM is part of Experiment 2
> after selecting the best MLP-based algorithm.

Ordered by implementation priority. Each milestone ends with: tests green + small artifact saved.

---

## M0 — Planning docs ✅ Complete

- [x] Create repo directory tree
- [x] Move `eurusd_m1_2024.csv` → `data/raw/`
- [x] `configs/env.yaml` (with five corrections explicit)
- [x] `reports/implementation_plan.md`
- [x] `reports/data_schema.md` (UTC source + UTC+7 view documented)
- [x] `reports/experimental_design.md`
- [x] `TODO.md`
- [x] `requirements.txt`
- [x] `README.md`
- [x] `.gitignore`

---

## M1 — Scaffolding ✅ Complete

- [x] `pip install -r requirements.txt` clean in a fresh venv
- [x] `pytest -q` discovers 0 tests successfully (sanity)
- [x] Empty `__init__.py` files where needed under `src/`

---

## M2 — Data pipeline ✅ Complete

- [x] `src/data/load_mt5.py` — CSV reader, UTC localisation, dtype coercion
- [x] `src/data/preprocess.py` — dedup, session mask via UTC+7 view, per-day reindex, gap-fill rules
- [x] `src/data/split.py` — chronological train/test by `session_day`
- [x] `scripts/prepare_data.py` — orchestrate above, log timestamps, write parquet + `metadata.json`
- [x] Verified row counts: train = 215,100 rows (239 sessions), test = 18,000 rows (20 sessions)
- [x] 1 dropped day: 2024-12-26 (Boxing Day, 360 missing bars > threshold 30)
- [x] `tests/test_data_pipeline.py` — 13 tests passing

---

## M3 — Feature engineering ✅ Complete

- [x] `src/features/indicators.py` — RSI, MACD-hist, Stoch %K, ATR (Wilder), log returns
- [x] `src/features/state_builder.py` — 15-D vector (10 price + 5 positional), per-session_day
      computation, no cross-session leakage
- [x] `src/features/normalization.py` — `PriceFeatureScaler` (StandardScaler, fit-on-train-only,
      joblib persistence)
- [x] `tests/test_features.py` — 18 tests passing (indicator values, no-lookahead, state
      shape/order, scaler train-only)

---

## M4 — Trading environment ✅ Complete

- [x] `src/envs/eurusd_intraday_env.py` — Gymnasium env (Discrete(3), Box(15,)), current_close
      execution, mtm-based reward, force-close at last bar using no next-bar data
- [x] `gymnasium.utils.env_checker.check_env` passes
- [x] `tests/test_env.py` — 14 tests passing
- [x] `tests/test_reward.py` — 12 tests passing (sign, unit consistency, force-close, magnitude,
      spread_cost_factor)

---

## M5 — Baselines + metrics ✅ Complete

- [x] `src/evaluation/metrics.py` — Sharpe, Sortino, MDD, win rate, exposure time, trade PnL
      aggregation, cumulative_return_curve, equity_curve_account_ccy
- [x] `src/evaluation/baselines.py` — long-only, short-only, flat policies
- [x] `src/evaluation/backtest.py` — deterministic rollout → trade_log DataFrame with time_utc,
      time_local, bar_idx_in_day
- [x] `src/evaluation/plots.py` — equity curve, drawdown, daily return histogram
      (non-compounded labeling)
- [x] `scripts/run_baselines.py` — runs all 3 baselines on test split, saves artefacts to
      `results/baselines/`
- [x] `tests/test_metrics.py` — 20 tests passing

Baseline results (Dec 2024 test split, 20 sessions):
- long-only:  total_return = −14.94%, Sharpe = −4.09, MDD = −16.50%, txn_cost = $713
- short-only: total_return = +0.68%,  Sharpe = +0.19, MDD = −11.50%, txn_cost = $713
- flat:        total_return = 0.00%,   all zeros,                      txn_cost = $0

---

## Cleanup pass ✅ Complete

- [x] All `__init__.py` files confirmed with double-underscore names
- [x] All three baseline trade_logs confirmed at 16,800 rows (20 sessions × 840 trading bars)
- [x] Added `time_utc`, `time_local`, `bar_idx_in_day`, `close`, `spread` to env info dict and
      trade_log
- [x] Added `costs.spread_cost_factor: 1.0` to `configs/env.yaml` with full documentation
- [x] Equity curve naming fixed: `cumulative_return_curve` (dimensionless) +
      `equity_curve_account_ccy` (account ccy); `equity_curve_from_log` kept as alias; plots
      labeled "Non-compounded equity (additive across daily resets)"
- [x] Transaction cost convention documented in module docstring and env.yaml
- [x] **Tests passing: 78/78**

---

## M6 — Custom Double DQN + MLP

- [ ] Implement custom PyTorch Double DQN in `src/agents/double_dqn.py`
  - QNetwork: input_dim=15, hidden_sizes=[128, 128], output_dim=3, ReLU
  - Replay buffer: capacity 100,000, batch size 64
  - Target network: hard-copy sync every 1,000 steps (Polyak configurable)
  - ε-greedy: linear decay 1.0 → 0.05 over configurable steps
  - DDQN target: `y = r + γ × Q_target(s', argmax_a Q_online(s', a))`
  - Adam optimizer, MSE loss, gradient clip 10
  - Checkpoint save/load for online + target networks
- [ ] `configs/dqn.yaml` — all hyperparameters
- [ ] `src/agents/train_dqn.py` — training loop with TensorBoard + CSV logging, periodic eval on
      test split, best-checkpoint tracking
- [ ] Smoke run: 1,000 steps without error; loss decreases

---

## M7 — A2C + MLP and PPO + MLP via Stable-Baselines3

- [ ] `configs/a2c.yaml` — A2C hyperparameters
- [ ] `configs/ppo.yaml` — PPO hyperparameters
- [ ] `src/agents/train_a2c.py` — SB3 A2C wrapper with MlpPolicy, hidden_sizes=[128, 128],
      Monitor + EvalCallback
- [ ] `src/agents/train_ppo.py` — SB3 PPO wrapper with MlpPolicy, hidden_sizes=[128, 128],
      Monitor + EvalCallback
- [ ] Smoke run: 1,000 steps each without error; no training crash

---

## M8 — Experiment 1: Algorithm comparison using the same MLP architecture

Compare Double DQN + MLP vs A2C + MLP vs PPO + MLP using the same environment, reward, state
space, cost model, train/test split, and evaluation metrics.

- [ ] `scripts/run_experiment_1.py` — drives all three algos × seeds {42, 123, 2024}
- [ ] At least 3 seeds per algorithm
- [ ] Full training on train split; evaluate on test split (Dec 2024)
- [ ] Save per run: `models/<algo>/<run-id>/{best, final}.{pt,zip}` + config, scaler, train_log,
      TensorBoard events
- [ ] Save per run: `results/<algo>/<run-id>/{equity_curve.png, drawdown.png, trade_log.csv}`
- [ ] `scripts/evaluate_all.py` — aggregate into `results/comparison_table.csv`
- [ ] Select the best algorithm based on Sharpe, total return, and MDD

---

## M9 — LSTM support for Experiment 2

Add recurrent/sequence model support only after Experiment 1 identifies the best MLP-based
algorithm.

- [ ] If best algorithm is PPO or A2C: use `sb3-contrib RecurrentPPO` (MlpLstmPolicy) with
      comparable hidden size
- [ ] If best algorithm is DDQN: implement recurrent Q-network with LSTM layer if feasible;
      otherwise document the limitation clearly
- [ ] Match parameter count between MLP and LSTM variants where possible for fair comparison
- [ ] Update relevant training script and config

---

## M10 — Experiment 2: Network comparison under the same best algorithm

Compare BestAlgo + MLP vs BestAlgo + LSTM using the same environment, reward, cost model, and
test split.

- [ ] `scripts/run_experiment_2.py` — runs best algo × {MLP, LSTM} × seeds {42, 123, 2024}
- [ ] Same evaluation protocol as Experiment 1
- [ ] Append results to `results/comparison_table.csv`

---

## M11 — Final report and baseline comparison

Compare the best DRL model against all baselines (long-only, short-only, flat).

- [ ] `reports/results_summary.md` — final tables, plots, conclusions, limitations
- [ ] Full comparison including MDD, Sharpe, win rate, transaction costs, exposure time
- [ ] Address limitations: single test month, no slippage model, no held-out validation set
- [ ] Honour scope: simulation and backtesting only; no live trading
