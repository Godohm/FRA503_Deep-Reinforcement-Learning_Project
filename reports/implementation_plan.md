# Implementation Plan — CFD Trading Agent (EURUSD, DRL)

**Project:** CFD Trading Agent Using Deep Reinforcement Learning for EURUSD Intraday Trading
**Scope:** Simulation & backtesting only — no live execution. Academic / research project.
**Original course slides** used XAUUSD / gold; this implementation uses **EURUSD** consistently.
**Locked-in decisions (user Q&A):** custom PyTorch Double DQN; SB3 A2C & PPO; session **09:00–00:00 UTC+7**; broker spread + configurable commission; **TensorBoard + CSV** logging.

> The full plan-mode document lives at `C:\Users\omza3\.claude\plans\you-are-claude-code-linear-nest.md`. This file is the repo-resident, user-facing version.

---

## 1. Overall Architecture

```
┌────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│ MT5 CSV (raw)  │ →  │ Data Pipeline      │ →  │ Train / Test split │
│ eurusd_m1_2024 │    │ - clean / dedupe   │    │ 11 mo / 1 mo       │
└────────────────┘    │ - session filter   │    └─────────┬──────────┘
                      │ - missing-bar fix  │              │
                      └────────────────────┘              ▼
                                                ┌────────────────────┐
                                                │ Feature Engineering│
                                                │ - returns 1/5/15/  │
                                                │   30/60 min        │
                                                │ - MACD, RSI, Stoch,│
                                                │   ATR, spread      │
                                                │ - fit scaler (train│
                                                │   only)            │
                                                └─────────┬──────────┘
                                                          │
                       ┌──────────────────────────────────┘
                       ▼
        ┌───────────────────────────────────┐
        │ EURUSDIntradayTradingEnv (gym)    │
        │ state s_t ∈ R^15                  │
        │ action a_t ∈ {-1, 0, 1}           │
        │ reward = net pnl after costs      │
        │ 1 episode = 1 day                 │
        └─────────────┬─────────────────────┘
                      │
   ┌──────────────────┼──────────────────┐
   ▼                  ▼                  ▼
┌──────────┐   ┌──────────────┐   ┌────────────┐
│ DDQN MLP │   │ SB3 A2C MLP  │   │ SB3 PPO MLP│
│ (custom) │   │              │   │            │
└────┬─────┘   └──────┬───────┘   └─────┬──────┘
     └────────────────┼──────────────────┘
                      ▼
            ┌────────────────────┐
            │ Backtest + Metrics │
            │ Sharpe, MDD, win%, │
            │ equity curves      │
            └────────────────────┘
```

## 2. Data Pipeline

**Source:** `data/raw/eurusd_m1_2024.csv` — 372,673 1-min bars (2024-01-02 → 2024-12-31).

- `src/data/load_mt5.py`: `mt5.copy_rates_range` returns **UTC**. Parse `time` as UTC with `tz='UTC'`. Coerce dtypes: prices `float32`, `tick_volume`/`spread`/`real_volume` `int32`.
- `src/data/preprocess.py`:
  1. Drop duplicate timestamps.
  2. Keep canonical index in UTC. Convert to `Asia/Bangkok` **only for session masking**.
  3. Apply session window (default 09:00–24:00 UTC+7).
  4. Reindex within each session_day to a complete 1-min grid; FFill OHLC for gaps ≤ 5; drop days with > 30 missing bars.
  5. Tag each row with `session_day` (UTC+7 calendar date) and `bar_idx_in_day`.
  6. Log first/last UTC and UTC+7 timestamps to `data_schema.md`.
- `src/data/split.py`: chronological split — train = first 11 months (Jan–Nov 2024), test = month 12 (Dec 2024). Configurable via `env.yaml`.

Outputs: `data/processed/train.parquet`, `data/processed/test.parquet`, `metadata.json`.

## 3. Feature Engineering Pipeline

`src/features/indicators.py` — pure, vectorised, **no look-ahead**:
- `rsi(close, period=14)`
- `macd(close, fast=12, slow=26, signal=9)` → MACD − signal
- `stochastic(high, low, close, k=14, d=3)` → %K
- `atr(high, low, close, period=14)` (Wilder)

`src/features/state_builder.py` — 15-D state vector at bar `t`:

| Idx | Feature           | Formula |
|-----|-------------------|---|
| 0   | ret_1m            | log(close_t / close_{t-1}) |
| 1   | ret_5m            | log(close_t / close_{t-5}) |
| 2   | ret_15m           | log(close_t / close_{t-15}) |
| 3   | ret_30m           | log(close_t / close_{t-30}) |
| 4   | ret_60m           | log(close_t / close_{t-60}) |
| 5   | spread (feature)  | spread_t × point_size  *(price units; see §3a)* |
| 6   | macd_hist         | macd_t − signal_t |
| 7   | stoch_k           | %K_t |
| 8   | rsi               | RSI_t / 100 |
| 9   | atr               | ATR_t |
| 10  | TL                | bars_remaining_in_session / session_length |
| 11  | POS               | current position ∈ {−1, 0, 1} |
| 12  | PR                | unrealized return of open position |
| 13  | DR                | cumulative daily PnL / initial_equity |
| 14  | HT                | holding_time / session_length |

### 3a. Spread's dual role

The spec lists spread as feature #6 *and* it drives transaction cost. So the observation space is **{OHLCV-derived + indicators + broker spread + positional}**, not strictly OHLCV-only.

- **Feature path:** `spread_t × point_size` → price units → scaled by train-fit `StandardScaler`. Lets the policy condition on cost regime.
- **Cost path:** `spread_t × point_size × unit_size` → account currency → deducted only on position-change bars.

Two paths, two unit tests (`test_features.py::test_spread_in_state`, `test_reward.py::test_spread_cost_units`).

`src/features/normalization.py` — `StandardScaler` for features 0–9 (fit on train only, dumped via `joblib`). Features 10–14 are bounded by design; only clipped.

## 4. Environment API (`EURUSDIntradayTradingEnv`)

Gymnasium `Env`:

```python
observation_space = Box(low=-inf, high=inf, shape=(15,), dtype=float32)
action_space      = Discrete(3)   # 0→short(-1), 1→flat(0), 2→long(+1)
```

`reset(seed)`: pick next valid `session_day`, seek past `warmup_bars`, reset position=0, equity=`initial_equity`, daily_pnl=0.

### 4a. Step semantics (no look-ahead, no missing next bar)

At step `t`, observation `s_t` is built from bars `≤ t`. Agent picks `a_t` ∈ {short, flat, long}, treated as **target position**.

- **Execution** defaults to `current_close` (configurable to `next_open`).
- **Force-close** on the last bar of the session overrides the agent's choice with `target = 0` and **always uses `current_close`** regardless of the config.
- **PnL booking** is mark-to-market between known closes:

```
mtm_pnl_t  = prev_pos × (close_t − close_{t-1}) × unit_size       # carry PnL on bar t
txn_cost_t = |target − prev_pos| × (spread_t × point_size + commission_price) × unit_size
net_pnl_t  = mtm_pnl_t − txn_cost_t                               # account currency
```

This avoids any reference to `close_{t+1}` / `open_{t+1}`, so the final bar's force-close needs no non-existent next bar.

### 4b. Cost & reward units

All terms end in **account currency** before subtraction (both `mtm_pnl` and `txn_cost` are × `unit_size`):

| Quantity | Unit | Notes |
|---|---|---|
| `close_t` | quote ccy | from CSV |
| `point_size` | quote ccy / point | `1e-5` for 5-digit EURUSD |
| `unit_size` | base ccy notional per position unit | `100 000` EUR = 1 std lot |
| `spread_t × point_size × unit_size` | **account ccy** | cost path only |
| `commission_price × unit_size` | **account ccy** | per side per lot |
| `mtm_pnl_t` | **account ccy** | every step |
| `net_pnl_t` | **account ccy** | reward source |

**Reward:** `reward_t = net_pnl_t / initial_equity × reward_scaling`. Default `reward_scaling = 1.0`. Per-step magnitude ~`±1e-4` on a $10 000 / 1-lot setup. **No opaque `1e4` multiplier.**

### 4c. `step()` algorithm (final)

```
1. prev_pos    = self.position
2. target      = action - 1
3. forced_close = (bar_idx == last_bar_of_session)
4. if forced_close: target = 0
5. txn_cost = |target - prev_pos| * (spread_t * point_size + commission_price) * unit_size
6. mtm_pnl  = prev_pos * (close_t - close_{t-1}) * unit_size
7. net_pnl  = mtm_pnl - txn_cost
8. self.position    = target
9. self.equity     += net_pnl
10. self.daily_pnl += net_pnl
11. self.holding_time = 0 if target != prev_pos else self.holding_time + 1
12. reward    = net_pnl / initial_equity * reward_scaling
13. terminated = forced_close or self.equity <= stop_equity_floor
14. obs_next   = build_state(t + 1)   # else last obs on termination
15. info  = {raw_pnl, transaction_cost, net_pnl, current_position, action,
             equity, daily_return, forced_close, mtm_pnl}
16. return obs_next, reward, terminated, False, info
```

## 5. Algorithm Training Workflow

| Algo | Source | Net | Config |
|---|---|---|---|
| Double DQN | custom PyTorch (`src/agents/double_dqn.py`) | MLP 15→128→128→3 | `configs/dqn.yaml` |
| A2C        | `stable_baselines3.A2C`                     | `MlpPolicy` (128, 128) | `configs/a2c.yaml` |
| PPO        | `stable_baselines3.PPO`                     | `MlpPolicy` (128, 128) | `configs/ppo.yaml` |

Common: same env, same state, same action, same reward; matched hidden sizes for parity; reproducible seeds (Python / numpy / torch / gymnasium). TensorBoard + CSV logging via `src/utils/logging.py`.

Saves per run: `models/<algo>/<run-id>/{best.{pt,zip}, final.{pt,zip}, config.yaml, scaler.pkl, train_log.csv}` + TB events.

## 6. Evaluation Workflow

- `src/evaluation/backtest.py` — deterministic rollout on test set, writes `trade_log.csv`.
- `src/evaluation/metrics.py` — Total return, mean daily return, Sharpe (×√252), Sortino, MDD%, # trades, win rate, avg PnL/trade, total txn cost, exposure time, final equity.
- `src/evaluation/plots.py` — `equity_curve.png`, `drawdown.png`, `daily_returns_hist.png`, `cumret_overlay.png`.
- `src/evaluation/baselines.py` — long-only, short-only, flat (same env, same costs).
- `scripts/evaluate_all.py` writes `results/comparison_table.csv` and `reports/results_summary.md`.

## 7. Configuration Design

YAML loaded by `src/utils/config.py` (dataclass-validated). Files: `env.yaml`, `dqn.yaml`, `a2c.yaml`, `ppo.yaml`. See `configs/env.yaml` for the canonical, fully-commented spec.

## 8. Testing Strategy

`pytest tests/`. Fast unit tests, no full training.

| File | Covers |
|---|---|
| `test_features.py` | indicator values vs hand-computed; no-look-ahead (shift-input ⇒ shift-output); NaN at warmup; **spread in state is in price units**; **MT5 UTC parsing** (UTC+7 view shifts by 7h) |
| `test_env.py` | obs shape 15; step return tuple; episode terminates at session end; force-close at last bar **without next-bar lookup**; deterministic with seed |
| `test_reward.py` | hold-no-move ≈ 0; correct sign; cost ∝ \|Δpos\|; **unit consistency** (mtm and txn both ∝ unit_size); **force-close** reads no next bar; reward magnitude exact on synthetic days |
| `test_metrics.py` | Sharpe / Sortino / MDD on known curves; win-rate edges |

Plus a smoke training test (~1 000 steps) per algorithm.

## 9. Potential Risks & Assumptions

| Risk | Mitigation |
|---|---|
| Only 12 months of data → small test set (~30 days) | Document 11/1 split; rolling-window eval if time |
| Scaler leakage | Fit on train only; unit test verifies |
| Spread spikes around news | `spread_cap_quantile` in `env.yaml` (default 0.99) |
| **Timezone confusion** | MT5 returns UTC; canonical index stays UTC; UTC+7 only for session masking; sanity-checks logged |
| **Cost-unit mismatch** | Both `mtm_pnl` and `txn_cost` × `unit_size` (account ccy); test enforced |
| **Reward magnitude** | Default `net_pnl / initial_equity` with `reward_scaling=1.0`; no opaque `1e4` |
| **Force-close on last bar** | MtM uses `close_{t-1}` & `close_t` only; force-close always `current_close` |
| **Spread dual role** | Feature path = price units; cost path = account ccy; two tests |
| SB3 DQN ≠ Double DQN | Custom PyTorch DDQN |
| Slippage absent | v1 = spread + commission; hook left for slippage model |
| Reproducibility on Windows | Seed Python / numpy / torch (CUDA) / gymnasium |

## 10. Step-by-Step Milestones

| # | Milestone | Output | Gate |
|---|---|---|---|
| **M0** | Planning docs | `reports/implementation_plan.md`, `reports/experimental_design.md`, `reports/data_schema.md`, `configs/env.yaml`, `TODO.md`, `README.md`, `requirements.txt`, `.gitignore` | Five corrections explicit; data_schema documents UTC+UTC+7 |
| M1 | Scaffolding + data move | tree + `data/raw/eurusd_m1_2024.csv` | `pytest -q` discovers 0 tests successfully |
| M2 | Data pipeline | `load_mt5.py`, `preprocess.py`, `split.py`, `prepare_data.py` | parquet files exist; row counts logged |
| M3 | Features + tests | `indicators.py`, `state_builder.py`, `normalization.py` | no-lookahead test passes |
| M4 | Trading env + tests | `eurusd_intraday_env.py` | `gymnasium.utils.env_checker.check_env` passes |
| M5 | Baselines + metrics | `baselines.py`, `metrics.py`, `backtest.py`, `plots.py` | long-only backtest emits CSV + plots |
| M6 | Custom DDQN | `double_dqn.py`, `train_dqn.py`, `configs/dqn.yaml` | 1 k-step smoke run + TB curves |
| M7 | A2C + PPO via SB3 | `train_a2c.py`, `train_ppo.py`, configs | 1 k-step smoke runs |
| M8 | Experiment 1 | `run_experiment_1.py` (≥ 3 seeds × 3 algos) | `results/comparison_table.csv` populated |
| M9 | Reports | `results_summary.md` | comparison narrative complete |
| M10 | (Optional) Exp 2 | `run_experiment_2.py` | deferred |
