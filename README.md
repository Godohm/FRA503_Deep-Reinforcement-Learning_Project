# FRA503 — Deep Reinforcement Learning for EURUSD Intraday Trading

> **Simulation & backtesting only — no live trading.**

Academic research project for FRA503. Builds a Gymnasium-compatible intraday CFD trading environment on 1-minute EURUSD data and compares three DRL algorithms under identical conditions.

---

## Algorithms compared

| Algorithm | Implementation | Best Sharpe (Optuna 50-trial HPO) |
|---|---|---|
| **Double DQN** | Custom PyTorch | **+3.64** |
| **PPO** | Stable-Baselines3 | +0.39 |
| **A2C** | Stable-Baselines3 | +0.12 |

ε-decay ablation: **linear decay** outperforms exponential (Sharpe +3.64 vs +0.51) over 50-trial Bayesian search.

---

## Project structure

```
FRA503_DRL/
├── configs/            # YAML hyperparameters (env, dqn, a2c, ppo, hpo)
├── data/raw/           # eurusd_m1_2024.csv — 372,673 1-min bars (MT5 export)
├── reports/            # implementation_plan, experimental_design, data_schema
├── scripts/
│   ├── prepare_data.py         # CSV → parquet splits + scaler
│   ├── train_{dqn,a2c,ppo}.py  # single-algo training CLI
│   ├── run_experiment_1.py     # multi-seed orchestrator
│   ├── run_baselines.py        # long/short/flat baselines
│   ├── evaluate_all.py         # aggregate comparison table
│   ├── run_hpo.py              # OAT sensitivity sweep
│   ├── optuna_search.py        # Bayesian HPO (Optuna TPE)
│   └── analyze_optuna.py       # dose-response plots + fANOVA report
├── src/
│   ├── agents/         # double_dqn.py (custom DDQN), SB3 wrappers
│   ├── data/           # load_mt5, preprocess, split
│   ├── envs/           # EURUSDIntradayTradingEnv (Gymnasium)
│   ├── evaluation/     # metrics, backtest, plots, baselines
│   ├── features/       # indicators, state_builder, normalization
│   └── utils/          # config, seeding, logging
└── tests/              # pytest unit tests (env, reward, features, metrics, agents)
```

---

## Environment design

| Setting | Value |
|---|---|
| Instrument | EURUSD CFD (5-digit broker, point_size = 1e-5) |
| Bar interval | 1 minute |
| Session window | 09:00–00:00 UTC+7 (force-close at midnight) |
| Train / val / test split | Jan–Dec 2025 (train) · Jan–Feb 2026 (val) · Mar–Apr 2026 (test) |
| State space | 15-D: 5 log-returns, spread, MACD, Stoch %K, RSI, ATR, time-left, position, unrealised PnL, daily PnL, holding-time |
| Action space | Discrete(3): short(−1), flat(0), long(+1) |
| Reward | `net_pnl / initial_equity` (dimensionless) |
| Execution | Current-close (force-close always current-close) |
| Transaction cost | Broker spread (points → price) + configurable commission |
| Initial equity | $10,000 · 1 standard lot (100,000 EUR) |

---

## Quick start

```powershell
# 1. Install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Build dataset
python scripts/prepare_data.py --config configs/env.yaml

# 3. Run unit tests
pytest tests/ -q

# 4. Smoke train (5k steps each)
python scripts/train_dqn.py --smoke
python scripts/train_a2c.py --smoke
python scripts/train_ppo.py --smoke

# 5. Full experiment (3 seeds × 3 algos)
python scripts/run_experiment_1.py --seeds 42 123 2024 --steps 200000

# 6. Baselines + comparison table
python scripts/run_baselines.py
python scripts/evaluate_all.py
```

---

## Hyperparameter optimisation

```powershell
# Bayesian HPO — 50 trials × 50k steps per algo (TPE sampler)
python scripts/optuna_search.py --algos ddqn a2c ppo --trials 50 --steps 50000

# ε-decay ablation for DDQN (exponential vs linear)
python scripts/optuna_search.py --algos ddqn --trials 50 --steps 50000 --study-suffix _expdecay --ddqn-decay-type exponential

# Generate dose-response plots + fANOVA report
python scripts/analyze_optuna.py --tags ddqn a2c ppo ddqn_expdecay
```

**Key findings (fANOVA importance):**

- **DDQN:** `target_update_freq` (48%) + `lr` (39%) dominate — lower target-sync lag and higher LR both help
- **PPO:** `clip_range` (35%) + `learning_rate` (24%) most sensitive — tighter clip (≈0.09) with low LR preferred
- **A2C:** `gae_lambda` (42%) + `n_steps` (36%) — high GAE lambda, short rollout window wins

---

## DDQN implementation notes

- **Architecture:** MLP 15 → 128 → 128 → 3 (ReLU), online + target networks
- **Target update:** hard sync every `target_update_freq` learner steps
- **Double DQN target:** `y = r + γ · Q_target(s', argmax_a Q_online(s', a))`
- **ε-decay:** linear (default) or exponential — configurable via `eps_decay_type`
- **Best HPO config:** lr=4.5e-3, γ=0.956, eps_decay_steps=60k, target_update=1700, batch=64, min_buffer=7500 → Sharpe **+3.64**

---

## Data & Split

**Active dataset:** `data/raw/eurusd_m1_2025_2026.csv` — 492,694 rows of 1-min EURUSD bars (2025-01-02 → 2026-04-30), exported from MetaTrader 5 via `scripts/download_mt5.py`.

```
Jan 2025 ──────────── Dec 2025 │ Jan 2026 ─ Feb 2026 │ Mar 2026 ─ Apr 2026
         TRAIN (12 mo)          │   VAL (2 mo)         │   TEST (2 mo)
         257 sessions           │   40 sessions         │   44 sessions
         learn weights          │   HP selection        │   final eval (never touched during training)
```

| Split | Period | Sessions | Rows |
|---|---|---|---|
| Train | Jan–Dec 2025 | 257 | 231,300 |
| Val | Jan–Feb 2026 | 40 | 36,000 |
| Test | Mar–Apr 2026 | 44 | 39,600 |

Timestamps are UTC; session masking converts to UTC+7 (Asia/Bangkok) for 09:00–00:00 window filtering only. See [reports/data_schema.md](reports/data_schema.md) for full schema and cost-unit derivation.

---

## Documents

| File | Content |
|---|---|
| [reports/implementation_plan.md](reports/implementation_plan.md) | Full architecture, env API, reward derivation, testing strategy |
| [reports/experimental_design.md](reports/experimental_design.md) | Experiment setup, metrics, reproducibility, limitations |
| [reports/data_schema.md](reports/data_schema.md) | Raw/processed data schema, timezone handling, cost units |
| [configs/env.yaml](configs/env.yaml) | Single source of truth for env / cost / session / portfolio |

---

## Disclaimer

This is a university course project. Do not use this code, model, or methodology for live trading. The authors accept no liability for losses incurred from doing so.
