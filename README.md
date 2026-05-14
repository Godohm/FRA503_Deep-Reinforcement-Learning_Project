# CFD Trading Agent Using Deep Reinforcement Learning for EURUSD Intraday Trading

Academic / research project. **Simulation & backtesting only — no live trading.** The original course material used XAUUSD / gold; this implementation uses **EURUSD CFD** consistently.

## What this project does

Builds a Gymnasium-compatible intraday trading environment for EURUSD on 1-minute MT5 historical data, then trains and compares three DRL agents under identical conditions:

1. **Double DQN** (custom PyTorch)
2. **A2C** (Stable-Baselines3)
3. **PPO** (Stable-Baselines3)

Performance is evaluated with profitability and risk-adjusted metrics (Sharpe, Sortino, max drawdown, win rate, etc.) and compared against long-only / short-only / flat baselines.

## Status

| Milestone | State |
|---|---|
| M0 — Planning docs | ✅ Done |
| M1 — Scaffolding | pending sign-off |
| M2 — Data pipeline | pending |
| M3 — Features + tests | pending |
| M4 — Trading env + tests | pending |
| M5 — Baselines + metrics | pending |
| M6 — Double DQN | pending |
| M7 — A2C + PPO | pending |
| M8 — Experiment 1 | pending |
| M9 — Reports | pending |
| M10 — (Optional) Exp 2 | deferred |

See [TODO.md](TODO.md).

## Documents

- [reports/implementation_plan.md](reports/implementation_plan.md) — architecture, env, reward, training, testing strategy
- [reports/experimental_design.md](reports/experimental_design.md) — experiments, metrics, reproducibility, limitations
- [reports/data_schema.md](reports/data_schema.md) — raw / processed data shape, timezone handling, cost units
- [configs/env.yaml](configs/env.yaml) — single source of truth for env / cost / session / split

## Key decisions

| Decision | Choice |
|---|---|
| Instrument | EURUSD CFD (5-digit) |
| Bar interval | 1 minute |
| Session window | 09:00–00:00 UTC+7 (force-close before midnight UTC+7) |
| State | 15-D (5 returns, spread, MACD, Stoch, RSI, ATR, TL, POS, PR, DR, HT) |
| Action | Discrete target position ∈ {−1, 0, +1} |
| Reward | `net_pnl / initial_equity` (dimensionless fractional return) |
| Execution | `current_close` (force-close always `current_close`) |
| Spread | broker `spread` column + configurable commission |
| Train / test split | 11 months / 1 month (chronological, Dec 2024 = test) |
| Double DQN | Custom PyTorch (SB3 doesn't ship DDQN) |
| Logging | TensorBoard + CSV |

## Quick start (after M1+ is implemented)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/prepare_data.py --config configs/env.yaml
pytest tests/ -q

python scripts/train_all.py --steps 5000 --seed 42      # smoke
python scripts/run_experiment_1.py --seeds 42 123 2024   # full
python scripts/run_baselines.py
python scripts/evaluate_all.py
```

## Data

- `data/raw/eurusd_m1_2024.csv` — 372,673 1-min bars (2024-01-02 → 2024-12-31, UTC), exported via `Test1` / `test.txt` (MT5 Python API).
- See [reports/data_schema.md](reports/data_schema.md) for the full schema and timezone notes.

## License / intended use

This is a course / research project. Do not use any of this code, model, or methodology for live trading without your own due diligence. The authors accept no liability for losses incurred from doing so.
