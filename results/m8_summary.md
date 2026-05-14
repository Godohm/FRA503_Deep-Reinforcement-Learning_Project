# M8 — Experiment 1: Algorithm Comparison (Full Run)

**Setup:** 200,000 env steps × 3 seeds {42, 123, 2024} × 3 algorithms (DDQN, A2C, PPO).  
**Test set:** Dec 2024 (1 month, 20 trading sessions). Evaluation is deterministic (greedy policy).  
**Baselines:** long-only, short-only, flat — same env and cost model, hard-coded policy.

---

## Cross-seed summary (mean ± std over seeds 42 / 123 / 2024)

| algorithm | mean Sharpe | std Sharpe | mean return % | std return % |
|---|---|---|---|---|
| **DDQN** | **+0.426** | ±1.289 | **+1.11%** | ±4.63% |
| A2C | -0.399 | ±0.704 | -1.57% | ±2.76% |
| PPO | -7.642 | ±0.928 | -29.05% | ±6.82% |
| baseline: short-only | +0.187 | — | +0.68% | — |
| baseline: flat | 0.000 | — | 0.00% | — |
| baseline: long-only | -4.091 | — | -14.94% | — |

**Winner: DDQN** — only algorithm with positive mean Sharpe and positive mean return across all 3 seeds. High std reflects seed sensitivity (seed 42 was the strong performer).

---

## Per-seed detail

### DDQN

| seed | Sharpe | total return | MDD % | trades | win rate |
|---|---|---|---|---|---|
| 42 | **+1.427** | +4.41% | -5.99% | 42 | 50.0% |
| 123 | -1.029 | -4.19% | -17.63% | 54 | 48.1% |
| 2024 | +0.881 | +3.10% | -7.06% | 120 | 52.4% |
| **mean** | **+0.426** | **+1.11%** | | | |

### A2C

| seed | Sharpe | total return | MDD % | trades | win rate |
|---|---|---|---|---|---|
| 42 | -1.211 | -4.75% | -11.27% | 72 | 53.7% |
| 123 | -0.022 | -0.08% | -11.50% | 42 | 42.9% |
| 2024 | +0.037 | +0.13% | -10.05% | 62 | 45.2% |
| **mean** | **-0.399** | **-1.57%** | | | |

### PPO

| seed | Sharpe | total return | MDD % | trades | win rate |
|---|---|---|---|---|---|
| 42 | -7.429 | -27.01% | -31.58% | 250 | 29.7% |
| 123 | -8.658 | -36.65% | -42.17% | 193 | 33.0% |
| 2024 | -6.839 | -23.48% | -29.44% | 72 | 40.0% |
| **mean** | **-7.642** | **-29.05%** | | | |

---

## Key observations

- **DDQN** is the only algorithm that consistently outperforms the short-only baseline (Sharpe +0.187) in mean, and beats all baselines including flat. High variance across seeds suggests the policy found useful signal but is sensitive to initialisation.
- **A2C** converges near breakeven — small losses, low drawdown (~11%), consistent trade counts. May benefit from longer training or further HPO tuning.
- **PPO** shows the worst performance with excessive overtrading (250 trades/month at seed 42) and large drawdowns (~30–42%). The policy likely did not converge in 200k steps without HPO-tuned hyperparameters.
- **Short-only baseline** (Sharpe +0.187) is a surprisingly competitive benchmark — Dec 2024 EURUSD was a bearish month, rewarding short positions without any learning.
- **Long-only** is clearly the worst strategy for this test period (Sharpe -4.09, -14.94% return).

---

## HPO context

Bayesian HPO (Optuna TPE, 50 trials × 50k steps) produced:

| algorithm | HPO best Sharpe | key finding |
|---|---|---|
| DDQN | +3.64 | `target_update_freq` (48%) + `lr` (39%) most important |
| PPO | +0.39 | `clip_range` (35%) + `learning_rate` (24%) most important |
| A2C | +0.12 | `gae_lambda` (42%) + `n_steps` (36%) most important |

The M8 full runs above use **baseline hyperparameters** (not HPO-tuned). Applying the Optuna best configs for DDQN (lr=4.5e-3, target_update=1700, batch=64, min_buffer=7500) to the full 200k × seed sweep is the recommended next step.

Full HPO report: [`results/optuna/optuna_report.md`](optuna/optuna_report.md)  
OAT sensitivity report: [`results/hpo/hpo_report.md`](hpo/hpo_report.md)
