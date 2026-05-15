# Bayesian HPO Report — Optuna TPE (2025–2026 Dataset)

## 1. What is Optuna?

**Optuna** is an open-source automatic hyperparameter optimisation (HPO) framework developed by Preferred Networks. Unlike grid search (exhaustive but exponential) or random search (simple but wasteful), Optuna builds a *probabilistic surrogate model* of the objective function and uses it to decide which hyperparameter configurations to evaluate next. This makes it far more sample-efficient: it finds good configurations in far fewer trials than grid or random search would need.

### 1.1 How TPE Works

The sampler used here is **TPE (Tree-structured Parzen Estimator)**, the default Optuna sampler and the same algorithm used in Hyperopt. It works as follows:

1. **Startup phase** (first `n_startup_trials` trials, here 10): sample configurations *uniformly at random* to warm up the model — no prior knowledge yet.
2. **Model phase** (from trial 11 onwards): split all past trials into two groups based on a quantile γ of the objective (Sharpe):
   - **l(x)** — the *good* distribution: density model fitted on the top      γ fraction of configurations (high Sharpe).
   - **g(x)** — the *bad* distribution: density model fitted on the bottom      1−γ fraction.
   TPE then proposes the next configuration by maximising the ratio **l(x) / g(x)** — i.e. it favours regions that are likely under the *good* distribution and unlikely under the *bad* one.
3. Each proposed configuration is then **evaluated** by training a fresh agent for 50 000 environment steps and measuring Sharpe on the *validation* split (Jan–Feb 2026). This Sharpe score is fed back to update l(x) and g(x).

The key advantage over random search is that TPE concentrates evaluations in promising regions, converging to better configurations with the same budget.

### 1.2 Study Setup

| Setting | Value |
| --- | --- |
| Sampler | TPE (Optuna 4.8.0) |
| Trials per study | 50 |
| Steps per trial | 50 000 |
| Startup (random) trials | 10 |
| Objective metric | Sharpe ratio (annualised, on *validation* split) |
| Seed | 42 |
| Train data | Jan–Dec 2025 (257 sessions) |
| Eval (HPO selection) | Jan–Feb 2026 *validation* split (40 sessions) |
| Test (held out) | Mar–Apr 2026 — never touched during HPO |
| Studies | DDQN-linear, DDQN-exponential, A2C, PPO |

> **Why separate validation?** The old 2-way split used the test set to select checkpoints during training, leaking information into HPO. The 3-way split fixes this: validation guides HPO, test is reserved for the final unbiased comparison reported in the paper.

## 2. How to Read the Artefacts

**Dose-response plot** (`<algo>_dose_response.png`) — for each sampled hyperparameter, a scatter of (parameter value, observed Sharpe) across all completed trials. A quadratic trend line is overlaid when ≥4 distinct values were sampled. Categorical parameters are shown as boxplots. The plot reveals *which parameter values correlate with higher Sharpe* and in which direction — useful for understanding the landscape the agent operates in.

**fANOVA importance** (`<algo>_importance.png`) — the Hutter/Hoos fANOVA algorithm partitions variance in the Sharpe scores attributable to each hyperparameter (and their interactions). A parameter with importance = 0.48 explains ~48 % of the variation in Sharpe across all 50 trials. Note: *importance ≠ direction* — a highly important parameter might be one where the wrong value hurts badly. Consult the dose-response plot for sign.

**History plot** (`<algo>_history.png`) — Sharpe vs trial number with the cumulative best-so-far overlaid. A rising best-so-far curve indicates TPE is successfully focusing on good regions. A flat curve after trial 10 suggests the search space is too wide or the objective is too noisy for TPE to learn from 50 trials.

## 3. HPO Results

The sections below cover each algorithm. The **Ablation** section (if present) compares variants of the same algorithm (e.g. linear vs exponential ε-decay for DDQN) trained under identical conditions.


## 3a. Ablation — DDQN variants

Each variant is a separate Optuna study over the *same search space* (see `scripts/optuna_search.py`) with one design knob held to a different value (encoded in the variant name). Best Sharpe is the single best trial across 50 random/TPE samples — read it as an *upper-bound* on what the design choice can deliver in 50k steps, not as a converged measurement.

| variant | best Sharpe | trial # | best total_return | n_trades | trials |
| --- | --- | --- | --- | --- | --- |
| `ddqn` | **2.542** | #52 | 0.0039 | 2 | 53 |
| `expdecay` | **0.000** | #19 | 0.0000 | 0 | 52 |

**Verdict:** `ddqn` wins the best-of-50 with Sharpe = 2.542, beating `expdecay` (Sharpe = 0.000) by Δ = +2.542.

### DDQN

- Trials completed: **53 / 54**
- Best Sharpe: **2.542** (trial #52)
- Best total_return: 0.0039
- Best MDD%: 0.0000
- Best n_trades: 2

### Best hyperparameters
- `lr`: **0.0047485256934171865**
- `gamma`: **0.9407006269765872**
- `eps_decay_steps`: **70000**
- `target_update_freq`: **2000**
- `batch_size`: **64**
- `min_buffer_to_learn`: **2000**

### Parameter importance (fANOVA)

| rank | param | importance | direction (linear slope, R²) |
| --- | --- | --- | --- |
| 1 | `eps_decay_steps` | 0.4473 | ↓ slope=-0.000, R²=0.12 |
| 2 | `gamma` | 0.2099 | ↓ slope=-13.470, R²=0.00 |
| 3 | `target_update_freq` | 0.1419 | ↓ slope=-0.006, R²=0.14 |
| 4 | `lr` | 0.1181 | ↑ slope=+14.270, R²=0.18 |
| 5 | `min_buffer_to_learn` | 0.0559 | ↑ slope=+0.000, R²=0.00 |
| 6 | `batch_size` | 0.0270 | ↓ slope=-0.143, R²=0.24 |

### Parameter glossary (what each knob does)
- **`lr`** — Adam step-size for the online Q-net. Too high → unstable bootstrapped TD targets; too low → never absorbs new transitions in the trial budget.
- **`gamma`** — Discount on future reward. Intraday bankroll resets each session, so the effective horizon is short — values near 0.95 are theoretically defensible.
- **`eps_decay_steps`** — Length (in agent action-steps) of linear ε decay 1.0 → 0.05. Short → exploit early (good if env signal is clear); long → keeps exploring.
- **`target_update_freq`** — Hard-sync interval (in learner steps) between online and target Q-net. Smaller → target tracks online closely (fast but less stable).
- **`batch_size`** — Replay minibatch size. Larger → smoother gradient, slower wall-time per step.
- **`min_buffer_to_learn`** — Wait this many transitions before the first gradient update. Higher = warmer cold-start, but burns exploration budget.

### Dose-response (linear) reading

For numeric params with R² ≥ 0.1, the direction below is informative; below that threshold the slope is dominated by noise.
- `batch_size`: Sharpe **trends decreasing** with the parameter (slope = -0.143, R² = 0.24).
- `lr` (log scale): Sharpe **trends increasing** with the parameter (slope = +14.270, R² = 0.18).
- `target_update_freq`: Sharpe **trends decreasing** with the parameter (slope = -0.006, R² = 0.14).
- `eps_decay_steps`: Sharpe **trends decreasing** with the parameter (slope = -0.000, R² = 0.12).

### Plots
- Optimisation history: `ddqn_history.png`
- Per-parameter dose-response: `ddqn_dose_response.png`
- Importance bars: `ddqn_importance.png`

### A2C

- Trials completed: **50 / 50**
- Best Sharpe: **-2.188** (trial #27)
- Best total_return: -0.1267
- Best MDD%: -0.1687
- Best n_trades: 123

### Best hyperparameters
- `learning_rate`: **0.00014937349991237846**
- `n_steps`: **30**
- `gamma`: **0.960544860724635**
- `ent_coef`: **0.00013822943038989413**
- `vf_coef`: **0.8213253187900392**
- `gae_lambda`: **0.9158805310594623**

### Parameter importance (fANOVA)

| rank | param | importance | direction (linear slope, R²) |
| --- | --- | --- | --- |
| 1 | `gae_lambda` | 0.2928 | ↓ slope=-106.868, R²=0.02 |
| 2 | `n_steps` | 0.2050 | ↑ slope=+0.017, R²=0.00 |
| 3 | `vf_coef` | 0.1670 | ↑ slope=+10.849, R²=0.01 |
| 4 | `gamma` | 0.1520 | ↑ slope=+18.060, R²=0.00 |
| 5 | `learning_rate` | 0.1146 | ↑ slope=+1.254, R²=0.00 |
| 6 | `ent_coef` | 0.0685 | ↑ slope=+3.691, R²=0.02 |

### Parameter glossary (what each knob does)
- **`learning_rate`** — Adam step-size shared by actor and critic. A2C is notoriously sensitive — too high can collapse the policy to a single action.
- **`n_steps`** — n-step return window per gradient update (rollout length). Small → frequent, noisy updates; large → smoother advantage but stale data.
- **`gamma`** — Discount factor. See DDQN row.
- **`ent_coef`** — Entropy bonus in the loss. Higher → forces stochastic policy (more exploration); zero → policy can collapse to deterministic early.
- **`vf_coef`** — Weight of the value-function loss vs the policy gradient loss.
- **`gae_lambda`** — Bias/variance trade-off in GAE. 1.0 = unbiased Monte-Carlo, 0.95 = standard smoothing.

### Dose-response (linear) reading

For numeric params with R² ≥ 0.1, the direction below is informative; below that threshold the slope is dominated by noise.
- `ent_coef` (log scale): Sharpe **trends increasing** with the parameter (slope = +3.691, R² = 0.02).
- `gae_lambda`: Sharpe **trends decreasing** with the parameter (slope = -106.868, R² = 0.02).

### Plots
- Optimisation history: `a2c_history.png`
- Per-parameter dose-response: `a2c_dose_response.png`
- Importance bars: `a2c_importance.png`

### PPO

- Trials completed: **50 / 50**
- Best Sharpe: **-0.483** (trial #30)
- Best total_return: -0.0405
- Best MDD%: -0.2434
- Best n_trades: 88

### Best hyperparameters
- `learning_rate`: **1.6599123590638213e-05**
- `n_steps`: **512**
- `batch_size`: **32**
- `n_epochs`: **16**
- `gamma`: **0.9705780547450751**
- `clip_range`: **0.13653497696379968**
- `ent_coef`: **0.004404197461177922**
- `vf_coef`: **0.7517324854742864**
- `gae_lambda`: **0.977676729539208**

### Parameter importance (fANOVA)

| rank | param | importance | direction (linear slope, R²) |
| --- | --- | --- | --- |
| 1 | `gae_lambda` | 0.2705 | ↑ slope=+78.832, R²=0.09 |
| 2 | `ent_coef` | 0.2594 | ↓ slope=-3.040, R²=0.07 |
| 3 | `batch_size` | 0.2395 | ↓ slope=-0.085, R²=0.25 |
| 4 | `vf_coef` | 0.0813 | ↓ slope=-8.375, R²=0.01 |
| 5 | `clip_range` | 0.0560 | ↓ slope=-32.797, R²=0.03 |
| 6 | `gamma` | 0.0351 | ↑ slope=+22.489, R²=0.00 |
| 7 | `n_epochs` | 0.0286 | ↑ slope=+0.338, R²=0.01 |
| 8 | `n_steps` | 0.0162 | ↓ slope=-0.001, R²=0.00 |
| 9 | `learning_rate` | 0.0135 | ↑ slope=+3.937, R²=0.03 |

### Parameter glossary (what each knob does)
- **`learning_rate`** — Adam step-size. PPO tolerates a moderate range; very high lr defeats the clip stabilisation.
- **`n_steps`** — Rollout buffer size per policy update. Small → many updates per total step budget; large → fewer but each update sees more data.
- **`batch_size`** — Minibatch within each rollout pass.
- **`n_epochs`** — Passes over each rollout. More epochs → re-use data hard (overfit risk); fewer → less sample-efficient.
- **`gamma`** — Discount factor. See DDQN row.
- **`clip_range`** — PPO importance-ratio clip ε. Tight (0.1) → conservative; loose (0.3) → faster but riskier policy moves.
- **`ent_coef`** — Entropy bonus, see A2C row.
- **`vf_coef`** — Value-loss weight, see A2C row.
- **`gae_lambda`** — GAE smoothing.

### Dose-response (linear) reading

For numeric params with R² ≥ 0.1, the direction below is informative; below that threshold the slope is dominated by noise.
- `batch_size`: Sharpe **trends decreasing** with the parameter (slope = -0.085, R² = 0.25).
- `gae_lambda`: Sharpe **trends increasing** with the parameter (slope = +78.832, R² = 0.09).
- `ent_coef` (log scale): Sharpe **trends decreasing** with the parameter (slope = -3.040, R² = 0.07).
- `clip_range`: Sharpe **trends decreasing** with the parameter (slope = -32.797, R² = 0.03).
- `learning_rate` (log scale): Sharpe **trends increasing** with the parameter (slope = +3.937, R² = 0.03).
- `vf_coef`: Sharpe **trends decreasing** with the parameter (slope = -8.375, R² = 0.01).

### Plots
- Optimisation history: `ppo_history.png`
- Per-parameter dose-response: `ppo_dose_response.png`
- Importance bars: `ppo_importance.png`

### DDQN — expdecay

- Trials completed: **52 / 53**
- Best Sharpe: **0.000** (trial #19)
- Best total_return: 0.0000
- Best MDD%: 0.0000
- Best n_trades: 0

### Best hyperparameters
- `lr`: **0.0036235551642241387**
- `gamma`: **0.9722336434303289**
- `eps_decay_steps`: **80000**
- `target_update_freq`: **4900**
- `batch_size`: **32**
- `min_buffer_to_learn`: **2500**

### Parameter importance (fANOVA)

| rank | param | importance | direction (linear slope, R²) |
| --- | --- | --- | --- |
| 1 | `eps_decay_steps` | 0.4157 | ↓ slope=-0.000, R²=0.00 |
| 2 | `lr` | 0.2651 | ↑ slope=+9.588, R²=0.32 |
| 3 | `batch_size` | 0.1263 | ↓ slope=-0.050, R²=0.17 |
| 4 | `gamma` | 0.1094 | ↑ slope=+20.263, R²=0.00 |
| 5 | `min_buffer_to_learn` | 0.0699 | ↓ slope=-0.001, R²=0.25 |
| 6 | `target_update_freq` | 0.0136 | ↑ slope=+0.000, R²=0.00 |

### Parameter glossary (what each knob does)
- **`lr`** — Adam step-size for the online Q-net. Too high → unstable bootstrapped TD targets; too low → never absorbs new transitions in the trial budget.
- **`gamma`** — Discount on future reward. Intraday bankroll resets each session, so the effective horizon is short — values near 0.95 are theoretically defensible.
- **`eps_decay_steps`** — Length (in agent action-steps) of linear ε decay 1.0 → 0.05. Short → exploit early (good if env signal is clear); long → keeps exploring.
- **`target_update_freq`** — Hard-sync interval (in learner steps) between online and target Q-net. Smaller → target tracks online closely (fast but less stable).
- **`batch_size`** — Replay minibatch size. Larger → smoother gradient, slower wall-time per step.
- **`min_buffer_to_learn`** — Wait this many transitions before the first gradient update. Higher = warmer cold-start, but burns exploration budget.

### Dose-response (linear) reading

For numeric params with R² ≥ 0.1, the direction below is informative; below that threshold the slope is dominated by noise.
- `lr` (log scale): Sharpe **trends increasing** with the parameter (slope = +9.588, R² = 0.32).
- `min_buffer_to_learn`: Sharpe **trends decreasing** with the parameter (slope = -0.001, R² = 0.25).
- `batch_size`: Sharpe **trends decreasing** with the parameter (slope = -0.050, R² = 0.17).

### Plots
- Optimisation history: `ddqn_expdecay_history.png`
- Per-parameter dose-response: `ddqn_expdecay_dose_response.png`
- Importance bars: `ddqn_expdecay_importance.png`