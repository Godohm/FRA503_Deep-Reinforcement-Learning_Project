# Bayesian HPO Report (Optuna TPE)

Per-algorithm short-horizon search using TPE sampler over the search spaces declared in `scripts/optuna_search.py`. Each trial trains a fresh agent at the proposed hyperparameters, evaluates on the test split, and returns Sharpe as the maximisation objective.

**Reading the artefacts:**
- *Dose-response plot* — for each parameter, scatter of (parameter value, observed Sharpe) over all completed trials. A quadratic trend line is overlaid when there are ≥4 distinct sampled values. Categorical params are shown as boxplots per category.
- *fANOVA importance* — fraction of the variance in Sharpe attributable to each parameter via the Hutter/Hoos fANOVA. Importance ≠ direction; consult the dose-response plot for sign.
- *History plot* — Sharpe vs trial number, with the cumulative max (best-so-far) overlaid.

**Caveats:**
- This is a *short-horizon* search (steps per trial < 50k typically). Rankings reflect early-training behaviour, not asymptotic performance.
- Single seed per trial → seed-induced variance is folded into the objective noise. Cross-seed validation is the M8 full run's job.


## Ablation — DDQN variants

Each variant is a separate Optuna study over the *same search space* (see `scripts/optuna_search.py`) with one design knob held to a different value (encoded in the variant name). Best Sharpe is the single best trial across 50 random/TPE samples — read it as an *upper-bound* on what the design choice can deliver in 50k steps, not as a converged measurement.

| variant | best Sharpe | trial # | best total_return | n_trades | trials |
| --- | --- | --- | --- | --- | --- |
| `ddqn` | **3.642** | #11 | 0.0095 | 4 | 50 |
| `expdecay` | **0.513** | #44 | 0.0187 | 44 | 50 |

**Verdict:** `ddqn` wins the best-of-50 with Sharpe = 3.642, beating `expdecay` (Sharpe = 0.513) by Δ = +3.128.

## DDQN

- Trials completed: **50 / 50**
- Best Sharpe: **3.642** (trial #11)
- Best total_return: 0.0095
- Best MDD%: 0.0000
- Best n_trades: 4

### Best hyperparameters
- `lr`: **0.004491635087462903**
- `gamma`: **0.9562436584229151**
- `eps_decay_steps`: **60000**
- `target_update_freq`: **1700**
- `batch_size`: **64**
- `min_buffer_to_learn`: **7500**

### Parameter importance (fANOVA)

| rank | param | importance | direction (linear slope, R²) |
| --- | --- | --- | --- |
| 1 | `target_update_freq` | 0.5497 | ↓ slope=-0.013, R²=0.24 |
| 2 | `lr` | 0.3623 | ↑ slope=+38.950, R²=0.72 |
| 3 | `gamma` | 0.0346 | ↓ slope=-118.585, R²=0.01 |
| 4 | `eps_decay_steps` | 0.0252 | ↓ slope=-0.000, R²=0.17 |
| 5 | `batch_size` | 0.0150 | ↓ slope=-0.132, R²=0.12 |
| 6 | `min_buffer_to_learn` | 0.0133 | ↓ slope=-0.002, R²=0.11 |

### Parameter glossary (what each knob does)
- **`lr`** — Adam step-size for the online Q-net. Too high → unstable bootstrapped TD targets; too low → never absorbs new transitions in the trial budget.
- **`gamma`** — Discount on future reward. Intraday bankroll resets each session, so the effective horizon is short — values near 0.95 are theoretically defensible.
- **`eps_decay_steps`** — Length (in agent action-steps) of linear ε decay 1.0 → 0.05. Short → exploit early (good if env signal is clear); long → keeps exploring.
- **`target_update_freq`** — Hard-sync interval (in learner steps) between online and target Q-net. Smaller → target tracks online closely (fast but less stable).
- **`batch_size`** — Replay minibatch size. Larger → smoother gradient, slower wall-time per step.
- **`min_buffer_to_learn`** — Wait this many transitions before the first gradient update. Higher = warmer cold-start, but burns exploration budget.

### Dose-response (linear) reading

For numeric params with R² ≥ 0.1, the direction below is informative; below that threshold the slope is dominated by noise.
- `lr` (log scale): Sharpe **trends increasing** with the parameter (slope = +38.950, R² = 0.72).
- `target_update_freq`: Sharpe **trends decreasing** with the parameter (slope = -0.013, R² = 0.24).
- `eps_decay_steps`: Sharpe **trends decreasing** with the parameter (slope = -0.000, R² = 0.17).
- `batch_size`: Sharpe **trends decreasing** with the parameter (slope = -0.132, R² = 0.12).
- `min_buffer_to_learn`: Sharpe **trends decreasing** with the parameter (slope = -0.002, R² = 0.11).

### Plots
- Optimisation history: `ddqn_history.png`
- Per-parameter dose-response: `ddqn_dose_response.png`
- Importance bars: `ddqn_importance.png`

## A2C

- Trials completed: **50 / 50**
- Best Sharpe: **0.124** (trial #48)
- Best total_return: 0.0045
- Best MDD%: -0.1150
- Best n_trades: 42

### Best hyperparameters
- `learning_rate`: **0.003781157394484058**
- `n_steps`: **5**
- `gamma`: **0.9660241943754099**
- `ent_coef`: **1.4447435100735788e-05**
- `vf_coef`: **0.6744294188400484**
- `gae_lambda`: **0.9163773620872995**

### Parameter importance (fANOVA)

| rank | param | importance | direction (linear slope, R²) |
| --- | --- | --- | --- |
| 1 | `n_steps` | 0.6367 | ↓ slope=-0.632, R²=0.20 |
| 2 | `gae_lambda` | 0.1364 | ↓ slope=-63.053, R²=0.00 |
| 3 | `learning_rate` | 0.1179 | ↑ slope=+0.154, R²=0.00 |
| 4 | `vf_coef` | 0.0553 | ↑ slope=+1.262, R²=0.00 |
| 5 | `gamma` | 0.0493 | ↑ slope=+239.524, R²=0.02 |
| 6 | `ent_coef` | 0.0045 | ↑ slope=+2.579, R²=0.01 |

### Parameter glossary (what each knob does)
- **`learning_rate`** — Adam step-size shared by actor and critic. A2C is notoriously sensitive — too high can collapse the policy to a single action.
- **`n_steps`** — n-step return window per gradient update (rollout length). Small → frequent, noisy updates; large → smoother advantage but stale data.
- **`gamma`** — Discount factor. See DDQN row.
- **`ent_coef`** — Entropy bonus in the loss. Higher → forces stochastic policy (more exploration); zero → policy can collapse to deterministic early.
- **`vf_coef`** — Weight of the value-function loss vs the policy gradient loss.
- **`gae_lambda`** — Bias/variance trade-off in GAE. 1.0 = unbiased Monte-Carlo, 0.95 = standard smoothing.

### Dose-response (linear) reading

For numeric params with R² ≥ 0.1, the direction below is informative; below that threshold the slope is dominated by noise.
- `n_steps`: Sharpe **trends decreasing** with the parameter (slope = -0.632, R² = 0.20).
- `gamma`: Sharpe **trends increasing** with the parameter (slope = +239.524, R² = 0.02).

### Plots
- Optimisation history: `a2c_history.png`
- Per-parameter dose-response: `a2c_dose_response.png`
- Importance bars: `a2c_importance.png`

## PPO

- Trials completed: **50 / 50**
- Best Sharpe: **0.385** (trial #17)
- Best total_return: 0.0126
- Best MDD%: -0.1343
- Best n_trades: 68

### Best hyperparameters
- `learning_rate`: **7.598095692444107e-05**
- `n_steps`: **512**
- `batch_size`: **128**
- `n_epochs`: **16**
- `gamma`: **0.9428021245649091**
- `clip_range`: **0.09436252009903716**
- `ent_coef`: **0.0054965248586139985**
- `vf_coef`: **0.8212638525264315**
- `gae_lambda`: **0.9707601740889749**

### Parameter importance (fANOVA)

| rank | param | importance | direction (linear slope, R²) |
| --- | --- | --- | --- |
| 1 | `learning_rate` | 0.2975 | ↓ slope=-4.454, R²=0.02 |
| 2 | `ent_coef` | 0.2396 | ↓ slope=-1.769, R²=0.01 |
| 3 | `clip_range` | 0.2367 | ↓ slope=-58.513, R²=0.05 |
| 4 | `gae_lambda` | 0.0829 | ↑ slope=+41.170, R²=0.01 |
| 5 | `batch_size` | 0.0417 | ↓ slope=-0.020, R²=0.00 |
| 6 | `n_steps` | 0.0311 | ↓ slope=-0.001, R²=0.00 |
| 7 | `gamma` | 0.0307 | ↓ slope=-157.762, R²=0.03 |
| 8 | `n_epochs` | 0.0290 | ↑ slope=+0.242, R²=0.00 |
| 9 | `vf_coef` | 0.0109 | ↑ slope=+9.837, R²=0.01 |

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
- `clip_range`: Sharpe **trends decreasing** with the parameter (slope = -58.513, R² = 0.05).
- `gamma`: Sharpe **trends decreasing** with the parameter (slope = -157.762, R² = 0.03).
- `learning_rate` (log scale): Sharpe **trends decreasing** with the parameter (slope = -4.454, R² = 0.02).
- `ent_coef` (log scale): Sharpe **trends decreasing** with the parameter (slope = -1.769, R² = 0.01).
- `gae_lambda`: Sharpe **trends increasing** with the parameter (slope = +41.170, R² = 0.01).

### Plots
- Optimisation history: `ppo_history.png`
- Per-parameter dose-response: `ppo_dose_response.png`
- Importance bars: `ppo_importance.png`

## DDQN — expdecay

- Trials completed: **50 / 50**
- Best Sharpe: **0.513** (trial #44)
- Best total_return: 0.0187
- Best MDD%: -0.1043
- Best n_trades: 44

### Best hyperparameters
- `lr`: **0.003481105232809787**
- `gamma`: **0.900540398962354**
- `eps_decay_steps`: **60000**
- `target_update_freq`: **4200**
- `batch_size`: **32**
- `min_buffer_to_learn`: **10500**

### Parameter importance (fANOVA)

| rank | param | importance | direction (linear slope, R²) |
| --- | --- | --- | --- |
| 1 | `lr` | 0.4874 | ↑ slope=+48.087, R²=0.60 |
| 2 | `gamma` | 0.2902 | ↓ slope=-305.224, R²=0.04 |
| 3 | `eps_decay_steps` | 0.1135 | ↓ slope=-0.000, R²=0.08 |
| 4 | `target_update_freq` | 0.0661 | ↓ slope=-0.003, R²=0.01 |
| 5 | `batch_size` | 0.0242 | ↓ slope=-0.078, R²=0.02 |
| 6 | `min_buffer_to_learn` | 0.0187 | ↓ slope=-0.001, R²=0.01 |

### Parameter glossary (what each knob does)
- **`lr`** — Adam step-size for the online Q-net. Too high → unstable bootstrapped TD targets; too low → never absorbs new transitions in the trial budget.
- **`gamma`** — Discount on future reward. Intraday bankroll resets each session, so the effective horizon is short — values near 0.95 are theoretically defensible.
- **`eps_decay_steps`** — Length (in agent action-steps) of linear ε decay 1.0 → 0.05. Short → exploit early (good if env signal is clear); long → keeps exploring.
- **`target_update_freq`** — Hard-sync interval (in learner steps) between online and target Q-net. Smaller → target tracks online closely (fast but less stable).
- **`batch_size`** — Replay minibatch size. Larger → smoother gradient, slower wall-time per step.
- **`min_buffer_to_learn`** — Wait this many transitions before the first gradient update. Higher = warmer cold-start, but burns exploration budget.

### Dose-response (linear) reading

For numeric params with R² ≥ 0.1, the direction below is informative; below that threshold the slope is dominated by noise.
- `lr` (log scale): Sharpe **trends increasing** with the parameter (slope = +48.087, R² = 0.60).
- `eps_decay_steps`: Sharpe **trends decreasing** with the parameter (slope = -0.000, R² = 0.08).
- `gamma`: Sharpe **trends decreasing** with the parameter (slope = -305.224, R² = 0.04).
- `batch_size`: Sharpe **trends decreasing** with the parameter (slope = -0.078, R² = 0.02).

### Plots
- Optimisation history: `ddqn_expdecay_history.png`
- Per-parameter dose-response: `ddqn_expdecay_dose_response.png`
- Importance bars: `ddqn_expdecay_importance.png`