# HPO short-sweep report — DDQN / A2C / PPO

Generated from `results/hpo/hpo_results.csv` (14 runs across 3 algorithms).

**Design:** one-at-a-time (OAT) variation around a per-algo baseline, 20k env steps per run, seed=42, eval every 5k steps. Each variation flips ONE parameter from the baseline, so each row's Δ Sharpe is attributable to that single change.

**Caveats:**
- 20k steps is a short horizon — agents are NOT converged; rankings are directional.
- Single seed (42) → seed-induced variance is not measured here. The M8 full run   (200k steps × 3 seeds) is the source of statistical claims.
- Test split is Dec 2024 only (20 sessions). Test-Sharpe is noisy on small windows.


## DDQN

### Parameters explained

- **`lr`** — Adam learning rate for the online Q-network. High lr → faster learning but unstable bootstrapping; low lr → more stable but slow to absorb new transitions.
- **`eps_decay_steps`** — Length (in action-steps) of the linear ε decay from 1.0 → 0.05. Short decay → exploit sooner (good if env signal is clear); long decay → keep exploring (avoids early policy collapse).
- **`target_update_freq`** — Hard-sync interval (in learner updates) between online and target Q-net. Smaller → target tracks online closely, faster TD propagation, but loses the stabilising lag; larger → slower convergence, more stable.
- **`gamma`** — Discount factor for future reward. Intraday sessions reset bankroll each day → 0.95 is defensible (shorter horizon); 0.99 weights end-of-day reward roughly the same as next-bar reward.

### Variation results (vs baseline)

| variant | param changed | value | Sharpe | Δ Sharpe | total_return | Δ return | MDD% | Δ MDD | trades | Δ trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | (baseline) | nan | -44.2929 | 0.0000 | -2.9500 | 0.0000 | -3.2319 | 0.0000 | 1503 | 0 |
| lr_low | lr | 0.0003 | -23.9194 | +20.3734 | -1.3096 | +1.6404 | -1.3752 | +1.8567 | 719 | -784 |
| eps_decay_short | eps_decay_steps | 20000.0 | -70.9748 | -26.6819 | -7.8321 | -4.8821 | -12.2407 | -9.0088 | 3321 | ++1818 |
| target_sync_fast | target_update_freq | 500.0 | -29.1716 | +15.1213 | -2.6562 | +0.2938 | -3.1582 | +0.0737 | 1550 | ++47 |
| gamma_low | gamma | 0.95 | -6.5902 | +37.7027 | -0.2769 | +2.6731 | -0.3292 | +2.9027 | 188 | -1315 |

### Sensitivity ranking (by |Δ Sharpe| vs baseline)

| rank | param | value | Δ Sharpe | Δ return | direction |
| --- | --- | --- | --- | --- | --- |
| 1 | gamma | 0.95 | +37.7027 | +2.6731 | ↑ helped |
| 2 | eps_decay_steps | 20000.0 | -26.6819 | -4.8821 | ↓ hurt |
| 3 | lr | 0.0003 | +20.3734 | +1.6404 | ↑ helped |
| 4 | target_update_freq | 500.0 | +15.1213 | +0.2938 | ↑ helped |

### Discussion

- Most impactful single change in this sweep: **`gamma` → 0.95** (Δ Sharpe = +37.7027). 
- Improved over baseline: `gamma=0.95`, `lr=0.0003`, `target_update_freq=500.0`.
- Worse than baseline: `eps_decay_steps=20000.0`.
- **Recommendation:** apply `gamma = 0.95` for the M8 full run.
- *Caveat:* 20k steps with single seed=42; signal is directional, not statistically robust. Validate the recommended config across the M8 seed sweep before trusting any ranking.


## A2C

### Parameters explained

- **`n_steps`** — Number of env steps collected per gradient update (n-step return window). Small → frequent updates, high variance; large → smoother advantage estimates but stale on-policy data.
- **`learning_rate`** — Adam learning rate for both actor and critic. A2C is sensitive — too high can collapse the policy to a single action.
- **`ent_coef`** — Entropy bonus weight in the loss. Higher → forces stochastic policy (more exploration); zero → policy can become deterministic early.

### Variation results (vs baseline)

| variant | param changed | value | Sharpe | Δ Sharpe | total_return | Δ return | MDD% | Δ MDD | trades | Δ trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | (baseline) | nan | -23.0171 | 0.0000 | -1.2063 | 0.0000 | -1.2299 | 0.0000 | 503 | 0 |
| n_steps_long | n_steps | 20.0 | -103.4767 | -80.4596 | -2.5018 | -1.2955 | -2.7161 | -1.4862 | 809 | ++306 |
| ent_coef_on | ent_coef | 0.01 | -79.0845 | -56.0673 | -4.4420 | -3.2357 | -5.1823 | -3.9523 | 2389 | ++1886 |
| lr_low | learning_rate | 0.0003 | -74.9572 | -51.9401 | -3.8439 | -2.6376 | -5.0419 | -3.8120 | 2081 | ++1578 |

### Sensitivity ranking (by |Δ Sharpe| vs baseline)

| rank | param | value | Δ Sharpe | Δ return | direction |
| --- | --- | --- | --- | --- | --- |
| 1 | n_steps | 20.0 | -80.4596 | -1.2955 | ↓ hurt |
| 2 | ent_coef | 0.01 | -56.0673 | -3.2357 | ↓ hurt |
| 3 | learning_rate | 0.0003 | -51.9401 | -2.6376 | ↓ hurt |

### Discussion

- Most impactful single change in this sweep: **`n_steps` → 20.0** (Δ Sharpe = -80.4596). 
- Worse than baseline: `n_steps=20.0`, `ent_coef=0.01`, `learning_rate=0.0003`.
- **Recommendation:** keep the baseline — no variation improved Sharpe in this short sweep.
- *Caveat:* 20k steps with single seed=42; signal is directional, not statistically robust. Validate the recommended config across the M8 seed sweep before trusting any ranking.


## PPO

### Parameters explained

- **`n_steps`** — Rollout buffer size per policy update. Small → many updates with stale-but-fresh data; large → fewer updates per total budget but each update sees a more representative batch.
- **`n_epochs`** — Number of passes over each rollout. More epochs → re-use data harder, risk overfitting to a single rollout; fewer → less sample-efficient.
- **`clip_range`** — PPO importance-ratio clip. Tighter (e.g. 0.1) → more conservative updates, less policy drift; looser (0.3) → faster but riskier policy moves.
- **`learning_rate`** — Adam learning rate. PPO tolerates a moderate range; very high lr defeats the clip stabilisation.

### Variation results (vs baseline)

| variant | param changed | value | Sharpe | Δ Sharpe | total_return | Δ return | MDD% | Δ MDD | trades | Δ trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | (baseline) | nan | -35.5265 | 0.0000 | -1.6513 | 0.0000 | -1.7714 | 0.0000 | 915 | 0 |
| n_steps_short | n_steps | 512.0 | -51.8472 | -16.3206 | -2.7550 | -1.1037 | -3.1788 | -1.4074 | 1368 | ++453 |
| n_epochs_low | n_epochs | 5.0 | -26.7461 | +8.7804 | -1.0776 | +0.5737 | -1.0870 | +0.6844 | 627 | -288 |
| clip_tight | clip_range | 0.1 | -34.0868 | +1.4398 | -1.5612 | +0.0901 | -1.6392 | +0.1322 | 768 | -147 |
| lr_low | learning_rate | 0.0001 | -33.6188 | +1.9077 | -2.8592 | -1.2079 | -3.2574 | -1.4860 | 1568 | ++653 |

### Sensitivity ranking (by |Δ Sharpe| vs baseline)

| rank | param | value | Δ Sharpe | Δ return | direction |
| --- | --- | --- | --- | --- | --- |
| 1 | n_steps | 512.0 | -16.3206 | -1.1037 | ↓ hurt |
| 2 | n_epochs | 5.0 | +8.7804 | +0.5737 | ↑ helped |
| 3 | learning_rate | 0.0001 | +1.9077 | -1.2079 | ↑ helped |
| 4 | clip_range | 0.1 | +1.4398 | +0.0901 | ↑ helped |

### Discussion

- Most impactful single change in this sweep: **`n_steps` → 512.0** (Δ Sharpe = -16.3206). 
- Improved over baseline: `n_epochs=5.0`, `learning_rate=0.0001`, `clip_range=0.1`.
- Worse than baseline: `n_steps=512.0`.
- **Recommendation:** apply `n_epochs = 5.0` for the M8 full run.
- *Caveat:* 20k steps with single seed=42; signal is directional, not statistically robust. Validate the recommended config across the M8 seed sweep before trusting any ranking.


---
## Suggested M8 overrides

Per-algo, apply these one-line overrides to the relevant config before running `scripts/run_experiment_1.py`:

- **DDQN** (`configs/dqn.yaml`): set under the `dqn:` section: `{'gamma': 0.95}`
- **PPO** (`configs/ppo.yaml`): set under the `ppo:` section: `{'n_epochs': 5}`