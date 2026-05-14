# M8 — Experiment 1 Comparison

Run prefix filter: `m8_full`

Rows: **3** (0 RL, 3 baseline)

## Ranked by Sharpe (higher is better)

| algorithm | seed | run_id | total_return_pct | sharpe | sortino | max_drawdown_pct | num_trades | win_rate | exposure_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:short_only |  | short_only | 0.6800 | 0.1868 | 0.2677 | -11.4953 | 40 | 0.4500 | 0.9988 |
| baseline:flat |  | flat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 |
| baseline:long_only |  | long_only | -14.9400 | -4.0905 | -4.7331 | -16.5014 | 40 | 0.4500 | 0.9988 |

## Ranked by total return (higher is better)

| algorithm | seed | run_id | total_return_pct | sharpe | sortino | max_drawdown_pct | num_trades | win_rate | exposure_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:short_only |  | short_only | 0.6800 | 0.1868 | 0.2677 | -11.4953 | 40 | 0.4500 | 0.9988 |
| baseline:flat |  | flat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 |
| baseline:long_only |  | long_only | -14.9400 | -4.0905 | -4.7331 | -16.5014 | 40 | 0.4500 | 0.9988 |

## Ranked by max drawdown (closer to 0 is better)

| algorithm | seed | run_id | total_return_pct | sharpe | sortino | max_drawdown_pct | num_trades | win_rate | exposure_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:flat |  | flat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 |
| baseline:short_only |  | short_only | 0.6800 | 0.1868 | 0.2677 | -11.4953 | 40 | 0.4500 | 0.9988 |
| baseline:long_only |  | long_only | -14.9400 | -4.0905 | -4.7331 | -16.5014 | 40 | 0.4500 | 0.9988 |
