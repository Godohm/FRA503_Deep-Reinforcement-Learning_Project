# Data Schema — EURUSD 1-Minute Bars

## 1. Source File

**Path:** `data/raw/eurusd_m1_2024.csv`
**Origin:** Exported via the MT5 Python API (`MetaTrader5.copy_rates_range`) using the script `Test1` / `test.txt` at the repo root.
**Size:** ~23 MB.
**Format:** CSV, comma-separated, ASCII, header on row 1.

### 1.1 Row counts (from raw file inspection)

| Metric | Value |
|---|---|
| Header rows | 1 |
| Data rows | 372,673 |
| Total lines | 372,674 |

### 1.2 Columns (raw)

| Column | Type | Unit | Notes |
|---|---|---|---|
| `time` | string → datetime | **UTC** (see §2) | `YYYY-MM-DD HH:MM:SS`, naive in the file |
| `open` | float | quote ccy (USD per EUR) | 5-digit broker |
| `high` | float | quote ccy | |
| `low` | float | quote ccy | |
| `close` | float | quote ccy | |
| `tick_volume` | int | tick count | not real volume |
| `spread` | int | **points** (1 point = `point_size` price units; `point_size = 1e-5`) | broker spread snapshot at bar close |
| `real_volume` | int | lots | typically 0 for retail Forex |

## 2. Timestamp & Timezone (CRITICAL)

`mt5.copy_rates_range` **always returns UTC** timestamps, even though the MT5 terminal UI displays broker time. The strings in the CSV are therefore **UTC, written as naive datetime**.

**Loader behaviour (`src/data/load_mt5.py`):**
1. Parse `time` with `pd.to_datetime(...)`.
2. Localise as UTC: `df.index = df.index.tz_localize('UTC')`.
3. Return a DataFrame with a UTC-aware `DatetimeIndex` as the canonical index.

**Session masking (`src/data/preprocess.py`):**
- The canonical index stays UTC throughout the pipeline.
- For session filtering only, we compute a UTC+7 view: `df.index.tz_convert('Asia/Bangkok')`.
- Session window defaults to **09:00–00:00 UTC+7** (15 hours; force-close before midnight UTC+7).
- `session_day` is assigned from the **UTC+7 calendar date** so a single session never spans two day labels.

### 2.1 Sanity-check timestamps (from current raw file)

| | UTC (as stored) | UTC+7 (display) |
|---|---|---|
| First bar | 2024-01-02 00:00:00 | 2024-01-02 07:00:00 |
| Last bar  | 2024-12-31 20:00:00 | 2024-12-31 27:00:00 (= 2025-01-01 03:00) |

These are recomputed and logged by `scripts/prepare_data.py` on every dataset build so the assumption stays auditable.

## 3. Processed Outputs

`scripts/prepare_data.py` writes:

| File | Index | Columns added on top of raw |
|---|---|---|
| `data/processed/train.parquet` | UTC `DatetimeIndex` | `session_day` (date, UTC+7), `bar_idx_in_day` (int) |
| `data/processed/test.parquet`  | UTC `DatetimeIndex` | same |
| `data/processed/metadata.json` | — | row count, date range (UTC + UTC+7), split config, env config fingerprint, scaler hash |

**Train / test split.** Chronological, by `session_day`:
- Train = months 1–11 of 2024 (Jan 2024 → Nov 2024).
- Test  = month 12 of 2024 (Dec 2024).
- Configurable via `configs/env.yaml` (`split.train_months`, `split.test_months`).

## 4. Cleaning Rules

1. **Dedup**: drop rows with duplicate timestamps (`keep='first'`).
2. **Session mask**: drop bars outside the 09:00–00:00 UTC+7 window (using the UTC+7 view).
3. **Re-index per day**: reindex each `session_day` to a complete 1-min grid.
4. **Gap fill**: forward-fill OHLC across gaps of ≤ 5 minutes. Spread is **not** ffilled — set to the previous valid value bounded by `spread_cap_quantile` (see env config). `tick_volume` set to 0 on filled bars.
5. **Drop bad days**: any `session_day` with > 30 missing bars (post-fill) is removed.

## 5. Cost Conversion Quick Reference

For 5-digit EURUSD with `point_size = 1e-5` and `unit_size = 100 000` (1 std lot):

| Quantity | Conversion |
|---|---|
| `spread = 10` points | = `10 × 1e-5` = 0.0001 in quote ccy = 1 pip |
| 1 pip on 1 lot | = `0.0001 × 100 000` = **$10** in account ccy |
| `spread_cost = spread_t × point_size × unit_size` | = account ccy cost for one full-side spread move on 1 lot |

Used identically in `src/envs/eurusd_intraday_env.py` (cost path) and `src/features/state_builder.py` index 5 (feature path, **without** `unit_size`).

## 6. Known Quirks

- **Negative-volume bars:** none observed in the 2024 file.
- **Spread spikes:** common during news (US CPI / NFP / FOMC). The first 60 bars after midnight UTC+7 (low liquidity) routinely show 70–90-point spreads (see header sample). The 0.99-quantile cap in `configs/env.yaml` dampens these.
- **Weekend gaps:** Friday 23:59 UTC → Sunday 22:00 UTC. After session masking, the data simply has no Saturday/Sunday rows; no special handling is needed because episodes are per-session_day.
- **Holiday-thin days:** US holidays (Independence Day, Thanksgiving, etc.) appear as low-volume, wide-spread bars but are not removed by default. Add to the `holidays` block in `env.yaml` if exclusion is desired in future experiments.
- **5-digit pricing assumption:** confirmed by inspection (`1.10418`, `1.03472`). If a different broker exports 4-digit prices, set `point_size = 1e-4` and `commission_price` accordingly.

## 7. Re-deriving the Dataset

```powershell
python scripts/prepare_data.py --config configs/env.yaml
```

This will: load CSV → tz_localize UTC → log first/last UTC and UTC+7 timestamps → session-mask → reindex → split → write parquet + `metadata.json`.
