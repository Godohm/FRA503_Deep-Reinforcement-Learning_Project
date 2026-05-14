# Data Schema — EURUSD 1-Minute Bars

## 1. Source Files

| File | Period | Rows | Size | Purpose |
|---|---|---|---|---|
| `data/raw/eurusd_m1_2024.csv` | 2024-01-02 → 2024-12-31 | 372,673 | ~23 MB | Original experiment (archived) |
| `data/raw/eurusd_m1_2025_2026.csv` | 2025-01-02 → 2026-04-30 | 492,694 | ~30 MB | **Active dataset** |

**Origin:** Exported via the MT5 Python API (`MetaTrader5.copy_rates_range`) using the script `Test1` / `test.txt` at the repo root. New data downloaded with `scripts/download_mt5.py`.

### 1.1 Columns (raw)

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

---

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

### 2.1 Sanity-check timestamps (active dataset)

| | UTC (as stored) | UTC+7 (display) |
|---|---|---|
| First bar | 2025-01-02 08:00:00+00:00 | 2025-01-02 15:00:00+07:00 |
| Last bar  | 2026-05-01 00:00:00+00:00 | 2026-05-01 07:00:00+07:00 |

These are recomputed and logged by `scripts/prepare_data.py` on every dataset build so the assumption stays auditable.

---

## 3. Train / Val / Test Split

```
Raw data: Jan 2025 ─────────────────────────── Apr 2026
          │                                         │
          ├── TRAIN (Jan–Dec 2025) ─────────────────┤ 12 months │ 257 sessions │ 231,300 rows
          │                                         │
          ├── VAL   (Jan–Feb 2026) ─────────────────┤  2 months │  40 sessions │  36,000 rows
          │                                         │
          └── TEST  (Mar–Apr 2026) ─────────────────┘  2 months │  44 sessions │  39,600 rows
```

| Split | Months | Sessions | Rows | Date range (UTC+7) | Purpose |
|---|---|---|---|---|---|
| **Train** | Jan–Dec 2025 | 257 | 231,300 | 2025-01-03 → 2025-12-31 | Learn weights |
| **Val** | Jan–Feb 2026 | 40 | 36,000 | 2026-01-05 → 2026-02-27 | HP selection, early stopping |
| **Test** | Mar–Apr 2026 | 44 | 39,600 | 2026-03-02 → 2026-04-30 | Final unbiased evaluation |

**Why 3-way split?**
The original 2-way split (train / test only) used the test set during training for checkpoint selection — technically leaking signal into HP tuning. The 3-way split fixes this: val is used during training (eval every N steps), test is **never touched** until the final reporting run.

**Why Mar–Apr 2026 for test?**
- Covers end-of-Q1 rebalancing (March), Easter holiday thin-market period, and early Q2.
- Spring quarter is generally higher liquidity than summer (Jun–Aug) — a more challenging and representative test window than low-liquidity summer.

**Configurable** via `configs/env.yaml` (`split.train_months`, `split.val_months`, `split.test_months`). Set `val_months: 0` to revert to the old 2-way behaviour.

---

## 4. Processed Outputs

`scripts/prepare_data.py` writes:

| File | Index | Columns added on top of raw |
|---|---|---|
| `data/processed/train.parquet` | UTC `DatetimeIndex` | `session_day` (date, UTC+7), `bar_idx_in_day` (int) |
| `data/processed/val.parquet`   | UTC `DatetimeIndex` | same |
| `data/processed/test.parquet`  | UTC `DatetimeIndex` | same |
| `data/processed/metadata.json` | — | row counts, date ranges (UTC + UTC+7), split config, dropped days |

---

## 5. Cleaning Rules

1. **Dedup**: drop rows with duplicate timestamps (`keep='first'`).
2. **Session mask**: drop bars outside the 09:00–00:00 UTC+7 window.
3. **Re-index per day**: reindex each `session_day` to a complete 1-min grid.
4. **Gap fill**: forward-fill OHLC across gaps of ≤ 5 minutes. Spread not ffilled. `tick_volume` = 0 on filled bars.
5. **Drop bad days**: any `session_day` with > 30 missing bars (post-fill) is removed.

**Dropped sessions in active dataset:** 2025-01-02 (New Year), 2025-12-26 (Christmas), 2026-01-02 (New Year) — all have 360 missing bars (full holiday close).

---

## 6. Cost Conversion Quick Reference

For 5-digit EURUSD with `point_size = 1e-5` and `unit_size = 100 000` (1 std lot):

| Quantity | Conversion |
|---|---|
| `spread = 10` points | = `10 × 1e-5` = 0.0001 in quote ccy = 1 pip |
| 1 pip on 1 lot | = `0.0001 × 100 000` = **$10** in account ccy |
| `spread_cost = spread_t × point_size × unit_size` | = account ccy cost for one full-side spread move on 1 lot |

Used identically in `src/envs/eurusd_intraday_env.py` (cost path) and `src/features/state_builder.py` index 5 (feature path, **without** `unit_size`).

---

## 7. Known Quirks

- **Spread spikes:** common during news (US CPI / NFP / FOMC). The 0.99-quantile cap in `configs/env.yaml` dampens these.
- **Weekend gaps:** Friday 23:59 UTC → Sunday 22:00 UTC. After session masking, no Saturday/Sunday rows exist; no special handling needed.
- **Holiday-thin days:** US/EU holidays appear as low-volume, wide-spread bars but are not removed by default.
- **5-digit pricing assumption:** confirmed by inspection. If a different broker exports 4-digit prices, set `point_size = 1e-4`.

---

## 8. Re-deriving the Dataset

```powershell
# Download fresh data from MT5
python scripts/download_mt5.py --from 2025-01-01 --to 2026-05-01 --out data/raw/eurusd_m1_2025_2026.csv

# Build processed splits
python scripts/prepare_data.py --config configs/env.yaml
```
