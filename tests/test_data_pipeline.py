"""M2 unit tests: load_mt5 → preprocess → split.

Uses tiny synthetic frames so the tests run in milliseconds and don't depend on
the actual 23 MB raw file. A single 'real raw file' smoke test is gated on the
file's existence so the suite stays portable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load_mt5 import EXPECTED_COLUMNS, load_mt5_csv
from src.data.preprocess import preprocess
from src.data.split import split_chronological


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def env_cfg() -> dict:
    """Env config used by tests. Matches configs/env.yaml defaults."""
    return {
        "data": {"raw_csv": "data/raw/eurusd_m1_2024.csv", "processed_dir": "data/processed"},
        "session": {
            "display_tz": "Asia/Bangkok",
            "start": "09:00",
            "end": "00:00",
            "warmup_bars": 60,
            "max_missing_bars_per_day": 30,
            "max_ffill_gap": 5,
        },
        "split": {"train_months": 2, "test_months": 1},
    }


def _make_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a tiny CSV mimicking the MT5 export schema."""
    df = pd.DataFrame(rows, columns=list(EXPECTED_COLUMNS))
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _synthetic_session(day_str: str, n_bars: int = 900, start_utc_hour: int = 2) -> list[dict]:
    """Generate `n_bars` 1-min EURUSD-like bars starting at session open in UTC+7.

    UTC+7 09:00 = UTC 02:00. Default produces a full session.
    """
    start = pd.Timestamp(f"{day_str} {start_utc_hour:02d}:00:00")
    rows = []
    price = 1.10000
    for i in range(n_bars):
        ts = start + pd.Timedelta(minutes=i)
        rows.append({
            "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": price,
            "high": price + 0.00005,
            "low":  price - 0.00005,
            "close": price + 0.00001,
            "tick_volume": 10,
            "spread": 5,
            "real_volume": 0,
        })
        price += 0.00001
    return rows


# --------------------------------------------------------------------------- load_mt5


def test_load_returns_dataframe(tmp_path: Path):
    csv = _make_csv(tmp_path, _synthetic_session("2024-01-08", n_bars=10))
    df = load_mt5_csv(csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


def test_load_index_is_utc_aware(tmp_path: Path):
    csv = _make_csv(tmp_path, _synthetic_session("2024-01-08", n_bars=5))
    df = load_mt5_csv(csv)
    assert df.index.tz is not None, "Index must be tz-aware."
    assert str(df.index.tz) == "UTC", f"Index tz must be UTC, got {df.index.tz}."


def test_load_required_columns_present(tmp_path: Path):
    csv = _make_csv(tmp_path, _synthetic_session("2024-01-08", n_bars=5))
    df = load_mt5_csv(csv)
    for col in ("open", "high", "low", "close", "tick_volume", "spread", "real_volume"):
        assert col in df.columns, f"missing column: {col}"


def test_load_missing_column_raises(tmp_path: Path):
    rows = _synthetic_session("2024-01-08", n_bars=2)
    df = pd.DataFrame(rows)
    df = df.drop(columns=["spread"])  # remove a required column
    csv = tmp_path / "broken.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_mt5_csv(csv)


def test_load_drops_duplicate_timestamps(tmp_path: Path):
    rows = _synthetic_session("2024-01-08", n_bars=3)
    rows.append(dict(rows[0]))  # duplicate the first row exactly
    csv = _make_csv(tmp_path, rows)
    df = load_mt5_csv(csv)
    assert df.index.is_unique
    assert len(df) == 3


# --------------------------------------------------------------------------- preprocess


def test_preprocess_adds_session_day_and_bar_idx(tmp_path: Path, env_cfg: dict):
    csv = _make_csv(tmp_path, _synthetic_session("2024-01-08", n_bars=900))
    df = load_mt5_csv(csv)
    processed, stats = preprocess(df, env_cfg)
    assert "session_day" in processed.columns
    assert "bar_idx_in_day" in processed.columns
    # bar_idx_in_day should be contiguous within the single day
    assert processed["bar_idx_in_day"].tolist() == list(range(len(processed)))
    assert stats.final_sessions == 1


def test_preprocess_index_stays_utc(tmp_path: Path, env_cfg: dict):
    csv = _make_csv(tmp_path, _synthetic_session("2024-01-08", n_bars=20))
    df = load_mt5_csv(csv)
    processed, _ = preprocess(df, env_cfg)
    assert str(processed.index.tz) == "UTC"


def test_preprocess_session_mask_excludes_pre_open(tmp_path: Path, env_cfg: dict):
    # Bars at UTC 00:00 == UTC+7 07:00 (before 09:00 session start) must be dropped.
    early = _synthetic_session("2024-01-08", n_bars=300, start_utc_hour=0)  # 07:00-11:59 UTC+7
    csv = _make_csv(tmp_path, early)
    df = load_mt5_csv(csv)
    processed, _ = preprocess(df, env_cfg)
    bkk = processed.index.tz_convert("Asia/Bangkok")
    assert (bkk.hour >= 9).all(), "Bars before 09:00 UTC+7 leaked through the session mask."


def test_preprocess_drops_day_with_too_many_missing(tmp_path: Path, env_cfg: dict):
    # 850 bars in a 900-slot session -> 50 missing > default 30 -> day dropped.
    csv = _make_csv(tmp_path, _synthetic_session("2024-01-08", n_bars=850))
    df = load_mt5_csv(csv)
    processed, stats = preprocess(df, env_cfg)
    assert stats.final_sessions == 0
    assert len(processed) == 0 or processed.empty


def test_preprocess_ffills_small_gaps(tmp_path: Path, env_cfg: dict):
    rows = _synthetic_session("2024-01-08", n_bars=900)
    # Delete bars 100..103 (4-minute gap; ≤ max_ffill_gap=5) and 500..510 (11-minute gap; > max).
    keep = [r for i, r in enumerate(rows) if not (100 <= i <= 103 or 500 <= i <= 510)]
    csv = _make_csv(tmp_path, keep)
    df = load_mt5_csv(csv)
    processed, stats = preprocess(df, env_cfg)
    # Total missing pre-fill: 4 + 11 = 15 (≤ max_missing 30, so day kept).
    assert stats.missing_bars_total == 15
    # 4 short-gap bars get filled; 11 long-gap bars remain dropped post-fill.
    assert stats.filled_bars_total == 4
    assert stats.rows_dropped_post_fill == 11
    # bar_idx_in_day still contiguous after row-drops.
    assert processed["bar_idx_in_day"].tolist() == list(range(len(processed)))


# --------------------------------------------------------------------------- split


def test_split_is_chronological_and_disjoint(tmp_path: Path, env_cfg: dict):
    # 3 months of single-day-per-month synthetic data (Jan 8 / Feb 8 / Mar 8).
    rows = (
        _synthetic_session("2024-01-08", n_bars=900)
        + _synthetic_session("2024-02-08", n_bars=900)
        + _synthetic_session("2024-03-08", n_bars=900)
    )
    csv = _make_csv(tmp_path, rows)
    df = load_mt5_csv(csv)
    processed, _ = preprocess(df, env_cfg)
    # env_cfg.split: train_months=2, test_months=1 → Jan+Feb train, Mar test.
    result = split_chronological(processed, env_cfg)
    assert result.train_months == ["2024-01", "2024-02"]
    assert result.test_months == ["2024-03"]
    # Disjoint session_days.
    train_days = set(result.train["session_day"])
    test_days = set(result.test["session_day"])
    assert train_days.isdisjoint(test_days)
    # All train timestamps strictly precede all test timestamps.
    assert result.train.index.max() < result.test.index.min()


def test_split_raises_when_not_enough_months(tmp_path: Path, env_cfg: dict):
    # Only one month → not enough for train_months=2 + test_months=1 = 3.
    csv = _make_csv(tmp_path, _synthetic_session("2024-01-08", n_bars=900))
    df = load_mt5_csv(csv)
    processed, _ = preprocess(df, env_cfg)
    with pytest.raises(ValueError, match="Not enough months"):
        split_chronological(processed, env_cfg)


# --------------------------------------------------------------------------- real-file smoke


REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "eurusd_m1_2024.csv"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="real raw CSV not present in checkout")
def test_real_csv_loads_with_utc_index():
    df = load_mt5_csv(REAL_CSV)
    assert len(df) > 300_000, f"expected >300k bars in real file, got {len(df)}"
    assert str(df.index.tz) == "UTC"
    assert df.index.is_unique
    # Spot-check first/last documented timestamps.
    assert df.index.min() == pd.Timestamp("2024-01-02 00:00:00+00:00")
    assert df.index.max() == pd.Timestamp("2024-12-31 20:00:00+00:00")
