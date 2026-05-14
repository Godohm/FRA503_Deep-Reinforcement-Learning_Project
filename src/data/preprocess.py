"""Preprocess MT5 EURUSD data: session masking, gap-fill, day-drop, day tagging.

Design notes
------------
- The canonical timestamp index stays in UTC throughout the pipeline.
- We compute a UTC+7 (Asia/Bangkok) view ONLY for session masking and for
  assigning ``session_day``.
- A "session" runs from ``[start, end)`` in display_tz. With the project
  defaults that is ``[09:00, 24:00) UTC+7`` (15 hours = 900 one-minute bars
  per day, max).
- ``session_day`` is the Asia/Bangkok calendar date of a bar, so a single
  session never spans two day labels.
- For each session_day we reindex to a complete 1-min grid in the session
  window. Missing bars are forward-filled within a configurable run-length
  limit; days with too many missing bars (raw, before fill) are dropped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessStats:
    """Summary statistics returned alongside the preprocessed DataFrame."""

    raw_rows: int = 0
    session_filtered_rows: int = 0
    final_rows: int = 0
    total_sessions_seen: int = 0
    final_sessions: int = 0
    dropped_days: list[str] = field(default_factory=list)
    missing_bars_total: int = 0     # NaN bars after grid reindex (pre-fill)
    filled_bars_total: int = 0      # successfully ffilled
    rows_dropped_post_fill: int = 0  # bars still NaN after ffill, dropped row-wise


def _parse_session_window(start: str, end: str) -> tuple[int, int]:
    """Convert 'HH:MM' strings to integer minutes-since-midnight in display_tz.

    The session is treated as ``[start, end)``. ``"00:00"`` for the end is
    interpreted as midnight of the *next* day, i.e. minute 1440.
    """
    def _to_min(s: str) -> int:
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm)

    start_min = _to_min(start)
    end_min = _to_min(end)
    if end_min == 0:
        end_min = 24 * 60  # next-day midnight
    if not (0 <= start_min < end_min <= 24 * 60):
        raise ValueError(
            f"Invalid session window: start={start} (={start_min}min), "
            f"end={end} (={end_min}min). Expect 0 <= start < end <= 1440."
        )
    return start_min, end_min


def _session_mask(index_local: pd.DatetimeIndex, start_min: int, end_min: int) -> np.ndarray:
    """Return a boolean mask for bars whose local time is in [start, end)."""
    minutes = index_local.hour * 60 + index_local.minute
    return (minutes >= start_min) & (minutes < end_min)


def _nan_run_lengths(mask: pd.Series) -> pd.Series:
    """For each True entry in ``mask``, return the length of the consecutive run.

    Each row's value = length of the maximal contiguous True run that contains it.
    Rows where ``mask`` is False get 0.
    """
    if mask.empty:
        return mask.astype("int64")
    groups = (mask != mask.shift()).cumsum()
    counts = mask.groupby(groups).transform("sum")
    return counts.where(mask, 0).astype("int64")


def _build_day_grid(
    session_day: pd.Timestamp,
    display_tz: str,
    start_min: int,
    end_min: int,
) -> pd.DatetimeIndex:
    """Build the complete 1-min UTC grid for a given session_day.

    Parameters
    ----------
    session_day : pd.Timestamp
        Asia/Bangkok calendar date (naive or date-like).
    display_tz : str
        IANA timezone string for the session window.
    start_min, end_min : int
        Minutes-since-midnight in display_tz; session is [start, end).
    """
    base = pd.Timestamp(session_day).tz_localize(display_tz, ambiguous=False, nonexistent="shift_forward")
    start_local = base + pd.Timedelta(minutes=start_min)
    end_local = base + pd.Timedelta(minutes=end_min)  # exclusive
    grid_local = pd.date_range(start=start_local, end=end_local - pd.Timedelta(minutes=1), freq="1min")
    return grid_local.tz_convert("UTC")


def preprocess(df: pd.DataFrame, env_cfg: dict[str, Any]) -> tuple[pd.DataFrame, PreprocessStats]:
    """Apply session filter, complete-grid reindex, gap-fill, and day-drop.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``load_mt5_csv`` (UTC-indexed; columns open/high/low/close/
        tick_volume/spread/real_volume).
    env_cfg : dict
        Parsed ``configs/env.yaml`` content.

    Returns
    -------
    tuple
        (processed_df, stats). ``processed_df`` has the original columns plus:
          - ``session_day`` (object dtype, holds ``datetime.date``)
          - ``bar_idx_in_day`` (int64, 0..N-1 per session_day, contiguous after drops)
        and a UTC-aware DatetimeIndex.
    """
    session_cfg = env_cfg["session"]
    display_tz: str = session_cfg["display_tz"]
    start_min, end_min = _parse_session_window(session_cfg["start"], session_cfg["end"])
    max_missing: int = int(session_cfg.get("max_missing_bars_per_day", 30))
    max_ffill_gap: int = int(session_cfg.get("max_ffill_gap", 5))

    stats = PreprocessStats(raw_rows=len(df))

    if df.index.tz is None:
        raise ValueError("preprocess() requires a UTC-aware DatetimeIndex. "
                         "Did you load via load_mt5_csv()?")
    if str(df.index.tz) != "UTC":
        raise ValueError(f"Canonical index must be UTC; got {df.index.tz}.")

    # 1. Drop duplicates defensively (load_mt5 already does this; cheap insurance).
    dup_mask = df.index.duplicated(keep="first")
    if dup_mask.any():
        logger.warning("Dropping %d duplicate timestamps inside preprocess.", int(dup_mask.sum()))
        df = df.loc[~dup_mask]

    # 2. Session mask using UTC+7 view (canonical index remains UTC).
    local_index = df.index.tz_convert(display_tz)
    in_session = _session_mask(local_index, start_min, end_min)
    df = df.loc[in_session].copy()
    stats.session_filtered_rows = len(df)

    # 3. Tag with session_day (UTC+7 calendar date).
    local_index = df.index.tz_convert(display_tz)
    session_days = pd.Series(local_index.date, index=df.index, name="session_day")
    df["session_day"] = session_days.values

    unique_days = sorted({d for d in session_days.values})
    stats.total_sessions_seen = len(unique_days)
    logger.info("Session-masked %d rows across %d unique session_days.",
                len(df), len(unique_days))

    # 4. Per-day: reindex to complete 1-min grid, ffill, drop bad days.
    kept_frames: list[pd.DataFrame] = []
    for day in unique_days:
        day_df = df.loc[df["session_day"] == day].drop(columns=["session_day"])
        grid = _build_day_grid(day, display_tz, start_min, end_min)
        reidx = day_df.reindex(grid)

        missing = int(reidx[["open", "high", "low", "close"]].isna().any(axis=1).sum())
        stats.missing_bars_total += missing

        if missing > max_missing:
            logger.info("Dropping session_day %s (missing=%d > max=%d).",
                        day, missing, max_missing)
            stats.dropped_days.append(str(day))
            continue

        # OHLC ffill: ONLY runs whose length <= max_ffill_gap are filled.
        # (pandas' .ffill(limit=N) would fill the first N of a longer run,
        # which is unsafe — we instead skip long runs entirely.)
        ohlc_missing = reidx[["open", "high", "low", "close"]].isna().any(axis=1)
        run_lens = _nan_run_lengths(ohlc_missing)
        safe_fill_mask = ohlc_missing & (run_lens <= max_ffill_gap)
        unsafe_mask = ohlc_missing & (run_lens > max_ffill_gap)

        # Apply ffill to OHLC + spread, then null out unsafe rows so they get
        # dropped below (the "safe" rows keep their forward-filled values).
        for col in ["open", "high", "low", "close", "spread"]:
            reidx[col] = reidx[col].ffill()
        for col in ["open", "high", "low", "close", "spread"]:
            reidx.loc[unsafe_mask, col] = pd.NA
        # tick_volume on synthesized bars = 0 (no real ticks).
        reidx["tick_volume"] = reidx["tick_volume"].fillna(0).astype("int64")
        reidx["real_volume"] = reidx["real_volume"].fillna(0).astype("int64")
        # spread fallback: if a leading NaN remains because no prior value exists
        # in a safe run, backfill once with the first observed spread of the day.
        if reidx.loc[safe_fill_mask, "spread"].isna().any():
            reidx["spread"] = reidx["spread"].bfill()

        stats.filled_bars_total += int(safe_fill_mask.sum())

        # Drop bars in unsafe runs (they remain NaN).
        n_unsafe = int(unsafe_mask.sum())
        if n_unsafe > 0:
            stats.rows_dropped_post_fill += n_unsafe
            reidx = reidx.loc[~unsafe_mask]

        # Cast spread to int (it may have become float from NaN-fill).
        reidx["spread"] = reidx["spread"].fillna(0).astype("int64")

        reidx["session_day"] = day
        reidx["bar_idx_in_day"] = np.arange(len(reidx), dtype="int64")
        kept_frames.append(reidx)

    if not kept_frames:
        logger.warning(
            "All %d session_days were dropped during preprocessing — "
            "returning an empty frame. Verify the data window and "
            "max_missing_bars_per_day=%d setting.",
            stats.total_sessions_seen, max_missing,
        )
        empty = df.iloc[0:0].copy()
        empty["session_day"] = pd.Series(dtype="object")
        empty["bar_idx_in_day"] = pd.Series(dtype="int64")
        stats.final_rows = 0
        stats.final_sessions = 0
        return empty, stats

    processed = pd.concat(kept_frames, axis=0).sort_index()
    processed.index.name = "time"
    stats.final_rows = len(processed)
    stats.final_sessions = stats.total_sessions_seen - len(stats.dropped_days)

    logger.info(
        "preprocess complete: raw=%d sessioned=%d final=%d "
        "sessions_seen=%d kept=%d dropped=%d missing_bars=%d filled=%d residual_dropped=%d",
        stats.raw_rows, stats.session_filtered_rows, stats.final_rows,
        stats.total_sessions_seen, stats.final_sessions, len(stats.dropped_days),
        stats.missing_bars_total, stats.filled_bars_total, stats.rows_dropped_post_fill,
    )

    return processed, stats
