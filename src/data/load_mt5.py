"""Load MT5-exported EURUSD 1-minute CSV into a UTC-indexed pandas DataFrame.

MetaTrader5's Python API (``mt5.copy_rates_range``) returns UTC timestamps even
though the terminal UI displays broker time. We therefore parse the ``time``
column as UTC and store a tz-aware ``DatetimeIndex(tz='UTC')`` as the canonical
index. Any conversion to UTC+7 happens downstream, in preprocessing, for
session masking only.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: tuple[str, ...] = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
)

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
INTEGER_COLUMNS: tuple[str, ...] = ("tick_volume", "spread", "real_volume")


def load_mt5_csv(path: str | Path) -> pd.DataFrame:
    """Load an MT5 CSV export and return a UTC-indexed DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the raw CSV (e.g. ``data/raw/eurusd_m1_2024.csv``).

    Returns
    -------
    pd.DataFrame
        Columns: ``open, high, low, close, tick_volume, spread, real_volume``.
        Index: ``pd.DatetimeIndex`` with ``tz='UTC'``, sorted ascending, unique.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If any required column is missing, prices cannot be parsed as floats,
        or duplicate timestamps remain after the first dedupe pass.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")

    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    for col in INTEGER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    df = df.set_index("time").sort_index()
    df.index.name = "time"

    # Validate no duplicate timestamps. The raw file should already be unique;
    # if it isn't, drop with a warning rather than silently keep the wrong row.
    dup_mask = df.index.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        logger.warning(
            "Found %d duplicate timestamps in %s; keeping first occurrence.",
            n_dup,
            path,
        )
        df = df.loc[~dup_mask]

    if df.index.duplicated().any():
        raise ValueError("Duplicate timestamps remain after dedupe — refusing to continue.")

    return df[list(EXPECTED_COLUMNS[1:])]  # drop the original 'time' col reference; index has it
