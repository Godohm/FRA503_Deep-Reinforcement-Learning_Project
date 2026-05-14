"""Chronological train/test split by session_day.

The split groups session_days by ``(year, month)`` and takes the first
``train_months`` calendar months as train, the next ``test_months`` as test.
With the 2024 dataset and the default 11/1 setting that yields:

  train = Jan 2024 ... Nov 2024
  test  = Dec 2024

No shuffling. No overlap. Both frames preserve the UTC index from preprocess.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    train_months: list[str]   # e.g. ['2024-01', ..., '2024-11']
    test_months: list[str]    # e.g. ['2024-12']
    train_session_days: list[str]
    test_session_days: list[str]


def split_chronological(processed: pd.DataFrame, env_cfg: dict[str, Any]) -> SplitResult:
    """Split preprocessed bars into train/test by month, chronologically.

    Parameters
    ----------
    processed : pd.DataFrame
        Output of ``preprocess()``. Must contain a ``session_day`` column.
    env_cfg : dict
        Parsed ``configs/env.yaml`` content. Reads ``split.train_months`` and
        ``split.test_months``.
    """
    if "session_day" not in processed.columns:
        raise ValueError("split_chronological expects a 'session_day' column.")

    split_cfg = env_cfg["split"]
    n_train = int(split_cfg["train_months"])
    n_test = int(split_cfg["test_months"])
    if n_train < 1 or n_test < 1:
        raise ValueError(f"train_months and test_months must be >= 1; got {n_train}/{n_test}.")

    sd_series = processed["session_day"]
    if sd_series.empty:
        raise ValueError("No session_days present in the processed frame.")

    unique_days: list[date] = sorted(set(sd_series.tolist()))

    # Group session_days by (year, month) preserving chronological order.
    month_to_days: dict[str, list[date]] = {}
    for d in unique_days:
        key = f"{d.year:04d}-{d.month:02d}"
        month_to_days.setdefault(key, []).append(d)
    months_sorted = sorted(month_to_days.keys())

    if len(months_sorted) < n_train + n_test:
        raise ValueError(
            f"Not enough months for the requested split: have {len(months_sorted)}, "
            f"need train_months + test_months = {n_train + n_test}. "
            f"Months seen: {months_sorted}"
        )

    train_months = months_sorted[:n_train]
    test_months = months_sorted[n_train:n_train + n_test]

    train_days = [d for m in train_months for d in month_to_days[m]]
    test_days = [d for m in test_months for d in month_to_days[m]]

    train_day_set = set(train_days)
    test_day_set = set(test_days)
    overlap = train_day_set & test_day_set
    if overlap:
        # Should be impossible because months are disjoint, but the check is cheap.
        raise RuntimeError(f"train/test session_day overlap detected: {sorted(overlap)}")

    train_mask = processed["session_day"].isin(train_day_set)
    test_mask = processed["session_day"].isin(test_day_set)

    train_df = processed.loc[train_mask].copy().sort_index()
    test_df = processed.loc[test_mask].copy().sort_index()

    logger.info(
        "Split: train_months=%s (%d days, %d rows) | test_months=%s (%d days, %d rows)",
        train_months, len(train_days), len(train_df),
        test_months, len(test_days), len(test_df),
    )

    return SplitResult(
        train=train_df,
        test=test_df,
        train_months=train_months,
        test_months=test_months,
        train_session_days=[str(d) for d in train_days],
        test_session_days=[str(d) for d in test_days],
    )
