"""Chronological train / val / test split by session_day.

Splits groups session_days by (year, month) and takes months in order:
  first  train_months  -> train
  next   val_months    -> val   (0 = no validation set, backward-compatible)
  next   test_months   -> test

Example with the 2025-2026 dataset and default 12/2/2 setting:
  train = Jan 2025 ... Dec 2025   (12 months)
  val   = Jan 2026 ... Feb 2026   ( 2 months)
  test  = Mar 2026 ... Apr 2026   ( 2 months)

No shuffling. No overlap between any two splits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame           # empty DataFrame when val_months == 0
    test: pd.DataFrame
    train_months: list[str]     # e.g. ['2025-01', ..., '2025-12']
    val_months: list[str]       # e.g. ['2026-01', '2026-02']  (empty if no val)
    test_months: list[str]      # e.g. ['2026-03', '2026-04']
    train_session_days: list[str]
    val_session_days: list[str]
    test_session_days: list[str]


def split_chronological(processed: pd.DataFrame, env_cfg: dict[str, Any]) -> SplitResult:
    """Split preprocessed bars into train / val / test by calendar month.

    Parameters
    ----------
    processed : pd.DataFrame
        Output of ``preprocess()``. Must contain a ``session_day`` column.
    env_cfg : dict
        Parsed ``configs/env.yaml`` content. Reads:
          split.train_months  (required, int >= 1)
          split.val_months    (optional, int >= 0, default 0)
          split.test_months   (required, int >= 1)
    """
    if "session_day" not in processed.columns:
        raise ValueError("split_chronological expects a 'session_day' column.")

    split_cfg = env_cfg["split"]
    n_train = int(split_cfg["train_months"])
    n_val   = int(split_cfg.get("val_months", 0))
    n_test  = int(split_cfg["test_months"])

    if n_train < 1:
        raise ValueError(f"train_months must be >= 1; got {n_train}.")
    if n_val < 0:
        raise ValueError(f"val_months must be >= 0; got {n_val}.")
    if n_test < 1:
        raise ValueError(f"test_months must be >= 1; got {n_test}.")

    sd_series = processed["session_day"]
    if sd_series.empty:
        raise ValueError("No session_days present in the processed frame.")

    unique_days: list[date] = sorted(set(sd_series.tolist()))

    # Group session_days by (year, month) in chronological order.
    month_to_days: dict[str, list[date]] = {}
    for d in unique_days:
        key = f"{d.year:04d}-{d.month:02d}"
        month_to_days.setdefault(key, []).append(d)
    months_sorted = sorted(month_to_days.keys())

    n_needed = n_train + n_val + n_test
    if len(months_sorted) < n_needed:
        raise ValueError(
            f"Not enough months for the requested split: have {len(months_sorted)}, "
            f"need train+val+test = {n_train}+{n_val}+{n_test} = {n_needed}. "
            f"Months in data: {months_sorted}"
        )

    train_months_list = months_sorted[:n_train]
    val_months_list   = months_sorted[n_train:n_train + n_val]
    test_months_list  = months_sorted[n_train + n_val:n_train + n_val + n_test]

    train_days = [d for m in train_months_list for d in month_to_days[m]]
    val_days   = [d for m in val_months_list   for d in month_to_days[m]]
    test_days  = [d for m in test_months_list  for d in month_to_days[m]]

    # Overlap guard (should be impossible with disjoint month slices).
    all_sets = [set(train_days), set(val_days), set(test_days)]
    for i, a in enumerate(all_sets):
        for j, b in enumerate(all_sets):
            if i >= j:
                continue
            overlap = a & b
            if overlap:
                names = ["train", "val", "test"]
                raise RuntimeError(
                    f"session_day overlap between {names[i]} and {names[j]}: {sorted(overlap)}"
                )

    def _mask(days):
        return processed["session_day"].isin(set(days))

    train_df = processed.loc[_mask(train_days)].copy().sort_index()
    val_df   = processed.loc[_mask(val_days)].copy().sort_index() if val_days else processed.iloc[0:0].copy()
    test_df  = processed.loc[_mask(test_days)].copy().sort_index()

    logger.info(
        "Split: train=%s..%s (%d days, %d rows) | val=%s (%d days, %d rows) | "
        "test=%s..%s (%d days, %d rows)",
        train_months_list[0], train_months_list[-1], len(train_days), len(train_df),
        f"{val_months_list[0]}..{val_months_list[-1]}" if val_months_list else "none",
        len(val_days), len(val_df),
        test_months_list[0], test_months_list[-1], len(test_days), len(test_df),
    )

    return SplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        train_months=train_months_list,
        val_months=val_months_list,
        test_months=test_months_list,
        train_session_days=[str(d) for d in train_days],
        val_session_days=[str(d) for d in val_days],
        test_session_days=[str(d) for d in test_days],
    )
