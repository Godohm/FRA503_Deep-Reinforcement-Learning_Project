"""End-to-end data pipeline: raw CSV -> session-filtered, gap-filled,
chronologically split parquet files + metadata.json.

Usage
-----
    python scripts/prepare_data.py --config configs/env.yaml

Writes
------
    data/processed/train.parquet
    data/processed/test.parquet
    data/processed/metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load_mt5 import load_mt5_csv
from src.data.preprocess import preprocess
from src.data.split import split_chronological
from src.utils.config import load_config


def _format_range(idx) -> str:
    if len(idx) == 0:
        return "<empty>"
    return f"{idx.min()} -> {idx.max()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EURUSD M1 training & test data.")
    parser.add_argument("--config", default="configs/env.yaml", help="Path to env config")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    logger = logging.getLogger("prepare_data")

    cfg = load_config(args.config)
    raw_csv = Path(cfg["data"]["raw_csv"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    display_tz = cfg["session"]["display_tz"]

    logger.info("Loading raw CSV: %s", raw_csv)
    raw = load_mt5_csv(raw_csv)
    raw_rows = len(raw)
    logger.info("Raw loaded: %d rows | UTC range %s",
                raw_rows, _format_range(raw.index))
    logger.info("Raw range (UTC+7): %s",
                _format_range(raw.index.tz_convert(display_tz)))

    logger.info("Preprocessing (session=%s %s-%s)...",
                display_tz, cfg["session"]["start"], cfg["session"]["end"])
    processed, stats = preprocess(raw, cfg)
    logger.info("Preprocessed: %d rows across %d sessions (dropped %d)",
                stats.final_rows, stats.final_sessions, len(stats.dropped_days))

    split_result = split_chronological(processed, cfg)

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    # Persist session_day as ISO strings so parquet round-trips reliably across
    # pandas/pyarrow versions (avoids object-dtype date warnings).
    train_to_save = split_result.train.copy()
    test_to_save = split_result.test.copy()
    train_to_save["session_day"] = train_to_save["session_day"].astype("string")
    test_to_save["session_day"] = test_to_save["session_day"].astype("string")
    train_to_save.to_parquet(train_path)
    test_to_save.to_parquet(test_path)

    metadata = {
        "config_path": str(Path(args.config).resolve()),
        "raw_csv": str(raw_csv),
        "raw_rows": raw_rows,
        "raw_range_utc": [str(raw.index.min()), str(raw.index.max())],
        "raw_range_display_tz": [
            str(raw.index.tz_convert(display_tz).min()),
            str(raw.index.tz_convert(display_tz).max()),
        ],
        "display_tz": display_tz,
        "session_window": [cfg["session"]["start"], cfg["session"]["end"]],
        "session_filtered_rows": stats.session_filtered_rows,
        "processed_rows": stats.final_rows,
        "total_sessions_seen": stats.total_sessions_seen,
        "final_sessions": stats.final_sessions,
        "dropped_days": stats.dropped_days,
        "missing_bars_total_pre_fill": stats.missing_bars_total,
        "filled_bars_total": stats.filled_bars_total,
        "rows_dropped_post_fill": stats.rows_dropped_post_fill,
        "split": {
            "train_months_config": int(cfg["split"]["train_months"]),
            "test_months_config": int(cfg["split"]["test_months"]),
            "train_months_actual": split_result.train_months,
            "test_months_actual": split_result.test_months,
            "train_rows": int(len(split_result.train)),
            "test_rows": int(len(split_result.test)),
            "train_sessions": len(split_result.train_session_days),
            "test_sessions": len(split_result.test_session_days),
            "train_range_utc": [
                str(split_result.train.index.min()),
                str(split_result.train.index.max()),
            ],
            "test_range_utc": [
                str(split_result.test.index.min()),
                str(split_result.test.index.max()),
            ],
            "train_range_display_tz": [
                str(split_result.train.index.tz_convert(display_tz).min()),
                str(split_result.train.index.tz_convert(display_tz).max()),
            ],
            "test_range_display_tz": [
                str(split_result.test.index.tz_convert(display_tz).min()),
                str(split_result.test.index.tz_convert(display_tz).max()),
            ],
        },
    }
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # ------------------------------------------------------------------ summary
    print()
    print("=" * 72)
    print("DATA PIPELINE SUMMARY")
    print("=" * 72)
    print(f"  Raw CSV               : {raw_csv}")
    print(f"  Raw rows              : {raw_rows:,}")
    print(f"  Raw range (UTC)       : {_format_range(raw.index)}")
    print(f"  Raw range (UTC+7)     : {_format_range(raw.index.tz_convert(display_tz))}")
    print(f"  Session window        : {cfg['session']['start']}-{cfg['session']['end']} {display_tz}")
    print()
    print(f"  Session-masked rows   : {stats.session_filtered_rows:,}")
    print(f"  Final processed rows  : {stats.final_rows:,}")
    print(f"  Sessions seen         : {stats.total_sessions_seen}")
    print(f"  Sessions kept         : {stats.final_sessions}")
    print(f"  Sessions dropped      : {len(stats.dropped_days)}")
    if stats.dropped_days:
        preview = ", ".join(stats.dropped_days[:5])
        print(f"    first 5 dropped     : {preview}{' ...' if len(stats.dropped_days) > 5 else ''}")
    print(f"  Missing bars pre-fill : {stats.missing_bars_total:,}")
    print(f"  Forward-filled bars   : {stats.filled_bars_total:,}")
    print(f"  Rows dropped post-fill: {stats.rows_dropped_post_fill:,}")
    print()
    print(f"  Train months          : {split_result.train_months[0]} .. {split_result.train_months[-1]}")
    print(f"  Train rows            : {len(split_result.train):,}")
    print(f"  Train sessions        : {len(split_result.train_session_days)}")
    print(f"  Train range (UTC)     : {_format_range(split_result.train.index)}")
    print(f"  Train range (UTC+7)   : {_format_range(split_result.train.index.tz_convert(display_tz))}")
    print()
    print(f"  Test months           : {split_result.test_months[0]} .. {split_result.test_months[-1]}")
    print(f"  Test rows             : {len(split_result.test):,}")
    print(f"  Test sessions         : {len(split_result.test_session_days)}")
    print(f"  Test range (UTC)      : {_format_range(split_result.test.index)}")
    print(f"  Test range (UTC+7)    : {_format_range(split_result.test.index.tz_convert(display_tz))}")
    print()
    print(f"  Wrote: {train_path}")
    print(f"  Wrote: {test_path}")
    print(f"  Wrote: {processed_dir / 'metadata.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
