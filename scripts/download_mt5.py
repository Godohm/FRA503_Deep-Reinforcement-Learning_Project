"""Download EURUSD M1 bars from MetaTrader 5 for a given date range.

Usage
-----
    python scripts/download_mt5.py --from 2025-01-01 --to 2026-05-01 --out data/raw/eurusd_m1_2025_2026.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download EURUSD M1 from MT5.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--from", dest="date_from", default="2025-01-01",
                        help="Start date YYYY-MM-DD (inclusive, UTC)")
    parser.add_argument("--to", dest="date_to", default="2026-05-01",
                        help="End date YYYY-MM-DD (exclusive, UTC)")
    parser.add_argument("--out", default="data/raw/eurusd_m1_2025_2026.csv")
    args = parser.parse_args()

    try:
        import MetaTrader5 as mt5
        import pandas as pd
    except ImportError as e:
        print(f"[error] Missing package: {e}")
        return 1

    if not mt5.initialize():
        print(f"[error] MT5 init failed: {mt5.last_error()}")
        return 1

    info = mt5.terminal_info()
    print(f"[MT5] connected — broker: {info.company}")

    dt_from = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt_to   = datetime.strptime(args.date_to,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"[MT5] downloading {args.symbol} M1  {args.date_from} to {args.date_to} ...")
    rates = mt5.copy_rates_range(args.symbol, mt5.TIMEFRAME_M1, dt_from, dt_to)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("[error] No data returned. Check symbol name and date range.")
        return 1

    import pandas as pd
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[done] {len(df):,} bars  {df['time'].iloc[0]}  to  {df['time'].iloc[-1]}")
    print(f"       saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
