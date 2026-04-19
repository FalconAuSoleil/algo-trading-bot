"""Mini grid search — 2 combos x 1 interval x 1 day. Smoke test du pipeline."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.engine.backtest import HistoricalDataLoader
import optimize


def save_csv(ticks, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "price"])
        for t in ticks:
            w.writerow([t.timestamp, t.price])


def main():
    print("Fetching 1 day of BTC data...")
    raw = HistoricalDataLoader.from_binance_rest("BTCUSDT", 1)
    ticks = HistoricalDataLoader.interpolate_to_1s(raw)
    print(f"Ticks: {len(ticks)}")

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    save_csv(ticks, tmp.name)
    tmp.close()

    test_combos = [
        {
            "BTC_EDGE_MIN": 0.060,
            "BTC_OFFPEAK_EDGE_MULT": 1.3,
            "BTC_STABILITY_MIN_SAMPLES": 5,
            "BTC_VOLATILITY_MAX": 0.0015,
        },
        {
            "BTC_EDGE_MIN": 0.080,
            "BTC_OFFPEAK_EDGE_MULT": 1.0,
            "BTC_STABILITY_MIN_SAMPLES": 3,
            "BTC_VOLATILITY_MAX": 0.0020,
        },
    ]

    print("\n--- Running mini grid (2 combos, 15m) ---")
    for i, params in enumerate(test_combos, 1):
        t0 = time.time()
        result = optimize._run_one(params, tmp.name, "15m", 1000.0)
        dt = time.time() - t0
        if "error" in result:
            print(f"\n  combo #{i}: ERROR ({dt:.1f}s)")
            print(f"    error: {result['error']}")
            print(f"    stdout_tail: {result.get('stdout_tail', '')[:200]}")
        else:
            print(f"\n  combo #{i}: OK ({dt:.1f}s)")
            print(f"    params: {params}")
            print(f"    WR={result['win_rate']:.1%}  ROI={result['roi']:+.1%}  "
                  f"bets={result['bets']}  score={result.get('score'):.3f}")

    os.unlink(tmp.name)
    print("\n--- Grid search pipeline: OK ---")


if __name__ == "__main__":
    main()
