"""Weekly parameter grid search — optimize.py

Downloads 7 days of BTC price data once, then runs 108 backtest
configurations in parallel (4 axes × 3–4 values each) and writes the
ranked results to reports/grid_search_YYYY-MM-DD.json.

Usage:
    python optimize.py                        # 7-day BTC, both intervals
    python optimize.py --days 14              # extend lookback
    python optimize.py --interval 5m          # single interval only
    python optimize.py --workers 16           # more parallel workers
    python optimize.py --capital 500          # match your live balance

Output:
    reports/grid_search_YYYY-MM-DD.json       # full ranked results
    (consumed by weekly_report.py to build the Claude prompt)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from itertools import product
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Grid definition ──────────────────────────────────────────────────────────
# 4 axes × {4,3,3,3} = 108 combinations per interval.
#
# IMPORTANT: these are BTC-SPECIFIC overrides (BTC_EDGE_MIN etc.).
# They apply to BTC engines only — ETH/SOL/XRP always use the global
# EDGE_MIN / OFFPEAK_EDGE_MULT from .env, which must NOT be changed
# because those assets are performing well with their current settings.
#
# Kelly / risk-management params (KELLY_FRACTION, MAX_DAILY_DRAWDOWN,
# MAX_CONSECUTIVE_LOSSES) are intentionally excluded.
GRID = {
    "BTC_EDGE_MIN":              [0.050, 0.060, 0.070, 0.080],
    "BTC_OFFPEAK_EDGE_MULT":     [1.0,   1.3,   1.6],
    "BTC_STABILITY_MIN_SAMPLES": [3,     5,     7],
    "BTC_VOLATILITY_MAX":        [0.0010, 0.0015, 0.0020],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_combos() -> list[dict]:
    keys = list(GRID.keys())
    return [dict(zip(keys, vals)) for vals in product(*GRID.values())]


def _save_ticks_to_csv(ticks, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price"])
        for t in ticks:
            writer.writerow([t.timestamp, t.price])


def _fetch_ticks(days: int, symbol: str = "BTCUSDT"):
    """Download historical ticks using the project's own data loader."""
    sys.path.insert(0, str(BASE_DIR))
    from src.engine.backtest import HistoricalDataLoader

    print(f"[optimize] Fetching {days}d of {symbol} data from Binance…")
    raw = HistoricalDataLoader.from_binance_rest(symbol, days)
    ticks = HistoricalDataLoader.interpolate_to_1s(raw)
    print(f"[optimize] Got {len(ticks):,} ticks ({len(raw):,} raw candles)")
    return ticks


def _parse_backtest_output(stdout: str) -> tuple[dict | None, str]:
    """Extract key metrics from backtest stdout.

    Returns (metrics_dict, error_msg). On success, error_msg is "".
    On failure, metrics_dict is None and error_msg describes what went wrong.
    """
    patterns = {
        "win_rate":  r"Win rate\s*:\s*([\d.]+)%",
        "bets":      r"Bets placed\s*:\s*(\d+)",
        "roi":       r"ROI flat\(\$100\)\s*:\s*([+-]?[\d.]+)%",
        "total_pnl": r"Total PnL\s*:\s*\$([+-]?[\d.]+)",
        "avg_edge":  r"Avg edge\s*:\s*([\d.]+)%",
    }
    try:
        matches = {k: re.search(p, stdout) for k, p in patterns.items()}
        missing = [k for k, m in matches.items() if m is None]
        if missing:
            return None, f"missing_fields:{','.join(missing)}"
        mkt_m = re.search(r"Markets tested\s*:\s*([\d,]+)", stdout)
        return {
            "win_rate":  float(matches["win_rate"].group(1))  / 100,
            "bets":      int(matches["bets"].group(1)),
            "roi":       float(matches["roi"].group(1)) / 100,
            "total_pnl": float(matches["total_pnl"].group(1)),
            "avg_edge":  float(matches["avg_edge"].group(1)) / 100,
            "markets":   int(mkt_m.group(1).replace(",", "")) if mkt_m else 0,
        }, ""
    except Exception as exc:
        return None, f"parse_exception:{type(exc).__name__}:{exc}"


def _score(r: dict) -> float:
    """Composite score: win_rate is primary, roi breaks ties."""
    if r["bets"] < 5:
        return -999.0   # not enough trades to trust
    return r["win_rate"] * 2.0 + r["roi"]


def _run_one(params: dict, csv_path: str, interval: str, capital: float) -> dict:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in params.items()})

    cmd = [
        sys.executable, "-m", "src.engine.backtest",
        "--csv", csv_path,
        "--interval", interval,
        "--bet", "100",          # flat bet for clean ROI comparison
        "--capital", str(capital),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(BASE_DIR)
    )
    metrics, parse_err = _parse_backtest_output(result.stdout)
    if metrics is None:
        # Return detailed error so user can see why this combo failed
        err = parse_err or (result.stderr[:300] if result.stderr else "unknown")
        return {
            "params": params, "interval": interval,
            "error": err,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-300:] if result.stdout else "",
        }
    return {"params": params, "interval": interval, **metrics, "score": _score(metrics)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel parameter grid search")
    parser.add_argument("--days",     type=int,   default=7)
    parser.add_argument("--interval", choices=["5m", "15m", "both"], default="both")
    parser.add_argument("--workers",  type=int,   default=12)
    parser.add_argument("--capital",  type=float, default=1000.0)
    parser.add_argument("--symbol",   default="BTCUSDT")
    args = parser.parse_args()

    intervals = ["5m", "15m"] if args.interval == "both" else [args.interval]
    combos    = _all_combos()
    total     = len(combos) * len(intervals)

    print(f"[optimize] Grid: {len(combos)} combos × {len(intervals)} interval(s) = {total} runs")
    print(f"[optimize] Workers: {args.workers}  |  Days: {args.days}")

    # Download price data once, reuse for all runs
    ticks = _fetch_ticks(args.days, args.symbol)
    tmp   = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    _save_ticks_to_csv(ticks, Path(tmp.name))
    tmp.close()
    csv_path = tmp.name

    t0 = time.time()
    results: list[dict] = []
    completed = 0

    tasks = [
        (params, csv_path, interval, args.capital)
        for interval in intervals
        for params in combos
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, *t): t for t in tasks}
        for fut in as_completed(futures):
            completed += 1
            r = fut.result()
            results.append(r)
            if completed % 20 == 0 or completed == total:
                elapsed = time.time() - t0
                eta = elapsed / completed * (total - completed)
                print(
                    f"  {completed}/{total}  |  "
                    f"elapsed {elapsed:.0f}s  eta {eta:.0f}s"
                )

    os.unlink(csv_path)

    # Sort by score descending, errors at bottom
    valid   = [r for r in results if "score" in r]
    errors  = [r for r in results if "score" not in r]
    valid.sort(key=lambda r: r["score"], reverse=True)

    output = {
        "generated": str(date.today()),
        "days":      args.days,
        "symbol":    args.symbol,
        "intervals": intervals,
        "grid":      GRID,
        "top":       valid[:20],   # top 20 stored in full
        "all":       valid,
        "errors":    errors,
    }

    out_dir = BASE_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"grid_search_{date.today()}.json"
    out_path.write_text(json.dumps(output, indent=2))

    elapsed = time.time() - t0
    print(f"\n[optimize] Done in {elapsed:.0f}s  →  {out_path}")
    print(f"[optimize] Valid runs: {len(valid)}  |  Errors: {len(errors)}")

    if valid:
        best = valid[0]
        print("\n  ── Top 5 configurations ───────────────────────────────────────")
        for i, r in enumerate(valid[:5], 1):
            p = r["params"]
            print(
                f"  #{i:2d}  WR={r['win_rate']:.1%}  ROI={r['roi']:+.1%}  "
                f"bets={r['bets']}  [{r['interval']}]"
                f"  BTC_EDGE_MIN={p['BTC_EDGE_MIN']}  BTC_OFFPEAK_MULT={p['BTC_OFFPEAK_EDGE_MULT']}  "
                f"BTC_STAB={p['BTC_STABILITY_MIN_SAMPLES']}  BTC_VOL_MAX={p['BTC_VOLATILITY_MAX']}"
            )
        print(f"\n  Run weekly_report.py to generate the Claude prompt.")


if __name__ == "__main__":
    main()
