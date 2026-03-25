"""Trade Analytics CLI — v4.0.

Analyzes resolved trades from the SQLite database and prints:
  - Overall win rate, PnL, Sharpe, max drawdown
  - Win rate by strategy, asset, time-remaining bucket, entry price
  - Calibration curve: predicted p_true vs actual win rate

Usage:
    python -m src.utils.analytics
    python -m src.utils.analytics --db data/trades.db --mode paper
    python -m src.utils.analytics --db data/trades.db --mode live
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


# ---- helpers ---------------------------------------------------------------

def load_trades(db_path: str, mode: str | None = None) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    query = "SELECT * FROM trades WHERE outcome IN ('won', 'lost') ORDER BY timestamp"
    cur.execute(query)
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    if mode:
        rows = [r for r in rows if r.get("mode") == mode]
    return rows


def _bucket(val: float, buckets: list) -> str:
    for lo, hi, label in buckets:
        if lo <= val < hi:
            return label
    return "other"


def _winrate_table(
    groups: dict[str, list[bool]],
    indent: str = "  ",
) -> None:
    print(f"{indent}{'Category':<25} {'N':>6} {'Wins':>6} {'WR':>8}")
    print(indent + "-" * 50)
    for name, results in sorted(groups.items(), key=lambda x: -len(x[1])):
        n = len(results)
        if n == 0:
            continue
        wins = sum(results)
        wr = wins / n * 100
        bar = "█" * int(wr / 5)  # 5% per block
        print(f"{indent}{name:<25} {n:>6} {wins:>6} {wr:>7.1f}%  {bar}")


def _sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    n = len(pnls)
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(var)
    return (mean / std * math.sqrt(n)) if std > 0 else 0.0


def _max_drawdown(pnls: list[float]) -> float:
    peak = 0.0
    cur = 0.0
    mdd = 0.0
    for p in pnls:
        cur += p
        if cur > peak:
            peak = cur
        if peak > 0:
            dd = (peak - cur) / peak
            if dd > mdd:
                mdd = dd
    return mdd


# ---- calibration ------------------------------------------------------------

def print_calibration(trades: list[dict]) -> None:
    """Print calibration table: predicted p_true buckets vs actual win rate."""
    buckets_def = [
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
        (0.70, 0.75, "70-75%"),
        (0.75, 0.80, "75-80%"),
        (0.80, 1.00, "80%+"),
    ]
    groups: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        p = t.get("p_true") or 0.5
        b = _bucket(p, buckets_def)
        groups[b].append(t["outcome"] == "won")

    print("\n  📊 Calibration (predicted p_true vs actual win rate):")
    print(f"  {'Range':<12} {'N':>5} {'Predicted':>11} {'Actual':>10} {'Bias':>8}")
    print("  " + "-" * 52)
    for lo, hi, label in buckets_def:
        results = groups.get(label, [])
        n = len(results)
        if n == 0:
            continue
        actual_wr = sum(results) / n
        midpoint = (lo + hi) / 2
        bias = actual_wr - midpoint
        sign = "+" if bias >= 0 else ""
        print(
            f"  {label:<12} {n:>5} {midpoint*100:>9.1f}% {actual_wr*100:>9.1f}%"
            f" {sign}{bias*100:>6.1f}%"
        )
    print()
    print("  Interpretation:")
    print("    Bias > 0 means the model UNDER-estimates win probability (conservative).")
    print("    Bias < 0 means the model OVER-estimates win probability (overconfident).")
    print("    Ideal: all biases close to 0%.")


# ---- main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trade analytics for algo-trading-bot (v4.0)"
    )
    parser.add_argument(
        "--db", default="data/trades.db", help="Path to SQLite DB"
    )
    parser.add_argument(
        "--mode", default=None, choices=["paper", "live"],
        help="Filter by trading mode",
    )
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"\n  ERROR: DB not found: {db_path}")
        print("  Run the bot in paper mode first to generate trade data.")
        return

    trades = load_trades(db_path, mode=args.mode)
    if not trades:
        print("  No resolved trades found in the database.")
        return

    n = len(trades)
    wins = sum(1 for t in trades if t["outcome"] == "won")
    losses = n - wins
    pnls = [t.get("pnl", 0.0) or 0.0 for t in trades]
    total_pnl = sum(pnls)
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    avg_win = sum(win_pnls) / max(len(win_pnls), 1)
    avg_loss = sum(loss_pnls) / max(len(loss_pnls), 1)
    wr = wins / n * 100
    sh = _sharpe(pnls)
    mdd = _max_drawdown(pnls)

    mode_label = args.mode or "all"
    print(f"\n{'='*65}")
    print(f"  🤖  ALGO-TRADING-BOT — Trade Analytics  (mode: {mode_label})")
    print(f"  DB: {db_path}  |  Total resolved: {n}")
    print(f"{'='*65}")

    print(f"\n  🎯 Overall Performance:")
    print(f"     Win rate    : {wr:.1f}%  ({wins}W / {losses}L)")
    print(f"     Total PnL   : ${total_pnl:+.2f}")
    print(f"     Avg Win     : ${avg_win:+.2f}")
    print(f"     Avg Loss    : ${avg_loss:+.2f}")
    if avg_loss != 0:
        print(f"     P:L Ratio   : {abs(avg_win / avg_loss):.2f}x")
    print(f"     Sharpe      : {sh:.2f}")
    print(f"     Max Drawdown: {mdd*100:.1f}%")

    # --- by strategy ---------------------------------------------------------
    by_strat: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        strat = t.get("strategy_used") or "unknown"
        by_strat[strat].append(t["outcome"] == "won")
    print("\n  📈 By Strategy:")
    _winrate_table(by_strat)

    # --- by asset ------------------------------------------------------------
    by_asset: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        slug = t.get("slug") or ""
        asset = slug.split("-")[0].upper() if slug else "UNK"
        by_asset[asset].append(t["outcome"] == "won")
    print("\n  🪙 By Asset:")
    _winrate_table(by_asset)

    # --- by time remaining ---------------------------------------------------
    tr_buckets = [
        (0, 65, "0-65s (late)"),
        (65, 100, "65-100s"),
        (100, 150, "100-150s"),
        (150, 300, "150-300s"),
        (300, 600, "300-600s"),
    ]
    by_time: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        tr = t.get("time_remaining_sec") or 0
        b = _bucket(tr, tr_buckets)
        by_time[b].append(t["outcome"] == "won")
    print("\n  ⏱  By Time Remaining at Entry:")
    _winrate_table(by_time)
    print("     (note: v4.0 raises time_min_5m to 65s, cutting the '0-65s' bucket)")

    # --- by entry price ------------------------------------------------------
    ep_buckets = [
        (0.30, 0.45, "0.30-0.45"),
        (0.45, 0.50, "0.45-0.50"),
        (0.50, 0.55, "0.50-0.55"),
        (0.55, 0.60, "0.55-0.60"),
        (0.60, 0.65, "0.60-0.65"),
        (0.65, 0.75, "0.65-0.75"),
    ]
    by_entry: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        ep = t.get("entry_price") or t.get("p_market") or 0.5
        b = _bucket(ep, ep_buckets)
        by_entry[b].append(t["outcome"] == "won")
    print("\n  💰 By Entry Price:")
    _winrate_table(by_entry)

    # --- by market duration --------------------------------------------------
    dur_buckets = [(0, 400, "5m (300s)"), (400, 1000, "15m (900s)")]
    by_dur: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        slug = t.get("slug") or ""
        dur = 900 if "15m" in slug else 300
        b = _bucket(dur, dur_buckets)
        by_dur[b].append(t["outcome"] == "won")
    print("\n  ⏰ By Market Duration:")
    _winrate_table(by_dur)

    # --- calibration ---------------------------------------------------------
    print_calibration(trades)

    # --- recommendations -----------------------------------------------------
    print("\n  💡 Recommendations:")
    if n < 50:
        print("     - Insufficient data for reliable calibration (need 50+ trades).")
        print("       Continue paper trading before going live.")
    else:
        late_trades = [t for t in trades if (t.get("time_remaining_sec") or 0) < 65]
        if late_trades:
            lwr = sum(1 for t in late_trades if t["outcome"] == "won") / len(late_trades)
            if lwr < 0.55:
                print(
                    f"     - Late-window trades (<65s) win rate: {lwr*100:.1f}%."
                    " Consider raising time_min_5m further."
                )
        if wr > 0 and abs(avg_loss) > avg_win * 1.5:
            print(
                "     - Loss magnitude significantly exceeds win magnitude. "
                "Review bet sizing — consider reducing max_bet_fraction."
            )
        if mdd > 0.10:
            print(
                f"     - Max drawdown {mdd*100:.1f}% exceeds 10%. "
                "Consider reducing max_open_positions or max_daily_risk."
            )

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
