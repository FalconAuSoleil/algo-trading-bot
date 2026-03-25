"""Offline backtester for the ChainlinkArb strategy — v4.0.

Tests whether the ChainlinkArb edge is statistically real on
historical BTC price data, and supports parameter sensitivity analysis.

Usage:
    # Fetch 7 days of live BTC data from Binance REST and backtest 5m markets:
    python -m src.engine.backtest --days 7 --interval 5m

    # Backtest 15m markets with verbose output:
    python -m src.engine.backtest --days 14 --interval 15m -v

    # Use a pre-downloaded CSV (columns: timestamp, price):
    python -m src.engine.backtest --csv prices.csv --interval 5m

    # Fix bet size to $100 per trade (removes Kelly sizing noise):
    python -m src.engine.backtest --days 7 --bet 100

NOTE: This is a SIGNAL backtest, not a full execution backtest.
  - Market depth is assumed constant (500 USD on each side).
  - Orderbook is symmetric (bid/ask = 0.49/0.51).
  - No slippage, no queue position.
  The purpose is to validate that the signal (edge) is real, not that
  execution would be perfect. Execution backtesting requires live CLOB data.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import config
from src.engine.signal import (
    MarketState,
    _ChainlinkArbEngine,
    p_brownian,
    clamp,
)


# ---- data structures -------------------------------------------------------

@dataclass
class TickData:
    timestamp: float
    price: float


@dataclass
class BacktestMarket:
    start_time: float
    end_time: float
    reference_price: float
    duration: int               # 300 or 900 seconds
    resolved_up: Optional[bool] = None
    resolution_price: float = 0.0

    @property
    def slug(self) -> str:
        d = "5m" if self.duration == 300 else "15m"
        return f"btc-{d}-backtest"


@dataclass
class BacktestResult:
    total: int = 0
    bets: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    predicted_probs: list = field(default_factory=list)
    actual_outcomes: list = field(default_factory=list)
    edge_values: list = field(default_factory=list)
    time_remaining_values: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.bets if self.bets else 0.0

    @property
    def avg_edge(self) -> float:
        return sum(self.edge_values) / len(self.edge_values) if self.edge_values else 0.0

    @property
    def roi_flat(self) -> float:
        """ROI per $100 flat bet."""
        if self.bets == 0:
            return 0.0
        return self.total_pnl / (self.bets * 100)


# ---- data loading ----------------------------------------------------------

class HistoricalDataLoader:
    """Load BTC price ticks from CSV or Binance REST API."""

    @staticmethod
    def from_csv(path: str) -> list[TickData]:
        ticks = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_key = next(
                    (k for k in row if k.lower() in ("timestamp", "time", "t")), None
                )
                p_key = next(
                    (k for k in row if k.lower() in ("price", "close", "p")), None
                )
                if ts_key and p_key:
                    ticks.append(TickData(float(row[ts_key]), float(row[p_key])))
        return sorted(ticks, key=lambda x: x.timestamp)

    @staticmethod
    def from_binance_rest(
        symbol: str = "BTCUSDT", days: int = 7
    ) -> list[TickData]:
        """Fetch 1-minute kline close prices from Binance REST API."""
        import urllib.request
        import json

        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86_400 * 1000
        url_base = "https://api.binance.com/api/v3/klines"
        ticks: list[TickData] = []
        cur = start_ms

        while cur < end_ms:
            url = (
                f"{url_base}?symbol={symbol}&interval=1m"
                f"&startTime={cur}&endTime={end_ms}&limit=1000"
            )
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    data = json.loads(resp.read())
            except Exception as exc:
                raise RuntimeError(
                    f"Binance REST fetch failed: {exc}. "
                    "Use --csv for offline backtesting."
                )
            if not data:
                break
            for candle in data:
                ts = candle[0] / 1000.0   # open time → seconds
                close = float(candle[4])  # close price
                ticks.append(TickData(ts, close))
            cur = int(data[-1][0]) + 60_000

        return ticks

    @staticmethod
    def interpolate_to_1s(
        ticks: list[TickData], noise_pct: float = 0.0001
    ) -> list[TickData]:
        """Linear interpolation of minute candles to ~1-second ticks.

        Adds Gaussian microstructure noise (default 0.01%/s) to avoid
        a perfectly smooth signal that would be unrealistic.
        """
        if len(ticks) < 2:
            return ticks
        result: list[TickData] = []
        for i in range(len(ticks) - 1):
            t0, p0 = ticks[i].timestamp, ticks[i].price
            t1, p1 = ticks[i + 1].timestamp, ticks[i + 1].price
            n = max(1, int(t1 - t0))
            for j in range(n):
                frac = j / n
                price = p0 + (p1 - p0) * frac
                noise = random.gauss(0, price * noise_pct)
                result.append(TickData(t0 + j, price + noise))
        result.append(ticks[-1])
        return result


# ---- backtester ------------------------------------------------------------

class Backtester:
    """Replay historical BTC data through the ChainlinkArb engine."""

    def __init__(
        self,
        market_interval: int = 300,
        chainlink_period: float = 27.0,
        chainlink_jitter: float = 5.0,
        orderbook_depth: float = 500.0,
        capital: float = 1000.0,
        flat_bet_usd: Optional[float] = None,
        verbose: bool = False,
    ):
        self.market_interval = market_interval
        self.cl_period = chainlink_period
        self.cl_jitter = chainlink_jitter
        self.depth = orderbook_depth
        self.capital = capital
        self.flat_bet = flat_bet_usd
        self.verbose = verbose

        asset_cfg = config.get_asset_config("BTC")
        self.engine = _ChainlinkArbEngine(
            cfg=config.signal,
            sigma_fallback=asset_cfg.sigma_fallback,
            delta_min_abs=asset_cfg.delta_min_abs,
        )

    def _generate_markets(
        self, ticks: list[TickData]
    ) -> list[BacktestMarket]:
        if not ticks or len(ticks) < 2:
            return []
        t_start = ticks[0].timestamp
        t_end = ticks[-1].timestamp
        ts_map = {t.timestamp: t.price for t in ticks}
        sorted_ts = sorted(ts_map)

        def _price_at(ts: float) -> float:
            # Binary search for nearest tick
            lo, hi = 0, len(sorted_ts) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_ts[mid] < ts:
                    lo = mid + 1
                else:
                    hi = mid
            return ts_map[sorted_ts[lo]]

        markets = []
        t = t_start
        while t + self.market_interval <= t_end:
            ref = _price_at(t)
            end_t = t + self.market_interval
            close = _price_at(end_t)
            markets.append(BacktestMarket(
                start_time=t,
                end_time=end_t,
                reference_price=ref,
                duration=self.market_interval,
                resolved_up=(close >= ref),
                resolution_price=close,
            ))
            t += self.market_interval
        return markets

    def _simulate_chainlink(
        self, ticks: list[TickData]
    ) -> list[tuple[float, float]]:
        """Generate (timestamp, price) oracle update events."""
        if not ticks:
            return []
        ts_map = {t.timestamp: t.price for t in ticks}
        sorted_ts = sorted(ts_map)

        def _price_at(ts: float) -> float:
            lo, hi = 0, len(sorted_ts) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_ts[mid] < ts:
                    lo = mid + 1
                else:
                    hi = mid
            return ts_map[sorted_ts[lo]]

        updates = []
        last_ts = ticks[0].timestamp - self.cl_period
        end_ts = ticks[-1].timestamp
        cur = ticks[0].timestamp

        while cur <= end_ts:
            next_ts = last_ts + self.cl_period + random.gauss(0, self.cl_jitter)
            if next_ts <= cur:
                next_ts = cur + 1.0
            updates.append((next_ts, _price_at(next_ts)))
            last_ts = next_ts
            cur = next_ts

        return updates

    def run(self, ticks: list[TickData]) -> BacktestResult:
        """Execute the full backtest loop."""
        result = BacktestResult()
        markets = self._generate_markets(ticks)
        cl_updates = self._simulate_chainlink(ticks)
        result.total = len(markets)

        if self.verbose:
            print(
                f"[Backtest] {len(ticks):,} ticks  |  "
                f"{len(markets)} markets  |  "
                f"{len(cl_updates)} oracle events"
            )

        # Build fast oracle lookup
        cl_queue = list(cl_updates)
        cl_idx = 0
        last_cl_price = ticks[0].price
        last_cl_ts = ticks[0].timestamp

        ts_map = {t.timestamp: t.price for t in ticks}
        sorted_ts = sorted(ts_map)

        def _price_at(ts: float) -> float:
            lo, hi = 0, len(sorted_ts) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_ts[mid] < ts:
                    lo = mid + 1
                else:
                    hi = mid
            return ts_map[sorted_ts[lo]]

        for market in markets:
            # Evaluate at intervals within the valid betting window
            eval_start = market.start_time + self.market_interval * 0.25
            eval_end = (
                market.end_time - config.signal.time_min_5m
                if market.duration == 300
                else market.end_time - config.signal.time_min_15m
            )
            t = eval_start
            bet_placed = False

            while t <= eval_end and not bet_placed:
                # Advance oracle state
                while cl_idx < len(cl_queue) and cl_queue[cl_idx][0] <= t:
                    last_cl_ts, last_cl_price = cl_queue[cl_idx]
                    cl_idx += 1

                binance_price = _price_at(t)
                self.engine.update_price(binance_price, t)
                self.engine.update_chainlink(last_cl_price, last_cl_ts)

                state = MarketState(
                    market_id=f"bt_{market.start_time:.0f}",
                    reference_price=market.reference_price,
                    end_time=market.end_time,
                    btc_chainlink=last_cl_price,
                    btc_binance=binance_price,
                    p_market_yes=0.50,
                    depth_yes=self.depth,
                    depth_no=self.depth,
                    best_bid_yes=0.49,
                    best_ask_yes=0.51,
                    best_bid_no=0.49,
                    best_ask_no=0.51,
                    spread_yes=0.02,
                    spread_no=0.02,
                    slug=market.slug,
                    start_time=market.start_time,
                    duration_seconds=self.market_interval,
                )

                sig = self.engine.evaluate(
                    state, self.capital,
                    consecutive_losses=0,
                    daily_pnl_pct=0.0,
                    open_positions=0,
                    has_position_on_market=False,
                )

                if sig.action == "BUY" and sig.filters_passed:
                    bet_placed = True
                    result.bets += 1
                    size = self.flat_bet if self.flat_bet else sig.size_usd
                    entry = sig.entry_price
                    shares = size / max(entry, 1e-9)

                    won = (
                        market.resolved_up
                        if sig.side == "YES"
                        else not market.resolved_up
                    )
                    if won:
                        pnl = shares * (1 - entry)
                        result.wins += 1
                    else:
                        pnl = -size
                        result.losses += 1

                    result.total_pnl += pnl
                    result.edge_values.append(sig.edge)
                    result.predicted_probs.append(sig.p_true)
                    result.actual_outcomes.append(won)
                    result.time_remaining_values.append(sig.time_remaining_sec)

                    if self.verbose:
                        marker = "✓" if won else "✗"
                        print(
                            f"  {marker} T-{sig.time_remaining_sec:.0f}s "
                            f"{sig.side:3s} @{entry:.2f} "
                            f"p={sig.p_true:.2f} edge={sig.edge:.3f} "
                            f"pnl=${pnl:+.2f}"
                        )

                    self.engine.reset_stability(market.slug)

                t += 2.0  # 2s evaluation step (matches live signal loop)

        return result

    def report(self, result: BacktestResult) -> None:
        """Print formatted backtest report."""
        dur = f"{self.market_interval // 60}m"
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS — {dur} markets")
        print(f"{'='*60}")
        print(f"  Markets tested : {result.total:,}")
        bet_pct = result.bets / max(result.total, 1) * 100
        print(f"  Bets placed    : {result.bets} ({bet_pct:.1f}% of markets)")
        print(f"  Win rate       : {result.win_rate*100:.1f}%  "
              f"({result.wins}W / {result.losses}L)")
        print(f"  Total PnL      : ${result.total_pnl:+.2f}")
        print(f"  ROI flat($100) : {result.roi_flat*100:+.2f}%")
        print(f"  Avg edge       : {result.avg_edge*100:.2f}%")

        if result.time_remaining_values:
            avg_tr = sum(result.time_remaining_values) / len(result.time_remaining_values)
            print(f"  Avg T-remaining: {avg_tr:.0f}s")

        if len(result.predicted_probs) >= 5:
            print(f"\n  Calibration (predicted vs actual win rate):")
            paired = sorted(
                zip(result.predicted_probs, result.actual_outcomes),
                key=lambda x: x[0],
            )
            n = len(paired)
            bucket_size = max(1, n // 4)
            for i in range(0, n, bucket_size):
                chunk = paired[i : i + bucket_size]
                avg_pred = sum(p for p, _ in chunk) / len(chunk)
                avg_act = sum(1 for _, a in chunk if a) / len(chunk)
                bias = avg_act - avg_pred
                sign = "+" if bias >= 0 else ""
                bar = "█" * int(avg_act * 20)
                print(
                    f"    pred={avg_pred:.0%}  actual={avg_act:.0%} "
                    f"bias={sign}{bias:.0%}  {bar}"
                )

        if result.bets > 0 and result.win_rate < 0.50:
            print("\n  ⚠ Win rate below 50% — strategy not profitable at this parameter set.")
            print("    Consider adjusting: edge_min, delta_min_abs, stability_min_samples.")
        elif result.bets > 0 and result.win_rate >= 0.60:
            print("\n  ✅ Win rate above 60% — signal shows positive edge.")

        print(f"{'='*60}\n")


# ---- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest ChainlinkArb strategy on historical BTC data"
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", choices=["5m", "15m"], default="5m")
    parser.add_argument("--csv", default=None, help="Path to CSV price file")
    parser.add_argument(
        "--bet", type=float, default=None,
        help="Flat bet size USD (overrides Kelly). Useful for clean ROI comparison."
    )
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    market_interval = 300 if args.interval == "5m" else 900

    print(f"[Backtest] Loading {args.days}d of {args.symbol} price data...")

    if args.csv:
        ticks = HistoricalDataLoader.from_csv(args.csv)
        print(f"[Backtest] Loaded {len(ticks):,} ticks from {args.csv}")
    else:
        try:
            raw = HistoricalDataLoader.from_binance_rest(args.symbol, args.days)
            print(f"[Backtest] Fetched {len(raw):,} minute candles from Binance")
            ticks = HistoricalDataLoader.interpolate_to_1s(raw)
            print(f"[Backtest] Interpolated to {len(ticks):,} second-resolution ticks")
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return

    bt = Backtester(
        market_interval=market_interval,
        capital=args.capital,
        flat_bet_usd=args.bet,
        verbose=args.verbose,
    )
    result = bt.run(ticks)
    bt.report(result)


if __name__ == "__main__":
    main()
