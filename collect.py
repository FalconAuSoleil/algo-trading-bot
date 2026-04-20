"""Data collector entry point — runs 24/7 continuously, accumulates forever.

Usage:
    python collect.py                   # default DB: data/collector.db
    python collect.py --db foo.db
    python collect.py --stats-interval 60

RETENTION: unlimited. The DB grows monotonically; nothing is ever deleted.
Run it for a week, a month, a year — every BTC/ETH/SOL/XRP market (5m and
15m) is accumulated.

What gets recorded (see src/collector/schema.py for full details):
    - markets             — every discovered Up/Down market + resolution
    - orderbook_snapshots — REST /book @ 2s + WS book events (on trade)
    - orderbook_deltas    — WS price_change (level adds / cancels)
    - trades              — WS last_trade_price (public trade tape)
    - tick_changes        — WS tick_size_change
    - market_states       — Gamma /events dynamic snapshot (~30s)
    - price_ticks         — Binance + Chainlink @ 1 Hz per asset
    - shadow_signals      — every SignalEngine.evaluate() result
    - ws_events           — raw WS event log (catch-all for replay safety)

Feeds wired:
    - Polymarket: REST /book poll (2s) + market-channel WebSocket
    - Binance: WS trade stream per asset
    - Chainlink: direct oracle reads on Polygon (~3s)

Why a separate collect.py (not a flag on main.py):
    - Passive — no wallet needed, no Polymarket writes. Safe on any host.
    - Independent DB (data/collector.db) so main.py keeps its clean trade
      history and the collector's volume doesn't bloat it.
    - Can run in parallel with main.py on another host.

After any N days of collection:
    sqlite3 data/collector.db 'SELECT COUNT(*) FROM shadow_signals'
    # replay backtester (TODO):
    python optimize.py --replay data/collector.db --workers 12
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiohttp

from src.config import config
from src.utils.logger import setup_logger
from src.feeds.binance import BinanceFeed
from src.feeds.chainlink import ChainlinkFeed
from src.feeds.polymarket import PolymarketFeed, MarketInfo
from src.engine.signal import SignalEngine, MarketState
from src.collector.recorder import CollectorRecorder
from src.collector.polymarket_ws import PolymarketMarketWS

log = setup_logger("collect", config.log_level)

GAMMA_POLL_INTERVAL = 30.0   # seconds between market_states snapshots


def _parse_iso(s: str) -> float | None:
    if not s:
        return None
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return None


class Collector:
    """Shadow-mode orchestrator: same wiring as Orchestrator but no trader."""

    def __init__(self, db_path: str, stats_interval: int = 300):
        self.recorder = CollectorRecorder(db_path=db_path)
        self._stats_interval = stats_interval

        self._chainlink_feeds: dict[str, ChainlinkFeed] = {}
        self._binance_feeds: dict[str, BinanceFeed] = {}
        self._signal_engines: dict[str, SignalEngine] = {}
        self._asset_prices: dict[str, dict] = {}

        for asset in config.assets:
            if not asset.enabled:
                continue
            self._chainlink_feeds[asset.symbol] = ChainlinkFeed(
                symbol=asset.symbol,
                contract_address=asset.chainlink_address,
                binance_symbol=asset.binance_symbol,
                on_price=self._on_price,
            )
            self._binance_feeds[asset.symbol] = BinanceFeed(
                symbol=asset.symbol,
                url=f"wss://stream.binance.com:9443/ws/{asset.binance_symbol.lower()}@trade",
                on_price=self._on_price,
            )
            self._signal_engines[asset.symbol] = SignalEngine(
                cfg=config.signal,
                sigma_fallback=asset.sigma_fallback,
                delta_min_abs=asset.delta_min_abs,
                asset_symbol=asset.symbol,
            )
            self._asset_prices[asset.symbol] = {
                "chainlink": 0.0,
                "binance": 0.0,
                "chainlink_ts": 0.0,
            }

        asset_intervals = {
            a.polymarket_prefix: a.supported_intervals
            for a in config.assets if a.enabled
        }
        self.polymarket_feed = PolymarketFeed(
            gamma_url=config.polymarket.gamma_url,
            clob_url=config.polymarket.clob_url,
            clob_ws_url=config.polymarket.clob_ws_url,
            asset_intervals=asset_intervals,
            on_market_update=self._on_market_update,
            on_orderbook_update=self._on_orderbook_update,
        )

        self._active_markets: dict[str, MarketInfo] = {}
        self._orderbooks: dict = {}
        self._last_tick_log: dict[str, int] = {}
        self._running = False

        # WS subscriber for market-channel events
        self.poly_ws = PolymarketMarketWS(recorder=self.recorder)
        # Aiohttp session used by the Gamma snapshot poller
        self._gamma_session: aiohttp.ClientSession | None = None

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        log.info("=" * 60)
        log.info("  COLLECTOR v1.0 - Shadow-mode data recorder")
        enabled = [a for a in config.assets if a.enabled]
        for a in enabled:
            log.info(
                "  %s: %s | CL=%s...",
                a.symbol,
                "5m+15m" if len(a.supported_intervals) > 1 else "15m only",
                a.chainlink_address[:10],
            )
        log.info("  DB: %s", self.recorder.db_path)
        log.info("  Stats interval: %ds", self._stats_interval)
        log.info("=" * 60)

        Path("data").mkdir(exist_ok=True)
        await self.recorder.start()

        self._running = True
        self._gamma_session = aiohttp.ClientSession()

        tasks = [
            asyncio.create_task(self.polymarket_feed.start(), name="polymarket_feed"),
            asyncio.create_task(self.poly_ws.run(),           name="polymarket_ws"),
            asyncio.create_task(self._signal_loop(),          name="signal_loop"),
            asyncio.create_task(self._resolution_loop(),      name="resolution_loop"),
            asyncio.create_task(self._gamma_poll_loop(),      name="gamma_poll_loop"),
            asyncio.create_task(self._stats_loop(),           name="stats_loop"),
        ]
        for s, f in self._chainlink_feeds.items():
            tasks.append(asyncio.create_task(f.start(), name=f"chainlink_{s}"))
        for s, f in self._binance_feeds.items():
            tasks.append(asyncio.create_task(f.start(), name=f"binance_{s}"))

        if sys.platform != "win32":
            import signal as _sig
            loop = asyncio.get_event_loop()
            for nm in (_sig.SIGINT, _sig.SIGTERM):
                loop.add_signal_handler(
                    nm, lambda: asyncio.create_task(self.shutdown())
                )

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        log.info("[Collect] Shutting down — draining queues...")
        await self.poly_ws.stop()
        await self.polymarket_feed.stop()
        for f in self._chainlink_feeds.values():
            await f.stop()
        for f in self._binance_feeds.values():
            await f.stop()
        if self._gamma_session:
            await self._gamma_session.close()
        await self.recorder.stop()
        log.info("[Collect] Shutdown complete")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    # ─── Feed callbacks ───────────────────────────────────────────────

    async def _on_price(
        self, source: str, price: float, timestamp: float, symbol: str = "BTC"
    ) -> None:
        if symbol not in self._asset_prices:
            return
        bucket = self._asset_prices[symbol]
        if source == "binance":
            bucket["binance"] = price
            if symbol in self._signal_engines:
                self._signal_engines[symbol].update_price(price, timestamp)
        elif source in ("chainlink", "chainlink_binance_fallback"):
            bucket["chainlink"] = price
            bucket["chainlink_ts"] = timestamp
            if symbol in self._signal_engines:
                self._signal_engines[symbol].update_chainlink_price(price, timestamp)

        # Downsample ticks — 1 row per (asset, whole second)
        sec = int(time.time())
        last = self._last_tick_log.get(symbol, 0)
        if sec != last:
            self._last_tick_log[symbol] = sec
            self.recorder.record_price_tick(
                asset=symbol,
                timestamp=float(sec),
                binance_price=bucket["binance"],
                chainlink_price=bucket["chainlink"],
                chainlink_ts=bucket["chainlink_ts"],
            )

    async def _on_market_update(self, markets: dict[str, MarketInfo]) -> None:
        self._active_markets = markets
        for m in markets.values():
            self.recorder.record_market(m)
        # Push the fresh set of token_ids to the WS subscriber — it will
        # reconnect with the new list if it changed.
        asset_ids: set[str] = set()
        for m in markets.values():
            if m.token_id_up:
                asset_ids.add(m.token_id_up)
            if m.token_id_down:
                asset_ids.add(m.token_id_down)
        self.poly_ws.set_assets(asset_ids)

    async def _on_orderbook_update(self, cid: str, ob) -> None:
        self._orderbooks[cid] = ob
        self.recorder.record_orderbook(cid, ob)

    # ─── Signal loop (shadow only — no trades) ────────────────────────

    async def _signal_loop(self) -> None:
        while self._running:
            try:
                await self._evaluate_all_markets()
            except Exception as exc:
                log.error("[Signal] Loop error: %s", exc, exc_info=True)
            await asyncio.sleep(2)

    async def _evaluate_all_markets(self) -> None:
        if not self._active_markets:
            return

        for cid, market in list(self._active_markets.items()):
            if market.is_expired:
                continue
            if market.reference_price <= 0:
                continue

            slug_asset = market.slug.split("-", 1)[0].upper() if market.slug else ""
            asset_cfg = next(
                (a for a in config.assets
                 if a.enabled and a.polymarket_prefix.upper() == slug_asset),
                None,
            )
            if asset_cfg is None:
                continue

            symbol = asset_cfg.symbol
            engine = self._signal_engines.get(symbol)
            if engine is None:
                continue

            prices = self._asset_prices.get(symbol, {})
            chainlink_price = prices.get("chainlink", 0.0)
            if chainlink_price <= 0:
                continue

            ob = self._orderbooks.get(cid)

            state = MarketState(
                market_id=cid,
                reference_price=market.reference_price,
                end_time=market.end_time,
                btc_chainlink=chainlink_price,
                btc_binance=prices.get("binance", 0.0),
                p_market_yes=ob.mid_yes if ob else 0.5,
                depth_yes=ob.depth_ask_yes if ob else 0,
                depth_no=ob.depth_ask_no if ob else 0,
                best_bid_yes=ob.best_bid_yes if ob else 0.0,
                best_ask_yes=ob.best_ask_yes if ob else 0.0,
                best_bid_no=ob.best_bid_no if ob else 0.0,
                best_ask_no=ob.best_ask_no if ob else 0.0,
                spread_yes=ob.spread_up if ob else 0.01,
                spread_no=ob.spread_down if ob else 0.01,
                slug=market.slug,
                start_time=market.start_time,
                duration_seconds=market.duration_seconds,
            )

            # Shadow eval: use a flat "capital" so Kelly sizing works sanely.
            # Values like consecutive_losses / daily_pnl_pct are stubbed at 0
            # because we don't have a real portfolio; the replay backtester
            # will apply these later from its own simulated state.
            sig = engine.evaluate(
                state=state,
                capital=1000.0,
                consecutive_losses=0,
                daily_pnl_pct=0.0,
                open_positions=0,
                has_position_on_market=False,
            )
            sig.slug = market.slug
            sig.market_start_time = market.start_time
            sig.market_duration = market.duration_seconds
            sig.token_id_yes = market.token_id_up
            sig.token_id_no = market.token_id_down

            self.recorder.record_signal(
                sig,
                chainlink_price=chainlink_price,
                binance_price=prices.get("binance", 0.0),
                ob=ob,
            )

    # ─── Resolution back-fill ─────────────────────────────────────────

    async def _resolution_loop(self) -> None:
        """Every 30s, look up markets whose end_time has passed but no
        resolved_outcome yet, and query past-results until Polymarket
        returns the final outcome. Usually available within 60s of close."""
        while self._running:
            try:
                now = time.time()
                pending = self.recorder.unresolved_markets(now - 30)
                for m in pending:
                    try:
                        outcome = await self.polymarket_feed.fetch_market_outcome(
                            slug=m["slug"],
                            start_time=m["start_time"],
                            duration=m["duration_seconds"],
                        )
                    except Exception as exc:
                        log.debug("[Resolution] fetch error %s: %s", m["slug"], exc)
                        continue
                    if outcome not in ("up", "down"):
                        continue
                    # Resolved price = current chainlink for that asset.
                    # The real resolution price is the chainlink snapshot at
                    # end_time, which Polymarket past-results closePrice
                    # gives us — so fetch that separately.
                    resolved_price = await self._fetch_resolved_price(m)
                    self.recorder.mark_resolved(
                        market_id=m["market_id"],
                        outcome=outcome,
                        resolved_price=resolved_price,
                    )
            except Exception as exc:
                log.error("[Resolution] Loop error: %s", exc, exc_info=True)
            await asyncio.sleep(30)

    # ─── Gamma dynamic-field poll (market_states) ─────────────────────

    async def _gamma_poll_loop(self) -> None:
        """Every GAMMA_POLL_INTERVAL seconds, re-fetch /events for each
        active market and record the dynamic fields (bestBid/bestAsk/
        lastTradePrice/volume24hrClob/openInterest/umaResolutionStatus/etc).

        This complements orderbook_snapshots (which only has BBO) with
        the market-wide context from Gamma (liquidity, volume, rewards
        params, resolution status) that can't be gotten from /book alone.
        """
        await asyncio.sleep(15)  # let discovery run first
        while self._running:
            try:
                markets = list(self._active_markets.values())
                for m in markets:
                    if m.is_expired:
                        continue
                    await self._fetch_gamma_state(m)
                    # Small breather between requests so we don't hammer Gamma.
                    await asyncio.sleep(0.25)
            except Exception as exc:
                log.error("[GammaPoll] Loop error: %s", exc, exc_info=True)
            await asyncio.sleep(GAMMA_POLL_INTERVAL)

    async def _fetch_gamma_state(self, m: MarketInfo) -> None:
        if self._gamma_session is None:
            return
        try:
            async with self._gamma_session.get(
                f"{config.polymarket.gamma_url}/events",
                params={"slug": m.slug},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return
                events = await resp.json()
        except Exception as exc:
            log.debug("[GammaPoll] fetch %s error: %s", m.slug, exc)
            return
        if not events:
            return
        event = events[0]
        markets = event.get("markets", []) or []
        if not markets:
            return
        g = markets[0]

        def _f(key):
            v = g.get(key)
            if v in (None, "", "null"):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        fields = {
            "best_bid":          _f("bestBid"),
            "best_ask":          _f("bestAsk"),
            "last_trade_price":  _f("lastTradePrice"),
            "spread":            _f("spread"),
            "volume_clob":       _f("volumeClob"),
            "volume_24h_clob":   _f("volume24hrClob"),
            "liquidity_clob":    _f("liquidityClob"),
            "open_interest":     _f("openInterest") or _f("openInterestAmount"),
            "one_hour_change":   _f("oneHourPriceChange"),
            "one_day_change":    _f("oneDayPriceChange"),
            "accepting_orders":  bool(g.get("acceptingOrders", False)),
            "uma_status":        g.get("umaResolutionStatus"),
            "closed_time":       _parse_iso(g.get("closedTime", "")),
            "rewards_min_size":  _f("rewardsMinSize"),
            "rewards_max_spread": _f("rewardsMaxSpread"),
            "min_tick_size":     _f("orderPriceMinTickSize"),
            "min_order_size":    _f("orderMinSize"),
            "maker_fee_bps":     _f("makerBaseFee"),
            "taker_fee_bps":     _f("takerBaseFee"),
            "neg_risk":          bool(g.get("negRisk", False)
                                      or event.get("negRisk", False)),
        }
        self.recorder.record_market_state(
            market_id=m.condition_id, timestamp=time.time(), fields=fields,
        )

    async def _fetch_resolved_price(self, m: dict) -> float:
        """Closing price at end_time — re-use past-results API via the feed.
        The feed's _fetch_price_to_beat fetches the exact chainlink
        snapshot for a given start_time, so calling it with start=end_time
        of the expired market returns that market's CLOSE price."""
        try:
            # end_time of this market = start_time of the NEXT window
            next_start = m["start_time"] + m["duration_seconds"]
            return await self.polymarket_feed._fetch_price_to_beat(
                slug=m["slug"],
                start_time=next_start,
                duration=m["duration_seconds"],
            )
        except Exception:
            return 0.0

    # ─── Stats loop ───────────────────────────────────────────────────

    async def _stats_loop(self) -> None:
        await asyncio.sleep(10)
        while self._running:
            try:
                s = self.recorder.stats()
                log.info(
                    "[Stats] markets=%d  ob=%d  deltas=%d  trades=%d  "
                    "mstates=%d  ticks=%d  signals=%d  ws_raw=%d  "
                    "| active=%d  ws_msgs=%d  db=%.1fMB",
                    s.get("markets", 0),
                    s.get("orderbook_snapshots", 0),
                    s.get("orderbook_deltas", 0),
                    s.get("trades", 0),
                    s.get("market_states", 0),
                    s.get("price_ticks", 0),
                    s.get("shadow_signals", 0),
                    s.get("ws_events", 0),
                    len(self._active_markets),
                    self.poly_ws.messages_received,
                    s.get("db_size_mb", 0.0),
                )
            except Exception as exc:
                log.debug("[Stats] error: %s", exc)
            await asyncio.sleep(self._stats_interval)


def main():
    parser = argparse.ArgumentParser(description="Polymarket data collector (shadow mode)")
    parser.add_argument(
        "--db", type=str, default="data/collector.db",
        help="SQLite output path (default: data/collector.db)",
    )
    parser.add_argument(
        "--stats-interval", type=int, default=300,
        help="Seconds between stats log lines (default: 300)",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def _run():
        loop = asyncio.get_event_loop()

        def _exc_handler(loop, context):
            exc = context.get("exception")
            msg = context.get("message", "")
            if isinstance(exc, OSError) and getattr(exc, "errno", None) == 11001:
                log.debug("[Net] DNS lookup échoué — transitoire")
                return
            if "Future exception was never retrieved" in msg:
                inner = exc or context.get("future")
                if isinstance(inner, OSError):
                    log.debug("[Net] WS future ignoré: %s", inner)
                    return
            if exc:
                log.warning("[Asyncio] %s — %s", type(exc).__name__, exc)
            else:
                log.warning("[Asyncio] %s", msg)

        loop.set_exception_handler(_exc_handler)
        collector = Collector(db_path=args.db, stats_interval=args.stats_interval)
        try:
            await collector.start()
        except (KeyboardInterrupt, asyncio.CancelledError):
            await collector.shutdown()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
