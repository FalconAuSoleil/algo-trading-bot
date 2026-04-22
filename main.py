"""BTC Sniper v4.1 - Multi-Asset Multi-Strategy Orchestrator.

v4.1 changes:
  - SignalEngine now receives asset_symbol for per-asset routing:
      BTC 15m  → _BTCStabilizationEngine (24/7, T=60-180s)
      BTC 5m   → ChainlinkArb (peak hours Mon-Fri 08-18h ET)
      ETH/SOL/XRP → ChainlinkArb (peak hours Mon-Fri 08-18h ET)

v3.8 fixes:
  - ETH/SOL/XRP restricted to 15m markets only (Polymarket 5m API
    returns error for non-BTC assets — verified 2026-03-21).
  - PolymarketFeed now uses per-asset interval dict.
  - Resolution loop bug fixed: was passing bool instead of float price.
  - Quant param fixes: source_coherence_max, time_max_15m,
    stability_min_samples, stability_edge_cv_max.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import config
from src.utils.logger import setup_logger
from src.utils.db import Database
from src.feeds.binance import BinanceFeed
from src.feeds.chainlink import ChainlinkFeed
from src.feeds.polymarket import PolymarketFeed, MarketInfo
from src.engine.signal import SignalEngine, MarketState, Signal, p_brownian
from src.engine.trend import market_trend
from src.trading.portfolio import Portfolio
from src.trading.paper import PaperTrader
from src.trading.live import LiveTrader
from src.dashboard.app import app as dashboard_app, dashboard_state
from src.collector.recorder import CollectorRecorder
from src.collector.polymarket_ws import PolymarketMarketWS

import aiohttp

log = setup_logger("main", config.log_level)

# Default collector DB path — separate from the bot's trade DB so the
# collector's firehose doesn't bloat the clean trade history.
DEFAULT_COLLECTOR_DB = "data/collector.db"
GAMMA_POLL_INTERVAL = 30.0


class Orchestrator:
    """Main system coordinator with multi-asset support.

    v5.3: embeds the data collector so a single `python main.py` run
    both trades AND records everything to data/collector.db in the
    background (unless --no-collect is passed).
    """

    def __init__(
        self,
        collect_enabled: bool = True,
        collector_db_path: str = DEFAULT_COLLECTOR_DB,
    ):
        self.db = Database(config.db_path)

        # ── Collector (shadow-mode recorder) ─────────────────────────────
        # Runs in parallel with trading, listening on the same feeds.
        # Writes to an independent SQLite so the bot's DB stays clean.
        self._collect_enabled = collect_enabled
        if collect_enabled:
            self.recorder: CollectorRecorder | None = CollectorRecorder(
                db_path=collector_db_path
            )
            self.poly_ws: PolymarketMarketWS | None = PolymarketMarketWS(
                recorder=self.recorder
            )
            self._last_tick_log: dict[str, int] = {}
            self._gamma_session: aiohttp.ClientSession | None = None
        else:
            self.recorder = None
            self.poly_ws = None
            self._last_tick_log = {}
            self._gamma_session = None

        # ── Multi-asset feeds ────────────────────────────────────────────────
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

            # v4.1: pass asset_symbol so the engine can route correctly
            # BTC → BTCStabilization (15m) + ChainlinkArb (5m, peak hours)
            # ETH/SOL/XRP → ChainlinkArb (15m only, peak hours)
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

            log.info(
                "[Multi-Asset] Initialized %s | intervals=%s | "
                "sigma_fallback=%.2e | delta_min_abs=%.4f",
                asset.symbol, asset.supported_intervals,
                asset.sigma_fallback, asset.delta_min_abs,
            )

        # v3.8: build per-asset interval dict for PolymarketFeed
        # ETH/SOL/XRP get (900,) only — 5m past-results API unsupported
        asset_intervals = {
            asset.polymarket_prefix: asset.supported_intervals
            for asset in config.assets
            if asset.enabled
        }
        self.polymarket_feed = PolymarketFeed(
            gamma_url=config.polymarket.gamma_url,
            clob_url=config.polymarket.clob_url,
            clob_ws_url=config.polymarket.clob_ws_url,
            asset_intervals=asset_intervals,
            on_market_update=self._on_market_update,
            on_orderbook_update=self._on_orderbook_update,
        )

        # Portfolio + Trader
        self.portfolio = Portfolio(
            initial_balance=config.paper_initial_balance,
            mode=config.trading_mode,
        )
        if config.is_paper:
            self.trader = PaperTrader(self.portfolio, self.db)
        else:
            self.trader = LiveTrader(
                self.portfolio, self.db, config.polymarket
            )

        self._active_markets: dict[str, MarketInfo] = {}
        self._orderbooks: dict = {}
        self._running = False
        self._snapshot_interval = 60
        self._strategy_by_trade: dict[int, str] = {}
        # v5.2: staged entry retired — kept as empty dict for back-compat
        # cooldown: market_id → expiry timestamp (no retry before this time)
        self._market_cooldowns: dict[str, float] = {}

    async def start(self) -> None:
        log.info("=" * 60)
        log.info("  BTC SNIPER v4.1 - Multi-Asset Multi-Strategy Engine")
        enabled = [a for a in config.assets if a.enabled]
        for a in enabled:
            log.info(
                "  %s: %s | CL=%s...",
                a.symbol,
                "5m+15m" if len(a.supported_intervals) > 1 else "15m only",
                a.chainlink_address[:10],
            )
        if config.is_live:
            log.info("  Mode: LIVE | Capital: syncing depuis wallet on-chain...")
        else:
            log.info("  Mode: PAPER | Capital: $%.2f", config.paper_initial_balance)
        log.info("  BTC 15m: BTCStabilization 24/7 (T=60-180s, 63-80¢)")
        log.info("  BTC 5m + ETH/SOL/XRP: ChainlinkArb peak hours only (Mon-Fri 08-18h ET)")
        log.info("=" * 60)

        Path("data").mkdir(exist_ok=True)
        await self.db.connect()

        # Start the embedded collector if enabled — non-blocking, won't
        # impact trading if it fails (guarded below).
        if self.recorder is not None:
            try:
                await self.recorder.start()
                self._gamma_session = aiohttp.ClientSession()
                log.info(
                    "  Collector: ENABLED -> %s (unlimited retention)",
                    self.recorder.db_path,
                )
            except Exception as exc:
                log.error(
                    "  Collector: FAILED to start (%s) — trading will "
                    "continue without data collection.", exc,
                )
                self.recorder = None
                self.poly_ws = None
        else:
            log.info("  Collector: DISABLED (--no-collect)")

        db_state = await self.db.load_portfolio_state(config.trading_mode)
        self.portfolio.restore_from_db(db_state)

        if hasattr(self.trader, 'restore_pending'):
            await self.trader.restore_pending()

        dashboard_state.set_db(self.db)
        await dashboard_state.refresh_from_db()
        await dashboard_state.update_portfolio(self.portfolio.get_stats())

        if config.is_live and hasattr(self.trader, "start"):
            await self.trader.start()
            dashboard_state.set_trader(self.trader)

        self._running = True

        tasks = [
            asyncio.create_task(self.polymarket_feed.start(), name="polymarket_feed"),
            asyncio.create_task(self._signal_loop(), name="signal_loop"),
            asyncio.create_task(self._resolution_loop(), name="resolution_loop"),
            asyncio.create_task(self._snapshot_loop(), name="snapshot_loop"),
            asyncio.create_task(self._monitor_positions(), name="position_monitor"),
            asyncio.create_task(self._dashboard_server(), name="dashboard"),
        ]
        if config.is_live:
            tasks.append(asyncio.create_task(
                self._wallet_balance_loop(), name="wallet_balance"
            ))
        for symbol, cl_feed in self._chainlink_feeds.items():
            tasks.append(asyncio.create_task(cl_feed.start(), name=f"chainlink_{symbol}"))
        for symbol, bn_feed in self._binance_feeds.items():
            tasks.append(asyncio.create_task(bn_feed.start(), name=f"binance_{symbol}"))

        # Collector background tasks (piggyback on the same feeds — no
        # duplicate Binance/Chainlink connections). Only the Polymarket
        # WebSocket and the Gamma snapshot poller are extra.
        if self.recorder is not None and self.poly_ws is not None:
            tasks.append(asyncio.create_task(
                self.poly_ws.run(), name="collector_poly_ws"
            ))
            tasks.append(asyncio.create_task(
                self._collector_gamma_poll_loop(), name="collector_gamma_poll"
            ))
            tasks.append(asyncio.create_task(
                self._collector_resolution_loop(), name="collector_resolution"
            ))
            tasks.append(asyncio.create_task(
                self._collector_stats_loop(), name="collector_stats"
            ))

        if sys.platform != "win32":
            import signal
            loop = asyncio.get_event_loop()
            for sig_name in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig_name, lambda: asyncio.create_task(self.shutdown())
                )

        log.info("[Main] Dashboard at http://localhost:%d", config.dashboard.port)
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        log.info("[Main] Shutting down...")
        if self.poly_ws is not None:
            await self.poly_ws.stop()
        await self.polymarket_feed.stop()
        for cl_feed in self._chainlink_feeds.values():
            await cl_feed.stop()
        for bn_feed in self._binance_feeds.values():
            await bn_feed.stop()
        if config.is_live and hasattr(self.trader, "stop"):
            await self.trader.stop()
        await self._save_snapshot()
        if self._gamma_session is not None:
            await self._gamma_session.close()
        if self.recorder is not None:
            await self.recorder.stop()
        await self.db.close()
        log.info("[Main] Shutdown complete")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

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

        # Collector: downsample to 1 row per (asset, whole second)
        if self.recorder is not None:
            sec = int(time.time())
            if self._last_tick_log.get(symbol, 0) != sec:
                self._last_tick_log[symbol] = sec
                self.recorder.record_price_tick(
                    asset=symbol,
                    timestamp=float(sec),
                    binance_price=bucket["binance"],
                    chainlink_price=bucket["chainlink"],
                    chainlink_ts=bucket["chainlink_ts"],
                )

        await dashboard_state.update_feeds({
            "binance": any(p["binance"] > 0 for p in self._asset_prices.values()),
            "chainlink": any(p["chainlink"] > 0 for p in self._asset_prices.values()),
            "polymarket": len(self._active_markets) > 0,
        })

    async def _on_market_update(self, markets: dict[str, MarketInfo]) -> None:
        self._active_markets = markets
        if self.recorder is not None and self.poly_ws is not None:
            for m in markets.values():
                self.recorder.record_market(m)
            asset_ids: set[str] = set()
            for m in markets.values():
                if m.token_id_up:
                    asset_ids.add(m.token_id_up)
                if m.token_id_down:
                    asset_ids.add(m.token_id_down)
            self.poly_ws.set_assets(asset_ids)
        await dashboard_state.update_feeds({
            "binance": any(p["binance"] > 0 for p in self._asset_prices.values()),
            "chainlink": any(p["chainlink"] > 0 for p in self._asset_prices.values()),
            "polymarket": len(markets) > 0,
        })

    async def _on_orderbook_update(self, cid: str, ob) -> None:
        self._orderbooks[cid] = ob
        if self.recorder is not None:
            self.recorder.record_orderbook(cid, ob)

    async def _signal_loop(self) -> None:
        while self._running:
            try:
                await self._evaluate_markets()
            except Exception as exc:
                log.error("[Signal] Loop error: %s", exc, exc_info=True)
            await asyncio.sleep(2)

    async def _evaluate_markets(self) -> None:
        if not self._active_markets:
            return

        markets_snapshot = list(self._active_markets.items())

        for cid, market in markets_snapshot:
            if market.is_expired:
                continue
            if market.reference_price <= 0:
                continue

            slug_parts = market.slug.split("-")
            asset_prefix = slug_parts[0].upper() if slug_parts else "BTC"

            asset_config = None
            for ac in config.assets:
                if ac.polymarket_prefix.upper() == asset_prefix:
                    asset_config = ac
                    break

            if asset_config is None or not asset_config.enabled:
                if asset_config is None:
                    log.warning(
                        "[Signal] Unknown asset prefix '%s' in slug %s — skipping",
                        asset_prefix, market.slug,
                    )
                continue

            symbol = asset_config.symbol
            if symbol not in self._signal_engines:
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

            consecutive_losses = await self.db.get_consecutive_losses(config.trading_mode)

            sig = self._signal_engines[symbol].evaluate(
                state=state,
                capital=self.portfolio.balance,
                consecutive_losses=consecutive_losses,
                daily_pnl_pct=self.portfolio.daily_pnl_pct,
                open_positions=self.portfolio.open_position_count,
                has_position_on_market=self.portfolio.has_position_on_market(cid),
            )

            sig.slug = market.slug
            sig.market_start_time = market.start_time
            sig.market_duration = market.duration_seconds
            sig.token_id_yes = market.token_id_up
            sig.token_id_no = market.token_id_down

            if self.recorder is not None:
                self.recorder.record_signal(
                    sig,
                    chainlink_price=chainlink_price,
                    binance_price=prices.get("binance", 0.0),
                    ob=ob,
                )

            filters = ",".join(sig.filter_reasons) if sig.filter_reasons else "ALL_PASS"
            log.info(
                "[Signal] %s T-%ds d=%.3f%% P=%.2f edge=%.3f -> %s [%s] | %s | "
                "strategy=%s | CL_age=%.0fs",
                market.slug[-14:], int(sig.time_remaining_sec),
                sig.delta_chainlink * 100, sig.p_true, sig.edge,
                sig.action, filters, sig.status,
                sig.strategy_used, sig.micro.oracle_age_sec,
            )

            await self.db.insert_signal({
                "timestamp": sig.timestamp,
                "market_id": sig.market_id,
                "btc_binance": sig.btc_binance,
                "btc_chainlink": sig.btc_chainlink,
                "reference_price": sig.reference_price,
                "delta_chainlink": sig.delta_chainlink,
                "delta_binance": sig.delta_binance,
                "sigma": sig.sigma,
                "time_remaining_sec": sig.time_remaining_sec,
                "p_true": sig.p_true,
                "p_market": sig.p_market,
                "edge": sig.edge,
                "filters_passed": 1 if sig.filters_passed else 0,
                "filter_details": ",".join(sig.filter_reasons),
                "action": sig.action,
                "oracle_age_sec": round(sig.micro.oracle_age_sec, 1),
            })

            await dashboard_state.update_signal({
                "market_id": sig.market_id,
                "slug": sig.slug,
                "action": sig.action,
                "side": sig.side,
                "btc_chainlink": sig.btc_chainlink,
                "btc_binance": sig.btc_binance,
                "reference_price": sig.reference_price,
                "delta_chainlink": sig.delta_chainlink,
                "delta_binance": sig.delta_binance,
                "sigma": sig.sigma,
                "time_remaining_sec": sig.time_remaining_sec,
                "p_true": sig.p_true,
                "p_market": sig.p_market,
                "edge": sig.edge,
                "taker_fee": sig.taker_fee,
                "size_usd": sig.size_usd,
                "entry_price": sig.entry_price,
                "filters_passed": sig.filters_passed,
                "filter_reasons": sig.filter_reasons,
                "status": sig.status,
                "confidence": sig.confidence,
                "strategy_used": sig.strategy_used,
                "strategies_agreeing": sig.strategies_agreeing,
                "oracle_age_sec": round(sig.micro.oracle_age_sec, 1),
                "micro": {
                    "chainlink_boost": round(sig.micro.chainlink_edge_boost, 4),
                    "ofi": round(sig.micro.ofi_raw, 4),
                    "kyle_quality": round(sig.micro.kyle_penalty, 4),
                    "hawkes_intensity": round(sig.micro.hawkes_intensity, 4),
                    "stability_ratio": round(sig.micro.stability_ratio, 3),
                    "stability_ok": sig.micro.stability_ok,
                    "taker_fee": round(sig.micro.taker_fee, 5),
                    "source_divergence": round(sig.micro.source_divergence, 6),
                    "time_decay": round(sig.micro.time_decay_factor, 3),
                    "oracle_age_sec": round(sig.micro.oracle_age_sec, 1),
                },
            })

            if sig.action == "BUY":
                # Cooldown: skip if we already tried this market and it failed
                cooldown_until = self._market_cooldowns.get(cid, 0)
                if time.time() < cooldown_until:
                    continue

                # v5.2: staged entry retire. Early exit gere les trades qui
                # tournent mal au lieu de doubler dessus.
                trade_id = await self.trader.execute(sig)

                # Live mode: push rejected/errored trades to dashboard
                # and set cooldown so we don't spam the same market
                if config.is_live and hasattr(self.trader, "pop_rejected_trades"):
                    rejected_ids = self.trader.pop_rejected_trades()
                    if rejected_ids:
                        # Cooldown = rest of the market window (min 60s)
                        cooldown = time.time() + max(60.0, sig.time_remaining_sec)
                        self._market_cooldowns[cid] = cooldown
                        log.info(
                            "[Cooldown] Market %s cooling down for %.0fs after rejection",
                            cid[:12], max(60.0, sig.time_remaining_sec),
                        )
                    for rejected_id in rejected_ids:
                        rejected_row = await self.db.get_trade(rejected_id)
                        if rejected_row:
                            await dashboard_state.update_trade(rejected_row)

                if trade_id:
                    self._strategy_by_trade[trade_id] = sig.strategy_used

                    full = await self.db.get_trade(trade_id)
                    if full:
                        await dashboard_state.update_trade(full)
                    await dashboard_state.update_portfolio(self.portfolio.get_stats())
                    self._signal_engines[symbol].reset_market_stability(market.slug)

        await dashboard_state.update_portfolio(self.portfolio.get_stats())

    async def _resolution_loop(self) -> None:
        while self._running:
            try:
                if self.trader.pending_count > 0:
                    # v3.8 fix: was `any(p["chainlink"] for p in ...)` which
                    # returned a bool (True=1.0 or False=0.0) instead of a price.
                    # Pass actual BTC chainlink price as the fallback price.
                    btc_price = self._asset_prices.get("BTC", {}).get("chainlink", 0.0)
                    resolved = await self.trader.check_resolutions(
                        btc_price,
                        fetch_outcome=self.polymarket_feed.fetch_market_outcome,
                    )
                    for r in resolved:
                        trade_id = r["trade_id"]
                        outcome = r["outcome"]
                        side = r["side"]
                        won = outcome == "won"

                        strategy = self._strategy_by_trade.pop(
                            trade_id, r.get("strategy_used", "chainlink_arb")
                        )

                        # Resolve asset symbol from trade slug
                        symbol = "BTC"
                        if "slug" in r:
                            slug_asset = r["slug"].split("-")[0].upper()
                            for ac in config.assets:
                                if ac.polymarket_prefix.upper() == slug_asset:
                                    symbol = ac.symbol
                                    break

                        if symbol in self._signal_engines:
                            self._signal_engines[symbol].record_result(strategy, won)

                        market_direction = (
                            ("up" if won else "down") if side == "YES"
                            else ("down" if won else "up")
                        )

                        try:
                            market_trend.record(market_direction)
                        except Exception:
                            pass

                        # Cross-market boost: 5m → 15m propagation (same asset)
                        if r.get("duration", 300) == 300 and symbol in self._signal_engines:
                            try:
                                self._signal_engines[symbol].record_5m_resolution(
                                    chainlink_price=r["btc_price"],
                                    reference_price=r["ref_price"],
                                    direction=market_direction,
                                )
                            except Exception as exc:
                                log.debug("[CrossMarket] record error: %s", exc)

                        log.info(
                            "[Resolution] trade=%d symbol=%s strategy=%s "
                            "won=%s pnl=$%.2f",
                            trade_id, symbol, strategy, won, r["pnl"],
                        )

                        full = await self.db.get_trade(trade_id)
                        await dashboard_state.update_trade(full if full else r)
                        await dashboard_state.update_portfolio(self.portfolio.get_stats())

            except Exception as exc:
                log.error("[Resolution] Loop error: %s", exc)
            await asyncio.sleep(3)

    async def _monitor_positions(self) -> None:
        """v5.2: Monitor open positions for early exit. Staged topup retired."""
        while self._running:
            try:
                if self.portfolio.open_position_count > 0:
                    await self._check_early_exits()
            except Exception as exc:
                log.error("[Monitor] Loop error: %s", exc, exc_info=True)
            await asyncio.sleep(4)

    async def _check_early_exits(self) -> None:
        """Check if any open positions should be sold early.

        Two exit modes:
        Mode A — p_true collapse: p_true drops below 35% AND >50% from entry.
        Mode B — delta erosion: delta shrunk >60% from entry AND is tiny (<0.15%)
                 while >70% of market elapsed. Catches trades where the move
                 is fading and a last-second reversal is likely.

        Shared conditions: t_rem > 30s, best_bid > 0.05, position is losing.
        """
        if not config.signal.early_exit_enabled:
            return

        now = time.time()

        for trade_id, pos in list(self.portfolio.open_positions.items()):
            t_rem = pos.market_end_time - now
            if t_rem < config.signal.early_exit_min_t_rem:
                continue  # too late, let it resolve normally

            elapsed = now - pos.entry_time

            # Get current price data for this asset
            symbol = "BTC"
            if pos.slug:
                slug_asset = pos.slug.split("-")[0].upper()
                for ac in config.assets:
                    if ac.polymarket_prefix.upper() == slug_asset:
                        symbol = ac.symbol
                        break

            prices = self._asset_prices.get(symbol, {})
            chainlink_now = prices.get("chainlink", 0.0)
            if chainlink_now <= 0:
                continue

            # Get reference price from pending resolutions
            pending_info = None
            if hasattr(self.trader, '_pending_resolutions'):
                pending_info = self.trader._pending_resolutions.get(trade_id)
            if not pending_info:
                continue

            ref_price = pending_info.get("reference_price", 0.0)
            if ref_price <= 0:
                continue

            # Recalculate delta (direction-adjusted for our side)
            delta_raw = (chainlink_now - ref_price) / ref_price
            delta_now = -delta_raw if pos.side == "NO" else delta_raw

            engine = self._signal_engines.get(symbol)
            sigma = engine.sigma_fallback if engine else 0.005 / (300 ** 0.5)

            p_true_now = p_brownian(abs(delta_now), t_rem, sigma)
            if delta_now < 0:
                p_true_now = 1.0 - p_true_now

            # ── Récupère best_bid en amont (nécessaire pour Mode D) ────
            ob = self._orderbooks.get(pos.market_id)
            if not ob:
                continue
            if pos.side == "YES":
                best_bid = ob.best_bid_yes if hasattr(ob, 'best_bid_yes') else (ob.best_bid_up if hasattr(ob, 'best_bid_up') else 0.0)
            else:
                best_bid = ob.best_bid_no if hasattr(ob, 'best_bid_no') else (ob.best_bid_down if hasattr(ob, 'best_bid_down') else 0.0)

            # ── Check exit modes ────────────────────────────────────────
            sell_reason = None

            # Mode F (v5.3): FORCE SELL absolu en fin de marché. Si le token
            # qu'on a (UP ou DOWN) cote sous 0.30 alors qu'il reste < 2 min,
            # on vend immédiatement — peu importe la liquidité minimale, peu
            # importe le loss_pct. À ce prix et ce timing, c'est perdu à 95%+.
            # Mieux vaut récupérer 20-25¢ par share que 0¢ à l'expiration.
            FORCE_SELL_PRICE = 0.30
            FORCE_SELL_TREM = 120.0
            if (
                best_bid > 0
                and best_bid < FORCE_SELL_PRICE
                and t_rem < FORCE_SELL_TREM
            ):
                sell_reason = "force_sell_endgame"

            # Mode E (v5.4): late-stage forced exit. Calibré via gridsearch
            # replay sur 691 marchés réels (reports/replay_grid_*.json) :
            #   - late_ratio=0.92 gagne la grille (Sharpe > 0.85 à ROI égal)
            #   - late_trem=180s > 90s (+0.5% ROI moyen)
            #   - elapsed_pct=0.70 cohérent avec entry_trem=180s optimal
            # Sans ce mode, le backtest donne -3.2% ROI moyen (held_losses à 0¢).
            if (
                pos.entry_price > 0
                and best_bid > 0
                and best_bid < pos.entry_price * 0.92
                and (t_rem < 180.0 or elapsed > pos.duration_seconds * 0.70)
            ):
                sell_reason = "late_stage_losing"

            # Mode D: token price collapse — le prix du token a chuté de X%
            # par rapport à notre prix d'entrée. Ex: acheté UP à 59¢, bid
            # maintenant à 35¢ = -41%. Le marché dit qu'on va perdre.
            # Déclenche dès 20% du temps écoulé pour couper tôt.
            if (
                pos.entry_price > 0
                and best_bid > 0
                and best_bid < pos.entry_price * 0.70   # token a perdu > 30%
                and elapsed > pos.duration_seconds * 0.20
            ):
                sell_reason = "token_price_collapse"

            # Mode C (v5.2): delta FLIP — le signal s'est inverse par rapport
            # a notre bet. Si on est UP et que delta est devenu negatif (BTC
            # est passe sous le strike), c'est un signal fort que le marche
            # a bouge contre nous. On vend tres vite (juste apres 20% ecoule).
            # Remplace le role pervers que staged-topup jouait : au lieu de
            # doubler quand l'ask plonge, on coupe.
            if sell_reason is None and (
                pos.delta_at_entry != 0
                and delta_now * pos.delta_at_entry < 0  # signe oppose
                and elapsed > pos.duration_seconds * 0.15
            ):
                sell_reason = "delta_flip"

            # Mode A: p_true collapse. Seuils assouplis pour couper plus tot.
            # On coupe des que :
            #   - >30% du market ecoule (avant : 50%)
            #   - p_true_now < early_exit_p_true_floor (0.45 par defaut v5.2)
            #   - p_true_now < p_true_entree x (1 - drop_pct)  (0.35 par defaut)
            if sell_reason is None and (
                elapsed > pos.duration_seconds * 0.30
                and p_true_now < config.signal.early_exit_p_true_floor
                and (
                    pos.p_true_at_entry <= 0
                    or p_true_now < pos.p_true_at_entry * (1.0 - config.signal.early_exit_p_true_drop_pct)
                )
            ):
                sell_reason = "p_true_collapse"

            # Mode B: delta erosion — move is fading, reversal likely
            if sell_reason is None and pos.delta_at_entry != 0:
                delta_entry_abs = abs(pos.delta_at_entry)
                delta_now_abs = abs(delta_now)
                erosion = 1.0 - (delta_now_abs / delta_entry_abs) if delta_entry_abs > 0 else 0.0
                elapsed_pct = elapsed / pos.duration_seconds if pos.duration_seconds > 0 else 0.0

                if (
                    elapsed_pct >= config.signal.early_exit_erosion_min_elapsed_pct
                    and erosion >= config.signal.early_exit_delta_erosion_pct
                    and delta_now_abs < config.signal.early_exit_delta_abs_floor
                ):
                    sell_reason = "delta_erosion"

            if sell_reason is None:
                continue

            # ── Shared conditions: liquidity + position losing ──────────
            # Mode F (force_sell_endgame) bypasse TOUT filtre : on vend même
            # avec liquidité minable, même si on n'a "perdu que" 5%. Le
            # principe : prix < 30¢ en fin de marché = jeu perdu.
            if sell_reason != "force_sell_endgame":
                if best_bid < config.signal.early_exit_min_bid:
                    continue

                current_value = best_bid * pos.shares
                if current_value >= pos.size_usd:
                    continue  # position is profitable, don't sell

                # Soft exit: ne pas vendre si on perd moins de 15% de la mise.
                # Exception : Mode E (late_stage_losing) exit dès 8% de perte.
                loss_pct = 1.0 - (current_value / pos.size_usd) if pos.size_usd > 0 else 0.0
                min_loss_pct = 0.08 if sell_reason == "late_stage_losing" else 0.15
                if loss_pct < min_loss_pct:
                    continue
            else:
                current_value = best_bid * pos.shares

            # ── Sell ────────────────────────────────────────────────────
            delta_entry_abs = abs(pos.delta_at_entry) * 100
            delta_now_abs = abs(delta_now) * 100
            log.warning(
                "[Monitor] EARLY EXIT (%s) trade %d | "
                "delta: %.3f%% -> %.3f%% | p_true: %.1f%% -> %.1f%% | "
                "bid=%.4f | saving ~$%.2f",
                sell_reason, trade_id,
                delta_entry_abs, delta_now_abs,
                pos.p_true_at_entry * 100, p_true_now * 100,
                best_bid, pos.size_usd - current_value,
            )

            pnl = await self.trader.sell_position(
                trade_id, best_bid, sell_reason
            )
            if pnl is not None:
                await dashboard_state.update_portfolio(self.portfolio.get_stats())

    # ─── Embedded collector loops ────────────────────────────────────────
    async def _collector_resolution_loop(self) -> None:
        while self._running:
            try:
                if self.recorder is None:
                    return
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
                        log.debug("[Collector][Res] fetch error %s: %s", m["slug"], exc)
                        continue
                    if outcome not in ("up", "down"):
                        continue
                    try:
                        next_start = m["start_time"] + m["duration_seconds"]
                        resolved_price = await self.polymarket_feed._fetch_price_to_beat(
                            slug=m["slug"],
                            start_time=next_start,
                            duration=m["duration_seconds"],
                        )
                    except Exception:
                        resolved_price = 0.0
                    self.recorder.mark_resolved(
                        market_id=m["market_id"],
                        outcome=outcome,
                        resolved_price=resolved_price,
                    )
            except Exception as exc:
                log.error("[Collector][Res] Loop error: %s", exc)
            await asyncio.sleep(30)

    async def _collector_gamma_poll_loop(self) -> None:
        await asyncio.sleep(15)
        while self._running:
            try:
                if self.recorder is None or self._gamma_session is None:
                    await asyncio.sleep(GAMMA_POLL_INTERVAL)
                    continue
                for m in list(self._active_markets.values()):
                    if m.is_expired:
                        continue
                    await self._collector_fetch_gamma_state(m)
                    await asyncio.sleep(0.25)
            except Exception as exc:
                log.error("[Collector][Gamma] Loop error: %s", exc)
            await asyncio.sleep(GAMMA_POLL_INTERVAL)

    async def _collector_fetch_gamma_state(self, m: MarketInfo) -> None:
        if self._gamma_session is None or self.recorder is None:
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
        except Exception:
            return
        if not events:
            return
        event = events[0]
        mkts = event.get("markets", []) or []
        if not mkts:
            return
        g = mkts[0]

        def _f(key):
            v = g.get(key)
            if v in (None, "", "null"):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _parse_iso(s: str):
            if not s:
                return None
            try:
                from datetime import datetime
                return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
            except ValueError:
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

    async def _collector_stats_loop(self) -> None:
        await asyncio.sleep(30)
        while self._running:
            try:
                if self.recorder is None:
                    return
                s = self.recorder.stats()
                log.info(
                    "[Collector][Stats] markets=%d ob=%d deltas=%d trades=%d "
                    "mstates=%d ticks=%d signals=%d ws_raw=%d | active=%d "
                    "ws_msgs=%d db=%.1fMB",
                    s.get("markets", 0),
                    s.get("orderbook_snapshots", 0),
                    s.get("orderbook_deltas", 0),
                    s.get("trades", 0),
                    s.get("market_states", 0),
                    s.get("price_ticks", 0),
                    s.get("shadow_signals", 0),
                    s.get("ws_events", 0),
                    len(self._active_markets),
                    self.poly_ws.messages_received if self.poly_ws else 0,
                    s.get("db_size_mb", 0.0),
                )
            except Exception as exc:
                log.debug("[Collector][Stats] error: %s", exc)
            await asyncio.sleep(300)

    async def _wallet_balance_loop(self) -> None:
        """Live mode only — poll real USDC.e balance from CLOB every 30s.
        Log au niveau INFO uniquement quand le solde change de plus de $1
        (evite le spam en logs, mais reste visible apres chaque trade).
        """
        await asyncio.sleep(5)  # laisser le client finir l'init d'abord
        _last_logged_balance: float = -1.0
        while self._running:
            try:
                client = getattr(self.trader, "_client", None)
                loop   = getattr(self.trader, "_loop", None)
                if client and loop:
                    def _fetch():
                        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                        return client.get_balance_allowance(
                            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                        )
                    bal  = await loop.run_in_executor(None, _fetch)
                    usdc = float(bal.get("balance", 0.0)) / 1_000_000
                    await dashboard_state.update_wallet_balance(usdc)
                    # Log INFO si le solde a change de plus de $1 depuis le dernier log
                    if abs(usdc - _last_logged_balance) >= 1.0:
                        log.info("[Wallet] USDC.e : $%.2f", usdc)
                        _last_logged_balance = usdc
                    else:
                        log.debug("[Wallet] USDC.e : $%.2f (inchange)", usdc)
            except Exception as exc:
                log.debug("[Wallet] Balance fetch error: %s", exc)
            await asyncio.sleep(30)

    async def _snapshot_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._snapshot_interval)
            await self._save_snapshot()

    async def _save_snapshot(self) -> None:
        try:
            await self.db.insert_snapshot(
                balance=self.portfolio.balance,
                open_positions=self.portfolio.open_position_count,
                daily_pnl=self.portfolio.daily_pnl,
                total_pnl=self.portfolio.total_pnl,
                mode=config.trading_mode,
            )
        except Exception as exc:
            log.error("[Snapshot] Error: %s", exc)

    async def _dashboard_server(self) -> None:
        server_config = uvicorn.Config(
            dashboard_app,
            host=config.dashboard.host,
            port=config.dashboard.port,
            log_level="warning",
            access_log=False,
        )
        await uvicorn.Server(server_config).serve()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTC Sniper v4.1")
    parser.add_argument("--mode", choices=["paper", "live", "collect"], default=None)
    parser.add_argument("--balance", type=float, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-collect", action="store_true",
                        help="Disable the embedded background data collector")
    parser.add_argument("--collector-db", type=str, default=DEFAULT_COLLECTOR_DB,
                        help=f"Collector SQLite path (default: {DEFAULT_COLLECTOR_DB})")
    args = parser.parse_args()

    if args.mode:
        config.trading_mode = args.mode
    if args.balance:
        config.paper_initial_balance = args.balance
    if args.port:
        config.dashboard = config.dashboard.__class__(
            host=config.dashboard.host, port=args.port,
        )

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def _run():
        # Supprime les tracebacks internes d'aiohttp sur gaierror (DNS flap) :
        # ces erreurs viennent de tâches "shielded" qu'on ne peut pas catch
        # normalement. On les downgrade en WARNING propre sans stacktrace.
        loop = asyncio.get_event_loop()

        def _asyncio_exception_handler(loop, context):
            exc = context.get("exception")
            msg = context.get("message", "")
            # gaierror = DNS lookup échoué pendant reconnect — transitoire, pas fatal
            if isinstance(exc, OSError) and getattr(exc, "errno", None) == 11001:
                log.debug("[Net] DNS lookup échoué (reconnect en cours) — transitoire")
                return
            # Filtre les "Future exception was never retrieved" liés aux WS
            if "Future exception was never retrieved" in msg:
                inner = exc or context.get("future")
                if isinstance(inner, OSError):
                    log.debug("[Net] WS future ignoré: %s", inner)
                    return
            # Tout le reste → log WARNING propre sans spammer le terminal
            if exc:
                log.warning("[Asyncio] Exception non rattrapée: %s — %s", type(exc).__name__, exc)
            else:
                log.warning("[Asyncio] %s", msg)

        loop.set_exception_handler(_asyncio_exception_handler)
        await Orchestrator(
            collect_enabled=not args.no_collect,
            collector_db_path=args.collector_db,
        ).start()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
