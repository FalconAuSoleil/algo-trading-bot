"""BTC Sniper v3.7 - Multi-Asset Multi-Strategy Orchestrator.

v3.7: Supports multiple assets (BTC, ETH, SOL, XRP) with per-asset
Chainlink feeds, Binance feeds, and optimized signal engines.

Coordinates all components: feeds, multi-strategy signal engine,
trading, trend tracking, and dashboard.
"""

from __future__ import annotations

import asyncio
import os
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
from src.engine.signal import SignalEngine, MarketState, Signal
from src.engine.trend import market_trend
from src.trading.portfolio import Portfolio
from src.trading.paper import PaperTrader
from src.trading.live import LiveTrader
from src.dashboard.app import app as dashboard_app, dashboard_state

log = setup_logger("main", config.log_level)


class Orchestrator:
    """Main system coordinator with multi-asset support."""

    def __init__(self):
        self.db = Database(config.db_path)

        # ── Multi-asset feeds ────────────────────────────────────────────────────────
        # v3.7: Create per-asset Chainlink and Binance feeds for all enabled assets.
        self._chainlink_feeds: dict[str, ChainlinkFeed] = {}
        self._binance_feeds: dict[str, BinanceFeed] = {}
        self._signal_engines: dict[str, SignalEngine] = {}
        self._asset_prices: dict[str, dict] = {}

        for asset in config.assets:
            if not asset.enabled:
                continue

            # Chainlink feed for this asset
            cl_feed = ChainlinkFeed(
                symbol=asset.symbol,
                contract_address=asset.chainlink_address,
                binance_symbol=asset.binance_symbol,
                on_price=self._on_price,
            )
            self._chainlink_feeds[asset.symbol] = cl_feed

            # Binance feed for this asset
            # Construct WS URL: wss://stream.binance.com:9443/ws/{symbol_lower}usdt@trade
            binance_ws_url = (
                f"wss://stream.binance.com:9443/ws/{asset.binance_symbol.lower()}@trade"
            )
            bn_feed = BinanceFeed(
                symbol=asset.symbol,
                url=binance_ws_url,
                on_price=self._on_price,
            )
            self._binance_feeds[asset.symbol] = bn_feed

            # Signal engine for this asset
            signal_engine = SignalEngine(
                cfg=config.signal,
                sigma_fallback=asset.sigma_fallback,
                delta_min_abs=asset.delta_min_abs,
            )
            self._signal_engines[asset.symbol] = signal_engine

            # Initialize price state
            self._asset_prices[asset.symbol] = {
                "chainlink": 0.0,
                "binance": 0.0,
                "chainlink_ts": 0.0,
            }

            log.info(
                "[Multi-Asset] Initialized %s | CL_addr=%s | "
                "sigma_fallback=%.2e | delta_min_abs=%.4f",
                asset.symbol, asset.chainlink_address[:10] + "...",
                asset.sigma_fallback, asset.delta_min_abs,
            )

        # Polymarket feed (multi-asset)
        asset_prefixes = tuple(
            asset.polymarket_prefix for asset in config.assets if asset.enabled
        )
        self.polymarket_feed = PolymarketFeed(
            gamma_url=config.polymarket.gamma_url,
            clob_url=config.polymarket.clob_url,
            clob_ws_url=config.polymarket.clob_ws_url,
            asset_prefixes=asset_prefixes,
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

        # Internal state
        self._active_markets: dict[str, MarketInfo] = {}
        self._orderbooks: dict = {}
        self._running = False
        self._snapshot_interval = 60

        # Maps trade_id → strategy_used, for record_result() after resolution
        self._strategy_by_trade: dict[int, str] = {}

    async def start(self) -> None:
        log.info("=" * 60)
        log.info("  BTC SNIPER v3.7 - Multi-Asset Multi-Strategy Engine")
        log.info("  Assets: %s", ", ".join(a.symbol for a in config.assets if a.enabled))
        log.info("  Strategies: ChainlinkArb | Momentum | MeanReversion")
        log.info("  Mode: %s", config.trading_mode.upper())
        log.info("  Capital: $%.2f", config.paper_initial_balance)
        log.info("=" * 60)

        Path("data").mkdir(exist_ok=True)
        await self.db.connect()

        db_state = await self.db.load_portfolio_state(config.trading_mode)
        self.portfolio.restore_from_db(db_state)

        if hasattr(self.trader, 'restore_pending'):
            await self.trader.restore_pending()

        dashboard_state.set_db(self.db)
        await dashboard_state.refresh_from_db()
        await dashboard_state.update_portfolio(self.portfolio.get_stats())

        if config.is_live and hasattr(self.trader, "start"):
            await self.trader.start()

        self._running = True

        # Create tasks for all feeds
        tasks = [
            asyncio.create_task(self.polymarket_feed.start(), name="polymarket_feed"),
            asyncio.create_task(self._signal_loop(), name="signal_loop"),
            asyncio.create_task(self._resolution_loop(), name="resolution_loop"),
            asyncio.create_task(self._snapshot_loop(), name="snapshot_loop"),
            asyncio.create_task(self._dashboard_server(), name="dashboard"),
        ]

        # Add per-asset feed tasks
        for symbol, cl_feed in self._chainlink_feeds.items():
            tasks.append(
                asyncio.create_task(cl_feed.start(), name=f"chainlink_feed_{symbol}")
            )
        for symbol, bn_feed in self._binance_feeds.items():
            tasks.append(
                asyncio.create_task(bn_feed.start(), name=f"binance_feed_{symbol}")
            )

        if sys.platform != "win32":
            import signal
            loop = asyncio.get_event_loop()
            for sig_name in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig_name,
                    lambda: asyncio.create_task(self.shutdown()),
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

        # Stop all feeds
        await self.polymarket_feed.stop()
        for cl_feed in self._chainlink_feeds.values():
            await cl_feed.stop()
        for bn_feed in self._binance_feeds.values():
            await bn_feed.stop()

        if config.is_live and hasattr(self.trader, "stop"):
            await self.trader.stop()
        await self._save_snapshot()
        await self.db.close()
        log.info("[Main] Shutdown complete")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    # ── Price callbacks ──────────────────────────────────────────────────────────────────

    async def _on_price(self, source: str, price: float, timestamp: float, symbol: str = "BTC") -> None:
        """Handle price updates from feeds. v3.7: routes by symbol."""
        if symbol not in self._asset_prices:
            return

        if source == "binance":
            self._asset_prices[symbol]["binance"] = price
            if symbol in self._signal_engines:
                self._signal_engines[symbol].update_price(price, timestamp)
        elif source in ("chainlink", "chainlink_binance_fallback"):
            self._asset_prices[symbol]["chainlink"] = price
            self._asset_prices[symbol]["chainlink_ts"] = timestamp
            if symbol in self._signal_engines:
                self._signal_engines[symbol].update_chainlink_price(price, timestamp)

        await dashboard_state.update_feeds({
            "binance": any(p["binance"] > 0 for p in self._asset_prices.values()),
            "chainlink": any(p["chainlink"] > 0 for p in self._asset_prices.values()),
            "polymarket": len(self._active_markets) > 0,
        })

    async def _on_market_update(self, markets: dict[str, MarketInfo]) -> None:
        self._active_markets = markets
        await dashboard_state.update_feeds({
            "binance": any(p["binance"] > 0 for p in self._asset_prices.values()),
            "chainlink": any(p["chainlink"] > 0 for p in self._asset_prices.values()),
            "polymarket": len(markets) > 0,
        })

    async def _on_orderbook_update(self, cid: str, ob) -> None:
        self._orderbooks[cid] = ob

    # ── Core loops ────────────────────────────────────────────────────────────────────────

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

        # Extract asset symbol from market slug to route to correct signal engine
        markets_snapshot = list(self._active_markets.items())

        for cid, market in markets_snapshot:
            if market.is_expired:
                continue
            if market.reference_price <= 0:
                continue

            # Parse asset from slug: "btc-updown-5m-..." → "BTC"
            slug_parts = market.slug.split("-")
            asset_prefix = slug_parts[0].upper() if slug_parts else "BTC"

            # Find the asset config for this slug
            asset_config = None
            for ac in config.assets:
                if ac.polymarket_prefix.upper() == asset_prefix:
                    asset_config = ac
                    break

            if asset_config is None or not asset_config.enabled:
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
                btc_chainlink=chainlink_price,  # named btc_* for compat, but is asset-generic
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

            # Populate market-level metadata
            sig.slug = market.slug
            sig.market_start_time = market.start_time
            sig.market_duration = market.duration_seconds
            sig.token_id_yes = market.token_id_up
            sig.token_id_no = market.token_id_down

            filters = ",".join(sig.filter_reasons) if sig.filter_reasons else "ALL_PASS"
            log.info(
                "[Signal] %s T-%ds d=%.3f%% P=%.2f edge=%.3f -> %s [%s] | %s | "
                "strategy=%s | CL_age=%.0fs",
                market.slug[-14:],
                int(sig.time_remaining_sec),
                sig.delta_chainlink * 100,
                sig.p_true,
                sig.edge,
                sig.action,
                filters,
                sig.status,
                sig.strategy_used,
                sig.micro.oracle_age_sec,
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
                trade_id = await self.trader.execute(sig)
                if trade_id:
                    self._strategy_by_trade[trade_id] = sig.strategy_used
                    full = await self.db.get_trade(trade_id)
                    if full:
                        await dashboard_state.update_trade(full)
                    await dashboard_state.update_portfolio(
                        self.portfolio.get_stats()
                    )
                    self._signal_engines[symbol].reset_market_stability(market.slug)

        await dashboard_state.update_portfolio(self.portfolio.get_stats())

    async def _resolution_loop(self) -> None:
        while self._running:
            try:
                if self.trader.pending_count > 0:
                    resolved = await self.trader.check_resolutions(
                        # For multi-asset, pass only the price of the asset being resolved
                        # The trader will determine the asset from market slug/ID
                        any(p["chainlink"] for p in self._asset_prices.values()) or 0.0,
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

                        # Find the signal engine for this trade's asset
                        # by parsing the market slug in the trade record
                        symbol = "BTC"  # default
                        if "slug" in r:
                            slug_asset = r["slug"].split("-")[0].upper()
                            for ac in config.assets:
                                if ac.polymarket_prefix.upper() == slug_asset:
                                    symbol = ac.symbol
                                    break

                        if symbol in self._signal_engines:
                            self._signal_engines[symbol].record_result(strategy, won)

                        if side == "YES":
                            market_direction = "up" if won else "down"
                        else:
                            market_direction = "down" if won else "up"

                        try:
                            market_trend.record(market_direction)
                        except Exception:
                            pass

                        if r.get("duration", 300) == 300:
                            try:
                                if symbol in self._signal_engines:
                                    self._signal_engines[symbol].record_5m_resolution(
                                        chainlink_price=r["btc_price"],
                                        reference_price=r["ref_price"],
                                        direction=market_direction,
                                    )
                            except Exception as exc:
                                log.debug(
                                    "[CrossMarket] record error: %s", exc
                                )

                        log.info(
                            "[Resolution] trade=%d strategy=%s won=%s "
                            "trend_direction=%s | pnl=$%.2f | symbol=%s",
                            trade_id, strategy, won,
                            market_direction, r["pnl"], symbol,
                        )

                        full = await self.db.get_trade(trade_id)
                        if full:
                            await dashboard_state.update_trade(full)
                        else:
                            await dashboard_state.update_trade(r)
                        await dashboard_state.update_portfolio(
                            self.portfolio.get_stats()
                        )
            except Exception as exc:
                log.error("[Resolution] Loop error: %s", exc)
            await asyncio.sleep(3)

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
        server = uvicorn.Server(server_config)
        await server.serve()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="BTC Sniper v3.7 - Multi-Asset Multi-Strategy Engine"
    )
    parser.add_argument("--mode", choices=["paper", "live", "collect"], default=None)
    parser.add_argument("--balance", type=float, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.mode:
        config.trading_mode = args.mode
    if args.balance:
        config.paper_initial_balance = args.balance
    if args.port:
        config.dashboard = config.dashboard.__class__(
            host=config.dashboard.host, port=args.port,
        )

    orchestrator = Orchestrator()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(orchestrator.start())


if __name__ == "__main__":
    main()
