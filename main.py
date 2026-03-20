"""BTC Sniper v3 - Multi-Strategy Orchestrator.

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
from src.engine.trend import market_trend          # module-level singleton
from src.trading.portfolio import Portfolio
from src.trading.paper import PaperTrader
from src.trading.live import LiveTrader
from src.dashboard.app import app as dashboard_app, dashboard_state

log = setup_logger("main", config.log_level)


class Orchestrator:
    """Main system coordinator."""

    def __init__(self):
        self.db = Database(config.db_path)

        # Feeds
        self.binance_feed = BinanceFeed(
            url=config.binance.ws_url,
            on_price=self._on_price,
        )
        self.chainlink_feed = ChainlinkFeed(
            on_price=self._on_price,
        )
        self.polymarket_feed = PolymarketFeed(
            gamma_url=config.polymarket.gamma_url,
            clob_url=config.polymarket.clob_url,
            clob_ws_url=config.polymarket.clob_ws_url,
            on_market_update=self._on_market_update,
            on_orderbook_update=self._on_orderbook_update,
        )

        # Multi-strategy signal engine
        self.signal_engine = SignalEngine(config.signal)

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
        self._btc_binance: float = 0.0
        self._btc_chainlink: float = 0.0
        self._active_markets: dict[str, MarketInfo] = {}
        self._orderbooks: dict = {}
        self._running = False
        self._snapshot_interval = 60

        # Maps trade_id → strategy_used, for record_result() after resolution
        self._strategy_by_trade: dict[int, str] = {}

    async def start(self) -> None:
        log.info("=" * 60)
        log.info("  BTC SNIPER v3 - Multi-Strategy Engine")
        log.info("  Strategies: ChainlinkArb | Momentum | MeanReversion")
        log.info("  Mode: %s", config.trading_mode.upper())
        log.info("  Capital: $%.2f", config.paper_initial_balance)
        log.info("  Chainlink: DIRECT on-chain oracle (Polygon)")
        log.info("=" * 60)

        Path("data").mkdir(exist_ok=True)
        await self.db.connect()

        db_state = await self.db.load_portfolio_state(config.trading_mode)
        self.portfolio.restore_from_db(db_state)

        dashboard_state.set_db(self.db)
        await dashboard_state.refresh_from_db()
        await dashboard_state.update_portfolio(self.portfolio.get_stats())

        if config.is_live and hasattr(self.trader, "start"):
            await self.trader.start()

        self._running = True

        tasks = [
            asyncio.create_task(self.binance_feed.start(), name="binance_feed"),
            asyncio.create_task(self.chainlink_feed.start(), name="chainlink_feed"),
            asyncio.create_task(self.polymarket_feed.start(), name="polymarket_feed"),
            asyncio.create_task(self._signal_loop(), name="signal_loop"),
            asyncio.create_task(self._resolution_loop(), name="resolution_loop"),
            asyncio.create_task(self._snapshot_loop(), name="snapshot_loop"),
            asyncio.create_task(self._dashboard_server(), name="dashboard"),
        ]

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
        await self.binance_feed.stop()
        await self.chainlink_feed.stop()
        await self.polymarket_feed.stop()
        if config.is_live and hasattr(self.trader, "stop"):
            await self.trader.stop()
        await self._save_snapshot()
        await self.db.close()
        log.info("[Main] Shutdown complete")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    # ── Price callbacks ──────────────────────────────────────────────────────────────────

    async def _on_price(self, source: str, price: float, timestamp: float) -> None:
        if source == "binance":
            self._btc_binance = price
            self.signal_engine.update_price(price, timestamp)
        elif source in ("chainlink", "chainlink_binance_fallback"):
            self._btc_chainlink = price
            self.signal_engine.update_chainlink_price(price, timestamp)

        await dashboard_state.update_feeds({
            "binance": self._btc_binance > 0,
            "chainlink": self._btc_chainlink > 0,
            "polymarket": len(self._active_markets) > 0,
        })

    async def _on_market_update(self, markets: dict[str, MarketInfo]) -> None:
        self._active_markets = markets
        await dashboard_state.update_feeds({
            "binance": self._btc_binance > 0,
            "chainlink": self._btc_chainlink > 0,
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
        if self._btc_chainlink <= 0:
            return

        consecutive_losses = await self.db.get_consecutive_losses(config.trading_mode)
        markets_snapshot = list(self._active_markets.items())

        for cid, market in markets_snapshot:
            if market.is_expired:
                continue
            if market.reference_price <= 0:
                continue

            ob = self._orderbooks.get(cid)

            state = MarketState(
                market_id=cid,
                reference_price=market.reference_price,
                end_time=market.end_time,
                btc_chainlink=self._btc_chainlink,
                btc_binance=self._btc_binance,
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

            sig = self.signal_engine.evaluate(
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

            # Populate outcome token IDs for live trader
            # (BUGFIX: previously conditionId was used as tokenID)
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
                    # Track which strategy fired this trade
                    self._strategy_by_trade[trade_id] = sig.strategy_used

                    full = await self.db.get_trade(trade_id)
                    if full:
                        await dashboard_state.update_trade(full)
                    await dashboard_state.update_portfolio(
                        self.portfolio.get_stats()
                    )
                    self.signal_engine.reset_market_stability(market.slug)

        await dashboard_state.update_portfolio(self.portfolio.get_stats())

    async def _resolution_loop(self) -> None:
        while self._running:
            try:
                if self.trader.pending_count > 0:
                    resolved = await self.trader.check_resolutions(
                        self._btc_chainlink,
                        fetch_outcome=self.polymarket_feed.fetch_market_outcome,
                    )
                    for r in resolved:
                        trade_id = r["trade_id"]
                        outcome = r["outcome"]   # "won" or "lost"
                        side = r["side"]          # "YES" or "NO"
                        won = outcome == "won"

                        # Update per-strategy performance tracker
                        strategy = self._strategy_by_trade.pop(
                            trade_id, r.get("strategy_used", "chainlink_arb")
                        )
                        self.signal_engine.record_result(strategy, won)

                        # Update market trend tracker
                        # YES bet won  → BTC ended ABOVE ref → market went UP
                        # YES bet lost → BTC ended BELOW ref → market went DOWN
                        # NO bet won   → BTC ended BELOW ref → market went DOWN
                        # NO bet lost  → BTC ended ABOVE ref → market went UP
                        if side == "YES":
                            market_direction = "up" if won else "down"
                        else:
                            market_direction = "down" if won else "up"
                        try:
                            market_trend.record(market_direction)
                        except Exception:
                            pass

                        # Cross-market propagation: notify booster if this was
                        # a 5-minute market. The booster will then apply a
                        # decaying confidence boost to 15m bets for 45 seconds.
                        if r.get("duration", 300) == 300:
                            try:
                                self.signal_engine.record_5m_resolution(
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
                            "trend_direction=%s | pnl=$%.2f",
                            trade_id, strategy, won,
                            market_direction, r["pnl"],
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
        description="BTC Sniper v3 - Multi-Strategy Engine"
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
