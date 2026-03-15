"""Paper trading engine - simulates trades against live data."""

from __future__ import annotations

import time
from typing import Callable, Optional

from src.engine.signal import Signal
from src.trading.portfolio import Portfolio, Position
from src.utils.db import Database, TradeRecord
from src.utils.logger import setup_logger

log = setup_logger("trading.paper")

# Wait this many seconds after expiry before first API check.
RESOLUTION_DELAY = 30


class PaperTrader:
    """Executes simulated trades using live market data.

    Simulates fills against the live orderbook snapshot. Positions
    are resolved when the market expires based on the actual Polymarket
    outcome fetched from the past-results API.

    IMPORTANT: No BTC price fallback. Only the official Polymarket
    API result is used for resolution. This ensures our PnL tracking
    matches what would actually happen on-chain.
    """

    def __init__(self, portfolio: Portfolio, db: Database):
        self.portfolio = portfolio
        self.db = db
        self._pending_resolutions: dict[int, dict] = {}

    async def execute(self, signal: Signal) -> Optional[int]:
        """Execute a paper trade based on signal.

        Returns:
            trade_id if executed, None if skipped.
        """
        if signal.action != "BUY" or not signal.filters_passed:
            return None

        if signal.size_usd < 1.0:
            return None

        entry_price = signal.entry_price
        if entry_price <= 0 or entry_price >= 1.0:
            log.warning("[Paper] Invalid entry price: %.4f", entry_price)
            return None

        shares = signal.size_usd / entry_price

        record = TradeRecord(
            market_id=signal.market_id,
            slug=signal.slug,
            side=signal.side,
            entry_price=entry_price,
            size_usd=signal.size_usd,
            delta=signal.delta_chainlink,
            sigma=signal.sigma,
            p_true=signal.p_true,
            p_market=signal.p_market,
            edge=signal.edge,
            time_remaining_sec=signal.time_remaining_sec,
            mode="paper",
        )
        trade_id = await self.db.insert_trade(record)

        pos = Position(
            trade_id=trade_id,
            market_id=signal.market_id,
            side=signal.side,
            entry_price=entry_price,
            size_usd=signal.size_usd,
            shares=shares,
            entry_time=time.time(),
            market_end_time=time.time() + signal.time_remaining_sec,
            delta_at_entry=signal.delta_chainlink,
            p_true_at_entry=signal.p_true,
            edge_at_entry=signal.edge,
        )

        if not self.portfolio.open_position(pos):
            await self.db.resolve_trade(trade_id, "cancelled", 0.0)
            return None

        self._pending_resolutions[trade_id] = {
            "market_id": signal.market_id,
            "side": signal.side,
            "end_time": pos.market_end_time,
            "reference_price": signal.reference_price,
            "start_time": signal.market_start_time,
            "duration": signal.market_duration,
        }

        log.info(
            "[Paper] EXECUTED %s %s | $%.2f @ $%.4f | "
            "edge=%.1f%% | P_true=%.1f%% | T=%.0fs | "
            "delta=%.4f%%",
            signal.side,
            signal.market_id[:16],
            signal.size_usd,
            entry_price,
            signal.edge * 100,
            signal.p_true * 100,
            signal.time_remaining_sec,
            signal.delta_chainlink * 100,
        )
        return trade_id

    async def check_resolutions(
        self,
        btc_chainlink_price: float,
        fetch_outcome: Optional[
            Callable[[float, int], object]
        ] = None,
    ) -> list[dict]:
        """Check if any pending positions should be resolved.

        Uses ONLY the real Polymarket API outcome. Retries
        indefinitely until the API returns "up" or "down".
        No BTC price fallback - only the official result counts.
        """
        now = time.time()
        resolved = []

        for trade_id, info in list(
            self._pending_resolutions.items()
        ):
            # Wait RESOLUTION_DELAY seconds after expiry
            if now < info["end_time"] + RESOLUTION_DELAY:
                continue

            side = info["side"]

            # Try to get the real outcome from Polymarket API
            real_outcome = None
            if fetch_outcome:
                try:
                    real_outcome = await fetch_outcome(
                        info.get("start_time", 0),
                        info.get("duration", 300),
                    )
                except Exception as e:
                    log.debug("[Paper] Outcome fetch error: %s", e)

            if real_outcome not in ("up", "down"):
                # API not ready yet - retry next cycle
                # Log periodically so we know it's still waiting
                wait_s = int(now - info["end_time"])
                if wait_s % 30 < 4:
                    log.info(
                        "[Paper] Waiting for API result on "
                        "trade %d (%ds since expiry)",
                        trade_id, wait_s,
                    )
                continue

            # Use the real Polymarket resolution
            if side == "YES":
                won = real_outcome == "up"
            else:
                won = real_outcome == "down"
            log.info(
                "[Paper] Resolved trade %d via API: "
                "outcome=%s, side=%s, won=%s",
                trade_id, real_outcome, side, won,
            )

            outcome, pnl = self.portfolio.close_position(
                trade_id, won
            )
            await self.db.resolve_trade(trade_id, outcome, pnl)

            del self._pending_resolutions[trade_id]
            resolved.append({
                "trade_id": trade_id,
                "outcome": outcome,
                "pnl": pnl,
                "side": side,
                "btc_price": btc_chainlink_price,
                "ref_price": info["reference_price"],
            })

        return resolved

    @property
    def pending_count(self) -> int:
        return len(self._pending_resolutions)
