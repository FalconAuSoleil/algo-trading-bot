"""Live trading engine — executes real orders on Polymarket CLOB."""

from __future__ import annotations

import time
from typing import Callable, Optional

import aiohttp

from src.config import PolymarketConfig
from src.engine.signal import Signal
from src.trading.portfolio import Portfolio, Position
from src.utils.db import Database, TradeRecord
from src.utils.logger import setup_logger

log = setup_logger("trading.live")


class LiveTrader:
    """Executes real trades on Polymarket via the CLOB API.

    Uses FOK (Fill-or-Kill) orders for immediate execution.
    Requires valid API credentials and a funded Polygon wallet.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        db: Database,
        poly_cfg: PolymarketConfig,
    ):
        self.portfolio = portfolio
        self.db = db
        self.cfg = poly_cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._pending_resolutions: dict[int, dict] = {}

    async def start(self) -> None:
        """Initialize HTTP session for CLOB API."""
        self._session = aiohttp.ClientSession(
            headers=self._build_headers()
        )
        log.info("[Live] Trader initialized")

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    def _build_headers(self) -> dict:
        """Build authentication headers for CLOB API.

        Note: Full Polymarket CLOB auth requires EIP-712 signing
        and HMAC-SHA256. This is a simplified placeholder.
        For production, use the official py-clob-client SDK.
        """
        return {
            "Content-Type": "application/json",
            "POLY-ADDRESS": self.cfg.wallet_address,
            "POLY-API-KEY": self.cfg.api_key,
            "POLY-PASSPHRASE": self.cfg.api_passphrase,
        }

    async def execute(self, signal: Signal) -> Optional[int]:
        """Execute a live trade based on signal.

        Returns:
            trade_id if executed, None if skipped.
        """
        if signal.action != "BUY" or not signal.filters_passed:
            return None

        if signal.size_usd < 1.0:
            return None

        if not self.cfg.api_key or not self.cfg.private_key:
            log.error(
                "[Live] Cannot execute: missing API credentials"
            )
            return None

        # Determine token ID and price
        entry_price = signal.entry_price
        if entry_price <= 0 or entry_price >= 1.0:
            return None

        # Record trade (pending) in database
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
            mode="live",
        )
        trade_id = await self.db.insert_trade(record)

        # Build FOK order
        shares = signal.size_usd / entry_price
        order_payload = {
            "tokenID": signal.market_id,
            "price": str(round(entry_price, 4)),
            "size": str(round(shares, 2)),
            "side": "BUY",
            "type": "FOK",
            "feeRateBps": "0",
        }

        try:
            log.info(
                "[Live] Sending FOK order: %s %s $%.2f @ %.4f",
                signal.side,
                signal.market_id[:16],
                signal.size_usd,
                entry_price,
            )

            async with self._session.post(
                f"{self.cfg.clob_url}/order",
                json=order_payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                result = await resp.json()

                if resp.status == 200 and result.get("success"):
                    log.info(
                        "[Live] Order FILLED: %s", result
                    )
                    # Update with actual fill price if available
                    fill_price = float(
                        result.get("price", entry_price)
                    )
                    actual_shares = signal.size_usd / fill_price

                    pos = Position(
                        trade_id=trade_id,
                        market_id=signal.market_id,
                        side=signal.side,
                        entry_price=fill_price,
                        size_usd=signal.size_usd,
                        shares=actual_shares,
                        entry_time=time.time(),
                        market_end_time=(
                            time.time()
                            + signal.time_remaining_sec
                        ),
                        delta_at_entry=signal.delta_chainlink,
                        p_true_at_entry=signal.p_true,
                        edge_at_entry=signal.edge,
                    )

                    if not self.portfolio.open_position(pos):
                        await self.db.resolve_trade(
                            trade_id, "cancelled", 0.0
                        )
                        return None

                    self._pending_resolutions[trade_id] = {
                        "market_id": signal.market_id,
                        "side": signal.side,
                        "end_time": pos.market_end_time,
                        "reference_price": signal.reference_price,
                        "start_time": signal.market_start_time,
                        "duration": signal.market_duration,
                    }
                    return trade_id
                else:
                    log.warning(
                        "[Live] Order REJECTED: %s (status=%d)",
                        result,
                        resp.status,
                    )
                    await self.db.resolve_trade(
                        trade_id, "rejected", 0.0
                    )
                    return None

        except Exception as e:
            log.error("[Live] Order execution error: %s", e)
            await self.db.resolve_trade(
                trade_id, "error", 0.0
            )
            return None

    async def check_resolutions(
        self,
        btc_chainlink_price: float,
        fetch_outcome: Optional[
            Callable[[float, int], object]
        ] = None,
    ) -> list[dict]:
        """Check if any pending positions should be resolved.

        Uses the real Polymarket outcome when available via
        fetch_outcome callback. Falls back to BTC price comparison.
        """
        now = time.time()
        resolved = []

        for trade_id, info in list(
            self._pending_resolutions.items()
        ):
            # Wait 10s after expiry for API to settle
            if now < info["end_time"] + 10:
                continue

            ref = info["reference_price"]
            side = info["side"]

            if ref <= 0:
                continue

            # Try real outcome from Polymarket API
            real_outcome = None
            if fetch_outcome:
                try:
                    real_outcome = await fetch_outcome(
                        info.get("start_time", 0),
                        info.get("duration", 300),
                    )
                except Exception:
                    pass

            if real_outcome in ("up", "down"):
                if side == "YES":
                    won = real_outcome == "up"
                else:
                    won = real_outcome == "down"
                log.info(
                    "[Live] Resolved trade %d via API: "
                    "outcome=%s, side=%s, won=%s",
                    trade_id, real_outcome, side, won,
                )
            else:
                if now < info["end_time"] + 30:
                    continue
                btc_above = btc_chainlink_price >= ref
                if side == "YES":
                    won = btc_above
                else:
                    won = not btc_above
                log.warning(
                    "[Live] Resolved trade %d via BTC price "
                    "FALLBACK: btc=$%.2f, ref=$%.2f, won=%s",
                    trade_id, btc_chainlink_price, ref, won,
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
                "ref_price": ref,
            })

        return resolved

    @property
    def pending_count(self) -> int:
        return len(self._pending_resolutions)
