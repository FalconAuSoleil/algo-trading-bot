"""Live trading engine — executes real orders on Polymarket CLOB.

Uses py-clob-client for proper EIP-712 / HMAC order signing.
Requires POLYMARKET_PRIVATE_KEY and POLYMARKET_WALLET_ADDRESS in .env.

Trade flow:
  1. Receive Signal with action="BUY" and filters_passed=True
  2. Determine correct outcome token_id from signal.token_id_yes / token_id_no
  3. Create a signed FOK order via ClobClient.create_order()
  4. Submit to CLOB; track as pending resolution
  5. Resolve after market close via Polymarket past-results API

BUGFIX (v3): previous version used conditionId as tokenID in the order
payload, which is wrong. The CLOB needs the YES or NO *outcome* token ID.
These are now carried on the Signal object and populated by main.py.

v3.6: BTC price fallback delay increased 30s→120s to reduce incorrect
resolutions when the API is temporarily slow. Also add restore_pending()
to reload in-flight trades from DB after a crash or restart.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from src.config import PolymarketConfig
from src.engine.signal import Signal
from src.trading.portfolio import Portfolio, Position
from src.utils.db import Database, TradeRecord
from src.utils.logger import setup_logger

log = setup_logger("trading.live")

# py-clob-client is required for live trading.
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.constants import POLYGON
    _HAS_CLOB = True
except ImportError:
    _HAS_CLOB = False
    log.warning(
        "[Live] py-clob-client not installed. "
        "Run: pip install py-clob-client>=0.6"
    )

# BUY side constant (avoids importing the enum in the order payload)
_BUY = "BUY"

# Seconds after expiry before falling back to BTC price for resolution.
# Increased from 30s to 120s (v3.6) to reduce incorrect resolutions
# when the Polymarket API is temporarily slow to publish outcomes.
_FALLBACK_DELAY_SEC = 120


class LiveTrader:
    """
    Executes real trades on Polymarket via the CLOB API.

    Uses FOK (Fill-or-Kill) orders for immediate execution.
    Requires valid Polygon wallet private key and USDC balance.
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
        self._client: Optional[object] = None
        self._pending_resolutions: dict[int, dict] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Initialise the CLOB client and derive/set API credentials."""
        self._loop = asyncio.get_event_loop()

        if not _HAS_CLOB:
            log.error(
                "[Live] Cannot start: py-clob-client not installed."
            )
            return

        if not self.cfg.private_key:
            log.error(
                "[Live] Cannot start: POLYMARKET_PRIVATE_KEY not set in .env"
            )
            return

        if not self.cfg.wallet_address:
            log.error(
                "[Live] Cannot start: POLYMARKET_WALLET_ADDRESS not set in .env"
            )
            return

        try:
            # Run synchronous ClobClient init in executor to avoid blocking
            def _init_client():
                client = ClobClient(
                    host=self.cfg.clob_url,
                    chain_id=POLYGON,
                    key=self.cfg.private_key,
                    signature_type=2,       # POLY_GNOSIS_SAFE
                    funder=self.cfg.wallet_address,
                )
                # Use pre-configured API key if available, else derive it
                if self.cfg.api_key and self.cfg.api_secret and self.cfg.api_passphrase:
                    from py_clob_client.clob_types import ApiCreds
                    creds = ApiCreds(
                        api_key=self.cfg.api_key,
                        api_secret=self.cfg.api_secret,
                        api_passphrase=self.cfg.api_passphrase,
                    )
                else:
                    creds = client.derive_api_creds()
                    log.info(
                        "[Live] Derived API creds from private key "
                        "(no explicit creds in .env)"
                    )
                client.set_api_creds(creds)
                return client

            self._client = await self._loop.run_in_executor(None, _init_client)
            log.info(
                "[Live] ClobClient ready | wallet=%s",
                self.cfg.wallet_address[:10] + "...",
            )
        except Exception as exc:
            log.error("[Live] ClobClient init failed: %s", exc)
            self._client = None

    async def stop(self) -> None:
        self._client = None
        log.info("[Live] Trader stopped")

    async def restore_pending(self) -> None:
        """Reload in-flight trades from DB after a crash or restart.

        Called by main.py on startup. Without this, any trade that was
        pending when the process died stays as outcome='pending' in the
        DB forever and the capital is never returned to the portfolio.

        v3.6 addition.
        """
        pending = await self.db.get_pending_trades(mode="live")
        restored = 0
        for row in pending:
            trade_id = row["id"]
            if trade_id in self._pending_resolutions:
                continue  # already tracked
            approx_start = row["timestamp"] - row.get("time_remaining_sec", 300)
            end_time = row["timestamp"] + row.get("time_remaining_sec", 300)
            self._pending_resolutions[trade_id] = {
                "market_id": row["market_id"],
                "side": row["side"],
                "end_time": end_time,
                "reference_price": 0.0,  # not stored in trades table
                "start_time": approx_start,
                "duration": 300,
                "slug": row.get("slug", ""),
                "strategy_used": "chainlink_arb",
            }
            restored += 1
        if restored:
            log.info(
                "[Live] Restored %d pending trade(s) from DB after restart",
                restored,
            )

    async def execute(self, signal: Signal) -> Optional[int]:
        """
        Execute a live trade based on signal.

        Returns:
            trade_id if order was filled, None if skipped or rejected.
        """
        if signal.action != "BUY" or not signal.filters_passed:
            return None
        if signal.size_usd < 1.0:
            return None
        if self._client is None:
            log.error("[Live] Client not initialized — check credentials")
            return None

        # Determine the correct outcome token ID
        if signal.side == "YES":
            token_id = signal.token_id_yes
        else:
            token_id = signal.token_id_no

        if not token_id:
            log.error(
                "[Live] Missing token_id for side=%s market=%s. "
                "Ensure main.py populates sig.token_id_yes/no from MarketInfo.",
                signal.side, signal.market_id[:16],
            )
            return None

        entry_price = signal.entry_price
        if entry_price <= 0 or entry_price >= 1.0:
            log.error("[Live] Invalid entry price: %.4f", entry_price)
            return None

        shares = round(signal.size_usd / entry_price, 2)

        # Record as pending in DB before submitting (idempotent crash safety)
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
            oracle_age_sec=signal.oracle_age_sec,
            mode="live",
        )
        trade_id = await self.db.insert_trade(record)

        # Submit signed FOK order in thread executor
        try:
            log.info(
                "[Live] Submitting FOK | %s %s | shares=%.2f @ %.4f | $%.2f | CL_age=%.0fs",
                signal.side, signal.market_id[:16], shares,
                entry_price, signal.size_usd, signal.oracle_age_sec,
            )

            def _place_order():
                from py_clob_client.clob_types import OrderArgs, OrderType
                order_args = OrderArgs(
                    token_id=token_id,
                    price=round(entry_price, 4),
                    size=shares,
                    side=_BUY,
                )
                signed = self._client.create_order(order_args)
                return self._client.post_order(signed, OrderType.FOK)

            resp = await self._loop.run_in_executor(None, _place_order)

            if resp and resp.get("success"):
                fill_price = float(resp.get("price", entry_price))
                actual_shares = signal.size_usd / fill_price

                log.info(
                    "[Live] FILLED | trade_id=%d | fill=%.4f | shares=%.2f",
                    trade_id, fill_price, actual_shares,
                )

                pos = Position(
                    trade_id=trade_id,
                    market_id=signal.market_id,
                    side=signal.side,
                    entry_price=fill_price,
                    size_usd=signal.size_usd,
                    shares=actual_shares,
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
                    "slug": signal.slug,
                    "strategy_used": signal.strategy_used,
                }
                return trade_id

            else:
                err = resp.get("errorMsg", str(resp)) if resp else "no response"
                log.warning(
                    "[Live] Order REJECTED | trade_id=%d | %s",
                    trade_id, err,
                )
                await self.db.resolve_trade(trade_id, "rejected", 0.0)
                return None

        except Exception as exc:
            log.error("[Live] Order execution error: %s", exc, exc_info=True)
            await self.db.resolve_trade(trade_id, "error", 0.0)
            return None

    async def check_resolutions(
        self,
        btc_chainlink_price: float,
        fetch_outcome: Optional[Callable] = None,
    ) -> list[dict]:
        """
        Check if pending positions have resolved.

        Prefers the real Polymarket API outcome; falls back to BTC
        price comparison only after _FALLBACK_DELAY_SEC (120s) if
        the API is still pending. Delay increased from 30s in v3.6
        to reduce incorrect resolutions on slow API responses.
        """
        now = time.time()
        resolved = []

        for trade_id, info in list(self._pending_resolutions.items()):
            if now < info["end_time"] + 10:
                continue  # too early

            ref = info["reference_price"]
            side = info["side"]
            if ref <= 0 and now < info["end_time"] + _FALLBACK_DELAY_SEC:
                # No reference price (restored from DB) — wait for API only
                pass

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
                won = (real_outcome == "up") if side == "YES" else (real_outcome == "down")
                log.info(
                    "[Live] Resolved trade %d via API | outcome=%s side=%s won=%s",
                    trade_id, real_outcome, side, won,
                )
            elif ref > 0 and now >= info["end_time"] + _FALLBACK_DELAY_SEC:
                # API still not ready after _FALLBACK_DELAY_SEC — use BTC price
                # as last resort. Note: this uses the CURRENT price, not the
                # price at expiry, which may differ if BTC moved post-resolution.
                btc_above = btc_chainlink_price >= ref
                won = btc_above if side == "YES" else not btc_above
                log.warning(
                    "[Live] Resolved trade %d via BTC FALLBACK (%ds wait) | "
                    "btc=$%.2f ref=$%.2f side=%s won=%s — "
                    "verify manually if outcome seems wrong",
                    trade_id, int(now - info["end_time"]),
                    btc_chainlink_price, ref, side, won,
                )
            else:
                continue  # still waiting for API

            outcome, pnl = self.portfolio.close_position(trade_id, won)
            await self.db.resolve_trade(trade_id, outcome, pnl)
            del self._pending_resolutions[trade_id]

            resolved.append({
                "trade_id": trade_id,
                "outcome": outcome,
                "pnl": pnl,
                "side": side,
                "btc_price": btc_chainlink_price,
                "ref_price": ref,
                "slug": info.get("slug", ""),
                "duration": info.get("duration", 300),
                "strategy_used": info.get("strategy_used", "chainlink_arb"),
            })

        return resolved

    @property
    def pending_count(self) -> int:
        return len(self._pending_resolutions)
