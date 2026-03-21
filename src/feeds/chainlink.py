"""Chainlink BTC/USD price feed - direct on-chain oracle reading.

Reads the Chainlink BTC/USD aggregator contract on Polygon via
latestRoundData().  This is the same oracle that Polymarket uses
to resolve BTC Up/Down markets, so our price feed is now *exactly*
aligned with the resolution source.

Multiple free Polygon RPC endpoints are used with automatic rotation
on failure.  If all RPCs are down, falls back to Binance REST as a
last resort (logged as a warning so we know).

v3.6 fix: Binance fallback no longer updates last_chainlink_ts.
Previously setting last_chainlink_ts = now() during fallback would
reset oracle_age to 0, masking staleness and silencing ORACLE_STALE.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

import aiohttp

from src.utils.logger import setup_logger

log = setup_logger("feeds.chainlink")

# Chainlink BTC/USD Price Feed on Polygon
CHAINLINK_BTC_USD = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
CHAINLINK_DECIMALS = 8

# Free Polygon RPC endpoints (rotated on failure)
RPC_ENDPOINTS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://1rpc.io/matic",
    "https://polygon.drpc.org",
]


class ChainlinkFeed:
    """Polls Chainlink BTC/USD price directly from the Polygon oracle.

    Reads latestRoundData() from the on-chain aggregator every
    poll_interval seconds.  Falls back to Binance REST only when
    *all* RPC endpoints are unreachable.
    """

    def __init__(
        self,
        on_price: Optional[Callable] = None,
        poll_interval: float = 3.0,
    ):
        self.on_price = on_price
        self.poll_interval = poll_interval
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self.last_chainlink_price: float = 0.0
        self.last_chainlink_ts: float = 0.0
        self._rpc_index: int = 0
        self._consecutive_rpc_failures: int = 0

    async def start(self) -> None:
        """Start polling Chainlink price from Polygon."""
        self._running = True
        self._session = aiohttp.ClientSession()
        log.info(
            "[Chainlink] Starting DIRECT oracle feed on Polygon "
            "(contract=%s, poll=%.1fs)",
            CHAINLINK_BTC_USD[:10] + "...",
            self.poll_interval,
        )

        while self._running:
            try:
                price, updated_at = await self._fetch_chainlink()
                if price and price > 0:
                    self.last_chainlink_price = price
                    self.last_chainlink_ts = updated_at or time.time()
                    self._consecutive_rpc_failures = 0
                    if self.on_price:
                        await self.on_price(
                            source="chainlink",
                            price=price,
                            timestamp=self.last_chainlink_ts,
                        )
                else:
                    # All RPCs failed - try Binance as emergency fallback
                    self._consecutive_rpc_failures += 1
                    price = await self._fetch_binance_fallback()
                    if price and price > 0:
                        now = time.time()
                        self.last_chainlink_price = price
                        # v3.6 FIX: do NOT update last_chainlink_ts here.
                        # oracle_age = now - last_chainlink_ts is used by the
                        # ORACLE_STALE filter in signal.py. If we reset ts to
                        # now() during a Binance fallback, oracle_age drops to
                        # ~0 and the filter never fires — even when Chainlink
                        # has been silent for minutes. Preserve the real ts.
                        if self.on_price:
                            await self.on_price(
                                source="chainlink_binance_fallback",
                                price=price,
                                timestamp=now,
                            )
                        if self._consecutive_rpc_failures <= 1:
                            log.warning(
                                "[Chainlink] All RPCs failed, using "
                                "Binance fallback ($%.2f). "
                                "oracle_age preserved for ORACLE_STALE filter.",
                                price,
                            )
            except Exception as e:
                log.debug("[Chainlink] Poll error: %s", e)
            await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
        log.info("[Chainlink] Feed stopped")

    # ------------------------------------------------------------------
    # Chainlink on-chain read via eth_call JSON-RPC
    # ------------------------------------------------------------------

    async def _fetch_chainlink(self) -> tuple[float, Optional[float]]:
        """Try each RPC endpoint until one succeeds."""
        for attempt in range(len(RPC_ENDPOINTS)):
            rpc = RPC_ENDPOINTS[self._rpc_index]
            try:
                price, updated_at = await self._call_latest_round(rpc)
                if price > 0:
                    return price, updated_at
            except Exception as e:
                log.debug("[Chainlink] RPC %s failed: %s", rpc, e)
            self._rpc_index = (self._rpc_index + 1) % len(RPC_ENDPOINTS)
        return 0.0, None

    async def _call_latest_round(self, rpc_url: str) -> tuple[float, float]:
        """Call latestRoundData() via eth_call JSON-RPC."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [
                {
                    "to": CHAINLINK_BTC_USD,
                    "data": "0xfeaf968c",  # latestRoundData()
                },
                "latest",
            ],
        }

        async with self._session.post(
            rpc_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                return 0.0, 0.0
            result = await resp.json()

        hex_data = result.get("result", "")
        if not hex_data or hex_data == "0x" or len(hex_data) < 66:
            return 0.0, 0.0

        # Decode ABI: 5 x uint256 (32 bytes each)
        # [64:128]  answer (int256) - price * 10^8
        # [192:256] updatedAt (uint256)
        data = hex_data[2:]  # strip 0x

        answer = int(data[64:128], 16)
        if answer >= 2**255:
            answer -= 2**256
        if answer <= 0:
            return 0.0, 0.0

        price = answer / (10**CHAINLINK_DECIMALS)
        updated_at = int(data[192:256], 16)

        return price, float(updated_at)

    # ------------------------------------------------------------------
    # Binance fallback (emergency only)
    # ------------------------------------------------------------------

    async def _fetch_binance_fallback(self) -> float:
        """Fetch BTC price from Binance REST as emergency fallback."""
        try:
            async with self._session.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data["price"])
        except (aiohttp.ClientError, TimeoutError, KeyError, ValueError):
            pass
        return 0.0
