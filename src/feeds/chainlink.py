"""Chainlink BTC/USD price feed — RPC Polygon + RTDS + Binance fallback.

Stratégie principale : RPC Polygon
  - Poll latestRoundData() toutes les POLL_INTERVAL secondes
  - Rotation automatique sur 3 endpoints
  - Expose get_price_at(ts) pour snapshoter le PTB exact à l'ouverture

Stratégie secondaire : RTDS Polymarket (en parallèle, si dispo)
  - Stream push vs poll → prix plus frais entre les ticks RPC

Fallback : Binance REST si tous les RPC sont down.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Callable, Optional

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from src.utils.logger import setup_logger

log = setup_logger("feeds.chainlink")

# ── Chainlink BTC/USD sur Polygon ────────────────────────
CHAINLINK_BTC_USD = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
CHAINLINK_DECIMALS = 8
LATEST_ROUND_DATA = "0xfeaf968c"

RPC_ENDPOINTS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://1rpc.io/matic",
    "https://polygon.drpc.org",
]

# ── RTDS Polymarket ──────────────────────────────────────
RTDS_WS_URL = "wss://ws-live-data.polymarket.com"
RTDS_SUBSCRIBE = {
    "action": "subscribe",
    "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*",
                       "filters": json.dumps({"symbol": "btc/usd"})}],
}
PING_MSG = json.dumps({"action": "PING"})

# ── Timings ──────────────────────────────────────────────
POLL_INTERVAL = 3.0
STALE_THRESHOLD = 20.0
HISTORY_WINDOW = 120.0


class ChainlinkFeed:
    """Price feed Chainlink BTC/USD — RPC Polygon + RTDS + Binance fallback."""

    def __init__(
        self,
        url: str = "wss://ws-live-data.polymarket.com",
        on_price: Optional[Callable] = None,
        poll_interval: float = POLL_INTERVAL,
    ):
        self.url = url
        self.on_price = on_price
        self.poll_interval = poll_interval
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self.last_chainlink_price: float = 0.0
        self.last_chainlink_ts: float = 0.0
        self._rpc_index = 0
        self._rpc_failures = 0
        self._history: deque[tuple[float, float]] = deque()

    def get_price_at(self, ts: float) -> float:
        """Return the Chainlink price at the given timestamp."""
        for tick_ts, tick_price in reversed(self._history):
            if tick_ts <= ts:
                return tick_price
        return self._history[0][1] if self._history else self.last_chainlink_price

    async def start(self) -> None:
        self._running = True
        self._session = aiohttp.ClientSession()
        log.info("[Chainlink] Starting — RPC Polygon (primary) + RTDS (bonus)")
        await asyncio.gather(
            self._rpc_loop(),
            self._rtds_loop(),
            self._stale_watchdog(),
        )

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
        log.info("[Chainlink] Feed stopped")

    # ── RPC poll loop ────────────────────────────────────

    async def _rpc_loop(self) -> None:
        while self._running:
            price, updated_at = await self._fetch_rpc()
            if price > 0:
                self._rpc_failures = 0
                await self._emit(price, updated_at or time.time(), "chainlink")
            else:
                self._rpc_failures += 1
                if self._rpc_failures == 1:
                    log.warning("[Chainlink] All Polygon RPCs failed")
            await asyncio.sleep(self.poll_interval)

    async def _fetch_rpc(self) -> tuple[float, Optional[float]]:
        for _ in range(len(RPC_ENDPOINTS)):
            rpc = RPC_ENDPOINTS[self._rpc_index]
            try:
                price, updated_at = await self._call_rpc(rpc)
                if price > 0:
                    return price, updated_at
            except Exception as e:
                log.debug("[Chainlink] RPC %s failed: %s", rpc, e)
            self._rpc_index = (self._rpc_index + 1) % len(RPC_ENDPOINTS)
        return 0.0, None

    async def _call_rpc(self, rpc_url: str) -> tuple[float, float]:
        async with self._session.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                  "params": [{"to": CHAINLINK_BTC_USD, "data": LATEST_ROUND_DATA}, "latest"]},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                return 0.0, 0.0
            result = await resp.json()
        hex_data = result.get("result", "")
        if not hex_data or hex_data == "0x" or len(hex_data) < 66:
            return 0.0, 0.0
        data = hex_data[2:]
        answer = int(data[64:128], 16)
        if answer >= 2**255:
            answer -= 2**256
        if answer <= 0:
            return 0.0, 0.0
        price = answer / (10 ** CHAINLINK_DECIMALS)
        updated_at = float(int(data[192:256], 16))
        return price, updated_at

    # ── RTDS loop ────────────────────────────────────────

    async def _rtds_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(
                    RTDS_WS_URL, ping_interval=None, open_timeout=10
                ) as ws:
                    await ws.send(json.dumps(RTDS_SUBSCRIBE))
                    log.info("[Chainlink] RTDS connected")
                    await asyncio.gather(
                        self._rtds_reader(ws),
                        self._rtds_pinger(ws),
                    )
            except Exception as e:
                if self._running:
                    log.debug("[Chainlink] RTDS unavailable (%s) — RPC only", e)
                    await asyncio.sleep(30)

    async def _rtds_reader(self, ws) -> None:
        async for raw in ws:
            if not self._running:
                break
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except Exception:
                    continue
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if msg.get("topic") != "crypto_prices_chainlink":
                continue
            payload = msg.get("payload", {})
            if not payload or payload.get("symbol", "").lower() != "btc/usd":
                continue
            try:
                price = float(payload["value"])
                ts_ms = float(payload.get("timestamp", time.time() * 1000))
            except (KeyError, ValueError):
                continue
            if price > 0:
                await self._emit(price, ts_ms / 1000.0, "chainlink_rtds")

    async def _rtds_pinger(self, ws) -> None:
        while self._running:
            await asyncio.sleep(5)
            try:
                await ws.send(PING_MSG)
            except Exception:
                break

    # ── Emit ─────────────────────────────────────────────

    async def _emit(self, price: float, ts: float, source: str) -> None:
        now = time.time()
        self.last_chainlink_price = price
        self.last_chainlink_ts = now
        oracle_ts = ts if ts < now + 60 else now
        self._history.append((oracle_ts, price))
        cutoff = now - HISTORY_WINDOW
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        log.debug("[Chainlink] BTC/USD = $%.2f (src=%s)", price, source)
        if self.on_price:
            await self.on_price(source="chainlink", price=price, timestamp=ts)

    # ── Stale watchdog ───────────────────────────────────

    async def _stale_watchdog(self) -> None:
        await asyncio.sleep(STALE_THRESHOLD + 5)
        while self._running:
            await asyncio.sleep(5)
            age = time.time() - self.last_chainlink_ts
            if age > STALE_THRESHOLD:
                log.warning("[Chainlink] No tick for %.0fs — Binance fallback", age)
                price = await self._fetch_binance()
                if price > 0:
                    await self._emit(price, time.time(), "chainlink_binance_fallback")

    async def _fetch_binance(self) -> float:
        if not self._session or self._session.closed:
            return 0.0
        try:
            async with self._session.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    return float((await resp.json())["price"])
        except Exception:
            pass
        return 0.0
