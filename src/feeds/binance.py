"""Binance BTC/USDT real-time WebSocket feed."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from src.utils.logger import setup_logger

log = setup_logger("feeds.binance")


class BinanceFeed:
    """Streams BTC/USDT trades from Binance WebSocket."""

    def __init__(
        self,
        url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade",
        on_price: Optional[Callable] = None,
    ):
        self.url = url
        self.on_price = on_price
        self._ws = None
        self._running = False
        self.last_price: float = 0.0
        self.last_timestamp: float = 0.0

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                log.info("[Binance] Connecting to %s", self.url)
                async with websockets.connect(
                    self.url, ping_interval=20, ping_timeout=10
                ) as ws:
                    self._ws = ws
                    log.info("[Binance] Connected")
                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)
            except ConnectionClosed:
                if self._running:
                    log.warning(
                        "[Binance] Connection closed, reconnecting..."
                    )
                    await asyncio.sleep(2)
            except Exception as e:
                if self._running:
                    log.error("[Binance] Error: %s", e)
                    await asyncio.sleep(5)

    async def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
            price = float(data["p"])
            ts = data["T"] / 1000.0  # ms to seconds
            self.last_price = price
            self.last_timestamp = ts
            if self.on_price:
                await self.on_price(
                    source="binance",
                    price=price,
                    timestamp=ts,
                )
        except (KeyError, ValueError, TypeError) as e:
            log.debug("[Binance] Parse error: %s", e)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        log.info("[Binance] Feed stopped")
