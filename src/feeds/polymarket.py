"""Polymarket market discovery and orderbook feed.

Uses deterministic slug-based discovery for Up/Down markets:
  5-min:  {prefix}-updown-5m-{floor(unix_ts / 300) * 300}
  15-min: {prefix}-updown-15m-{floor(unix_ts / 900) * 900}

v3.8: Uses per-asset interval dict so ETH/SOL/XRP only discover
15-minute markets (Polymarket past-results API does not support
5-minute markets for non-BTC assets — verified 2026-03-21).

v3.7: Generalized to support multiple asset prefixes (btc, eth, sol, xrp).
Reference price (\"price to beat\") is fetched from the Polymarket
past-results API, which provides the exact Chainlink snapshot price
at each market window boundary.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional
from dataclasses import dataclass

import aiohttp

from src.utils.logger import setup_logger

log = setup_logger("feeds.polymarket")

# Market intervals in seconds
INTERVAL_5M = 300
INTERVAL_15M = 900

# Variant names for past-results API
VARIANT_MAP = {
    INTERVAL_5M: "fiveminute",
    INTERVAL_15M: "fifteen",
}


@dataclass
class MarketInfo:
    """Represents an active Up/Down market on Polymarket."""

    condition_id: str = ""
    question: str = ""
    token_id_up: str = ""
    token_id_down: str = ""
    reference_price: float = 0.0
    end_time: float = 0.0
    start_time: float = 0.0
    duration_seconds: int = 300
    slug: str = ""
    active: bool = False
    accepting_orders: bool = False
    outcome_prices: tuple = (0.5, 0.5)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - time.time())

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.end_time

    @property
    def duration_minutes(self) -> int:
        return self.duration_seconds // 60


@dataclass
class OrderbookState:
    """Current state of the orderbook for a market."""

    best_bid_up: float = 0.0
    best_ask_up: float = 0.0
    best_bid_down: float = 0.0
    best_ask_down: float = 0.0
    depth_bid_up: float = 0.0
    depth_ask_up: float = 0.0
    depth_bid_down: float = 0.0
    depth_ask_down: float = 0.0
    mid_up: float = 0.5
    spread_up: float = 1.0
    spread_down: float = 1.0
    timestamp: float = 0.0

    @property
    def mid_down(self) -> float:
        return 1.0 - self.mid_up

    @property
    def best_bid_yes(self) -> float:
        return self.best_bid_up

    @property
    def best_ask_yes(self) -> float:
        return self.best_ask_up

    @property
    def best_bid_no(self) -> float:
        return self.best_bid_down

    @property
    def best_ask_no(self) -> float:
        return self.best_ask_down

    @property
    def depth_ask_yes(self) -> float:
        return self.depth_ask_up

    @property
    def depth_ask_no(self) -> float:
        return self.depth_ask_down

    @property
    def mid_yes(self) -> float:
        return self.mid_up


def compute_slug(asset_prefix: str, interval_sec: int, ts: float = 0) -> str:
    """Compute the deterministic market slug for a given time."""
    if ts <= 0:
        ts = time.time()
    aligned = int(ts // interval_sec) * interval_sec
    label = "5m" if interval_sec == INTERVAL_5M else "15m"
    return f"{asset_prefix}-updown-{label}-{aligned}"


class PolymarketFeed:
    """Discovers Up/Down markets using slug-based lookup
    and polls orderbooks for active markets.

    v3.8: accepts asset_intervals dict {prefix: (interval1, interval2, ...)}
    so each asset can have different supported intervals.
    ETH/SOL/XRP: (900,) only. BTC: (300, 900).
    """

    def __init__(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/",
        # v3.8: primary parameter — per-asset intervals.
        # e.g. {"btc": (300, 900), "eth": (900,), "sol": (900,), "xrp": (900,)}
        asset_intervals: Optional[dict] = None,
        # Legacy params kept for backward compat — ignored if asset_intervals provided
        asset_prefixes: tuple = ("btc",),
        intervals: tuple = (INTERVAL_5M, INTERVAL_15M),
        on_market_update: Optional[Callable] = None,
        on_orderbook_update: Optional[Callable] = None,
    ):
        self.gamma_url = gamma_url
        self.clob_url = clob_url
        self.clob_ws_url = clob_ws_url
        self.on_market_update = on_market_update
        self.on_orderbook_update = on_orderbook_update

        # v3.8: build internal asset_intervals dict
        if asset_intervals is not None:
            self._asset_intervals: dict = asset_intervals
        else:
            # Backward compat: expand legacy asset_prefixes × intervals
            self._asset_intervals = {
                prefix: tuple(intervals) for prefix in asset_prefixes
            }

        self.active_markets: dict = {}
        self.orderbooks: dict = {}
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def asset_prefixes(self) -> tuple:
        """All asset prefixes being monitored."""
        return tuple(self._asset_intervals.keys())

    async def start(self) -> None:
        self._running = True
        self._session = aiohttp.ClientSession()
        log.info(
            "[Polymarket] Feed starting | asset_intervals=%s",
            {k: v for k, v in self._asset_intervals.items()},
        )
        await asyncio.gather(
            self._market_discovery_loop(),
            self._orderbook_poll_loop(),
            self._lifecycle_loop(),
        )

    async def _market_discovery_loop(self) -> None:
        while self._running:
            try:
                await self._discover_markets()
            except Exception as e:
                log.error("[Polymarket] Discovery error: %s", e)
            await asyncio.sleep(3)

    async def _lifecycle_loop(self) -> None:
        while self._running:
            now = time.time()

            for cid, market in list(self.active_markets.items()):
                if market.is_expired:
                    if cid in self.active_markets:
                        log.info("[Polymarket] Market expired: %s", market.slug)
                        del self.active_markets[cid]
                        self.orderbooks.pop(cid, None)
                elif market.reference_price <= 0 and now >= market.start_time:
                    ptb = await self._fetch_price_to_beat(
                        market.slug, market.start_time, market.duration_seconds
                    )
                    if ptb > 0:
                        market.reference_price = ptb
                        log.info(
                            "[Polymarket] Price to beat for %s: $%.4f",
                            market.slug, ptb,
                        )

                if not market.accepting_orders and now >= market.start_time - 5:
                    market.accepting_orders = True

            if self.on_market_update and self.active_markets:
                await self.on_market_update(self.active_markets)

            await asyncio.sleep(1)

    async def _discover_markets(self) -> None:
        """Discover markets for each asset using its supported intervals.

        v3.8: iterates over asset_intervals dict so ETH/SOL/XRP only
        look for 15m markets, while BTC looks for both 5m and 15m.
        """
        now = time.time()
        found = 0

        for asset_prefix, prefix_intervals in self._asset_intervals.items():
            for interval in prefix_intervals:
                aligned = int(now // interval) * interval
                for offset in range(3):
                    window_start = aligned + (offset * interval)
                    slug = compute_slug(asset_prefix, interval, window_start)

                    existing = [
                        m for m in self.active_markets.values() if m.slug == slug
                    ]
                    if existing:
                        continue

                    market_info = await self._fetch_market_by_slug(slug)
                    if market_info and not market_info.is_expired:
                        self.active_markets[market_info.condition_id] = market_info
                        found += 1
                        log.info(
                            "[Polymarket] Discovered: %s | accepting=%s | T-%ds",
                            slug, market_info.accepting_orders,
                            int(market_info.time_remaining),
                        )

        expired = [cid for cid, m in self.active_markets.items() if m.is_expired]
        for cid in expired:
            del self.active_markets[cid]
            self.orderbooks.pop(cid, None)

        if found and self.on_market_update:
            await self.on_market_update(self.active_markets)

    async def _fetch_market_by_slug(self, slug: str) -> Optional[MarketInfo]:
        try:
            async with self._session.get(
                f"{self.gamma_url}/events",
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return None
                events = await resp.json()

            if not events:
                return None

            event = events[0]
            markets = event.get("markets", [])
            if not markets:
                return None

            market = markets[0]
            cid = market.get("conditionId", "")
            if not cid:
                return None

            clob_tokens = market.get("clobTokenIds", "")
            if isinstance(clob_tokens, str) and clob_tokens:
                clob_tokens = json.loads(clob_tokens)
            token_up = clob_tokens[0] if len(clob_tokens) > 0 else ""
            token_down = clob_tokens[1] if len(clob_tokens) > 1 else ""

            prices_raw = market.get("outcomePrices", "[]")
            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw
            p_up = float(prices[0]) if prices else 0.5
            p_down = float(prices[1]) if len(prices) > 1 else 0.5

            end_ts = self._parse_iso(market.get("endDate", ""))
            start_ts = self._parse_iso(market.get("eventStartTime", ""))

            if end_ts <= 0 or start_ts <= 0:
                parts = slug.rsplit("-", 1)
                try:
                    slug_ts = int(parts[-1])
                    label = "5m" if "5m" in slug else "15m"
                    interval_s = 300 if label == "5m" else 900
                    start_ts = float(slug_ts)
                    end_ts = float(slug_ts + interval_s)
                except (ValueError, IndexError):
                    pass

            duration = int(end_ts - start_ts) if (end_ts > 0 and start_ts > 0) else 300

            ref_price = 0.0
            if start_ts > 0 and time.time() >= start_ts:
                ref_price = await self._fetch_price_to_beat(slug, start_ts, duration)
            if ref_price > 0:
                log.info("[Polymarket] Price to beat for %s: $%.4f", slug, ref_price)

            return MarketInfo(
                condition_id=cid,
                question=market.get("question", ""),
                token_id_up=token_up,
                token_id_down=token_down,
                reference_price=ref_price,
                end_time=end_ts,
                start_time=start_ts,
                duration_seconds=duration,
                slug=slug,
                active=market.get("active", False),
                accepting_orders=market.get("acceptingOrders", False),
                outcome_prices=(p_up, p_down),
            )
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            log.debug("[Polymarket] Fetch error for %s: %s", slug, e)
            return None

    async def _orderbook_poll_loop(self) -> None:
        while self._running:
            for cid, market in list(self.active_markets.items()):
                if market.is_expired:
                    continue
                if not market.accepting_orders:
                    continue
                max_ob_window = 900 if market.duration_seconds > 300 else 420
                if market.time_remaining > max_ob_window:
                    continue
                try:
                    await self._fetch_orderbook(cid, market)
                except Exception as e:
                    log.debug("[Polymarket] OB error: %s", e)
            await asyncio.sleep(2)

    async def _fetch_orderbook(self, cid: str, market: MarketInfo) -> None:
        ob = self.orderbooks.setdefault(cid, OrderbookState())
        ob.timestamp = time.time()

        for side, token_id in [
            ("up", market.token_id_up),
            ("down", market.token_id_down),
        ]:
            if not token_id:
                continue
            try:
                async with self._session.get(
                    f"{self.clob_url}/book",
                    params={"token_id": token_id},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        continue
                    book = await resp.json()

                bids = sorted(
                    book.get("bids", []),
                    key=lambda x: -float(x.get("price", 0)),
                )
                asks = sorted(
                    book.get("asks", []),
                    key=lambda x: float(x.get("price", 0)),
                )

                best_bid = float(bids[0]["price"]) if bids else 0
                best_ask = float(asks[0]["price"]) if asks else 0
                depth_bid = sum(
                    float(b.get("size", 0)) * float(b.get("price", 0))
                    for b in bids[:10]
                )
                depth_ask = sum(
                    float(a.get("size", 0)) * float(a.get("price", 0))
                    for a in asks[:10]
                )

                if side == "up":
                    ob.best_bid_up = best_bid
                    ob.best_ask_up = best_ask
                    ob.depth_bid_up = depth_bid
                    ob.depth_ask_up = depth_ask
                    if best_bid and best_ask:
                        ob.mid_up = (best_bid + best_ask) / 2
                        ob.spread_up = best_ask - best_bid
                else:
                    ob.best_bid_down = best_bid
                    ob.best_ask_down = best_ask
                    ob.depth_bid_down = depth_bid
                    ob.depth_ask_down = depth_ask
                    if best_bid and best_ask:
                        ob.spread_down = best_ask - best_bid

            except (aiohttp.ClientError, KeyError, ValueError):
                pass

        if self.on_orderbook_update:
            await self.on_orderbook_update(cid, ob)

    @staticmethod
    def _parse_iso(iso_str: str) -> float:
        if not iso_str:
            return 0.0
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return 0.0

    async def _fetch_price_to_beat(
        self, slug: str, start_time: float, duration: int
    ) -> float:
        """Fetch the reference price for a market from the past-results API.

        NOTE: As of 2026-03-21, Polymarket only supports 5-minute markets
        for BTC. ETH/SOL/XRP only work with variant='fifteen' (15m).
        """
        parts = slug.split("-")
        symbol = parts[0].upper() if parts else "BTC"

        variant = VARIANT_MAP.get(duration, "fiveminute")
        start_iso = datetime.fromtimestamp(
            start_time, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        try:
            async with self._session.get(
                "https://polymarket.com/api/past-results",
                params={
                    "symbol": symbol,
                    "variant": variant,
                    "assetType": "crypto",
                    "currentEventStartTime": start_iso,
                },
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()

            if data.get("status") != "success":
                # Log API-level errors at debug (e.g. non-BTC 5m rejection)
                err = data.get("error", "unknown")
                log.debug(
                    "[Polymarket] past-results error for %s (variant=%s): %s",
                    slug, variant, err,
                )
                return 0.0

            results = data.get("data", {}).get("results", [])
            if not results:
                return 0.0

            return float(results[-1]["closePrice"])

        except (aiohttp.ClientError, TimeoutError, KeyError, ValueError, IndexError) as e:
            log.debug("[Polymarket] Price-to-beat fetch error (%s): %s", slug, e)
            return 0.0

    async def fetch_market_outcome(
        self, slug: str, start_time: float, duration: int
    ) -> Optional[str]:
        """Fetch the outcome of a resolved market."""
        parts = slug.split("-")
        symbol = parts[0].upper() if parts else "BTC"

        next_start = start_time + duration
        variant = VARIANT_MAP.get(duration, "fiveminute")
        next_iso = datetime.fromtimestamp(
            next_start, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        try:
            async with self._session.get(
                "https://polymarket.com/api/past-results",
                params={
                    "symbol": symbol,
                    "variant": variant,
                    "assetType": "crypto",
                    "currentEventStartTime": next_iso,
                },
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            if data.get("status") != "success":
                return None

            results = data.get("data", {}).get("results", [])
            if not results:
                return None

            expected_start = datetime.fromtimestamp(
                start_time, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            last = results[-1]
            if last.get("startTime") != expected_start:
                log.debug(
                    "[Polymarket] Outcome not ready: got %s, expected %s",
                    last.get("startTime"), expected_start,
                )
                return None
            outcome = last.get("outcome", "").lower()
            if outcome in ("up", "down"):
                return outcome
            return None

        except (aiohttp.ClientError, TimeoutError, KeyError, ValueError, IndexError):
            return None

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
        log.info("[Polymarket] Feed stopped")
