"""Polymarket market discovery and orderbook feed.

Uses deterministic slug-based discovery for BTC Up/Down markets:
  5-min:  btc-updown-5m-{floor(unix_ts / 300) * 300}
  15-min: btc-updown-15m-{floor(unix_ts / 900) * 900}

Reference price ("price to beat") is fetched from the Polymarket
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
    """Represents an active BTC Up/Down market on Polymarket."""

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
    outcome_prices: tuple[float, float] = (0.5, 0.5)

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
    timestamp: float = 0.0

    @property
    def mid_down(self) -> float:
        return 1.0 - self.mid_up

    # Aliases for compatibility with signal engine
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


def compute_slug(interval_sec: int, ts: float = 0) -> str:
    """Compute the deterministic market slug for a given time."""
    if ts <= 0:
        ts = time.time()
    aligned = int(ts // interval_sec) * interval_sec
    label = "5m" if interval_sec == INTERVAL_5M else "15m"
    return f"btc-updown-{label}-{aligned}"


class PolymarketFeed:
    """Discovers BTC Up/Down markets using slug-based lookup
    and polls orderbooks for active markets."""

    def __init__(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        clob_ws_url: str = (
            "wss://ws-subscriptions-clob.polymarket.com/ws/"
        ),
        intervals: tuple[int, ...] = (INTERVAL_5M, INTERVAL_15M),
        on_market_update: Optional[Callable] = None,
        on_orderbook_update: Optional[Callable] = None,
    ):
        self.gamma_url = gamma_url
        self.clob_url = clob_url
        self.clob_ws_url = clob_ws_url
        self.intervals = intervals
        self.on_market_update = on_market_update
        self.on_orderbook_update = on_orderbook_update

        self.active_markets: dict[str, MarketInfo] = {}
        self.orderbooks: dict[str, OrderbookState] = {}
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._running = True
        self._session = aiohttp.ClientSession()
        log.info("[Polymarket] Feed starting (slug-based discovery)")
        await asyncio.gather(
            self._market_discovery_loop(),
            self._orderbook_poll_loop(),
            self._lifecycle_loop(),
        )

    async def _market_discovery_loop(self) -> None:
        """Discover current and upcoming BTC Up/Down markets.

        Runs every 3 seconds to ensure we catch new markets
        within seconds of their start time.
        """
        while self._running:
            try:
                await self._discover_markets()
            except Exception as e:
                log.error("[Polymarket] Discovery error: %s", e)
            await asyncio.sleep(3)

    async def _lifecycle_loop(self) -> None:
        """Track market transitions: pending → active → expired.

        This loop ensures we react quickly when:
        - A new market window opens (capture reference price)
        - A market expires (trigger resolution)
        """
        while self._running:
            now = time.time()

            for cid, market in list(self.active_markets.items()):
                # Clean up expired markets
                if market.is_expired:
                    if cid in self.active_markets:
                        log.info(
                            "[Polymarket] Market expired: %s",
                            market.slug,
                        )
                        del self.active_markets[cid]
                        self.orderbooks.pop(cid, None)

                # Fetch price to beat once market has started
                elif (
                    market.reference_price <= 0
                    and now >= market.start_time
                ):
                    ptb = await self._fetch_price_to_beat(
                        market.start_time, market.duration_seconds
                    )
                    if ptb > 0:
                        market.reference_price = ptb
                        log.info(
                            "[Polymarket] Price to beat for "
                            "%s: $%.2f",
                            market.slug,
                            ptb,
                        )

                # Refresh accepting_orders status near start
                if (
                    not market.accepting_orders
                    and now >= market.start_time - 5
                ):
                    market.accepting_orders = True

            if self.on_market_update and self.active_markets:
                await self.on_market_update(self.active_markets)

            await asyncio.sleep(1)

    async def _discover_markets(self) -> None:
        """Find active markets using deterministic slug computation."""
        now = time.time()
        found = 0

        for interval in self.intervals:
            aligned = int(now // interval) * interval
            # Check current window and next 2 windows
            for offset in range(3):
                window_start = aligned + (offset * interval)
                slug = compute_slug(interval, window_start)

                # Skip if we already know this market
                existing = [
                    m for m in self.active_markets.values()
                    if m.slug == slug
                ]
                if existing:
                    continue

                market_info = await self._fetch_market_by_slug(slug)
                if market_info and not market_info.is_expired:
                    self.active_markets[market_info.condition_id] = (
                        market_info
                    )
                    found += 1
                    log.info(
                        "[Polymarket] Discovered: %s | "
                        "accepting=%s | T-%ds",
                        slug,
                        market_info.accepting_orders,
                        int(market_info.time_remaining),
                    )

        # Clean expired
        expired = [
            cid
            for cid, m in self.active_markets.items()
            if m.is_expired
        ]
        for cid in expired:
            del self.active_markets[cid]
            self.orderbooks.pop(cid, None)

        if found and self.on_market_update:
            await self.on_market_update(self.active_markets)

    async def _fetch_market_by_slug(
        self, slug: str
    ) -> Optional[MarketInfo]:
        """Fetch a single market from Gamma API by slug."""
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

            # Parse token IDs
            clob_tokens = market.get("clobTokenIds", "")
            if isinstance(clob_tokens, str) and clob_tokens:
                clob_tokens = json.loads(clob_tokens)
            token_up = clob_tokens[0] if len(clob_tokens) > 0 else ""
            token_down = (
                clob_tokens[1] if len(clob_tokens) > 1 else ""
            )

            # Parse outcome prices
            prices_raw = market.get("outcomePrices", "[]")
            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw
            p_up = float(prices[0]) if prices else 0.5
            p_down = float(prices[1]) if len(prices) > 1 else 0.5

            # Parse timestamps — use endDate (full datetime),
            # NOT endDateIso (just a date string)
            end_ts = self._parse_iso(market.get("endDate", ""))
            start_ts = self._parse_iso(
                market.get("eventStartTime", "")
            )

            # Fallback: compute from slug timestamp
            if end_ts <= 0 or start_ts <= 0:
                # Slug format: btc-updown-{5m|15m}-{unix_ts}
                parts = slug.rsplit("-", 1)
                try:
                    slug_ts = int(parts[-1])
                    label = "5m" if "5m" in slug else "15m"
                    interval_s = 300 if label == "5m" else 900
                    start_ts = float(slug_ts)
                    end_ts = float(slug_ts + interval_s)
                except (ValueError, IndexError):
                    pass

            duration = int(end_ts - start_ts) if (
                end_ts > 0 and start_ts > 0
            ) else 300

            # Fetch real Chainlink reference price from
            # past-results API. Only fetch if market has started
            # (price to beat is the Chainlink snapshot at start).
            ref_price = 0.0
            if start_ts > 0 and time.time() >= start_ts:
                ref_price = await self._fetch_price_to_beat(
                    start_ts, duration
                )
            if ref_price > 0:
                log.info(
                    "[Polymarket] Price to beat for %s: $%.2f",
                    slug,
                    ref_price,
                )

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
                accepting_orders=market.get(
                    "acceptingOrders", False
                ),
                outcome_prices=(p_up, p_down),
            )
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            log.debug("[Polymarket] Fetch error for %s: %s", slug, e)
            return None

    async def _orderbook_poll_loop(self) -> None:
        """Poll orderbooks for markets approaching expiry."""
        while self._running:
            for cid, market in list(self.active_markets.items()):
                if market.is_expired:
                    continue
                if not market.accepting_orders:
                    continue
                # Only poll orderbooks for markets < 5 min out
                # or always if it's a 5-min market
                if (
                    market.time_remaining > 300
                    and market.duration_seconds > 300
                ):
                    continue
                try:
                    await self._fetch_orderbook(cid, market)
                except Exception as e:
                    log.debug("[Polymarket] OB error: %s", e)
            await asyncio.sleep(2)

    async def _fetch_orderbook(
        self, cid: str, market: MarketInfo
    ) -> None:
        """Fetch orderbooks for Up and Down tokens."""
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

            except (aiohttp.ClientError, KeyError, ValueError):
                pass

        if self.on_orderbook_update:
            await self.on_orderbook_update(cid, ob)

    @staticmethod
    def _parse_iso(iso_str: str) -> float:
        """Parse ISO timestamp to unix seconds."""
        if not iso_str:
            return 0.0
        try:
            dt = datetime.fromisoformat(
                iso_str.replace("Z", "+00:00")
            )
            return dt.timestamp()
        except ValueError:
            return 0.0

    async def _fetch_price_to_beat(
        self,
        start_time: float,
        duration: int,
    ) -> float:
        """Fetch the Chainlink reference price from past-results API.

        The API returns historical market results including the exact
        Chainlink snapshot price at each window boundary. We pass
        the market's start_time as currentEventStartTime — the API
        returns completed windows up to that point, and the last
        closePrice equals the price to beat.

        Args:
            start_time: Unix timestamp of the market start.
            duration: Market duration in seconds (300 or 900).

        Returns:
            The Chainlink reference price, or 0.0 on failure.
        """
        variant = VARIANT_MAP.get(duration, "fiveminute")
        start_iso = datetime.fromtimestamp(
            start_time, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        try:
            async with self._session.get(
                "https://polymarket.com/api/past-results",
                params={
                    "symbol": "BTC",
                    "variant": variant,
                    "assetType": "crypto",
                    "currentEventStartTime": start_iso,
                },
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()

            if data.get("status") != "success":
                return 0.0

            results = data.get("data", {}).get("results", [])
            if not results:
                return 0.0

            # The last result's closePrice = price to beat
            return float(results[-1]["closePrice"])

        except (
            aiohttp.ClientError,
            TimeoutError,
            KeyError,
            ValueError,
            IndexError,
        ) as e:
            log.debug(
                "[Polymarket] Price-to-beat fetch error: %s", e
            )
            return 0.0

    async def fetch_market_outcome(
        self,
        start_time: float,
        duration: int,
    ) -> Optional[str]:
        """Fetch the actual resolved outcome from past-results API.

        To get the result of market starting at `start_time`, we query
        with the NEXT market's start time (start_time + duration), so
        that our market appears in the completed results.

        Returns:
            "up", "down", or None if not yet available.
        """
        next_start = start_time + duration
        variant = VARIANT_MAP.get(duration, "fiveminute")
        next_iso = datetime.fromtimestamp(
            next_start, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        try:
            async with self._session.get(
                "https://polymarket.com/api/past-results",
                params={
                    "symbol": "BTC",
                    "variant": variant,
                    "assetType": "crypto",
                    "currentEventStartTime": next_iso,
                },
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json",
                },
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

            # Verify the last result is actually OUR market
            # (not a previous window that's still the latest)
            expected_start = datetime.fromtimestamp(
                start_time, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            last = results[-1]
            if last.get("startTime") != expected_start:
                log.debug(
                    "[Polymarket] Outcome not ready: got %s, "
                    "expected %s",
                    last.get("startTime"), expected_start,
                )
                return None
            outcome = last.get("outcome", "").lower()
            if outcome in ("up", "down"):
                return outcome
            return None

        except (
            aiohttp.ClientError,
            TimeoutError,
            KeyError,
            ValueError,
            IndexError,
        ):
            return None

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
        log.info("[Polymarket] Feed stopped")
