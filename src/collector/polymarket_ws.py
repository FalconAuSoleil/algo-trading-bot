"""Polymarket market-channel WebSocket subscriber.

Closes the 2s polling gap: gives us event-driven orderbook dynamics,
the public trade tape (last_trade_price), tick-size flips, and
real-time resolution events.

Protocol (from docs.polymarket.com/developers/CLOB/websocket/market-channel):

    Connect: wss://ws-subscriptions-clob.polymarket.com/ws/market
    Subscribe:
      { "assets_ids": [token_id, ...], "type": "market",
        "custom_feature_enabled": true }

    Server emits JSON objects (sometimes a single object, sometimes a
    JSON array) with `event_type` in:
      - book             (on subscribe + after every trade)
      - price_change     (level add/modify/cancel — has price_changes[])
      - last_trade_price (public trade tape)
      - tick_size_change (price approached 0/1)
      - best_bid_ask     (BBO change; custom_feature_enabled)
      - new_market       (custom_feature_enabled)
      - market_resolved  (custom_feature_enabled)

Design:
  - One long-lived task `run()` with reconnect loop.
  - The collector periodically calls `set_assets(ids)`; if the set
    changed since last subscribe, we close & reopen the WS with the
    new subscription.
  - Ping every 10s to keep Polymarket from closing us (they respond
    with "PONG" text frames).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

import websockets
from websockets.exceptions import ConnectionClosed

from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from src.collector.recorder import CollectorRecorder

log = setup_logger("collector.polymarket_ws")

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL = 10.0
RECONNECT_BACKOFF_BASE = 2.0
RECONNECT_BACKOFF_MAX = 60.0


class PolymarketMarketWS:
    def __init__(self, recorder: "CollectorRecorder"):
        self.recorder = recorder
        self._target_assets: set[str] = set()
        self._current_subscribed: set[str] = set()
        self._ws = None
        self._running = False
        self._resubscribe_event = asyncio.Event()
        self.messages_received: int = 0

    # ─── Public API ───────────────────────────────────────────────────

    def set_assets(self, asset_ids: set[str]) -> None:
        """Update the set of asset_ids we should be subscribed to.
        Triggers a reconnect if the set changed."""
        new = {a for a in asset_ids if a}
        if new != self._target_assets:
            added = new - self._target_assets
            removed = self._target_assets - new
            log.info(
                "[PolyWS] Subscription target changed: +%d / -%d (target=%d)",
                len(added), len(removed), len(new),
            )
            self._target_assets = new
            self._resubscribe_event.set()

    async def run(self) -> None:
        """Main connect/subscribe/listen loop. Reconnects on any error."""
        self._running = True
        backoff = RECONNECT_BACKOFF_BASE
        while self._running:
            if not self._target_assets:
                # Nothing to subscribe to yet — wait and check again.
                try:
                    await asyncio.wait_for(self._resubscribe_event.wait(), timeout=5.0)
                    self._resubscribe_event.clear()
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self._connect_and_listen()
                backoff = RECONNECT_BACKOFF_BASE  # success → reset backoff
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as e:
                if self._running:
                    log.warning(
                        "[PolyWS] Closed (%s) — reconnect in %.0fs",
                        e, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, RECONNECT_BACKOFF_MAX)
            except Exception as e:
                if self._running:
                    log.warning(
                        "[PolyWS] %s — reconnect in %.0fs",
                        type(e).__name__, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, RECONNECT_BACKOFF_MAX)

    async def stop(self) -> None:
        self._running = False
        self._resubscribe_event.set()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        log.info("[PolyWS] Stopped")

    # ─── Internal ─────────────────────────────────────────────────────

    async def _connect_and_listen(self) -> None:
        assets = list(self._target_assets)
        log.info(
            "[PolyWS] Connecting + subscribing to %d asset_ids", len(assets)
        )
        async with websockets.connect(
            WS_URL, ping_interval=20, ping_timeout=10, max_size=2**22,
        ) as ws:
            self._ws = ws
            sub = {
                "assets_ids": assets,
                "type": "market",
                "custom_feature_enabled": True,
            }
            await ws.send(json.dumps(sub))
            self._current_subscribed = set(assets)
            self._resubscribe_event.clear()

            # Spawn a pinger. Polymarket's server closes idle connections.
            ping_task = asyncio.create_task(self._pinger(ws))
            try:
                async for raw in ws:
                    if not self._running:
                        break
                    if self._resubscribe_event.is_set() \
                            and self._target_assets != self._current_subscribed:
                        # New asset set — break out to reconnect
                        log.info("[PolyWS] Asset set changed, reconnecting")
                        break
                    await self._handle_raw(raw)
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._ws = None

    async def _pinger(self, ws) -> None:
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL)
                try:
                    await ws.send("PING")
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    async def _handle_raw(self, raw) -> None:
        if raw is None:
            return
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return
        s = raw.strip()
        if not s or s == "PONG":
            return
        try:
            payload = json.loads(s)
        except json.JSONDecodeError:
            log.debug("[PolyWS] Non-JSON frame: %s", s[:80])
            return

        self.messages_received += 1
        # Polymarket sends either a single object OR a JSON array of events.
        events = payload if isinstance(payload, list) else [payload]
        for ev in events:
            if isinstance(ev, dict):
                self._dispatch(ev)

    def _dispatch(self, ev: dict) -> None:
        et = ev.get("event_type") or ev.get("type") or ""
        ts = _parse_ts(ev.get("timestamp"))

        # Always archive the raw event for replay-safety.
        try:
            self.recorder.record_ws_event(event_type=et, timestamp=ts, payload=ev)
        except Exception as exc:
            log.debug("[PolyWS] ws_events archive failed: %s", exc)

        try:
            if et == "book":
                self._on_book(ev, ts)
            elif et == "price_change":
                self._on_price_change(ev, ts)
            elif et == "last_trade_price":
                self._on_trade(ev, ts)
            elif et == "tick_size_change":
                self._on_tick_change(ev, ts)
            elif et == "best_bid_ask":
                # Nothing extra to store beyond ws_events — BBO snapshots
                # already covered by market_states + orderbook_snapshots.
                pass
            elif et == "new_market":
                # Discovery flow handles metadata; just keep the raw trace.
                pass
            elif et == "market_resolved":
                self._on_resolved(ev, ts)
            else:
                log.debug("[PolyWS] unmapped event_type=%s", et)
        except Exception as exc:
            log.warning(
                "[PolyWS] dispatch error on %s: %s", et, exc,
            )

    # ─── Per-event handlers ───────────────────────────────────────────

    def _on_book(self, ev: dict, ts: float) -> None:
        """book event arrives after every trade affecting the book.
        We translate it to an orderbook_snapshots row.

        Since book events are per-asset_id (one side of Up/Down), we
        can't reconstruct a bilateral OrderbookState from a single event.
        What we DO want is the best_bid/best_ask on THIS side for this
        moment. We write a partial row — the other-side columns stay 0
        and the consumer (replay) joins against the latest REST snapshot
        to complete the picture."""
        from src.feeds.polymarket import OrderbookState
        asset_id = ev.get("asset_id") or ""
        market_id = ev.get("market") or ""
        bids = ev.get("bids", []) or []
        asks = ev.get("asks", []) or []
        # Bids come descending by convention; asks ascending. Defensive sort.
        best_bid = max((float(b.get("price", 0)) for b in bids), default=0.0)
        best_ask = min((float(a.get("price", 0)) for a in asks), default=0.0) \
            if asks else 0.0
        depth_bid = sum(
            float(b.get("size", 0)) * float(b.get("price", 0)) for b in bids[:10]
        )
        depth_ask = sum(
            float(a.get("size", 0)) * float(a.get("price", 0)) for a in asks[:10]
        )
        spread = (best_ask - best_bid) if (best_bid and best_ask) else 0.0
        book_hash = ev.get("hash", "") or ""

        ob = OrderbookState(timestamp=ts)
        # Figure out which side this asset_id is (up or down) via the map.
        mapping = self.recorder._asset_to_market.get(asset_id)
        outcome = mapping[1] if mapping else "up"
        if outcome == "up":
            ob.best_bid_up = best_bid
            ob.best_ask_up = best_ask
            ob.depth_bid_up = depth_bid
            ob.depth_ask_up = depth_ask
            ob.spread_up = spread
            if best_bid and best_ask:
                ob.mid_up = (best_bid + best_ask) / 2
            ob.book_hash_up = book_hash
        else:
            ob.best_bid_down = best_bid
            ob.best_ask_down = best_ask
            ob.depth_bid_down = depth_bid
            ob.depth_ask_down = depth_ask
            ob.spread_down = spread
            ob.book_hash_down = book_hash

        if market_id:
            self.recorder.record_orderbook(market_id, ob, source="ws")

    def _on_price_change(self, ev: dict, ts: float) -> None:
        """price_change carries a list of level modifications.

        Wire-format note (verified against live feed 2026-04-19):
          - asset_id is NESTED inside each price_changes[] item
            (one event can carry changes for both Up and Down tokens).
          - Top-level may have no asset_id at all.
          - side comes uppercase ("BUY"/"SELL").
        """
        top_asset = ev.get("asset_id") or ""
        # Docs show price_changes[] with {asset_id, price, side, size,
        # best_bid, best_ask, hash}
        changes = ev.get("price_changes")
        if changes is None:
            # Some formats inline the change on the top-level object.
            changes = [{
                "asset_id": top_asset,
                "price": ev.get("price"),
                "side": ev.get("side"),
                "size": ev.get("size"),
                "best_bid": ev.get("best_bid"),
                "best_ask": ev.get("best_ask"),
                "hash": ev.get("hash"),
            }]
        for c in changes:
            try:
                price = float(c.get("price") or 0)
                size = float(c.get("size") or 0)
            except (TypeError, ValueError):
                continue
            # Per-change asset_id with fallback to top-level
            asset_id = c.get("asset_id") or top_asset
            if not asset_id:
                continue
            side = c.get("side") or ""
            bb_raw = c.get("best_bid")
            ba_raw = c.get("best_ask")
            try:
                bb = float(bb_raw) if bb_raw not in (None, "") else None
            except (TypeError, ValueError):
                bb = None
            try:
                ba = float(ba_raw) if ba_raw not in (None, "") else None
            except (TypeError, ValueError):
                ba = None
            self.recorder.record_orderbook_delta(
                asset_id=asset_id,
                timestamp=ts,
                side=side,
                price=price,
                size=size,
                best_bid=bb,
                best_ask=ba,
                book_hash=str(c.get("hash") or ev.get("hash") or ""),
            )

    def _on_trade(self, ev: dict, ts: float) -> None:
        asset_id = ev.get("asset_id") or ""
        try:
            price = float(ev.get("price") or 0)
            size = float(ev.get("size") or 0)
        except (TypeError, ValueError):
            return
        side = ev.get("side") or ""
        try:
            fee_bps = float(ev.get("fee_rate_bps") or 0)
        except (TypeError, ValueError):
            fee_bps = 0.0
        self.recorder.record_trade(
            asset_id=asset_id, timestamp=ts,
            price=price, size=size, side=side, fee_rate_bps=fee_bps,
        )

    def _on_tick_change(self, ev: dict, ts: float) -> None:
        asset_id = ev.get("asset_id") or ""
        try:
            old = float(ev.get("old_tick_size") or 0)
            new = float(ev.get("new_tick_size") or 0)
        except (TypeError, ValueError):
            return
        self.recorder.record_tick_change(
            asset_id=asset_id, timestamp=ts,
            old_tick_size=old, new_tick_size=new,
        )

    def _on_resolved(self, ev: dict, ts: float) -> None:
        market_id = ev.get("market") or ev.get("id") or ""
        winner_asset = ev.get("winning_asset_id") or ""
        winner_outcome = ev.get("winning_outcome") or ""
        if not market_id:
            return
        self.recorder.handle_market_resolved_ws(
            market_id=market_id,
            winning_asset_id=winner_asset,
            winning_outcome=winner_outcome,
        )


def _parse_ts(v) -> float:
    """Polymarket timestamps come as int ms (or sometimes str). Normalise
    to float seconds. Fallback to wall clock."""
    if v is None:
        return time.time()
    try:
        v = float(v)
    except (TypeError, ValueError):
        return time.time()
    # Heuristic: >1e11 means milliseconds, else assume seconds.
    return v / 1000.0 if v > 1e11 else v
