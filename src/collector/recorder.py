"""Async batch writer for the collector SQLite DB.

Design:
  - Callers (feed callbacks + signal loop) call `record_*` methods which
    just append to in-memory queues. No blocking I/O on the hot path.
  - A background `_flush_loop` wakes up every FLUSH_INTERVAL seconds and
    writes the queued rows in a single transaction each. SQLite with WAL
    + one writer is happy.
  - On shutdown we drain the queues once more and close the connection.

All timestamps are float unix seconds. All prices are USD floats.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json

from src.collector.schema import init_db, table_counts, db_size_mb
from src.utils.logger import setup_logger

log = setup_logger("collector.recorder")


FLUSH_INTERVAL = 3.0  # seconds between batch writes

# ── Firehose throttles (disk-blowout guards) ──────────────────────────────
# The Polymarket market-channel WS pumps thousands of events/s; without
# throttling the DB grows ~5 GB/day. Defaults are tuned to keep it at
# ~500 MB/day while still providing enough data for replay.
#
# Knobs (env vars, all optional):
#   COLLECTOR_WS_RAW_ENABLED     "0" to drop the ws_events firehose entirely
#                                (default: "0" — you rarely need raw bytes).
#   COLLECTOR_DELTAS_SAMPLE_N    Keep 1 of every N price_change deltas per
#                                asset. Default: 10.
#   COLLECTOR_MAX_DB_MB          If the DB file exceeds this many MB, stop
#                                writing deltas/ws_events (keep the "clean"
#                                tables: markets/snapshots/trades/signals).
#                                Default: 4000 (~4 GB).
WS_RAW_ENABLED = os.environ.get("COLLECTOR_WS_RAW_ENABLED", "0") in ("1", "true", "True")
DELTAS_SAMPLE_N = max(1, int(os.environ.get("COLLECTOR_DELTAS_SAMPLE_N", "10")))
MAX_DB_MB = float(os.environ.get("COLLECTOR_MAX_DB_MB", "4000"))


@dataclass
class CollectorRecorder:
    db_path: str
    _conn: sqlite3.Connection | None = None

    _markets_q: list[dict[str, Any]] = field(default_factory=list)
    _market_updates_q: list[dict[str, Any]] = field(default_factory=list)
    _ob_q: list[tuple] = field(default_factory=list)
    _deltas_q: list[tuple] = field(default_factory=list)
    _trades_q: list[tuple] = field(default_factory=list)
    _tick_changes_q: list[tuple] = field(default_factory=list)
    _market_states_q: list[tuple] = field(default_factory=list)
    _ticks_q: dict[tuple[str, int], tuple] = field(default_factory=dict)
    _signals_q: list[tuple] = field(default_factory=list)
    _ws_raw_q: list[tuple] = field(default_factory=list)

    _known_markets: set[str] = field(default_factory=set)
    _resolved_markets: set[str] = field(default_factory=set)

    # asset_id -> (market_id, outcome) lookup for enriching WS events
    # (WS events carry asset_id + market; we map asset_id to our internal
    # up/down classification so downstream queries are easy)
    _asset_to_market: dict[str, tuple[str, str]] = field(default_factory=dict)

    _running: bool = False
    _flush_task: asyncio.Task | None = None

    # Per-asset counter for delta sampling (1 in N keeps index % N == 0).
    _delta_counter: dict[str, int] = field(default_factory=dict)
    # Cached "DB too big, stop firehosing" flag — refreshed by the flush
    # loop (cheap) to avoid stat()ing the file on every hot-path call.
    _firehose_blocked: bool = False
    _firehose_warned: bool = False

    # ─── Lifecycle ────────────────────────────────────────────────────

    def open(self) -> None:
        self._conn = init_db(self.db_path)
        # Hydrate known_markets so we don't re-insert on restart
        cur = self._conn.execute("SELECT market_id FROM markets")
        self._known_markets = {row[0] for row in cur.fetchall()}
        cur = self._conn.execute(
            "SELECT market_id FROM markets WHERE resolved_outcome IS NOT NULL"
        )
        self._resolved_markets = {row[0] for row in cur.fetchall()}
        # Hydrate asset_id -> (market_id, outcome) map so WS events from
        # markets carried over from a previous session can still be tagged.
        cur = self._conn.execute(
            "SELECT market_id, token_id_up, token_id_down FROM markets"
        )
        for mid, tup, tdn in cur.fetchall():
            if tup:
                self._asset_to_market[tup] = (mid, "up")
            if tdn:
                self._asset_to_market[tdn] = (mid, "down")
        size = db_size_mb(self._conn)
        log.info(
            "[Recorder] DB open at %s (existing: %d markets, %d resolved, %.1f MB)",
            self.db_path, len(self._known_markets), len(self._resolved_markets), size,
        )

    async def start(self) -> None:
        if not self._conn:
            self.open()
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop(), name="recorder_flush")
        log.info("[Recorder] Background flush loop started (%.1fs)", FLUSH_INTERVAL)

    async def stop(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final drain
        self._flush_sync()
        if self._conn:
            self._conn.close()
            self._conn = None
        log.info("[Recorder] Closed")

    # ─── Hot-path recording (non-blocking) ────────────────────────────

    def record_market(self, market) -> None:
        """Upsert a MarketInfo. Called from polymarket_feed.on_market_update."""
        cid = market.condition_id
        if not cid:
            return
        # First-time discovery → queue full insert + populate asset_id map
        if cid not in self._known_markets:
            self._markets_q.append({
                "market_id": cid,
                "slug": market.slug,
                "asset": self._asset_from_slug(market.slug),
                "interval_sec": market.duration_seconds,
                "start_time": market.start_time,
                "end_time": market.end_time,
                "duration_seconds": market.duration_seconds,
                "reference_price": market.reference_price,
                "token_id_up": market.token_id_up,
                "token_id_down": market.token_id_down,
                "question": market.question,
                "discovered_at": time.time(),
            })
            self._known_markets.add(cid)
            if market.token_id_up:
                self._asset_to_market[market.token_id_up] = (cid, "up")
            if market.token_id_down:
                self._asset_to_market[market.token_id_down] = (cid, "down")
        elif market.reference_price > 0:
            # Known market — ref_price may have been filled late by lifecycle loop
            self._market_updates_q.append({
                "market_id": cid,
                "reference_price": market.reference_price,
            })

    def record_orderbook(self, cid: str, ob, source: str = "rest") -> None:
        """Queue an orderbook snapshot. Called from on_orderbook_update
        (REST poll) and from the WS subscriber (on 'book' event).

        source='rest' comes from _fetch_orderbook; 'ws' from the market WS
        'book' event (which fires on subscribe + after every trade)."""
        if not cid or ob is None:
            return
        # tick_size/last_trade_price are per-side — use the up side as primary
        # (it always exists for valid Up/Down markets) and fall back to down.
        tick = getattr(ob, "tick_size_up", 0.01) or getattr(ob, "tick_size_down", 0.01)
        ltp  = getattr(ob, "last_trade_price_up", 0.0) \
            or getattr(ob, "last_trade_price_down", 0.0)
        bh   = getattr(ob, "book_hash_up", "") or getattr(ob, "book_hash_down", "")
        self._ob_q.append((
            cid,
            ob.timestamp or time.time(),
            ob.best_bid_up, ob.best_ask_up,
            ob.best_bid_down, ob.best_ask_down,
            ob.depth_bid_up, ob.depth_ask_up,
            ob.depth_bid_down, ob.depth_ask_down,
            ob.spread_up, ob.spread_down,
            ob.mid_up,
            tick, ltp, bh, source,
        ))

    # ─── WS-driven recording (trades, deltas, tick changes) ───────────

    def record_trade(
        self,
        asset_id: str,
        timestamp: float,
        price: float,
        size: float,
        side: str,
        fee_rate_bps: float = 0.0,
    ) -> None:
        """Record a WS last_trade_price event."""
        mid_outcome = self._asset_to_market.get(asset_id)
        if not mid_outcome:
            return
        market_id, outcome = mid_outcome
        self._trades_q.append((
            market_id, asset_id, outcome,
            float(timestamp), float(price), float(size),
            str(side or "").lower(), float(fee_rate_bps or 0),
        ))

    def record_orderbook_delta(
        self,
        asset_id: str,
        timestamp: float,
        side: str,
        price: float,
        size: float,
        best_bid: float | None = None,
        best_ask: float | None = None,
        book_hash: str = "",
    ) -> None:
        """Record a WS price_change event (one level add/modify/cancel)."""
        mid_outcome = self._asset_to_market.get(asset_id)
        if not mid_outcome:
            return
        # Disk-guard: drop firehose writes when the DB is past the cap.
        if self._firehose_blocked:
            return
        # Sample 1 of every N per asset — deltas are ~2 800/s, we don't
        # need every single level cancel for replay. Snapshots still go
        # through every trade event (book channel) so the book stays
        # reconstructable.
        if DELTAS_SAMPLE_N > 1:
            n = self._delta_counter.get(asset_id, 0)
            self._delta_counter[asset_id] = n + 1
            if n % DELTAS_SAMPLE_N != 0:
                return
        market_id, outcome = mid_outcome
        self._deltas_q.append((
            market_id, asset_id, outcome, float(timestamp),
            str(side or "").lower(), float(price), float(size),
            None if best_bid is None else float(best_bid),
            None if best_ask is None else float(best_ask),
            book_hash or "",
        ))

    def record_tick_change(
        self,
        asset_id: str,
        timestamp: float,
        old_tick_size: float,
        new_tick_size: float,
    ) -> None:
        """Record a WS tick_size_change event."""
        mid_outcome = self._asset_to_market.get(asset_id)
        if not mid_outcome:
            return
        market_id, _ = mid_outcome
        self._tick_changes_q.append((
            market_id, asset_id, float(timestamp),
            float(old_tick_size), float(new_tick_size),
        ))

    def record_market_state(
        self, market_id: str, timestamp: float, fields: dict[str, Any]
    ) -> None:
        """Record a Gamma dynamic snapshot. `fields` is a dict of optional
        columns — missing ones become NULL."""
        if not market_id:
            return
        self._market_states_q.append((
            market_id, float(timestamp),
            fields.get("best_bid"),
            fields.get("best_ask"),
            fields.get("last_trade_price"),
            fields.get("spread"),
            fields.get("volume_clob"),
            fields.get("volume_24h_clob"),
            fields.get("liquidity_clob"),
            fields.get("open_interest"),
            fields.get("one_hour_change"),
            fields.get("one_day_change"),
            1 if fields.get("accepting_orders") else 0,
            fields.get("uma_status"),
            fields.get("closed_time"),
            fields.get("rewards_min_size"),
            fields.get("rewards_max_spread"),
            fields.get("min_tick_size"),
            fields.get("min_order_size"),
            fields.get("maker_fee_bps"),
            fields.get("taker_fee_bps"),
            1 if fields.get("neg_risk") else 0,
        ))

    def record_ws_event(
        self, event_type: str, timestamp: float, payload: dict[str, Any],
    ) -> None:
        """Catch-all raw WS event log. Disabled by default (COLLECTOR_WS_RAW_ENABLED=1
        to turn it back on) — the parsed tables (book/deltas/trades) already
        cover replay; the raw log mainly doubles the disk footprint."""
        if not WS_RAW_ENABLED or self._firehose_blocked:
            return
        asset_id = payload.get("asset_id", "") or ""
        market_id = payload.get("market", "") or ""
        self._ws_raw_q.append((
            float(timestamp), str(event_type), market_id, asset_id,
            json.dumps(payload, separators=(",", ":")),
        ))

    def handle_market_resolved_ws(
        self, market_id: str, winning_asset_id: str, winning_outcome: str,
    ) -> None:
        """WS market_resolved arrives immediately at resolution — much
        faster than our 30s past-results poll. Resolve the markets row now."""
        if not market_id:
            return
        outcome_norm = (winning_outcome or "").lower()
        # winning_outcome from WS is typically "Up" or "Down"
        if outcome_norm not in ("up", "down"):
            # Try to infer from winning_asset_id
            mapping = self._asset_to_market.get(winning_asset_id or "")
            if mapping:
                outcome_norm = mapping[1]
        if outcome_norm not in ("up", "down"):
            return
        self.mark_resolved(market_id, outcome_norm, 0.0)  # price back-filled later

    def record_price_tick(
        self,
        asset: str,
        timestamp: float,
        binance_price: float,
        chainlink_price: float,
        chainlink_ts: float,
    ) -> None:
        """Queue a price tick. Dedup to 1 row per (asset, whole-second) for
        disk efficiency — newer ticks in same second overwrite older."""
        key = (asset, int(timestamp))
        self._ticks_q[key] = (
            float(timestamp), asset,
            float(binance_price), float(chainlink_price), float(chainlink_ts),
        )

    def record_signal(
        self,
        sig,
        chainlink_price: float,
        binance_price: float,
        ob=None,
    ) -> None:
        """Queue a shadow signal. Always records, regardless of action."""
        ob_bid_yes = ob_ask_yes = ob_bid_no = ob_ask_no = 0.0
        if ob is not None:
            ob_bid_yes = ob.best_bid_up
            ob_ask_yes = ob.best_ask_up
            ob_bid_no  = ob.best_bid_down
            ob_ask_no  = ob.best_ask_down

        self._signals_q.append((
            sig.timestamp,
            sig.market_id,
            sig.slug,
            self._asset_from_slug(sig.slug),
            sig.action,
            sig.side,
            sig.entry_price,
            sig.p_true,
            sig.p_market,
            sig.edge,
            sig.delta_chainlink,
            sig.delta_binance,
            sig.sigma,
            sig.time_remaining_sec,
            1 if sig.filters_passed else 0,
            ",".join(sig.filter_reasons) if sig.filter_reasons else "",
            sig.strategy_used,
            sig.size_usd,
            sig.confidence,
            chainlink_price,
            binance_price,
            sig.reference_price,
            sig.micro.oracle_age_sec,
            ob_bid_yes, ob_ask_yes, ob_bid_no, ob_ask_no,
        ))

    # ─── Resolution back-fill ─────────────────────────────────────────

    def mark_resolved(
        self, market_id: str, outcome: str, resolved_price: float
    ) -> None:
        """Update resolved_* for a market.

        Two call sites:
          - Past-results polling (resolution_loop) — knows the close price.
          - WS market_resolved event — immediate, but no price; we keep
            the column at 0 and let the past-results poll later UPDATE it
            (guarded so it only fills-in zero prices, never overwrites).
        """
        if not market_id or market_id in self._resolved_markets:
            # Already resolved — but allow a later poll to back-fill the
            # close price if the first source (WS) didn't have it.
            if market_id in self._resolved_markets and resolved_price > 0 \
                    and self._conn is not None:
                try:
                    self._conn.execute(
                        "UPDATE markets SET resolved_price=? "
                        "WHERE market_id=? AND (resolved_price IS NULL "
                        "OR resolved_price<=0)",
                        (float(resolved_price), market_id),
                    )
                except sqlite3.Error:
                    pass
            return
        if outcome not in ("up", "down"):
            return
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "UPDATE markets SET resolved_outcome=?, resolved_price=?, "
                "resolved_time=? WHERE market_id=?",
                (outcome, float(resolved_price), time.time(), market_id),
            )
            self._resolved_markets.add(market_id)
            log.info(
                "[Recorder] Resolved %s -> %s @ $%.4f",
                market_id[:12], outcome, resolved_price,
            )
        except sqlite3.Error as e:
            log.error("[Recorder] mark_resolved error: %s", e)

    def unresolved_markets(self, now: float | None = None) -> list[dict]:
        """Return markets past end_time but not yet resolved."""
        if self._conn is None:
            return []
        now = now or time.time()
        cur = self._conn.execute(
            "SELECT market_id, slug, start_time, duration_seconds "
            "FROM markets WHERE resolved_outcome IS NULL AND end_time <= ?",
            (now,),
        )
        return [
            {"market_id": r[0], "slug": r[1],
             "start_time": r[2], "duration_seconds": r[3]}
            for r in cur.fetchall()
        ]

    # ─── Stats (for periodic status log) ──────────────────────────────

    def stats(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "queued_markets":       len(self._markets_q),
            "queued_ob":            len(self._ob_q),
            "queued_deltas":        len(self._deltas_q),
            "queued_trades":        len(self._trades_q),
            "queued_tick_changes":  len(self._tick_changes_q),
            "queued_market_states": len(self._market_states_q),
            "queued_ticks":         len(self._ticks_q),
            "queued_signals":       len(self._signals_q),
            "queued_ws_raw":        len(self._ws_raw_q),
        }
        if self._conn is not None:
            base.update(table_counts(self._conn))
            base["db_size_mb"] = round(db_size_mb(self._conn), 1)
        return base

    # ─── Internal: disk guard ─────────────────────────────────────────

    def _check_disk_guard(self) -> None:
        """Flip _firehose_blocked if the DB is past MAX_DB_MB. Runs every
        FLUSH_INTERVAL so we don't stat() on every record_* call."""
        if self._conn is None:
            return
        try:
            size_mb = db_size_mb(self._conn)
        except Exception:
            return
        if size_mb >= MAX_DB_MB:
            if not self._firehose_blocked:
                self._firehose_blocked = True
                log.warning(
                    "[Recorder] DB %.0f MB >= cap %.0f MB -> pausing "
                    "deltas/ws_raw writes (snapshots/trades/signals continue)",
                    size_mb, MAX_DB_MB,
                )
        else:
            if self._firehose_blocked:
                self._firehose_blocked = False
                log.info(
                    "[Recorder] DB back under cap (%.0f < %.0f MB) -> "
                    "firehose resumed", size_mb, MAX_DB_MB,
                )

    # ─── Internal: flush loop ─────────────────────────────────────────

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(FLUSH_INTERVAL)
                self._check_disk_guard()
                self._flush_sync()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("[Recorder] Flush error: %s", exc, exc_info=True)

    def _flush_sync(self) -> None:
        if self._conn is None:
            return

        # Grab current queues and reset them before we start writing so the
        # hot path can keep appending to fresh containers.
        markets = self._markets_q
        self._markets_q = []
        updates = self._market_updates_q
        self._market_updates_q = []
        obs = self._ob_q
        self._ob_q = []
        deltas = self._deltas_q
        self._deltas_q = []
        trades = self._trades_q
        self._trades_q = []
        tick_changes = self._tick_changes_q
        self._tick_changes_q = []
        market_states = self._market_states_q
        self._market_states_q = []
        ticks = list(self._ticks_q.values())
        self._ticks_q = {}
        signals = self._signals_q
        self._signals_q = []
        ws_raw = self._ws_raw_q
        self._ws_raw_q = []

        if not (markets or updates or obs or deltas or trades or tick_changes
                or market_states or ticks or signals or ws_raw):
            return

        try:
            with self._conn:  # implicit transaction
                if markets:
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO markets "
                        "(market_id, slug, asset, interval_sec, start_time, "
                        "end_time, duration_seconds, reference_price, "
                        "token_id_up, token_id_down, question, discovered_at) "
                        "VALUES (:market_id, :slug, :asset, :interval_sec, "
                        ":start_time, :end_time, :duration_seconds, "
                        ":reference_price, :token_id_up, :token_id_down, "
                        ":question, :discovered_at)",
                        markets,
                    )
                if updates:
                    self._conn.executemany(
                        "UPDATE markets SET reference_price = :reference_price "
                        "WHERE market_id = :market_id AND reference_price <= 0",
                        updates,
                    )
                if obs:
                    self._conn.executemany(
                        "INSERT INTO orderbook_snapshots "
                        "(market_id, timestamp, best_bid_yes, best_ask_yes, "
                        "best_bid_no, best_ask_no, depth_bid_yes, depth_ask_yes, "
                        "depth_bid_no, depth_ask_no, spread_yes, spread_no, mid_yes, "
                        "tick_size, last_trade_price, book_hash, book_source) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        obs,
                    )
                if deltas:
                    self._conn.executemany(
                        "INSERT INTO orderbook_deltas "
                        "(market_id, asset_id, outcome, timestamp, side, price, "
                        "size, best_bid, best_ask, book_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        deltas,
                    )
                if trades:
                    self._conn.executemany(
                        "INSERT INTO trades "
                        "(market_id, asset_id, outcome, timestamp, price, size, "
                        "side, fee_rate_bps) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        trades,
                    )
                if tick_changes:
                    self._conn.executemany(
                        "INSERT INTO tick_changes "
                        "(market_id, asset_id, timestamp, old_tick_size, new_tick_size) "
                        "VALUES (?, ?, ?, ?, ?)",
                        tick_changes,
                    )
                if market_states:
                    self._conn.executemany(
                        "INSERT INTO market_states "
                        "(market_id, timestamp, best_bid, best_ask, last_trade_price, "
                        "spread, volume_clob, volume_24h_clob, liquidity_clob, "
                        "open_interest, one_hour_change, one_day_change, "
                        "accepting_orders, uma_status, closed_time, rewards_min_size, "
                        "rewards_max_spread, min_tick_size, min_order_size, "
                        "maker_fee_bps, taker_fee_bps, neg_risk) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                        "?, ?, ?, ?, ?, ?)",
                        market_states,
                    )
                if ticks:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO price_ticks "
                        "(timestamp, asset, binance_price, chainlink_price, chainlink_ts) "
                        "VALUES (?, ?, ?, ?, ?)",
                        ticks,
                    )
                if signals:
                    self._conn.executemany(
                        "INSERT INTO shadow_signals "
                        "(timestamp, market_id, slug, asset, action, side, "
                        "entry_price, p_true, p_market, edge, delta_chainlink, "
                        "delta_binance, sigma, time_remaining_sec, filters_passed, "
                        "filter_reasons, strategy_used, size_usd, confidence, "
                        "chainlink_price, binance_price, reference_price, "
                        "oracle_age_sec, best_bid_yes, best_ask_yes, "
                        "best_bid_no, best_ask_no) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        signals,
                    )
                if ws_raw:
                    self._conn.executemany(
                        "INSERT INTO ws_events "
                        "(timestamp, event_type, market_id, asset_id, payload_json) "
                        "VALUES (?, ?, ?, ?, ?)",
                        ws_raw,
                    )
        except sqlite3.Error as exc:
            log.error("[Recorder] DB write error: %s", exc)

    # ─── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _asset_from_slug(slug: str) -> str:
        if not slug:
            return ""
        return slug.split("-", 1)[0].upper()
