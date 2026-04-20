"""SQLite schema for the continuous data collector.

RETENTION: UNLIMITED. The DB accumulates forever — nothing deletes rows.
Designed to run 24/7 for months/years. WAL mode keeps writer + parallel
readers (replay backtester) non-blocking.

Tables (all BTC/ETH/SOL/XRP x 5m/15m markets):

  markets
    One row per Polymarket event discovered.
    Filled at discovery; resolved_* filled by WS market_resolved event OR
    by past-results polling fallback.

  orderbook_snapshots
    REST /book snapshots every 2s AND WebSocket book snapshots (emitted
    on trade). book_source='rest'|'ws'. Captures best bid/ask YES+NO,
    depth top-10, spreads, tick_size, last_trade_price, book_hash.

  orderbook_deltas
    WS price_change events: each row is one level add/modify/cancel
    (size=0 means level removed). Lets replay reconstruct the book
    between full snapshots at sub-second granularity.

  trades
    WS last_trade_price events — public trade tape. Critical: tells
    replay which side took liquidity and at what price. This is the
    single biggest fidelity win over the REST-only collector.

  tick_changes
    WS tick_size_change events. Load-bearing as prices approach 0/1
    (tick flips from 0.01 -> 0.001) — without this, replay can fill
    at invalid off-grid prices.

  market_states
    Periodic Gamma /events re-poll (~30s) capturing dynamic fields:
    bestBid/bestAsk/lastTradePrice/volume24hrClob/openInterest/
    umaResolutionStatus/acceptingOrders. Gives us liquidity context
    for the BBO and an authoritative resolution signal.

  price_ticks
    Downsampled (1 Hz) per-asset snapshot of Binance + Chainlink.
    Chainlink timestamp preserved — oracle_age replay-able exactly.

  shadow_signals
    Every signal.evaluate() result from the shadow engine, including
    non-BUY decisions. Lets a grid-search with LOWER edge thresholds
    turn past NONE into BUY and still have the underlying state.

  ws_events (catch-all)
    Raw JSON of every WS event (incl. best_bid_ask and any future
    event types). If we later realise a field matters, we can re-parse
    history from here without having lost anything.

Disk usage rough estimate (continuous):
  markets:            ~150 rows/day      ~50 MB/year
  orderbook_*:        ~0.5 M rows/day    ~30 GB/year
  trades:             ~100k rows/day     ~5 GB/year
  shadow_signals:     ~0.5 M rows/day    ~30 GB/year
  price_ticks:        ~345k rows/day     ~15 GB/year
  Total: ~80 GB/year. Add disk monitoring or archive old-then-vacuum
  if the collection PC runs short on space.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS markets (
    market_id         TEXT PRIMARY KEY,
    slug              TEXT NOT NULL,
    asset             TEXT NOT NULL,              -- BTC/ETH/SOL/XRP
    interval_sec      INTEGER NOT NULL,            -- 300 or 900
    start_time        REAL NOT NULL,               -- unix ts
    end_time          REAL NOT NULL,
    duration_seconds  INTEGER NOT NULL,
    reference_price   REAL NOT NULL DEFAULT 0,     -- price to beat
    token_id_up       TEXT,
    token_id_down     TEXT,
    question          TEXT,
    resolved_outcome  TEXT,                        -- 'up' | 'down' | NULL
    resolved_price    REAL,                        -- chainlink close
    resolved_time     REAL,
    discovered_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_markets_asset_start  ON markets(asset, start_time);
CREATE INDEX IF NOT EXISTS idx_markets_end_unresolved ON markets(end_time)
    WHERE resolved_outcome IS NULL;


CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id         TEXT NOT NULL,
    timestamp         REAL NOT NULL,
    best_bid_yes      REAL NOT NULL DEFAULT 0,
    best_ask_yes      REAL NOT NULL DEFAULT 0,
    best_bid_no       REAL NOT NULL DEFAULT 0,
    best_ask_no       REAL NOT NULL DEFAULT 0,
    depth_bid_yes     REAL NOT NULL DEFAULT 0,
    depth_ask_yes     REAL NOT NULL DEFAULT 0,
    depth_bid_no      REAL NOT NULL DEFAULT 0,
    depth_ask_no      REAL NOT NULL DEFAULT 0,
    spread_yes        REAL NOT NULL DEFAULT 0,
    spread_no         REAL NOT NULL DEFAULT 0,
    mid_yes           REAL NOT NULL DEFAULT 0.5,
    tick_size         REAL NOT NULL DEFAULT 0.01,
    last_trade_price  REAL NOT NULL DEFAULT 0,
    book_hash         TEXT,
    book_source       TEXT NOT NULL DEFAULT 'rest'    -- 'rest' | 'ws'
);
CREATE INDEX IF NOT EXISTS idx_ob_market_ts  ON orderbook_snapshots(market_id, timestamp);


-- WS price_change events: granular level changes between full snapshots.
-- One row per (price, side) level change. size=0 means level cancelled.
CREATE TABLE IF NOT EXISTS orderbook_deltas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id       TEXT NOT NULL,
    asset_id        TEXT NOT NULL,         -- token_id (up or down)
    outcome         TEXT NOT NULL,         -- 'up' | 'down'
    timestamp       REAL NOT NULL,
    side            TEXT NOT NULL,         -- 'buy' | 'sell'
    price           REAL NOT NULL,
    size            REAL NOT NULL,         -- 0 = level cancelled
    best_bid        REAL,
    best_ask        REAL,
    book_hash       TEXT
);
CREATE INDEX IF NOT EXISTS idx_deltas_market_ts ON orderbook_deltas(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_deltas_asset_ts  ON orderbook_deltas(asset_id, timestamp);


-- WS last_trade_price events: public trade tape.
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id       TEXT NOT NULL,
    asset_id        TEXT NOT NULL,
    outcome         TEXT NOT NULL,         -- 'up' | 'down'
    timestamp       REAL NOT NULL,
    price           REAL NOT NULL,
    size            REAL NOT NULL,
    side            TEXT NOT NULL,         -- 'buy' | 'sell' (taker side)
    fee_rate_bps    REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_trades_market_ts ON trades(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_asset_ts  ON trades(asset_id, timestamp);


-- WS tick_size_change events.
CREATE TABLE IF NOT EXISTS tick_changes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id       TEXT NOT NULL,
    asset_id        TEXT NOT NULL,
    timestamp       REAL NOT NULL,
    old_tick_size   REAL NOT NULL,
    new_tick_size   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tick_changes_market_ts ON tick_changes(market_id, timestamp);


-- Gamma dynamic-field snapshots (~every 30s per active market).
CREATE TABLE IF NOT EXISTS market_states (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id          TEXT NOT NULL,
    timestamp          REAL NOT NULL,
    best_bid           REAL,
    best_ask           REAL,
    last_trade_price   REAL,
    spread             REAL,
    volume_clob        REAL,
    volume_24h_clob    REAL,
    liquidity_clob     REAL,
    open_interest      REAL,
    one_hour_change    REAL,
    one_day_change     REAL,
    accepting_orders   INTEGER,
    uma_status         TEXT,               -- umaResolutionStatus
    closed_time        REAL,
    rewards_min_size   REAL,
    rewards_max_spread REAL,
    min_tick_size      REAL,
    min_order_size     REAL,
    maker_fee_bps      REAL,
    taker_fee_bps      REAL,
    neg_risk           INTEGER
);
CREATE INDEX IF NOT EXISTS idx_mstates_market_ts ON market_states(market_id, timestamp);


-- Catch-all raw WS log (for any event type we haven't mapped yet).
CREATE TABLE IF NOT EXISTS ws_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL NOT NULL,
    event_type   TEXT NOT NULL,
    market_id    TEXT,
    asset_id     TEXT,
    payload_json TEXT NOT NULL               -- full JSON body for replay
);
CREATE INDEX IF NOT EXISTS idx_wsevents_type_ts ON ws_events(event_type, timestamp);


CREATE TABLE IF NOT EXISTS price_ticks (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            REAL NOT NULL,
    asset                TEXT NOT NULL,
    binance_price        REAL NOT NULL DEFAULT 0,
    chainlink_price      REAL NOT NULL DEFAULT 0,
    chainlink_ts         REAL NOT NULL DEFAULT 0,   -- age = timestamp - chainlink_ts
    UNIQUE(asset, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_ticks_asset_ts ON price_ticks(asset, timestamp);


CREATE TABLE IF NOT EXISTS shadow_signals (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            REAL NOT NULL,
    market_id            TEXT NOT NULL,
    slug                 TEXT,
    asset                TEXT NOT NULL,
    action               TEXT NOT NULL,              -- BUY | NONE | WAIT
    side                 TEXT,                       -- YES | NO
    entry_price          REAL,
    p_true               REAL,
    p_market             REAL,
    edge                 REAL,
    delta_chainlink      REAL,
    delta_binance        REAL,
    sigma                REAL,
    time_remaining_sec   REAL,
    filters_passed       INTEGER NOT NULL,
    filter_reasons       TEXT,
    strategy_used        TEXT,
    size_usd             REAL,
    confidence           REAL,
    chainlink_price      REAL,
    binance_price        REAL,
    reference_price      REAL,
    oracle_age_sec       REAL,
    best_bid_yes         REAL,
    best_ask_yes         REAL,
    best_bid_no          REAL,
    best_ask_no          REAL
);
CREATE INDEX IF NOT EXISTS idx_sig_market_ts ON shadow_signals(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sig_asset_ts  ON shadow_signals(asset, timestamp);
CREATE INDEX IF NOT EXISTS idx_sig_action    ON shadow_signals(action, timestamp);
"""


def init_db(path: str | Path) -> sqlite3.Connection:
    """Open the collector DB, enable WAL, create schema if missing.

    Returns the connection — caller is responsible for closing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-65536;")  # 64 MiB page cache
    conn.executescript(SCHEMA_SQL)
    return conn


ALL_TABLES = (
    "markets", "orderbook_snapshots", "orderbook_deltas", "trades",
    "tick_changes", "market_states", "price_ticks", "shadow_signals",
    "ws_events",
)


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return row counts for each collector table — useful for stats."""
    out: dict[str, int] = {}
    for t in ALL_TABLES:
        cur = conn.execute(f"SELECT COUNT(*) FROM {t}")
        out[t] = cur.fetchone()[0]
    return out


def db_size_mb(conn: sqlite3.Connection) -> float:
    """Total DB size on disk in MiB (main + WAL)."""
    cur = conn.execute("PRAGMA page_count")
    pages = cur.fetchone()[0]
    cur = conn.execute("PRAGMA page_size")
    page_size = cur.fetchone()[0]
    return (pages * page_size) / (1024 * 1024)
