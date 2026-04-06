"""SQLite database layer for trade history and portfolio snapshots."""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    market_id TEXT NOT NULL,
    slug TEXT NOT NULL DEFAULT '',
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    delta REAL NOT NULL,
    sigma REAL NOT NULL,
    p_true REAL NOT NULL,
    p_market REAL NOT NULL,
    edge REAL NOT NULL,
    time_remaining_sec REAL NOT NULL,
    oracle_age_sec REAL DEFAULT 0.0,
    outcome TEXT DEFAULT 'pending',
    pnl REAL DEFAULT 0.0,
    mode TEXT NOT NULL DEFAULT 'paper',
    resolved_at REAL
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    balance REAL NOT NULL,
    open_positions INTEGER NOT NULL DEFAULT 0,
    daily_pnl REAL NOT NULL DEFAULT 0.0,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    mode TEXT NOT NULL DEFAULT 'paper'
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    market_id TEXT,
    btc_binance REAL,
    btc_chainlink REAL,
    reference_price REAL,
    delta_chainlink REAL,
    delta_binance REAL,
    sigma REAL,
    time_remaining_sec REAL,
    p_true REAL,
    p_market REAL,
    edge REAL,
    filters_passed INTEGER,
    filter_details TEXT,
    action TEXT,
    oracle_age_sec REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp
    ON portfolio_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
"""


@dataclass
class TradeRecord:
    market_id: str
    slug: str
    side: str  # "YES" or "NO"
    entry_price: float
    size_usd: float
    delta: float
    sigma: float
    p_true: float
    p_market: float
    edge: float
    time_remaining_sec: float
    oracle_age_sec: float = 0.0  # seconds since last Chainlink update at bet (v3.5)
    mode: str = "paper"
    timestamp: float = 0.0
    outcome: str = "pending"
    pnl: float = 0.0
    resolved_at: Optional[float] = None
    exit_price: float = 0.0
    exit_reason: str = "normal"
    is_topup: int = 0
    parent_trade_id: Optional[int] = None
    token_id: str = ""
    id: Optional[int] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class Database:
    """Async SQLite database for trade persistence."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row

        # v3.6: WAL mode allows concurrent reads during high-frequency signal
        # inserts (every 2s). NORMAL sync is safe and much faster than FULL.
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")

        await self._db.executescript(_SCHEMA)
        await self._db.commit()

        # v3.5 migration: add oracle_age_sec to existing tables.
        # CREATE TABLE IF NOT EXISTS does not add new columns to existing
        # tables, so we ALTER TABLE here. Safe to run on every startup.
        for _tbl in ("trades", "signals"):
            try:
                await self._db.execute(
                    f"ALTER TABLE {_tbl} "
                    f"ADD COLUMN oracle_age_sec REAL DEFAULT 0.0"
                )
                await self._db.commit()
            except Exception:
                pass  # column already exists — safe to ignore

        # v5 migration: early exit + staged entry support columns.
        for _col, _default in [
            ("exit_price REAL", "0.0"),
            ("exit_reason TEXT", "'normal'"),
            ("is_topup INTEGER", "0"),
            ("parent_trade_id INTEGER", "NULL"),
            ("token_id TEXT", "''"),
        ]:
            try:
                await self._db.execute(
                    f"ALTER TABLE trades ADD COLUMN {_col} DEFAULT {_default}"
                )
                await self._db.commit()
            except Exception:
                pass  # column already exists

        # v3.6: auto-purge signals older than 7 days to prevent unbounded growth.
        # At ~3 signals/s this table accumulates ~1.8M rows/week without cleanup.
        _signals_cutoff = time.time() - 7 * 86400
        await self._db.execute(
            "DELETE FROM signals WHERE timestamp < ?", (_signals_cutoff,)
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def insert_trade(self, trade: TradeRecord) -> int:
        d = asdict(trade)
        d.pop("id", None)
        cols = ", ".join(d.keys())
        placeholders = ", ".join("?" for _ in d)
        cursor = await self._db.execute(
            f"INSERT INTO trades ({cols}) VALUES ({placeholders})",
            list(d.values()),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def resolve_trade(
        self, trade_id: int, outcome: str, pnl: float
    ) -> None:
        await self._db.execute(
            "UPDATE trades SET outcome=?, pnl=?, resolved_at=? "
            "WHERE id=?",
            (outcome, pnl, time.time(), trade_id),
        )
        await self._db.commit()

    async def resolve_trade_early(
        self,
        trade_id: int,
        outcome: str,
        pnl: float,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Resolve a trade that was exited early (sold before expiry)."""
        await self._db.execute(
            "UPDATE trades SET outcome=?, pnl=?, resolved_at=?, "
            "exit_price=?, exit_reason=? WHERE id=?",
            (outcome, pnl, time.time(), exit_price, exit_reason, trade_id),
        )
        await self._db.commit()

    async def get_pending_trades(self, mode: str = "paper") -> list[dict]:
        """Return all trades still awaiting resolution for a given mode."""
        cursor = await self._db.execute(
            "SELECT * FROM trades WHERE outcome='pending' AND mode=? "
            "ORDER BY timestamp DESC",
            (mode,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_trade(self, trade_id: int) -> Optional[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM trades WHERE id=?", (trade_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_recent_trades(self, limit: int = 50) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_trade_stats(self, mode: str = "paper") -> dict:
        cursor = await self._db.execute(
            "SELECT "
            "  COUNT(*) as total, "
            "  SUM(CASE WHEN outcome='won' THEN 1 ELSE 0 END) as wins, "
            "  SUM(CASE WHEN outcome='lost' THEN 1 ELSE 0 END) as losses, "
            "  SUM(pnl) as total_pnl, "
            "  AVG(CASE WHEN outcome='won' THEN pnl ELSE NULL END) "
            "    as avg_win, "
            "  AVG(CASE WHEN outcome='lost' THEN pnl ELSE NULL END) "
            "    as avg_loss, "
            "  AVG(edge) as avg_edge "
            "FROM trades WHERE mode=? AND outcome != 'pending'",
            (mode,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}

    async def get_daily_pnl(
        self, mode: str = "paper", days: int = 30
    ) -> list[dict]:
        cutoff = time.time() - (days * 86400)
        cursor = await self._db.execute(
            "SELECT "
            "  date(timestamp, 'unixepoch') as day, "
            "  SUM(pnl) as daily_pnl, "
            "  COUNT(*) as trades "
            "FROM trades "
            "WHERE mode=? AND outcome != 'pending' "
            "  AND timestamp > ? "
            "GROUP BY day ORDER BY day",
            (mode, cutoff),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_consecutive_losses(self, mode: str = "paper") -> int:
        cursor = await self._db.execute(
            "SELECT outcome FROM trades "
            "WHERE mode=? AND outcome != 'pending' "
            "ORDER BY timestamp DESC LIMIT 20",
            (mode,),
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            if row["outcome"] == "lost":
                count += 1
            else:
                break
        return count

    async def insert_snapshot(
        self,
        balance: float,
        open_positions: int,
        daily_pnl: float,
        total_pnl: float,
        mode: str = "paper",
    ) -> None:
        await self._db.execute(
            "INSERT INTO portfolio_snapshots "
            "(timestamp, balance, open_positions, daily_pnl, "
            "total_pnl, mode) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), balance, open_positions, daily_pnl,
             total_pnl, mode),
        )
        await self._db.commit()

    async def get_equity_curve(
        self, mode: str = "paper", limit: int = 500
    ) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT timestamp, balance, total_pnl "
            "FROM portfolio_snapshots "
            "WHERE mode=? ORDER BY timestamp DESC LIMIT ?",
            (mode, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(list(rows))]

    async def load_portfolio_state(
        self, mode: str = "paper"
    ) -> dict:
        """Compute portfolio state from all historical trades."""
        cursor = await self._db.execute(
            "SELECT "
            "  COALESCE(SUM(CASE WHEN outcome='won' "
            "    THEN 1 ELSE 0 END), 0) as wins, "
            "  COALESCE(SUM(CASE WHEN outcome='lost' "
            "    THEN 1 ELSE 0 END), 0) as losses, "
            "  COALESCE(SUM(CASE WHEN outcome != 'pending' "
            "    THEN pnl ELSE 0 END), 0) as total_pnl, "
            "  COALESCE(SUM(CASE WHEN outcome = 'pending' "
            "    THEN size_usd ELSE 0 END), 0) as pending_capital "
            "FROM trades WHERE mode=?",
            (mode,),
        )
        row = await cursor.fetchone()
        stats = dict(row) if row else {
            "wins": 0, "losses": 0,
            "total_pnl": 0.0, "pending_capital": 0.0,
        }

        # Daily PnL (today only)
        import time as _time
        today = _time.strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(pnl), 0) as daily_pnl "
            "FROM trades WHERE mode=? AND outcome != 'pending' "
            "AND date(timestamp, 'unixepoch', 'localtime') = ?",
            (mode, today),
        )
        row = await cursor.fetchone()
        stats["daily_pnl"] = (
            dict(row)["daily_pnl"] if row else 0.0
        )

        # Consecutive losses
        stats["consecutive_losses"] = (
            await self.get_consecutive_losses(mode)
        )

        # Peak balance from snapshots
        cursor = await self._db.execute(
            "SELECT MAX(balance) as peak "
            "FROM portfolio_snapshots WHERE mode=?",
            (mode,),
        )
        row = await cursor.fetchone()
        stats["peak_balance"] = (
            dict(row)["peak"] if row and dict(row)["peak"]
            else 0.0
        )

        return stats

    async def insert_signal(self, signal: dict) -> None:
        cols = ", ".join(signal.keys())
        placeholders = ", ".join("?" for _ in signal)
        await self._db.execute(
            f"INSERT INTO signals ({cols}) VALUES ({placeholders})",
            list(signal.values()),
        )
        await self._db.commit()
