"""Portfolio management for both paper and live trading."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import setup_logger

log = setup_logger("trading.portfolio")


@dataclass
class Position:
    """An open position in a binary market."""

    trade_id: int
    market_id: str
    side: str  # "YES" or "NO"
    entry_price: float
    size_usd: float
    shares: float  # size_usd / entry_price
    entry_time: float
    market_end_time: float
    delta_at_entry: float = 0.0
    p_true_at_entry: float = 0.0
    edge_at_entry: float = 0.0

    @property
    def potential_profit(self) -> float:
        """Max profit if position wins."""
        return self.shares * (1.0 - self.entry_price)

    @property
    def potential_loss(self) -> float:
        """Max loss if position loses."""
        return self.size_usd

    @property
    def time_to_expiry(self) -> float:
        return max(0.0, self.market_end_time - time.time())


class Portfolio:
    """Tracks capital, positions, and PnL."""

    def __init__(self, initial_balance: float, mode: str = "paper"):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.mode = mode
        self.open_positions: dict[int, Position] = {}
        self.total_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self._daily_reset_date: str = ""
        self.total_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.consecutive_losses: int = 0
        self._max_consecutive_losses: int = 0
        self.max_drawdown: float = 0.0
        self._peak_balance: float = initial_balance

    def restore_from_db(self, state: dict) -> None:
        """Restore portfolio state from database history."""
        self.wins = state.get("wins", 0) or 0
        self.losses = state.get("losses", 0) or 0
        self.total_trades = self.wins + self.losses
        self.total_pnl = state.get("total_pnl", 0.0) or 0.0
        self.daily_pnl = state.get("daily_pnl", 0.0) or 0.0
        self.consecutive_losses = (
            state.get("consecutive_losses", 0) or 0
        )
        self._daily_reset_date = time.strftime("%Y-%m-%d")

        pending_capital = state.get("pending_capital", 0.0) or 0.0
        self.balance = (
            self.initial_balance + self.total_pnl - pending_capital
        )

        peak = state.get("peak_balance", 0.0) or 0.0
        self._peak_balance = max(
            self.balance, self.initial_balance, peak
        )

        if self._peak_balance > 0 and self.balance < self._peak_balance:
            self.max_drawdown = (
                (self._peak_balance - self.balance)
                / self._peak_balance
            )

        log.info(
            "[Portfolio] Restored from DB: $%.2f balance | "
            "%dW/%dL | PnL $%.2f",
            self.balance, self.wins, self.losses, self.total_pnl,
        )

    def open_position(self, pos: Position) -> bool:
        """Open a new position, deducting capital."""
        self._check_daily_reset()

        if pos.size_usd > self.balance:
            log.warning(
                "Insufficient balance: need $%.2f, have $%.2f",
                pos.size_usd,
                self.balance,
            )
            return False

        self.balance -= pos.size_usd
        self.open_positions[pos.trade_id] = pos
        log.info(
            "[Portfolio] Opened %s %s | $%.2f @ %.4f | "
            "market=%s",
            pos.side,
            "position",
            pos.size_usd,
            pos.entry_price,
            pos.market_id[:16],
        )
        return True

    def close_position(
        self, trade_id: int, won: bool
    ) -> tuple[str, float]:
        """Close a position and update PnL.

        Returns:
            (outcome, pnl) tuple
        """
        self._check_daily_reset()

        pos = self.open_positions.pop(trade_id, None)
        if pos is None:
            return ("error", 0.0)

        self.total_trades += 1

        if won:
            # Win: receive $1.00 per share
            payout = pos.shares * 1.0
            pnl = payout - pos.size_usd
            self.balance += payout
            self.wins += 1
            self.consecutive_losses = 0
            outcome = "won"
            log.info(
                "[Portfolio] WON %s | PnL: +$%.2f (%.1f%%) | "
                "balance: $%.2f",
                pos.market_id[:16],
                pnl,
                (pnl / pos.size_usd) * 100,
                self.balance,
            )
        else:
            # Loss: shares expire worthless
            pnl = -pos.size_usd
            self.losses += 1
            self.consecutive_losses += 1
            self._max_consecutive_losses = max(
                self._max_consecutive_losses,
                self.consecutive_losses,
            )
            outcome = "lost"
            log.warning(
                "[Portfolio] LOST %s | PnL: -$%.2f | "
                "balance: $%.2f | streak: %d",
                pos.market_id[:16],
                pos.size_usd,
                self.balance,
                self.consecutive_losses,
            )

        self.total_pnl += pnl
        self.daily_pnl += pnl

        # Track drawdown
        if self.balance > self._peak_balance:
            self._peak_balance = self.balance
        current_dd = (
            (self._peak_balance - self.balance) / self._peak_balance
        )
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        return (outcome, pnl)

    def _check_daily_reset(self) -> None:
        """Reset daily PnL at midnight."""
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self.daily_pnl = 0.0

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        if closed == 0:
            return 0.0
        return self.wins / closed

    @property
    def daily_pnl_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return self.daily_pnl / self.initial_balance

    @property
    def total_return_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return self.total_pnl / self.initial_balance

    @property
    def open_position_count(self) -> int:
        return len(self.open_positions)

    def has_position_on_market(self, market_id: str) -> bool:
        """Check if there's already an open position on this market."""
        return any(
            p.market_id == market_id
            for p in self.open_positions.values()
        )

    @property
    def capital_at_risk(self) -> float:
        return sum(p.size_usd for p in self.open_positions.values())

    def get_stats(self) -> dict:
        """Return comprehensive portfolio statistics."""
        return {
            "mode": self.mode,
            "balance": round(self.balance, 2),
            "initial_balance": round(self.initial_balance, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct * 100, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct * 100, 2),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate * 100, 1),
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self._max_consecutive_losses,
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "open_positions": self.open_position_count,
            "capital_at_risk": round(self.capital_at_risk, 2),
            "peak_balance": round(self._peak_balance, 2),
        }
