"""Market outcome trend tracker for BTC Up/Down Polymarket markets.

Maintains a ring buffer of the last N resolved market outcomes
("up" or "down"). Updated by main.py after each trade resolution.
Consumed by MomentumEngine to detect directional streaks.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Tuple


class MarketTrendTracker:
    """
    Ring buffer of BTC Up/Down market outcomes.
    Supports streak detection for trend-following logic.
    """

    def __init__(self, maxlen: int = 20):
        self._outcomes: deque = deque(maxlen=maxlen)

    def record(self, outcome: str) -> None:
        """
        Record a resolved market outcome.
        outcome: 'up' or 'down'
        """
        if outcome not in ("up", "down"):
            raise ValueError(f"Invalid outcome: {outcome!r}, expected 'up' or 'down'")
        self._outcomes.append((time.time(), outcome))

    def recent_streak(self, n: int = 4) -> Tuple[str, int]:
        """
        Returns the direction and length of the most recent streak
        over the last n markets.

        Returns:
            ("up", k)    if the last k markets all resolved UP
            ("down", k)  if the last k markets all resolved DOWN
            ("mixed", 0) if no clear streak
        """
        if len(self._outcomes) < 2:
            return "mixed", 0
        recent = [o for _, o in list(self._outcomes)[-n:]]
        if all(o == "up" for o in recent):
            return "up", len(recent)
        if all(o == "down" for o in recent):
            return "down", len(recent)
        return "mixed", 0

    def last_n(self, n: int = 10) -> list:
        """Return the last n outcomes as a list of strings."""
        return [o for _, o in list(self._outcomes)[-n:]]

    def __len__(self) -> int:
        return len(self._outcomes)


# Module-level singleton shared across the system.
# Populated by main.py after each trade resolution.
market_trend = MarketTrendTracker()
