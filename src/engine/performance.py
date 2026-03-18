"""Per-strategy rolling performance tracker.

Tracks win rate per strategy over a sliding window.
Used by SignalEngine to weight strategy scores.
"""
from __future__ import annotations

from collections import deque
from typing import Dict
import time


class PerformanceTracker:
    """
    Tracks win/loss history per strategy over a rolling window.
    Computes a weight multiplier used by the router:
      weight = 0.5 + win_rate  → range [0.5, 1.5]
    After < MIN_SAMPLES trades the weight is neutral (1.0).
    """

    MIN_SAMPLES = 5

    def __init__(self, window: int = 30):
        self._window = window
        self._results: Dict[str, deque] = {}
        self._last_ts: Dict[str, float] = {}

    def record(self, strategy: str, won: bool) -> None:
        """Record the outcome of a resolved trade."""
        if strategy not in self._results:
            self._results[strategy] = deque(maxlen=self._window)
        self._results[strategy].append(won)
        self._last_ts[strategy] = time.time()

    def win_rate(self, strategy: str) -> float:
        """Rolling win rate for a strategy (0.0–1.0)."""
        h = self._results.get(strategy, deque())
        if len(h) < self.MIN_SAMPLES:
            return 0.5  # neutral until enough data
        return sum(h) / len(h)

    def weight(self, strategy: str) -> float:
        """
        Score multiplier for edge weighting:
          50 % WR → 1.0  (neutral)
          65 % WR → 1.15
          80 % WR → 1.30
          35 % WR → 0.85
        """
        return 0.5 + self.win_rate(strategy)

    def sample_count(self, strategy: str) -> int:
        return len(self._results.get(strategy, []))

    def stats(self) -> dict:
        """Summary of all tracked strategies."""
        return {
            name: {
                "samples": len(h),
                "win_rate_pct": round(self.win_rate(name) * 100, 1),
                "weight": round(self.weight(name), 3),
            }
            for name, h in self._results.items()
        }
