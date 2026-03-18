"""Cross-Market Propagation Delay Exploiter.

Implements QR-PM-2026-0041 §6 — after a 5-minute BTC market resolves,
Polymarket takes 12-45 seconds to reprice the correlated 15-minute
market. During this window, the 15m market offers stale odds.

Usage:
    booster = CrossMarketBooster()

    # When a 5m market resolves:
    booster.record_5m_close(chainlink_price, ref_price, "up")

    # In evaluate() for a 15m market:
    cm_boost = booster.get_boost("YES", btc_chainlink, ref_price_15m)
    p_true = min(p_true + cm_boost, 0.97)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

# Empirical propagation window from QR-PM-2026-0041 §6: 12-45 seconds.
# We use 45s as the full decay window — boost is 0 at t=45s.
PROPAGATION_WINDOW_SEC = 45.0

# Maximum confidence boost added to p_true for 15m markets.
# 5% additive boost at t=0, decaying linearly to 0 at t=45s.
MAX_BOOST = 0.05


@dataclass
class FiveMinClose:
    """Record of a resolved 5-minute market."""
    timestamp: float
    chainlink_price: float
    reference_price_5m: float
    direction: str   # "up" or "down"
    delta_pct: float  # |chainlink_price - ref| / ref, in fraction


class CrossMarketBooster:
    """
    Tracks recent 5-minute market resolutions and computes an additive
    probability boost for 15-minute market bets during the propagation
    delay window (12-45 seconds after a 5m close).

    Boost logic:
    - Direction agreement required: 5m closed "up" → only boosts YES bets
    - Linear decay: boost = MAX_BOOST * (1 - elapsed / PROPAGATION_WINDOW_SEC)
    - Minimum delta filter: the 5m close must have moved at least 0.05%
      to be considered a meaningful directional signal
    - Only the most recent 5m close is used (stale signals are discarded)
    - Current 15m delta must agree with direction (no reversal)
    """

    # Minimum 5m delta to consider the signal meaningful
    MIN_DELTA_PCT = 0.0005  # 0.05%

    def __init__(self) -> None:
        self._last_close: Optional[FiveMinClose] = None

    def record_5m_close(
        self,
        chainlink_price: float,
        reference_price_5m: float,
        direction: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a resolved 5-minute market.

        Args:
            chainlink_price: Chainlink BTC price at resolution time.
            reference_price_5m: The 5m market's reference price (start-of-window BTC).
            direction: "up" or "down" — the actual resolution outcome.
            timestamp: Unix timestamp; defaults to time.time().
        """
        if timestamp is None:
            timestamp = time.time()
        if reference_price_5m <= 0:
            return
        delta_pct = abs(chainlink_price - reference_price_5m) / reference_price_5m
        self._last_close = FiveMinClose(
            timestamp=timestamp,
            chainlink_price=chainlink_price,
            reference_price_5m=reference_price_5m,
            direction=direction,
            delta_pct=delta_pct,
        )

    def get_boost(
        self,
        side: str,
        chainlink_now: float,
        m15_reference: float,
        now: Optional[float] = None,
    ) -> float:
        """
        Return additive p_true boost for a 15m market bet.

        Args:
            side: "YES" or "NO" — the proposed bet side for the 15m market.
            chainlink_now: Current Chainlink BTC price.
            m15_reference: Reference price for the 15m market.
            now: Current timestamp; defaults to time.time().

        Returns:
            Float in [0, MAX_BOOST]. Returns 0.0 if:
            - No 5m close was recorded yet
            - The propagation window has elapsed (>45s)
            - The 5m direction conflicts with the proposed bet side
            - The 5m close delta was too small (<0.05%)
            - The 15m market is already moving against the 5m direction
        """
        if now is None:
            now = time.time()

        c = self._last_close
        if c is None:
            return 0.0

        elapsed = now - c.timestamp
        if elapsed < 0 or elapsed >= PROPAGATION_WINDOW_SEC:
            return 0.0

        # Directional agreement: 5m "up" should only boost YES bets
        bet_up = side == "YES"
        close_up = c.direction == "up"
        if bet_up != close_up:
            return 0.0

        # Minimum signal strength filter
        if c.delta_pct < self.MIN_DELTA_PCT:
            return 0.0

        # Sanity check: current 15m delta must agree with direction
        # (BTC must not have already reversed vs the 15m reference)
        if m15_reference > 0 and chainlink_now > 0:
            current_delta = (chainlink_now - m15_reference) / m15_reference
            if bet_up and current_delta < 0:
                return 0.0  # BTC already reversed
            if not bet_up and current_delta > 0:
                return 0.0  # BTC already reversed

        # Linear decay over propagation window
        decay = 1.0 - (elapsed / PROPAGATION_WINDOW_SEC)
        return MAX_BOOST * decay
