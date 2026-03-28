"""Cross-Market Propagation Delay Exploiter.

After a 5-minute BTC market resolves, Polymarket takes 12-45 seconds
to reprice the correlated 15-minute market. During this window, the
15m market may offer temporarily stale odds.

Usage:
    booster = CrossMarketBooster()

    # When a 5m market resolves:
    booster.record_5m_close(chainlink_price, ref_price, "up")

    # In evaluate() for a 15m market:
    cm_boost = booster.get_boost("YES", btc_chainlink, ref_price_15m)
    p_true = min(p_true + cm_boost, 0.97)

v4.0 CHANGES:
  - MAX_BOOST reduced from 0.05 to 0.02.
    The original +5% boost was too aggressive for an empirically
    unvalidated signal. At +2%, the boost is meaningful but constrained:
    it can push a borderline 64% p_true to 66%, which is within the
    normal signal noise band, rather than creating artificial certainty.
  - Added MIN_OBSERVATIONS guard: the booster is only active once at
    least 3 five-minute resolutions have been recorded in the current
    session. This prevents the first trade of a session from being
    boosted by stale state from a previous run.
  - Propagation window empirically observed at 12-45s; kept as-is.

v4.2 CHANGES:
  - Trend exhaustion tracking: after 3+ consecutive resolutions in the
    same direction, a skepticism penalty is applied. Empirically, crypto
    micro-structure shows higher pullback probability after extended
    unidirectional runs (profit-taking, mean reversion).
  - consecutive_same_direction() method: returns the current streak count
    and direction. Used by SignalEngine to penalize p_true.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

# Empirically observed propagation window (12-45 seconds).
# Boost is MAX_BOOST at t=0, decays linearly to 0 at t=45s.
PROPAGATION_WINDOW_SEC = 45.0

# v4.0: Reduced from 0.05 to 0.02.
# +5% was too aggressive for an unvalidated signal source.
# +2% is meaningful (moves borderline signals) but not dominant.
MAX_BOOST = 0.02

# v4.0: Minimum number of 5m closes this session before boost is active.
# Prevents stale/cold-start state from influencing live trades.
MIN_OBSERVATIONS = 3


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
    - Only the most recent 5m close is used (stale signals are discarded)
    - Current 15m delta must agree with direction (no reversal)
    - MIN_OBSERVATIONS sessions guard (v4.0)
    """

    # Minimum 5m delta to consider the signal meaningful
    MIN_DELTA_PCT = 0.0005  # 0.05%

    def __init__(self) -> None:
        self._last_close: Optional[FiveMinClose] = None
        self._total_observations: int = 0
        # v4.2: track consecutive same-direction resolutions for exhaustion
        self._recent_directions: list = []  # last N directions ("up"/"down")

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
        self._total_observations += 1
        # v4.2: track direction streak (keep last 10)
        self._recent_directions.append(direction.lower())
        if len(self._recent_directions) > 10:
            self._recent_directions = self._recent_directions[-10:]

    def get_boost(
        self,
        side: str,
        chainlink_now: float,
        m15_reference: float,
        now: Optional[float] = None,
    ) -> float:
        """
        Return additive p_true boost for a 15m market bet.

        Returns:
            Float in [0, MAX_BOOST]. Returns 0.0 if:
            - Fewer than MIN_OBSERVATIONS 5m closes recorded this session
            - No 5m close was recorded yet
            - The propagation window has elapsed (>45s)
            - The 5m direction conflicts with the proposed bet side
            - The 5m close delta was too small (<0.05%)
            - The 15m market is already moving against the 5m direction
        """
        if now is None:
            now = time.time()

        # v4.0: guard against cold-start / insufficient history
        if self._total_observations < MIN_OBSERVATIONS:
            return 0.0

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
        if m15_reference > 0 and chainlink_now > 0:
            current_delta = (chainlink_now - m15_reference) / m15_reference
            if bet_up and current_delta < 0:
                return 0.0  # BTC already reversed
            if not bet_up and current_delta > 0:
                return 0.0  # BTC already reversed

        # Linear decay over propagation window
        decay = 1.0 - (elapsed / PROPAGATION_WINDOW_SEC)
        return MAX_BOOST * decay

    def consecutive_same_direction(self) -> tuple:
        """Return (streak_count, direction) of recent consecutive same-direction closes.

        v4.2: After 3+ consecutive resolutions in the same direction,
        the next bet in that direction should be penalized (trend exhaustion /
        pullback risk). Returns (0, "") if no streak or < 2 observations.

        Example: ["up","up","up","up"] → (4, "up")
                 ["up","up","down"]    → (1, "down")
        """
        if len(self._recent_directions) < 2:
            return 0, ""
        last_dir = self._recent_directions[-1]
        count = 0
        for d in reversed(self._recent_directions):
            if d == last_dir:
                count += 1
            else:
                break
        return count, last_dir

    @property
    def observations(self) -> int:
        """Number of 5m closes recorded this session."""
        return self._total_observations
