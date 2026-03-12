"""Mathematical utilities for probability and risk calculations."""

from __future__ import annotations

import math
from collections import deque
from scipy.stats import norm


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return float(norm.cdf(x))


def calc_true_probability(
    delta: float,
    sigma_1min: float,
    t_minutes: float,
) -> float:
    """Calculate true probability that current side holds until expiry.

    Args:
        delta: Relative price change from reference (e.g. 0.003 = 0.3%)
        sigma_1min: BTC 1-minute realized volatility
        t_minutes: Time remaining in minutes

    Returns:
        Probability [0, 1] that current sign of delta holds.
    """
    if t_minutes <= 0 or sigma_1min <= 0:
        return 0.5
    if abs(delta) < 1e-10:
        return 0.5

    sigma_t = sigma_1min * math.sqrt(t_minutes)
    z = abs(delta) / sigma_t
    p_reversal = normal_cdf(-z)
    return 1.0 - p_reversal


def calc_taker_fee(p_market: float, fee_rate: float = 0.25) -> float:
    """Calculate Polymarket dynamic taker fee.

    Formula: fee_rate * (p * (1 - p))^2
    """
    if p_market <= 0 or p_market >= 1:
        return 0.0
    return fee_rate * (p_market * (1.0 - p_market)) ** 2


def calc_edge(
    p_true: float,
    p_market: float,
    fee_rate: float = 0.25,
) -> float:
    """Calculate net edge after fees.

    Edge = P_true - P_market - taker_fee
    """
    fee = calc_taker_fee(p_market, fee_rate)
    return p_true - p_market - fee


def kelly_size(
    edge: float,
    entry_price: float,
    taker_fee: float,
    capital: float,
    fraction: float = 0.25,
    max_pct: float = 0.12,
) -> float:
    """Calculate fractional Kelly position size for a binary option.

    Derivation: buying 1 share at effective cost c = entry_price + fee,
    the binary pays $1 on win or $0 on loss.

        Win profit per $ risked: (1 - c) / c
        Loss:                    100% of wager

    Maximizing E[log(W)] gives:

        f* = (p_true - c) / (1 - c) = edge / (1 - c)

    where edge = p_true - entry_price - fee.

    Args:
        edge: Net edge after fees (= p_true - entry_price - fee).
        entry_price: Market ask price paid per share.
        taker_fee: Polymarket taker fee per share.
        capital: Current available capital.
        fraction: Kelly fraction (0.25 = quarter Kelly).
        max_pct: Hard cap as fraction of capital.

    Returns:
        Dollar amount to bet.
    """
    if edge <= 0 or capital <= 0:
        return 0.0

    c_eff = entry_price + taker_fee
    if c_eff >= 1.0 or c_eff <= 0:
        return 0.0

    # Full Kelly: f* = edge / (1 - c_eff)
    f_full = edge / (1.0 - c_eff)
    f_adjusted = fraction * f_full
    f_capped = min(f_adjusted, max_pct)
    return max(0.0, capital * f_capped)


class RollingVolatility:
    """Rolling realized volatility calculator for 1-minute returns."""

    def __init__(self, window_minutes: int = 30):
        self.window = window_minutes
        self._prices: deque[tuple[float, float]] = deque()
        self._returns: deque[float] = deque()
        self._last_minute_price: float | None = None
        self._last_minute_ts: float = 0.0

    def update(self, price: float, timestamp: float) -> None:
        """Update with a new price tick.

        Aggregates into 1-minute buckets before computing vol.
        """
        self._prices.append((timestamp, price))

        minute_ts = timestamp // 60.0
        if self._last_minute_price is None:
            self._last_minute_price = price
            self._last_minute_ts = minute_ts
            return

        if minute_ts > self._last_minute_ts:
            log_return = math.log(price / self._last_minute_price)
            self._returns.append(log_return)
            self._last_minute_price = price
            self._last_minute_ts = minute_ts

            while len(self._returns) > self.window:
                self._returns.popleft()

    # Minimum sigma floor — BTC 1-min vol rarely drops below
    # 0.05%.  Using a lower value causes P_true inflation and
    # overconfident bets.
    SIGMA_FLOOR = 0.0005

    @property
    def sigma(self) -> float:
        """Current 1-minute realized volatility."""
        if len(self._returns) < 5:
            return self.SIGMA_FLOOR
        n = len(self._returns)
        mean = sum(self._returns) / n
        variance = sum(
            (r - mean) ** 2 for r in self._returns
        ) / (n - 1)
        return max(math.sqrt(variance), self.SIGMA_FLOOR)

    @property
    def sample_count(self) -> int:
        return len(self._returns)
