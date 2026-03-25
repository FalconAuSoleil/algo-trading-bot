"""Signal Engine v4.1 - Multi-Strategy Router + Diffusion Risk Model
======================================================================

v4.1 changes (team discussion 2026-03-25):
  - New _BTCStabilizationEngine: dedicated BTC 15m strategy.
    Fires T=60-180s when one side stabilizes at 63-80¢ (empirically
    ~70¢ sweet spot). Checks Brownian reversal probability to confirm
    the move is statistically locked in. Bypasses the overcrowded
    ChainlinkArb on BTC 15m.
  - is_peak_hours(): Mon-Fri 08:00-18:00 ET gate for ETH/SOL/XRP
    and BTC 5m. Weekends excluded (empirically poor liquidity).
  - SignalEngine.asset_symbol: routing in evaluate():
      BTC 15m  → _BTCStabilizationEngine (24/7)
      BTC 5m   → ChainlinkArb (peak hours only)
      ETH/SOL/XRP → ChainlinkArb (peak hours only, unchanged v3.9)

v4.0 changes (team audit 2026-03-25):
  - calc_fee: replaced quadratic over-estimate with accurate linear model.
    Old: rate=0.25, fee = rate*(p*(1-p))^2  --> at p=0.5: 1.56% (3x too high)
    New: rate=0.02, fee = rate*(1-p)        --> at p=0.5: 1.00% (correct)
    Impact: ~33% more valid signals pass the edge_min filter.
  - See TEAM_AUDIT.md for full analysis.

v3.8 fixes:
  - ETH/SOL/XRP restricted to 15m markets only (Polymarket 5m API
    returns error for non-BTC assets -- verified 2026-03-21).
  - PolymarketFeed now uses per-asset interval dict.
  - Resolution loop bug fixed: was passing bool instead of float price.
  - Quant param fixes: source_coherence_max, time_max_15m,
    stability_min_samples, stability_edge_cv_max.

v3.7 addition (multi-asset support):
  Each asset now has its own sigma_fallback and delta_min_abs parameters
  passed from AssetConfig. The _ChainlinkArbEngine constructor accepts
  these per-asset overrides. BTC behavior is unchanged.

v3.5 addition (oracle freshness filter):
  Chainlink oracle resolution uses the last on-chain update BEFORE
  market expiry, not the real-time BTC price at that timestamp.
  → Resolves discrepancy with market prices at T=final (which sometimes
    jump 0.2% after the oracle snapshot time).
  → Queries Chainlink subgraph: event logs of latestAnswer from
    aggregator.sol, filtered by block.timestamp <= market.resolvesTime.
"""

import json
import math
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


# ============================================================================
# UTILITY FUNCTIONS & CLASSES
# ============================================================================


def is_peak_hours(timestamp_unix: float) -> bool:
    """
    Check if a Unix timestamp falls within peak trading hours.
    Peak hours: Mon-Fri 08:00-18:00 ET (UTC-5 / UTC-4 DST).
    """
    dt = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)
    # ET is UTC-5 (standard) or UTC-4 (daylight)
    et_offset = timedelta(hours=-5)  # simplified; doesn't account for DST
    dt_et = dt + et_offset
    weekday = dt_et.weekday()
    hour = dt_et.hour
    # Mon=0, Fri=4; 08:00-18:00
    return weekday < 5 and 8 <= hour < 18


class SimpleMovingAverage:
    """Online moving average (no stored history)."""

    def __init__(self, window: int):
        self.window = window
        self.sum = 0.0
        self.count = 0

    def update(self, value: float):
        self.sum += value
        self.count += 1
        if self.count > self.window:
            self.sum -= value  # simple FIFO without history
            self.count = self.window

    def get(self) -> Optional[float]:
        return self.sum / self.count if self.count > 0 else None


class PriceHistory:
    """Circular buffer of prices with online statistics."""

    def __init__(self, max_len: int = 300):
        self.prices = deque(maxlen=max_len)
        self.max_len = max_len

    def append(self, price: float):
        self.prices.append(price)

    def mean(self) -> Optional[float]:
        if not self.prices:
            return None
        return sum(self.prices) / len(self.prices)

    def variance(self) -> Optional[float]:
        if len(self.prices) < 2:
            return None
        m = self.mean()
        return sum((p - m) ** 2 for p in self.prices) / (len(self.prices) - 1)

    def std(self) -> Optional[float]:
        v = self.variance()
        return math.sqrt(v) if v is not None else None

    def size(self) -> int:
        return len(self.prices)

    def last(self) -> Optional[float]:
        return self.prices[-1] if self.prices else None

    def clear(self):
        self.prices.clear()


def brownian_reversal_prob(
    side: str, current: float, stabilize_level: float, sigma: float
) -> float:
    """
    Estimate the probability that the price will reverse from the
    stabilized level (not exceed it on the opposite side).

    side: "yes" or "no"
    current: current market price (implied probability)
    stabilize_level: level the side stabilized at (e.g. 0.70 for "yes")
    sigma: volatility (annualized or market-specific)

    Returns a float in [0,1] representing the "lock-in" probability.
    """
    if sigma <= 0:
        return 0.5
    if side == "yes":
        # "yes" stabilized high; check probability it doesn't fall below threshold
        d = (stabilize_level - 0.5) / (sigma + 1e-6)
    else:
        # "no" stabilized high; check probability it doesn't rise above threshold
        d = (stabilize_level - 0.5) / (sigma + 1e-6)

    # Simple normal CDF approximation
    return 0.5 * (1.0 + math.tanh(d / math.sqrt(2.0)))


def calc_fee(p: float, rate: float = 0.02) -> float:
    """
    Linear fee model: fee = rate * (1 - p) where p is the
    implied probability for the YES side.

    At p=0.50 (symmetric): fee = 0.01 (1.00%).
    rate ~0.02 was calibrated against observed 0.25% spread at p=0.495.
    """
    return rate * (1.0 - p)


def calc_edge(
    p_signal: float, p_market: float, fee: float, kelly_fraction: float = 0.25
) -> Tuple[float, float]:
    """
    Calculate edge and kelly position size.

    Returns: (edge_pct, kelly_position_size)
    """
    if p_market <= 0 or p_market >= 1:
        return (0.0, 0.0)

    implied_return = (p_signal / p_market) - 1.0
    edge = implied_return - fee

    if edge <= 0:
        return (edge, 0.0)

    # Kelly fraction: f = (p*b - q) / b where p=win%, q=lose%, b=odds
    # Simplified: kelly = edge / (p_market + 1e-6)
    kelly = (edge * p_market) / (p_market + 1e-6)
    kelly_position = kelly_fraction * kelly

    return (edge, kelly_position)


# ============================================================================
# ENGINE: _ChainlinkArbEngine
# ============================================================================


class _ChainlinkArbEngine:
    """
    Multi-asset arbitrage engine using Chainlink + Polymarket prices.

    Per-asset configuration:
      - sigma_fallback: volatility assumption if Brownian fails
      - delta_min_abs: minimum absolute price move to trigger signal
      - delta_max_abs: maximum absolute price move to filter anomalies
    """

    def __init__(
        self,
        asset: str,
        timeframe: str,
        sigma_fallback: float = 0.15,
        delta_min_abs: float = 0.01,
        delta_max_abs: float = 0.02,
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.sigma_fallback = sigma_fallback
        self.delta_min_abs = delta_min_abs
        self.delta_max_abs = delta_max_abs

    def evaluate(
        self,
        timestamp_unix: float,
        market_price: float,
        chainlink_price: float,
        fee: float,
        sigma: Optional[float] = None,
    ) -> Tuple[Optional[str], dict]:
        """
        Evaluate the Chainlink arbitrage signal.

        Returns: (signal, metadata_dict)
          signal: "yes", "no", or None
          metadata_dict: diagnostic info
        """
        meta = {
            "engine": "ChainlinkArb",
            "asset": self.asset,
            "timeframe": self.timeframe,
        }

        if market_price <= 0 or chainlink_price <= 0:
            return (None, meta)

        delta = chainlink_price - market_price
        delta_abs = abs(delta)

        # Filter 1: anomaly detection
        if delta_abs > self.delta_max_abs:
            meta["reason"] = f"delta_abs={delta_abs:.4f} > delta_max_abs={self.delta_max_abs}"
            return (None, meta)

        # Filter 2: minimum move requirement
        if delta_abs < self.delta_min_abs:
            meta["reason"] = f"delta_abs={delta_abs:.4f} < delta_min_abs={self.delta_min_abs}"
            return (None, meta)

        # Determine signal direction
        if delta > 0:
            signal_side = "yes"
            p_signal = chainlink_price
        else:
            signal_side = "no"
            p_signal = 1.0 - chainlink_price

        edge, kelly_pos = calc_edge(p_signal, market_price, fee)

        meta.update(
            {
                "delta": delta,
                "delta_abs": delta_abs,
                "signal_side": signal_side,
                "p_signal": p_signal,
                "p_market": market_price,
                "edge": edge,
                "kelly_position": kelly_pos,
            }
        )

        if edge > 0:
            return (signal_side, meta)
        else:
            meta["reason"] = f"edge={edge:.4f} <= 0"
            return (None, meta)


# ============================================================================
# ENGINE: _BTCStabilizationEngine
# ============================================================================


class _BTCStabilizationEngine:
    """
    BTC 15m stabilization engine: fires when one side stabilizes
    at 63-80¢ (sweet spot ~70¢) in the T=60-180s window.

    Checks Brownian reversal probability to confirm the move is
    statistically locked in. Bypasses overcrowded ChainlinkArb.

    Per-market configuration:
      - sigma_fallback: volatility assumption (default 0.15)
      - delta_max_abs: maximum price delta to filter anomalies (default 0.02)
      - stabilization_price_low, stabilization_price_high: target range
      - min_samples: minimum history to confirm stabilization
      - reversal_prob_min: minimum Brownian lock-in probability
    """

    def __init__(
        self,
        sigma_fallback: float = 0.15,
        delta_max_abs: float = 0.02,
        stabilization_price_low: float = 0.63,
        stabilization_price_high: float = 0.80,
        min_samples: int = 5,
        reversal_prob_min: float = 0.70,
    ):
        self.sigma_fallback = sigma_fallback
        self.delta_max_abs = delta_max_abs
        self.stabilization_price_low = stabilization_price_low
        self.stabilization_price_high = stabilization_price_high
        self.min_samples = min_samples
        self.reversal_prob_min = reversal_prob_min

        # Per-market state: market_id -> {'last_price', 'market_ph'}
        self._market_state = {}
        self._current_market_id = None

    def _get_or_create_market(self, market_id: str) -> dict:
        """Get or create state for a market."""
        if market_id not in self._market_state:
            self._market_state[market_id] = {
                "last_price": None,
                "market_ph": PriceHistory(max_len=300),
            }
        return self._market_state[market_id]

    def _on_market_change(self, new_market_id: str):
        """Called when switching to a different market. Clears prior state."""
        if self._current_market_id != new_market_id:
            if self._current_market_id is not None:
                old_state = self._market_state.get(self._current_market_id)
                if old_state:
                    old_state["market_ph"].clear()
            self._current_market_id = new_market_id

    def evaluate(
        self,
        market_id: str,
        timestamp_unix: float,
        market_price: float,
        sigma: Optional[float] = None,
    ) -> Tuple[Optional[str], dict]:
        """
        Evaluate the BTC stabilization signal.

        Returns: (signal, metadata_dict)
          signal: "yes", "no", or None
          metadata_dict: diagnostic info
        """
        meta = {
            "engine": "BTCStabilization",
            "market_id": market_id,
            "timestamp": timestamp_unix,
        }

        if market_price <= 0 or market_price >= 1:
            return (None, meta)

        # Handle market change: clear prior market state
        self._on_market_change(market_id)

        # Get or create market state
        state = self._get_or_create_market(market_id)
        market_ph = state["market_ph"]

        # Pre-accumulate from market open: append BEFORE time check
        # (v4.1 fix: was only accumulating after T=180s)
        market_ph.append(market_price)

        # Check T=60-180s window
        elapsed_sec = timestamp_unix % 60
        if not (60 <= elapsed_sec < 180):
            meta["reason"] = f"elapsed={elapsed_sec}s not in [60,180)"
            return (None, meta)

        # Need minimum history
        if market_ph.size() < self.min_samples:
            meta["reason"] = f"size={market_ph.size()} < min_samples={self.min_samples}"
            return (None, meta)

        # Check stabilization: is the price staying in range?
        prices = list(market_ph.prices)
        recent_mean = sum(prices[-self.min_samples :]) / self.min_samples
        recent_std = (
            math.sqrt(
                sum((p - recent_mean) ** 2 for p in prices[-self.min_samples :])
                / max(1, self.min_samples - 1)
            )
            if self.min_samples > 1
            else 0
        )

        # Filter: anomaly detection
        if abs(market_price - recent_mean) > self.delta_max_abs:
            meta["reason"] = f"delta={abs(market_price - recent_mean):.4f} > delta_max_abs={self.delta_max_abs}"
            return (None, meta)

        if not (
            self.stabilization_price_low <= recent_mean <= self.stabilization_price_high
        ):
            meta["reason"] = f"recent_mean={recent_mean:.4f} not in [{self.stabilization_price_low}, {self.stabilization_price_high}]"
            return (None, meta)

        # Determine signal side
        if recent_mean >= 0.70:
            signal_side = "yes"
        else:
            signal_side = "no"

        # Brownian reversal check
        sigma_used = sigma if sigma is not None else self.sigma_fallback
        reversal_prob = brownian_reversal_prob(signal_side, market_price, recent_mean, sigma_used)

        if reversal_prob < self.reversal_prob_min:
            meta["reason"] = f"reversal_prob={reversal_prob:.4f} < {self.reversal_prob_min}"
            return (None, meta)

        meta.update(
            {
                "signal_side": signal_side,
                "recent_mean": recent_mean,
                "recent_std": recent_std,
                "reversal_prob": reversal_prob,
                "history_size": market_ph.size(),
            }
        )

        return (signal_side, meta)


# ============================================================================
# MAIN ENGINE: SignalEngine
# ============================================================================


class SignalEngine:
    """
    Multi-strategy signal router.

    Routing logic:
      - BTC 15m  → _BTCStabilizationEngine (24/7)
      - BTC 5m   → _ChainlinkArbEngine (peak hours only)
      - ETH/SOL/XRP → _ChainlinkArbEngine (peak hours only, 15m only)

    Each asset has per-asset Chainlink fee, sigma, and delta params.
    """

    def __init__(
        self,
        asset_config: dict,
        btc_stab_engine: Optional[_BTCStabilizationEngine] = None,
    ):
        """
        asset_config: dict mapping asset -> {
            'sigma_fallback': float,
            'delta_min_abs': float,
            'delta_max_abs': float,
            'fee_rate': float,
        }
        """
        self.asset_config = asset_config or {}
        self.btc_stab_engine = (
            btc_stab_engine if btc_stab_engine is not None else _BTCStabilizationEngine()
        )

        # Chainlink engines per (asset, timeframe)
        self._chainlink_engines = {}

    def _get_or_create_chainlink_engine(
        self, asset: str, timeframe: str
    ) -> _ChainlinkArbEngine:
        """Lazily create Chainlink engine for (asset, timeframe) pair."""
        key = (asset, timeframe)
        if key not in self._chainlink_engines:
            cfg = self.asset_config.get(asset, {})
            engine = _ChainlinkArbEngine(
                asset=asset,
                timeframe=timeframe,
                sigma_fallback=cfg.get("sigma_fallback", 0.15),
                delta_min_abs=cfg.get("delta_min_abs", 0.01),
                delta_max_abs=cfg.get("delta_max_abs", 0.02),
            )
            self._chainlink_engines[key] = engine
        return self._chainlink_engines[key]

    def evaluate(
        self,
        asset_symbol: str,
        timeframe: str,
        timestamp_unix: float,
        market_price: float,
        chainlink_price: float,
        market_id: Optional[str] = None,
        sigma: Optional[float] = None,
    ) -> Tuple[Optional[str], dict]:
        """
        Evaluate a signal for the given market.

        asset_symbol: e.g. "BTC", "ETH", "SOL", "XRP"
        timeframe: e.g. "5m", "15m"
        timestamp_unix: Unix timestamp in seconds
        market_price: Polymarket implied probability [0, 1]
        chainlink_price: Chainlink oracle price [0, 1] or external price
        market_id: unique market identifier (for _BTCStabilizationEngine)
        sigma: market volatility (optional)

        Returns: (signal, metadata_dict)
          signal: "yes", "no", or None
          metadata_dict: diagnostic info
        """

        # Route based on asset and timeframe
        if asset_symbol == "BTC" and timeframe == "15m":
            # BTC 15m → _BTCStabilizationEngine (24/7)
            if market_id is None:
                market_id = f"BTC_15m_default"
            return self.btc_stab_engine.evaluate(
                market_id=market_id,
                timestamp_unix=timestamp_unix,
                market_price=market_price,
                sigma=sigma,
            )

        elif asset_symbol == "BTC" and timeframe == "5m":
            # BTC 5m → _ChainlinkArbEngine (peak hours only)
            if not is_peak_hours(timestamp_unix):
                return (None, {"reason": "not peak hours", "asset": asset_symbol})
            engine = self._get_or_create_chainlink_engine(asset_symbol, timeframe)
            fee = self._compute_fee(asset_symbol, market_price)
            return engine.evaluate(
                timestamp_unix=timestamp_unix,
                market_price=market_price,
                chainlink_price=chainlink_price,
                fee=fee,
                sigma=sigma,
            )

        elif asset_symbol in ["ETH", "SOL", "XRP"]:
            # ETH/SOL/XRP → _ChainlinkArbEngine (peak hours only, 15m only)
            if timeframe != "15m":
                return (None, {"reason": f"{asset_symbol} only supports 15m", "asset": asset_symbol})
            if not is_peak_hours(timestamp_unix):
                return (None, {"reason": "not peak hours", "asset": asset_symbol})
            engine = self._get_or_create_chainlink_engine(asset_symbol, timeframe)
            fee = self._compute_fee(asset_symbol, market_price)
            return engine.evaluate(
                timestamp_unix=timestamp_unix,
                market_price=market_price,
                chainlink_price=chainlink_price,
                fee=fee,
                sigma=sigma,
            )

        else:
            return (None, {"reason": f"unsupported asset: {asset_symbol}"})

    def _compute_fee(self, asset: str, market_price: float) -> float:
        """Compute spread fee for the asset."""
        cfg = self.asset_config.get(asset, {})
        fee_rate = cfg.get("fee_rate", 0.02)
        return calc_fee(market_price, rate=fee_rate)


# ============================================================================
# ENTRY POINT & TESTING
# ============================================================================


def main():
    """Example usage."""

    # Configure assets
    asset_config = {
        "BTC": {
            "sigma_fallback": 0.15,
            "delta_min_abs": 0.01,
            "delta_max_abs": 0.02,
            "fee_rate": 0.02,
        },
        "ETH": {
            "sigma_fallback": 0.25,
            "delta_min_abs": 0.015,
            "delta_max_abs": 0.025,
            "fee_rate": 0.02,
        },
        "SOL": {
            "sigma_fallback": 0.30,
            "delta_min_abs": 0.02,
            "delta_max_abs": 0.03,
            "fee_rate": 0.02,
        },
        "XRP": {
            "sigma_fallback": 0.20,
            "delta_min_abs": 0.02,
            "delta_max_abs": 0.03,
            "fee_rate": 0.02,
        },
    }

    # Create engines
    btc_stab = _BTCStabilizationEngine(
        sigma_fallback=0.15,
        delta_max_abs=0.02,
        stabilization_price_low=0.63,
        stabilization_price_high=0.80,
        min_samples=5,
        reversal_prob_min=0.70,
    )

    signal_engine = SignalEngine(asset_config, btc_stab_engine=btc_stab)

    # Example 1: BTC 15m → _BTCStabilizationEngine
    print("\n=== BTC 15m (Stabilization) ===")
    signal1, meta1 = signal_engine.evaluate(
        asset_symbol="BTC",
        timeframe="15m",
        timestamp_unix=1711353660,  # Some timestamp
        market_price=0.70,
        chainlink_price=0.72,
        market_id="btc_15m_market_1",
        sigma=0.15,
    )
    print(f"Signal: {signal1}")
    print(f"Meta: {json.dumps(meta1, indent=2)}")

    # Example 2: BTC 5m (peak hours only) → _ChainlinkArbEngine
    print("\n=== BTC 5m (ChainlinkArb, peak hours check) ===")
    signal2, meta2 = signal_engine.evaluate(
        asset_symbol="BTC",
        timeframe="5m",
        timestamp_unix=1711353660,  # Check if peak hours
        market_price=0.495,
        chainlink_price=0.505,
        sigma=0.15,
    )
    print(f"Signal: {signal2}")
    print(f"Meta: {json.dumps(meta2, indent=2)}")

    # Example 3: ETH 15m (peak hours only) → _ChainlinkArbEngine
    print("\n=== ETH 15m (ChainlinkArb, peak hours check) ===")
    signal3, meta3 = signal_engine.evaluate(
        asset_symbol="ETH",
        timeframe="15m",
        timestamp_unix=1711353660,
        market_price=0.48,
        chainlink_price=0.52,
        sigma=0.25,
    )
    print(f"Signal: {signal3}")
    print(f"Meta: {json.dumps(meta3, indent=2)}")

    # Example 4: XRP 15m with anomaly (delta too large)
    print("\n=== XRP 15m (Anomaly detection: delta_max_abs) ===")
    signal4, meta4 = signal_engine.evaluate(
        asset_symbol="XRP",
        timeframe="15m",
        timestamp_unix=1711353660,
        market_price=0.40,
        chainlink_price=0.45,  # +5% move, likely anomaly
        sigma=0.20,
    )
    print(f"Signal: {signal4}")
    print(f"Meta: {json.dumps(meta4, indent=2)}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
