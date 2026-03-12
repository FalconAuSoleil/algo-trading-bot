"""Signal engine — core decision logic."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import SignalConfig
from src.utils.math_utils import (
    calc_true_probability,
    calc_taker_fee,
    calc_edge,
    kelly_size,
    RollingVolatility,
)
from src.utils.logger import setup_logger

log = setup_logger("engine.signal")


@dataclass
class MarketState:
    """Current state of a BTC binary market."""

    market_id: str = ""
    reference_price: float = 0.0
    end_time: float = 0.0
    btc_chainlink: float = 0.0
    btc_binance: float = 0.0
    p_market_yes: float = 0.5
    depth_yes: float = 0.0
    depth_no: float = 0.0
    best_ask_yes: float = 0.0
    best_ask_no: float = 0.0


@dataclass
class Signal:
    """Output of the signal engine."""

    timestamp: float = 0.0
    market_id: str = ""
    # Calculated values
    delta_chainlink: float = 0.0
    delta_binance: float = 0.0
    sigma: float = 0.0
    time_remaining_sec: float = 0.0
    p_true: float = 0.5
    p_market: float = 0.5
    edge: float = 0.0
    taker_fee: float = 0.0
    kelly_pct: float = 0.0
    # Decision
    side: str = ""  # "YES", "NO", or ""
    action: str = "NO_TRADE"  # "BUY" or "NO_TRADE"
    entry_price: float = 0.0
    size_usd: float = 0.0
    # Filter details
    filters_passed: bool = False
    filter_reasons: list[str] = field(default_factory=list)
    # Raw state
    btc_chainlink: float = 0.0
    btc_binance: float = 0.0
    reference_price: float = 0.0
    slug: str = ""
    # Market timing (for resolution)
    market_start_time: float = 0.0
    market_duration: int = 300


class SignalEngine:
    """Calculates trading signals from market state."""

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self.volatility = RollingVolatility(
            window_minutes=cfg.volatility_window_minutes
        )

    def update_price(self, price: float, timestamp: float) -> None:
        """Feed price tick to volatility calculator."""
        self.volatility.update(price, timestamp)

    def evaluate(
        self,
        state: MarketState,
        capital: float,
        consecutive_losses: int = 0,
        daily_pnl_pct: float = 0.0,
        open_positions: int = 0,
        has_position_on_market: bool = False,
        macro_blackout: bool = False,
    ) -> Signal:
        """Evaluate current state and produce a signal."""
        now = time.time()
        sig = Signal(
            timestamp=now,
            market_id=state.market_id,
            btc_chainlink=state.btc_chainlink,
            btc_binance=state.btc_binance,
            reference_price=state.reference_price,
        )

        # --- Basic checks ---
        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            sig.filter_reasons.append("missing_price_data")
            return sig

        # --- Calculate deltas ---
        sig.delta_chainlink = (
            (state.btc_chainlink - state.reference_price)
            / state.reference_price
        )
        if state.btc_binance > 0:
            sig.delta_binance = (
                (state.btc_binance - state.reference_price)
                / state.reference_price
            )

        # --- Time remaining ---
        sig.time_remaining_sec = state.end_time - now
        t_minutes = sig.time_remaining_sec / 60.0

        # --- Volatility ---
        sig.sigma = self.volatility.sigma

        # --- True probability ---
        sig.p_true = calc_true_probability(
            sig.delta_chainlink, sig.sigma, t_minutes
        )

        # --- Determine side ---
        if sig.delta_chainlink > 0:
            sig.side = "YES"
            sig.p_market = state.best_ask_yes or state.p_market_yes
            sig.entry_price = state.best_ask_yes or state.p_market_yes
        elif sig.delta_chainlink < 0:
            sig.side = "NO"
            sig.p_market = state.best_ask_no or (
                1.0 - state.p_market_yes
            )
            sig.entry_price = state.best_ask_no or (
                1.0 - state.p_market_yes
            )
        else:
            sig.filter_reasons.append("delta_zero")
            return sig

        # --- Edge ---
        sig.taker_fee = calc_taker_fee(
            sig.p_market, self.cfg.fee_rate
        )
        sig.edge = calc_edge(
            sig.p_true, sig.p_market, self.cfg.fee_rate
        )

        # --- Apply filters ---
        sig.filter_reasons = self._apply_filters(
            sig,
            state,
            consecutive_losses,
            daily_pnl_pct,
            open_positions,
            has_position_on_market,
            macro_blackout,
        )
        sig.filters_passed = len(sig.filter_reasons) == 0

        if not sig.filters_passed:
            return sig

        # --- Position sizing ---
        from src.config import config

        depth = (
            state.depth_yes if sig.side == "YES" else state.depth_no
        )
        kelly_amt = kelly_size(
            sig.edge,
            sig.entry_price,
            sig.taker_fee,
            capital,
            fraction=config.risk.kelly_fraction,
            max_pct=config.risk.max_position_pct,
        )
        max_depth_size = depth * 0.3 if depth > 0 else kelly_amt
        sig.size_usd = min(kelly_amt, max_depth_size)
        sig.kelly_pct = sig.size_usd / capital if capital > 0 else 0

        if sig.size_usd < 1.0:
            sig.filter_reasons.append("size_too_small")
            sig.filters_passed = False
            return sig

        sig.action = "BUY"
        return sig

    def _apply_filters(
        self,
        sig: Signal,
        state: MarketState,
        consecutive_losses: int,
        daily_pnl_pct: float,
        open_positions: int,
        has_position_on_market: bool,
        macro_blackout: bool,
    ) -> list[str]:
        """Apply all trade filters. Returns list of failures."""
        reasons = []

        # One position per market
        if has_position_on_market:
            reasons.append("already_in_market")

        # Time window
        if sig.time_remaining_sec < self.cfg.time_min_seconds:
            reasons.append(
                f"time_too_short:{sig.time_remaining_sec:.0f}s"
            )
        if sig.time_remaining_sec > self.cfg.time_max_seconds:
            reasons.append(
                f"time_too_far:{sig.time_remaining_sec:.0f}s"
            )

        # Delta minimum
        if abs(sig.delta_chainlink) < self.cfg.delta_min:
            reasons.append(
                f"delta_too_small:{sig.delta_chainlink:.5f}"
            )

        # Source coherence
        if state.btc_binance > 0:
            divergence = abs(
                sig.delta_chainlink - sig.delta_binance
            )
            if divergence > self.cfg.source_coherence_max:
                reasons.append(
                    f"source_divergence:{divergence:.5f}"
                )

        # Edge minimum
        if sig.edge < self.cfg.edge_min:
            reasons.append(f"edge_too_low:{sig.edge:.4f}")

        # Entry price cap — above 0.85 the risk:reward is
        # terrible (risk $0.85 to win $0.15 = 5.7:1 against)
        if sig.entry_price > 0.85:
            reasons.append(
                f"entry_too_high:{sig.entry_price:.2f}"
            )

        # Orderbook depth
        depth = (
            state.depth_yes if sig.side == "YES" else state.depth_no
        )
        if depth < 2000:
            reasons.append(f"depth_low:{depth:.0f}")

        # Volatility cap
        if sig.sigma > self.cfg.volatility_max:
            reasons.append(f"vol_too_high:{sig.sigma:.6f}")

        # Macro blackout
        if macro_blackout:
            reasons.append("macro_blackout")

        # Circuit breaker
        from src.config import config

        risk = config.risk
        if consecutive_losses >= risk.max_consecutive_losses:
            reasons.append(
                f"circuit_breaker:{consecutive_losses}_losses"
            )

        # Daily loss limit
        if daily_pnl_pct < -risk.max_daily_drawdown:
            reasons.append(
                f"daily_loss_limit:{daily_pnl_pct:.2%}"
            )

        # Max open positions
        if open_positions >= risk.max_open_positions:
            reasons.append(
                f"max_positions:{open_positions}"
            )

        return reasons
