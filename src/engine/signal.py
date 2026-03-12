"""Signal Engine v2 — Market Microstructure Brain
================================================

4 composantes combinées en un score bayésien :

1. CHAINLINK LAG ARBITRAGE
   Chainlink update toutes ~20-40s sur Polygon.
   Dans les secondes avant/après un update, on connaît le prix
   de résolution avec quasi-certitude.
   → Fenêtre d'edge maximale détectable

2. ORDER FLOW IMBALANCE (OFI)
   Basé sur le modèle de Cont, Kukanov & Stoikov (2014).
   Si le carnet est asymétrique (plus de volume côté ASK-UP),
   les "smart money" ont déjà pricé le mouvement.
   → Prédicteur du mouvement de prix à court terme

3. KYLE LAMBDA — PRICE IMPACT
   Modèle de Kyle (1985) : l'impact prix d'un ordre révèle
   l'information privée. Un spread large + faible depth =
   market makers incertains = signal peu fiable.
   → Filtre de fiabilité du signal

4. HAWKES PROCESS — CLUSTERING D'ÉVÉNEMENTS
   Les mouvements de prix ne sont pas poissonniens — ils arrivent
   en clusters (Hawkes, 1971). On estime l'intensité conditionnelle
   du prochain move via un processus auto-excitateur.
   → Détecte les régimes de forte activité = edge plus fiable

Combinaison finale : score bayésien
   P(UP | data) = f(chainlink_edge, OFI, kyle_filter, hawkes_intensity)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.config import SignalConfig
from src.utils.logger import setup_logger

log = setup_logger("engine.signal")


# ─────────────────────────────────────────────────────────
# Math utilities
# ─────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 500)))
    e = math.exp(max(x, -500))
    return e / (1.0 + e)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def logit(p: float) -> float:
    """Logit = log(p / (1-p)). Inverse of sigmoid."""
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1.0 - p))


def shrink_logit(logit_val: float, factor: float) -> float:
    """Shrink logit toward zero by factor ∈ [0, 1].

    factor=1.0 → no change, factor=0.3 → pulls toward 0.5.
    This correctly handles both positive and negative logits
    (unlike naive multiplication which has asymmetric effects).
    """
    return logit_val * clamp(factor, 0.0, 1.0)


def calc_taker_fee(p_market: float, fee_rate: float = 0.25) -> float:
    """Polymarket dynamic taker fee: fee_rate × (p × (1-p))^2."""
    if p_market <= 0 or p_market >= 1:
        return 0.0
    return fee_rate * (p_market * (1.0 - p_market)) ** 2


# ─────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────

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
    best_bid_yes: float = 0.0
    best_ask_yes: float = 0.0
    best_bid_no: float = 0.0
    best_ask_no: float = 0.0
    spread_yes: float = 0.01
    spread_no: float = 0.01
    slug: str = ""
    start_time: float = 0.0
    duration_seconds: int = 300


@dataclass
class MicrostructureState:
    """Full microstructure state for debug/dashboard."""
    chainlink_lag_seconds: float = 0.0
    chainlink_edge_boost: float = 0.0
    ofi_raw: float = 0.0
    ofi_signal: float = 0.0
    kyle_lambda: float = 0.0
    kyle_penalty: float = 0.0
    hawkes_intensity: float = 0.0
    hawkes_boost: float = 0.0
    base_prob_up: float = 0.5
    final_prob_up: float = 0.5
    stability_ratio: float = 0.0
    stability_edge_cv: float = 0.0
    stability_ok: bool = False
    stability_ticks: int = 0
    taker_fee: float = 0.0
    source_divergence: float = 0.0
    time_decay_factor: float = 1.0
    components: dict = field(default_factory=dict)


@dataclass
class Signal:
    """Output of the signal engine."""
    timestamp: float = 0.0
    market_id: str = ""
    delta_chainlink: float = 0.0
    delta_binance: float = 0.0
    sigma: float = 0.0
    time_remaining_sec: float = 0.0
    p_true: float = 0.5
    p_market: float = 0.5
    edge: float = 0.0
    taker_fee: float = 0.0
    kelly_pct: float = 0.0
    side: str = ""  # "YES", "NO", or ""
    action: str = "NO_TRADE"
    entry_price: float = 0.0
    size_usd: float = 0.0
    filters_passed: bool = False
    filter_reasons: list[str] = field(default_factory=list)
    btc_chainlink: float = 0.0
    btc_binance: float = 0.0
    reference_price: float = 0.0
    slug: str = ""
    market_start_time: float = 0.0
    market_duration: int = 300
    micro: MicrostructureState = field(default_factory=MicrostructureState)
    confidence: str = "LOW"
    status: str = "WATCHING"


# ─────────────────────────────────────────────────────────
# Component 1 : Chainlink Lag Arbitrage
# ─────────────────────────────────────────────────────────

class ChainlinkArbModule:
    """Exploits the ~27s Chainlink update lag on Polygon."""

    def __init__(self, cfg: SignalConfig):
        self._last_chainlink_ts: float = 0.0
        self._last_chainlink_price: float = 0.0
        self._chainlink_updates: deque = deque(maxlen=20)
        self._estimated_period: float = cfg.chainlink_period
        self._edge_window: float = cfg.chainlink_edge_window

    def on_chainlink_update(self, price: float, ts: float) -> None:
        if self._last_chainlink_ts > 0:
            delta = ts - self._last_chainlink_ts
            if 5.0 < delta < 120.0:
                self._chainlink_updates.append(delta)
                if len(self._chainlink_updates) >= 3:
                    weights = [0.5 ** i for i in range(len(self._chainlink_updates))]
                    total_w = sum(weights)
                    vals = list(reversed(self._chainlink_updates))
                    self._estimated_period = sum(v * w for v, w in zip(vals, weights)) / total_w
        self._last_chainlink_ts = ts
        self._last_chainlink_price = price

    def compute(self, binance_price: float, chainlink_price: float, current_ts: float) -> tuple[float, float]:
        if self._last_chainlink_ts <= 0 or chainlink_price <= 0:
            return 0.0, 0.0
        lag = current_ts - self._last_chainlink_ts
        time_to_next_update = self._estimated_period - (lag % self._estimated_period)
        price_gap = (binance_price - chainlink_price) / chainlink_price
        proximity = clamp(1.0 - time_to_next_update / self._edge_window, 0.0, 1.0)
        # Calibrated: 0.1% gap at full proximity → ~0.15 logit boost
        edge_boost = proximity * price_gap * 1500.0
        # Cap the boost to avoid wild swings from stale data
        edge_boost = clamp(edge_boost, -1.5, 1.5)
        return lag, edge_boost


# ─────────────────────────────────────────────────────────
# Component 2 : Order Flow Imbalance (OFI)
# ─────────────────────────────────────────────────────────

class OFIModule:
    """Order Flow Imbalance — Cont, Kukanov & Stoikov (2014)."""

    def __init__(self, cfg: SignalConfig):
        self._ofi_history: deque = deque(maxlen=30)
        self._ofi_weight: float = cfg.ofi_weight

    def compute(self, bid_up: float, ask_up: float, bid_down: float, ask_down: float,
                depth_up: float, depth_down: float) -> tuple[float, float]:
        total_up = bid_up + ask_up
        total_down = bid_down + ask_down
        if total_up <= 0:
            return 0.0, 0.0
        ofi_up = (bid_up - ask_up) / max(total_up, 1e-6)
        ofi_down = (bid_down - ask_down) / max(total_down + 1e-6, 1e-6) if total_down > 0 else 0.0
        ofi_net = 0.5 * ofi_up - 0.5 * ofi_down
        total_depth = depth_up + depth_down
        depth_imbalance = (depth_up - depth_down) / total_depth if total_depth > 0 else 0.0
        ofi_combined = 0.6 * ofi_net + 0.4 * depth_imbalance
        self._ofi_history.append((time.time(), ofi_combined))
        ofi_momentum = 0.0
        if len(self._ofi_history) >= 5:
            recent = list(self._ofi_history)[-10:]
            ofi_momentum = recent[-1][1] - recent[0][1]
        ofi_signal = ofi_combined * self._ofi_weight + ofi_momentum * 0.1
        logit_adjustment = clamp(ofi_signal * 2.0, -0.8, 0.8)
        return ofi_combined, logit_adjustment


# ─────────────────────────────────────────────────────────
# Component 3 : Kyle Lambda
# ─────────────────────────────────────────────────────────

class KyleModule:
    """Kyle (1985) price impact model — signal reliability filter."""

    def __init__(self, cfg: SignalConfig):
        self._spread_history: deque = deque(maxlen=50)
        self._depth_history: deque = deque(maxlen=50)
        self._spread_penalty: float = cfg.kyle_spread_penalty

    def compute(self, spread_up: float, spread_down: float,
                depth_up: float, depth_down: float) -> tuple[float, float]:
        avg_spread = (spread_up + (spread_down or spread_up)) / 2.0
        avg_depth = (depth_up + (depth_down or depth_up)) / 2.0
        self._spread_history.append(avg_spread)
        self._depth_history.append(avg_depth)
        if avg_depth <= 0:
            return 0.0, 1.0
        kyle_lambda = avg_spread / (2.0 * math.sqrt(max(avg_depth, 1.0)))
        if len(self._spread_history) >= 5:
            hist_spread_mean = sum(self._spread_history) / len(self._spread_history)
            hist_depth_mean = sum(self._depth_history) / len(self._depth_history)
            relative_spread = avg_spread / max(hist_spread_mean, 1e-6)
            relative_depth = avg_depth / max(hist_depth_mean, 1e-6)
        else:
            relative_spread = 1.0
            relative_depth = 1.0
        spread_pen = clamp((relative_spread - 1.0) * self._spread_penalty, 0.0, self._spread_penalty)
        depth_bonus = clamp((relative_depth - 1.0) * 0.05, 0.0, 0.10)
        quality_factor = clamp(1.0 - spread_pen + depth_bonus, 0.3, 1.0)
        return kyle_lambda, quality_factor


# ─────────────────────────────────────────────────────────
# Component 4 : Hawkes Process
# ─────────────────────────────────────────────────────────

class HawkesModule:
    """Hawkes (1971) self-exciting process — regime detection."""

    def __init__(self, cfg: SignalConfig):
        self._events: deque = deque(maxlen=cfg.hawkes_history)
        self._last_mid: float = 0.5
        self._mu: float = cfg.hawkes_mu
        self._alpha: float = cfg.hawkes_alpha
        self._beta: float = cfg.hawkes_beta

    def on_price_event(self, ts: float, magnitude: float = 1.0) -> None:
        self._events.append((ts, magnitude))

    def on_mid_update(self, mid_up: float, ts: float) -> None:
        if self._last_mid > 0:
            change = abs(mid_up - self._last_mid)
            if change >= 0.005:
                magnitude = min(change / 0.005, 5.0)
                self.on_price_event(ts, magnitude)
        self._last_mid = mid_up

    def intensity(self, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
        lam = self._mu
        for ts_i, mag_i in self._events:
            dt = t - ts_i
            if dt < 0:
                continue
            lam += self._alpha * mag_i * math.exp(-self._beta * dt)
        return lam

    def regime_boost(self, t: Optional[float] = None) -> tuple[float, float]:
        lam = self.intensity(t)
        excess = max(0.0, lam - self._mu)
        boost = clamp(excess / (5.0 * self._alpha) * 0.3, 0.0, 0.3)
        return boost, lam


# ─────────────────────────────────────────────────────────
# Component 5 : Stability Filter
# ─────────────────────────────────────────────────────────

class StabilityFilter:
    """Directional stability filter — avoids betting on noise."""

    def __init__(self, cfg: SignalConfig):
        self._history: dict[str, deque] = {}
        self._window_sec: float = cfg.stability_window_sec
        self._min_samples: int = cfg.stability_min_samples
        self._min_ratio: float = cfg.stability_min_ratio
        self._edge_cv_max: float = cfg.stability_edge_cv_max

    def _get_buffer(self, market_slug: str) -> deque:
        if market_slug not in self._history:
            self._history[market_slug] = deque(maxlen=100)
        return self._history[market_slug]

    def record(self, market_slug: str, side: str, edge: float, ts: float) -> None:
        buf = self._get_buffer(market_slug)
        buf.append((ts, side, abs(edge)))

    def evaluate(self, market_slug: str, current_side: str) -> tuple[bool, float, float, int]:
        buf = self._get_buffer(market_slug)
        if not buf:
            return False, 0.0, 999.0, 0
        all_ticks = list(buf)
        n = len(all_ticks)
        if n < self._min_samples:
            return False, 0.0, 999.0, n
        same_dir = sum(1 for _, s, _ in all_ticks if s == current_side)
        direction_ratio = same_dir / n
        now = time.time()
        cutoff = now - self._window_sec
        recent = [(ts, side, e) for ts, side, e in all_ticks if ts >= cutoff]
        if not recent:
            recent = all_ticks
        edges = [e for _, _, e in recent]
        mean_edge = sum(edges) / len(edges)
        if mean_edge < 1e-6:
            return False, direction_ratio, 999.0, n
        variance = sum((e - mean_edge) ** 2 for e in edges) / len(edges)
        std_edge = math.sqrt(variance)
        edge_cv = std_edge / mean_edge
        is_stable = (direction_ratio >= self._min_ratio and edge_cv <= self._edge_cv_max)
        return is_stable, direction_ratio, edge_cv, n

    def reset_market(self, market_slug: str) -> None:
        self._history.pop(market_slug, None)


# ─────────────────────────────────────────────────────────
# Signal Engine v2 — Orchestrator
# ─────────────────────────────────────────────────────────

class SignalEngine:
    """Microstructure bayesian signal engine.

    Pipeline (all in logit space for proper bayesian composition):
      1. Prior from delta prix + short momentum
      2. + Chainlink lag arbitrage boost (additive logit)
      3. + OFI signal (additive logit)
      4. × Kyle quality shrinkage (toward 0.5 if uncertain)
      5. × (1 + Hawkes boost) on the RAW logit before Kyle
         (Hawkes amplifies the pre-filter signal, not the shrunk one)
      6. → sigmoid → final prob → edge (net of fees) → decision
    """

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self._price_history: deque = deque(maxlen=120)
        self.chainlink_arb = ChainlinkArbModule(cfg)
        self.ofi = OFIModule(cfg)
        self.kyle = KyleModule(cfg)
        self.hawkes = HawkesModule(cfg)
        self.stability = StabilityFilter(cfg)
        self._binance_price: float = 0.0
        self._chainlink_price: float = 0.0

    def update_price(self, price: float, timestamp: float) -> None:
        self._binance_price = price
        self._price_history.append((timestamp, price))

    def update_chainlink_price(self, price: float, timestamp: float) -> None:
        self._chainlink_price = price
        self.chainlink_arb.on_chainlink_update(price, timestamp)

    def _short_momentum(self, window_seconds: float = 15.0) -> float:
        if len(self._price_history) < 3:
            return 0.0
        now = time.time()
        cutoff = now - window_seconds
        recent = [(ts, p) for ts, p in self._price_history if ts >= cutoff]
        if len(recent) < 2:
            return 0.0
        first_p = recent[0][1]
        last_p = recent[-1][1]
        delta = (last_p - first_p) / first_p
        return clamp(delta / 0.002, -1.0, 1.0)

    def evaluate(
        self,
        state: MarketState,
        capital: float,
        consecutive_losses: int = 0,
        daily_pnl_pct: float = 0.0,
        open_positions: int = 0,
        has_position_on_market: bool = False,
    ) -> Signal:
        now = time.time()
        cfg = self.cfg

        sig = Signal(
            timestamp=now,
            market_id=state.market_id,
            btc_chainlink=state.btc_chainlink,
            btc_binance=state.btc_binance,
            reference_price=state.reference_price,
        )
        micro = sig.micro

        # ── Basic checks ──
        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            sig.filter_reasons.append("missing_price_data")
            return sig

        # ── Deltas ──
        sig.delta_chainlink = (state.btc_chainlink - state.reference_price) / state.reference_price
        if state.btc_binance > 0:
            sig.delta_binance = (state.btc_binance - state.reference_price) / state.reference_price

        # ── Source coherence check ──
        # If Binance and Chainlink diverge too much, data is unreliable
        if state.btc_binance > 0 and state.btc_chainlink > 0:
            source_div = abs(state.btc_binance - state.btc_chainlink) / state.btc_chainlink
            micro.source_divergence = source_div
            if source_div > cfg.source_coherence_max:
                sig.filter_reasons.append(f"source_divergence:{source_div:.5f}")
                sig.status = f"SOURCE_DIVERGENCE ({source_div*100:.3f}%)"
                return sig

        # ── Time remaining ──
        sig.time_remaining_sec = state.end_time - now
        market_slug = state.slug or state.market_id[:20]
        is_5m = "5m" in market_slug or sig.time_remaining_sec < 330

        min_tr = cfg.time_min_5m if is_5m else cfg.time_min_15m
        max_tr = cfg.time_max_5m if is_5m else cfg.time_max_15m
        max_tr_accum = cfg.time_max_5m_accum if is_5m else cfg.time_max_15m

        if sig.time_remaining_sec < min_tr:
            sig.filter_reasons.append(f"too_late:{sig.time_remaining_sec:.0f}s")
            sig.status = f"TOO_LATE ({sig.time_remaining_sec:.0f}s)"
            return sig

        if sig.time_remaining_sec > max_tr_accum:
            sig.filter_reasons.append(f"too_early:{sig.time_remaining_sec:.0f}s")
            sig.status = f"TOO_EARLY ({sig.time_remaining_sec:.0f}s)"
            return sig

        accumulation_only = (sig.time_remaining_sec > max_tr)

        # ── Time decay: scale edge threshold down near expiry ──
        # Closer to expiry = less time for reversal = lower edge needed
        # At max_tr: factor=1.0, at min_tr: factor=0.6
        time_range = max_tr - min_tr
        if time_range > 0:
            time_position = clamp((sig.time_remaining_sec - min_tr) / time_range, 0.0, 1.0)
            time_decay_factor = 0.6 + 0.4 * time_position
        else:
            time_decay_factor = 1.0
        micro.time_decay_factor = time_decay_factor
        effective_edge_min = cfg.edge_min * time_decay_factor

        # ── Step 1: Prior from delta prix ──
        price_delta = sig.delta_chainlink
        momentum = self._short_momentum(15.0)
        z_prior = cfg.momentum_factor * price_delta + 0.2 * momentum
        prob_prior = sigmoid(z_prior)
        micro.base_prob_up = prob_prior
        logit_score = logit(prob_prior)

        # ── Step 2: Chainlink Lag Arb (additive in logit) ──
        binance_p = self._binance_price if self._binance_price > 0 else state.btc_chainlink
        chain_p = self._chainlink_price if self._chainlink_price > 0 else state.reference_price
        lag_s, chainlink_boost = self.chainlink_arb.compute(binance_p, chain_p, now)
        micro.chainlink_lag_seconds = lag_s
        micro.chainlink_edge_boost = chainlink_boost
        logit_score += chainlink_boost

        # ── Step 3: OFI (additive in logit) ──
        self.hawkes.on_mid_update(state.p_market_yes, now)
        ofi_raw, ofi_logit_adj = self.ofi.compute(
            bid_up=state.best_bid_yes, ask_up=state.best_ask_yes,
            bid_down=state.best_bid_no, ask_down=state.best_ask_no,
            depth_up=state.depth_yes, depth_down=state.depth_no,
        )
        micro.ofi_raw = ofi_raw
        micro.ofi_signal = ofi_logit_adj
        logit_score += ofi_logit_adj

        # ── Step 4+5: Hawkes THEN Kyle (correct order) ──
        # Hawkes amplifies raw signal BEFORE Kyle shrinks it.
        # Rationale: if market is active (high Hawkes), the OFI
        # and chainlink signals are more informative → amplify.
        # Then Kyle shrinks based on orderbook quality.
        hawkes_boost, hawkes_intensity = self.hawkes.regime_boost()
        micro.hawkes_intensity = hawkes_intensity
        micro.hawkes_boost = hawkes_boost
        logit_score *= (1.0 + hawkes_boost)

        # Kyle shrinkage: pulls logit toward zero (0.5 in prob)
        kyle_lambda, kyle_quality = self.kyle.compute(
            spread_up=state.spread_yes, spread_down=state.spread_no,
            depth_up=state.depth_yes, depth_down=state.depth_no,
        )
        micro.kyle_lambda = kyle_lambda
        micro.kyle_penalty = kyle_quality
        logit_score = shrink_logit(logit_score, kyle_quality)

        # ── Final prob ──
        final_prob_up = sigmoid(logit_score)
        final_prob_down = 1.0 - final_prob_up
        micro.final_prob_up = final_prob_up
        sig.p_true = final_prob_up

        # ── Market prob and side ──
        market_prob_up = state.p_market_yes
        market_prob_down = 1.0 - market_prob_up
        sig.p_market = market_prob_up

        # ── Edge NET OF FEES ──
        # This is critical: without deducting fees you'll take
        # trades with negative real edge.
        if final_prob_up >= 0.5:
            raw_edge_up = final_prob_up - market_prob_up
            fee_up = calc_taker_fee(market_prob_up, cfg.fee_rate)
            net_edge_up = raw_edge_up - fee_up
            sig.taker_fee = fee_up
            sig.edge = net_edge_up
        else:
            raw_edge_down = final_prob_down - market_prob_down
            fee_down = calc_taker_fee(market_prob_down, cfg.fee_rate)
            net_edge_down = raw_edge_down - fee_down
            sig.taker_fee = fee_down
            sig.edge = -net_edge_down  # keep sign convention: positive = bet YES

        micro.taker_fee = sig.taker_fee

        # ── Liquidity guard ──
        min_depth = min(state.depth_yes or 0.0, state.depth_no or 0.0)
        if min_depth < cfg.min_market_liquidity:
            sig.filter_reasons.append(f"no_liquidity:{min_depth:.0f}")
            sig.status = f"NO_LIQUIDITY (${min_depth:.0f})"
            price_side = "YES" if price_delta > 0 else "NO"
            self.stability.record(market_slug, price_side, abs(price_delta) * 100, now)
            return sig

        # ── Determine candidate side (using net edge) ──
        net_edge_abs = abs(sig.edge)
        if sig.edge >= effective_edge_min:
            candidate_side = "YES"
            edge_abs = sig.edge
        elif -sig.edge >= effective_edge_min:
            candidate_side = "NO"
            edge_abs = -sig.edge
        else:
            candidate_side = ""
            edge_abs = net_edge_abs

        # Guard conviction
        if candidate_side == "YES" and final_prob_up < cfg.min_true_prob:
            candidate_side = ""
            sig.filter_reasons.append(f"low_conviction:{final_prob_up*100:.0f}%")
        elif candidate_side == "NO" and final_prob_down < cfg.min_true_prob:
            candidate_side = ""
            sig.filter_reasons.append(f"low_conviction:{final_prob_down*100:.0f}%")

        # Guard payout ratio
        if candidate_side:
            mp_side = market_prob_up if candidate_side == "YES" else market_prob_down
            if mp_side < cfg.min_market_prob_side:
                candidate_side = ""
                sig.filter_reasons.append(f"longshot:{mp_side*100:.0f}c")
            elif mp_side > cfg.max_market_prob_side:
                candidate_side = ""
                sig.filter_reasons.append(f"bad_odds:{mp_side*100:.0f}c")

        sig.side = candidate_side
        sig.entry_price = state.best_ask_yes if candidate_side == "YES" else state.best_ask_no if candidate_side == "NO" else 0.0

        # ── Stability Filter ──
        if candidate_side:
            self.stability.record(market_slug, candidate_side, edge_abs, now)
        stability_ok, direction_ratio, edge_cv, n_ticks = self.stability.evaluate(
            market_slug, candidate_side
        ) if candidate_side else (False, 0.0, 999.0, 0)

        micro.stability_ratio = direction_ratio
        micro.stability_edge_cv = edge_cv
        micro.stability_ok = stability_ok
        micro.stability_ticks = n_ticks

        should_bet = candidate_side != "" and stability_ok and not accumulation_only

        # ── Risk filters ──
        reasons = []
        if has_position_on_market:
            reasons.append("already_in_market")
        from src.config import config
        risk = config.risk
        if consecutive_losses >= risk.max_consecutive_losses:
            reasons.append(f"circuit_breaker:{consecutive_losses}")
        if daily_pnl_pct < -risk.max_daily_drawdown:
            reasons.append(f"daily_loss:{daily_pnl_pct:.2%}")
        if open_positions >= risk.max_open_positions:
            reasons.append(f"max_positions:{open_positions}")

        sig.filter_reasons.extend(reasons)

        if reasons or not should_bet:
            sig.filters_passed = False
            if candidate_side and not stability_ok:
                if n_ticks < self.cfg.stability_min_samples:
                    sig.status = f"STABILIZING ({n_ticks}/{self.cfg.stability_min_samples})"
                else:
                    sig.status = f"UNSTABLE ({direction_ratio*100:.0f}% {candidate_side})"
            elif not candidate_side:
                sig.status = "WATCHING"
            return sig

        # ── Kelly sizing (using net edge after fees) ──
        prob_win = final_prob_up if sig.side == "YES" else final_prob_down
        market_p = market_prob_up if sig.side == "YES" else market_prob_down
        entry_p = sig.entry_price if sig.entry_price > 0 else market_p
        fee = calc_taker_fee(entry_p, cfg.fee_rate)

        # Effective cost per share
        c_eff = entry_p + fee
        if c_eff >= 1.0 or c_eff <= 0:
            sig.filter_reasons.append("cost_exceeds_payout")
            sig.filters_passed = False
            return sig

        # Kelly for binary: f* = (p × (1-c) - (1-p) × c) / (1-c)
        #                     = (p - c) / (1 - c)
        kelly_full = (prob_win - c_eff) / (1.0 - c_eff)
        if kelly_full <= 0:
            sig.filter_reasons.append(f"negative_kelly:{kelly_full:.4f}")
            sig.filters_passed = False
            return sig

        fraction = clamp(kelly_full * 0.25, 0.0, cfg.max_bet_fraction)

        # Reduce in quiet regimes
        if hawkes_intensity < cfg.hawkes_mu * 1.5:
            fraction *= 0.7

        # Stability bonus
        stability_bonus = clamp((direction_ratio - cfg.stability_min_ratio) * 2.0, 0.0, 0.3)
        fraction *= (1.0 + stability_bonus)
        fraction = clamp(fraction, 0.0, cfg.max_bet_fraction)

        sig.size_usd = round(capital * fraction, 2)
        sig.kelly_pct = fraction

        # Depth limit: don't take more than 30% of visible depth
        depth = state.depth_yes if sig.side == "YES" else state.depth_no
        max_depth_size = depth * 0.3 if depth > 0 else sig.size_usd
        sig.size_usd = min(sig.size_usd, max_depth_size)

        if sig.size_usd < 1.0:
            sig.filter_reasons.append("size_too_small")
            sig.filters_passed = False
            return sig

        # ── Confidence ──
        confidence_score = (edge_abs * 0.4 + kyle_quality * 0.2 +
                           hawkes_boost * 0.2 + (direction_ratio if stability_ok else 0.0) * 0.2)
        sig.confidence = "HIGH" if confidence_score >= 0.12 else "MEDIUM" if confidence_score >= 0.07 else "LOW"

        sig.action = "BUY"
        sig.status = "BETTING"
        sig.filters_passed = True

        log.info(
            "[Micro] %s | side=%s edge=%+.1f%% (net) fee=%.3f%% | "
            "base=%.0f%% → final=%.0f%% | "
            "CL=%+.2f OFI=%+.2f Kyle=%.2f Hawkes=%.2f | "
            "stab=%.0f%% CV=%.2f (%dt) | $%.2f",
            market_slug[-20:], sig.side, sig.edge * 100, sig.taker_fee * 100,
            prob_prior * 100, final_prob_up * 100,
            chainlink_boost, ofi_raw, kyle_quality, hawkes_intensity,
            direction_ratio * 100, edge_cv, n_ticks, sig.size_usd,
        )

        return sig

    def reset_market_stability(self, market_slug: str) -> None:
        self.stability.reset_market(market_slug)
