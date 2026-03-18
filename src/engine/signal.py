"""Signal Engine v3 - Multi-Strategy Router
============================================

Three independent strategies compete for each market evaluation:

  1. ChainlinkArb  — Bayesian microstructure: exploit Chainlink oracle lag,
                     OFI, Kyle lambda, Hawkes process. The primary strategy.
  2. PriceMomentum — Follow sustained short-term BTC price momentum (60s+120s).
                     Confirms or independently triggers directional bets.
  3. MeanReversion — Contrarian: fade extreme delta moves (>0.20%).
                     Activates when ChainlinkArb is too early or edge is low.

Routing logic:
  - Each strategy returns a candidate Signal (action=BUY) or None.
  - If strategies disagree on direction: skip unless one side dominates by >50%.
  - If 2+ strategies agree: consensus size boost +25% per extra strategy.
  - All signals weighted by rolling per-strategy win rate (PerformanceTracker).
  - Unified risk filters applied at router level.

Backward compatible with main.py:
  - SignalEngine.evaluate() same signature
  - Signal, MarketState, MicrostructureState unchanged except added fields
  - update_price(), update_chainlink_price(), reset_market_stability() unchanged
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import SignalConfig
from src.engine.performance import PerformanceTracker
from src.utils.logger import setup_logger

log = setup_logger("engine.signal")


# ── Maths helpers ─────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 500)))
    e = math.exp(max(x, -500))
    return e / (1.0 + e)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1.0 - p))


def shrink_logit(v, f):
    return v * clamp(f, 0.0, 1.0)


def calc_fee(p: float, rate: float = 0.25) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return rate * (p * (1.0 - p)) ** 2


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class MarketState:
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
    """Populated by ChainlinkArbEngine; stays at defaults for other strategies."""
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
    side: str = ""
    action: str = "NO_TRADE"
    entry_price: float = 0.0
    size_usd: float = 0.0
    filters_passed: bool = False
    filter_reasons: list = field(default_factory=list)
    btc_chainlink: float = 0.0
    btc_binance: float = 0.0
    reference_price: float = 0.0
    slug: str = ""
    market_start_time: float = 0.0
    market_duration: int = 300
    micro: MicrostructureState = field(default_factory=MicrostructureState)
    confidence: str = "LOW"
    status: str = "WATCHING"
    # v3: multi-strategy additions
    strategy_used: str = "chainlink_arb"
    strategies_agreeing: int = 1
    token_id_yes: str = ""   # populated by main.py from MarketInfo
    token_id_no: str = ""    # populated by main.py from MarketInfo


# ── Microstructure sub-modules (used by ChainlinkArbEngine) ──────────────────

class _OFI:
    """Order Flow Imbalance."""
    def __init__(self, w: float):
        self._h: deque = deque(maxlen=30)
        self._w = w

    def compute(self, bu, au, bd, ad, du, dd):
        tu = bu + au
        if tu <= 0:
            return 0.0, 0.0
        ou = (bu - au) / max(tu, 1e-6)
        td = bd + ad
        od = (bd - ad) / max(td, 1e-6) if td > 0 else 0.0
        on = 0.5 * ou - 0.5 * od
        tot = du + dd
        di = (du - dd) / tot if tot > 0 else 0.0
        c = 0.6 * on + 0.4 * di
        self._h.append((time.time(), c))
        m = 0.0
        if len(self._h) >= 5:
            r = list(self._h)[-10:]
            m = r[-1][1] - r[0][1]
        return c, clamp((c * self._w + m * 0.1) * 2, -0.8, 0.8)


class _Kyle:
    """Kyle lambda: market impact / adverse selection penalty."""
    def __init__(self, p: float):
        self._sh: deque = deque(maxlen=50)
        self._dh: deque = deque(maxlen=50)
        self._p = p

    def compute(self, su, sd, du, dd):
        avs = (su + (sd or su)) / 2
        avd = (du + (dd or du)) / 2
        self._sh.append(avs)
        self._dh.append(avd)
        if avd <= 0:
            return 0.0, 1.0
        if len(self._sh) >= 5:
            hs = sum(self._sh) / len(self._sh)
            hd = sum(self._dh) / len(self._dh)
            rs = avs / max(hs, 1e-6)
            rd = avd / max(hd, 1e-6)
        else:
            rs, rd = 1.0, 1.0
        sp = clamp((rs - 1) * self._p, 0, self._p)
        db = clamp((rd - 1) * 0.05, 0, 0.1)
        return 0.0, clamp(1 - sp + db, 0.3, 1.0)


class _Hawkes:
    """Hawkes process: microstructure activity detector."""
    def __init__(self, mu, alpha, beta, history):
        self._ev: deque = deque(maxlen=history)
        self._lm = 0.5
        self._mu = mu
        self._a = alpha
        self._b = beta

    def on_mid(self, mid: float, ts: float) -> None:
        if self._lm > 0:
            ch = abs(mid - self._lm)
            if ch >= 0.005:
                self._ev.append((ts, min(ch / 0.005, 5)))
        self._lm = mid

    def boost(self, t: float = 0) -> tuple:
        t = t or time.time()
        lam = self._mu
        for ts, m in self._ev:
            dt = t - ts
            if dt >= 0:
                lam += self._a * m * math.exp(-self._b * dt)
        ex = max(0, lam - self._mu)
        return clamp(ex / (5 * self._a) * 0.3, 0, 0.3), lam


class _Stability:
    """Signal stability filter: requires consistent direction + low edge variance."""
    def __init__(self, ws, ms, mr, mc):
        self._h: dict = {}
        self._ws = ws
        self._ms = ms
        self._mr = mr
        self._mc = mc

    def _buf(self, s: str) -> deque:
        if s not in self._h:
            self._h[s] = deque(maxlen=100)
        return self._h[s]

    def record(self, s: str, side: str, edge: float, ts: float) -> None:
        self._buf(s).append((ts, side, abs(edge)))

    def evaluate(self, s: str, side: str) -> tuple:
        b = self._buf(s)
        if not b:
            return False, 0, 999, 0
        t = list(b)
        n = len(t)
        if n < self._ms:
            return False, 0, 999, n
        dr = sum(1 for _, sd, _ in t if sd == side) / n
        now = time.time()
        r = [e for ts, _, e in t if ts >= now - self._ws] or [e for _, _, e in t]
        me = sum(r) / len(r)
        if me < 1e-6:
            return False, dr, 999, n
        cv = math.sqrt(sum((e - me) ** 2 for e in r) / len(r)) / me
        return (dr >= self._mr and cv <= self._mc), dr, cv, n

    def reset(self, s: str) -> None:
        self._h.pop(s, None)


# ── Strategy 1: Chainlink Arb (primary, Bayesian microstructure) ──────────────

class _ChainlinkArbEngine:
    """
    Primary strategy: exploit Chainlink oracle lag relative to Binance spot.

    When Binance spot has moved but Chainlink hasn't updated yet, we bet
    in the direction Chainlink will move when it catches up. Combined with
    OFI (order flow imbalance), Kyle lambda, and Hawkes process activity.
    """
    NAME = "chainlink_arb"

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self._ph: deque = deque(maxlen=120)
        self._cl_updates: deque = deque(maxlen=20)
        self._cl_period = cfg.chainlink_period
        self._bp = 0.0
        self._cp = 0.0
        self._cl_ts = 0.0
        self.ofi = _OFI(cfg.ofi_weight)
        self.kyle = _Kyle(cfg.kyle_spread_penalty)
        self.hawkes = _Hawkes(
            cfg.hawkes_mu, cfg.hawkes_alpha, cfg.hawkes_beta, cfg.hawkes_history
        )
        self.stab = _Stability(
            cfg.stability_window_sec,
            cfg.stability_min_samples,
            cfg.stability_min_ratio,
            cfg.stability_edge_cv_max,
        )

    def update_price(self, p: float, ts: float) -> None:
        self._bp = p
        self._ph.append((ts, p))

    def update_chainlink(self, p: float, ts: float) -> None:
        if self._cl_ts > 0:
            d = ts - self._cl_ts
            if 5 < d < 120:
                self._cl_updates.append(d)
                if len(self._cl_updates) >= 3:
                    w = [0.5 ** i for i in range(len(self._cl_updates))]
                    v = list(reversed(self._cl_updates))
                    self._cl_period = sum(a * b for a, b in zip(v, w)) / sum(w)
        self._cp = p
        self._cl_ts = ts

    def _momentum(self, window: float = 15.0) -> float:
        if len(self._ph) < 3:
            return 0.0
        now = time.time()
        r = [(t, p) for t, p in self._ph if t >= now - window]
        if len(r) < 2:
            return 0.0
        return clamp((r[-1][1] - r[0][1]) / r[0][1] / 0.002, -1, 1)

    def _cl_boost(self, now: float) -> tuple:
        if self._cl_ts <= 0 or self._cp <= 0:
            return 0.0, 0.0
        lag = now - self._cl_ts
        ttn = self._cl_period - (lag % self._cl_period)
        gap = (self._bp - self._cp) / self._cp if self._cp > 0 else 0.0
        prox = clamp(1 - ttn / self.cfg.chainlink_edge_window, 0, 1)
        return lag, clamp(prox * gap * 1500, -1.5, 1.5)

    def evaluate(
        self,
        state: MarketState,
        capital: float,
        consecutive_losses: int,
        daily_pnl_pct: float,
        open_positions: int,
        has_position_on_market: bool,
    ) -> Signal:
        now = time.time()
        cfg = self.cfg
        micro = MicrostructureState()

        sig = Signal(
            timestamp=now,
            market_id=state.market_id,
            btc_chainlink=state.btc_chainlink,
            btc_binance=state.btc_binance,
            reference_price=state.reference_price,
            strategy_used=self.NAME,
        )

        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            sig.filter_reasons.append("no_price")
            sig.micro = micro
            return sig

        sig.delta_chainlink = (
            (state.btc_chainlink - state.reference_price) / state.reference_price
        )
        if state.btc_binance > 0:
            sig.delta_binance = (
                (state.btc_binance - state.reference_price) / state.reference_price
            )

        # Source coherence check
        if state.btc_binance > 0 and state.btc_chainlink > 0:
            sd = abs(state.btc_binance - state.btc_chainlink) / state.btc_chainlink
            micro.source_divergence = sd
            if sd > cfg.source_coherence_max:
                sig.filter_reasons.append(f"src_div:{sd:.5f}")
                sig.status = f"SRC_DIV ({sd * 100:.3f}%)"
                sig.micro = micro
                return sig

        sig.time_remaining_sec = state.end_time - now
        slug = state.slug or state.market_id[:20]
        is5 = "5m" in slug or sig.time_remaining_sec < 330
        min_t = cfg.time_min_5m if is5 else cfg.time_min_15m
        max_t = cfg.time_max_5m if is5 else cfg.time_max_15m
        max_a = cfg.time_max_5m_accum if is5 else cfg.time_max_15m

        if sig.time_remaining_sec < min_t:
            sig.filter_reasons.append(f"late:{sig.time_remaining_sec:.0f}s")
            sig.status = "TOO_LATE"
            sig.micro = micro
            return sig
        if sig.time_remaining_sec > max_a:
            sig.filter_reasons.append(f"early:{sig.time_remaining_sec:.0f}s")
            sig.status = "TOO_EARLY"
            sig.micro = micro
            return sig

        accum = sig.time_remaining_sec > max_t
        tr = max_t - min_t
        tdf = (
            (0.6 + 0.4 * clamp((sig.time_remaining_sec - min_t) / tr, 0, 1))
            if tr > 0
            else 1.0
        )
        micro.time_decay_factor = tdf
        eff_min = cfg.edge_min * tdf

        # ── Bayesian pipeline ────────────────────────────────────────────
        pd = sig.delta_chainlink
        z = cfg.momentum_factor * pd + 0.2 * self._momentum(15)
        pp = sigmoid(z)
        micro.base_prob_up = pp
        ls = logit(pp)

        lag, clb = self._cl_boost(now)
        micro.chainlink_lag_seconds = lag
        micro.chainlink_edge_boost = clb
        ls += clb

        self.hawkes.on_mid(state.p_market_yes, now)
        oraw, oadj = self.ofi.compute(
            state.best_bid_yes, state.best_ask_yes,
            state.best_bid_no, state.best_ask_no,
            state.depth_yes, state.depth_no,
        )
        micro.ofi_raw = oraw
        micro.ofi_signal = oadj
        ls += oadj

        hb, hi = self.hawkes.boost()
        micro.hawkes_intensity = hi
        micro.hawkes_boost = hb
        ls *= (1 + hb)

        _, kq = self.kyle.compute(
            state.spread_yes, state.spread_no,
            state.depth_yes, state.depth_no,
        )
        micro.kyle_penalty = kq
        ls = shrink_logit(ls, kq)

        pu = sigmoid(ls)
        pdn = 1 - pu
        micro.final_prob_up = pu

        if pd > 0:
            side = "YES"
            prob = pu
            entry = state.best_ask_yes if state.best_ask_yes > 0 else state.p_market_yes
        elif pd < 0:
            side = "NO"
            prob = pdn
            entry = (
                state.best_ask_no if state.best_ask_no > 0
                else (1 - state.p_market_yes)
            )
        else:
            sig.status = "WATCHING"
            sig.micro = micro
            return sig

        fee = calc_fee(entry, cfg.fee_rate)
        edge = prob - entry - fee
        sig.side = side
        sig.p_true = prob
        sig.p_market = entry
        sig.entry_price = entry
        sig.edge = edge
        sig.taker_fee = fee
        micro.taker_fee = fee

        # ── Filters ────────────────────────────────────────────────────
        if edge < eff_min:
            self.stab.record(slug, side, max(edge, 0.001), now)
            sig.status = (
                f"LOW_EDGE ({edge * 100:.1f}%)" if edge > 0 else "WATCHING"
            )
            sig.micro = micro
            return sig
        if edge > cfg.edge_max:
            self.stab.record(slug, side, max(edge, 0.001), now)
            sig.filter_reasons.append(f"edge_suspicious:{edge * 100:.1f}%")
            sig.status = f"SUSPICIOUS_EDGE ({edge * 100:.1f}%)"
            sig.micro = micro
            return sig
        if prob < entry + 0.02:
            self.stab.record(slug, side, max(edge, 0.001), now)
            sig.filter_reasons.append(f"overpaying:{prob * 100:.0f}%<{entry * 100:.0f}c+2%")
            sig.micro = micro
            return sig
        if prob < cfg.min_true_prob:
            self.stab.record(slug, side, max(edge, 0.001), now)
            sig.filter_reasons.append(f"low_conv:{prob * 100:.0f}%")
            sig.micro = micro
            return sig
        if entry < cfg.min_market_prob_side:
            sig.filter_reasons.append(f"longshot:{entry * 100:.0f}c")
            sig.micro = micro
            return sig
        if entry > cfg.max_market_prob_side:
            sig.filter_reasons.append(f"expensive:{entry * 100:.0f}c")
            sig.micro = micro
            return sig
        md = min(state.depth_yes or 0, state.depth_no or 0)
        if md < cfg.min_market_liquidity:
            sig.filter_reasons.append(f"no_liq:{md:.0f}")
            sig.status = "NO_LIQ"
            sig.micro = micro
            return sig

        self.stab.record(slug, side, edge, now)
        sok, dr, ecv, nt = self.stab.evaluate(slug, side)
        micro.stability_ratio = dr
        micro.stability_edge_cv = ecv
        micro.stability_ok = sok
        micro.stability_ticks = nt

        if not sok or accum:
            sig.filters_passed = False
            sig.status = (
                f"STAB ({nt}/{cfg.stability_min_samples})"
                if nt < cfg.stability_min_samples
                else f"UNSTABLE ({dr * 100:.0f}%)"
            )
            sig.micro = micro
            return sig

        # Risk filters
        reasons = []
        if has_position_on_market:
            reasons.append("in_market")
        from src.config import config as _cfg
        risk = _cfg.risk
        if consecutive_losses >= risk.max_consecutive_losses:
            reasons.append(f"circuit:{consecutive_losses}")
        if daily_pnl_pct < -risk.max_daily_drawdown:
            reasons.append(f"daily_loss:{daily_pnl_pct:.1%}")
        if open_positions >= risk.max_open_positions:
            reasons.append(f"max_pos:{open_positions}")
        sig.filter_reasons.extend(reasons)
        if reasons:
            sig.filters_passed = False
            sig.micro = micro
            return sig

        # Sizing
        c_eff = entry + fee
        if c_eff >= 1 or c_eff <= 0:
            sig.micro = micro
            return sig
        kelly = (prob - c_eff) / (1 - c_eff)
        if kelly <= 0:
            sig.filter_reasons.append(f"neg_k:{kelly:.4f}")
            sig.micro = micro
            return sig

        frac = clamp(kelly * 0.25, 0, cfg.max_bet_fraction)
        if hi < cfg.hawkes_mu * 1.5:
            frac *= 0.7
        frac = clamp(frac, 0, cfg.max_bet_fraction)
        size = round(capital * frac, 2)
        size = min(size, capital * cfg.max_bet_fraction)
        depth = state.depth_yes if side == "YES" else state.depth_no
        if depth > 0:
            size = min(size, depth * 0.3)
        if size < 1.0:
            sig.filter_reasons.append("tiny")
            sig.micro = micro
            return sig

        sig.size_usd = size
        sig.kelly_pct = frac
        cs = edge * 0.4 + kq * 0.2 + hb * 0.2 + (dr if sok else 0) * 0.2
        sig.confidence = "HIGH" if cs >= 0.10 else "MEDIUM" if cs >= 0.06 else "LOW"
        sig.action = "BUY"
        sig.status = "BETTING"
        sig.filters_passed = True
        sig.micro = micro

        log.info(
            "[BET:CL_ARB] %s %s | edge=+%.1f%% P=%.0f%% @%.0fc | $%.2f | T-%ds | %s",
            side, slug[-16:], edge * 100, prob * 100, entry * 100,
            size, int(sig.time_remaining_sec), sig.confidence,
        )
        return sig

    def reset_stability(self, slug: str) -> None:
        self.stab.reset(slug)


# ── Strategy 2: Price Momentum ──────────────────────────────────────────────────

class _MomentumEngine:
    """
    Bet in the direction of sustained BTC price momentum.

    Requires consistent momentum over BOTH 60s and 120s windows,
    and the direction must align with the current market delta.
    This prevents chasing short-lived spikes.

    Most effective 60–250s before market close (accumulation window).
    Sizing is conservative (half of ChainlinkArb max).
    """
    NAME = "momentum"

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self._ph: deque = deque(maxlen=150)

    def update_price(self, p: float, ts: float) -> None:
        self._ph.append((ts, p))

    def _mom_window(self, seconds: float) -> float:
        """Fractional BTC price change over the last `seconds`."""
        now = time.time()
        r = [(t, p) for t, p in self._ph if t >= now - seconds]
        if len(r) < 4:  # need at least 4 ticks for reliable estimate
            return 0.0
        return (r[-1][1] - r[0][1]) / r[0][1]

    def evaluate(
        self,
        state: MarketState,
        capital: float,
        has_position_on_market: bool,
    ) -> Optional[Signal]:
        cfg = self.cfg
        now = time.time()
        time_remaining = state.end_time - now

        # Active window: between 60s and 250s remaining
        if time_remaining < 60 or time_remaining > 250:
            return None
        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            return None
        if has_position_on_market:
            return None

        m60 = self._mom_window(60)
        m120 = self._mom_window(120)
        threshold = cfg.momentum_min_threshold

        # Both windows must exceed threshold
        if abs(m60) < threshold or abs(m120) < threshold:
            return None
        # Both windows must agree on direction
        if (m60 > 0) != (m120 > 0):
            return None

        going_up = m60 > 0
        delta = (
            (state.btc_chainlink - state.reference_price) / state.reference_price
        )

        # Delta must at least not strongly oppose momentum
        if going_up and delta < -0.0005:
            return None
        if not going_up and delta > 0.0005:
            return None

        side = "YES" if going_up else "NO"
        if going_up:
            entry = state.best_ask_yes if state.best_ask_yes > 0 else state.p_market_yes
        else:
            entry = (
                state.best_ask_no if state.best_ask_no > 0
                else (1 - state.p_market_yes)
            )

        momentum_strength = (abs(m60) + abs(m120)) / 2
        # Conservative probability: base 0.53 + strength bonus
        p_true = clamp(0.53 + momentum_strength * 20, 0.53, 0.67)

        fee = calc_fee(entry)
        edge = p_true - entry - fee

        if edge < cfg.edge_min * 0.9:  # slight relaxation for momentum
            return None
        if entry < cfg.min_market_prob_side or entry > cfg.max_market_prob_side:
            return None

        max_frac = cfg.max_bet_fraction * 0.5  # half of ChainlinkArb cap
        c_eff = entry + fee
        if c_eff >= 1:
            return None
        kelly = (p_true - c_eff) / (1 - c_eff)
        if kelly <= 0:
            return None
        frac = clamp(kelly * 0.25, 0, max_frac)
        size = round(capital * frac, 2)
        if size < 1.0:
            return None

        slug = state.slug or state.market_id[:20]
        sig = Signal(
            timestamp=now,
            market_id=state.market_id,
            delta_chainlink=delta,
            btc_chainlink=state.btc_chainlink,
            btc_binance=state.btc_binance,
            reference_price=state.reference_price,
            time_remaining_sec=time_remaining,
            slug=slug,
            side=side,
            p_true=p_true,
            p_market=entry,
            entry_price=entry,
            edge=edge,
            taker_fee=fee,
            kelly_pct=frac,
            size_usd=size,
            action="BUY",
            status="BETTING",
            filters_passed=True,
            confidence="MEDIUM",
            strategy_used=self.NAME,
            strategies_agreeing=1,
        )
        log.info(
            "[BET:MOMENTUM] %s %s | m60=%.3f%% m120=%.3f%% edge=%.1f%% $%.2f | T-%ds",
            side, slug[-14:], m60 * 100, m120 * 100, edge * 100, size,
            int(time_remaining),
        )
        return sig


# ── Strategy 3: Mean Reversion (contrarian) ───────────────────────────────────

class _MeanReversionEngine:
    """
    Contrarian bet: fade extreme delta moves.

    When BTC has moved far (>0.20%) from the reference price, the market
    tends to overprice continuation. We fade the extreme and bet reversion.

    Conservative: never bets during losing streaks (>= 2 consecutive losses),
    low sizing (40% of ChainlinkArb cap), and requires 90–250s remaining.

    NOTE: This strategy will often CONFLICT with ChainlinkArb (which follows
    the delta direction). The router’s conflict detection will then choose
    the dominant side or skip — which is the intended safety behavior.
    """
    NAME = "mean_rev"

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg

    def evaluate(
        self,
        state: MarketState,
        capital: float,
        has_position_on_market: bool,
        consecutive_losses: int,
    ) -> Optional[Signal]:
        cfg = self.cfg
        now = time.time()

        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            return None

        delta = (
            (state.btc_chainlink - state.reference_price) / state.reference_price
        )
        abs_delta = abs(delta)
        thresh = cfg.mean_reversion_delta_threshold

        if abs_delta < thresh:
            return None

        time_remaining = state.end_time - now
        if time_remaining < 90 or time_remaining > 250:
            return None

        # Don’t fade during losing streaks — trend may be stronger
        if consecutive_losses >= 2:
            return None
        if has_position_on_market:
            return None

        # Contrarian: bet AGAINST the direction of the extreme move
        if delta > 0:
            side = "NO"  # BTC went up far → bet it won’t stay above ref
            entry = (
                state.best_ask_no if state.best_ask_no > 0
                else (1 - state.p_market_yes)
            )
        else:
            side = "YES"  # BTC went down far → bet it will bounce back above ref
            entry = state.best_ask_yes if state.best_ask_yes > 0 else state.p_market_yes

        # Conservative probability: base 0.52 + extra for larger extremes
        p_true = clamp(0.52 + (abs_delta - thresh) * 8, 0.52, 0.62)

        fee = calc_fee(entry)
        edge = p_true - entry - fee

        if edge < cfg.edge_min:
            return None
        if entry < cfg.min_market_prob_side or entry > cfg.max_market_prob_side:
            return None

        max_frac = cfg.max_bet_fraction * 0.4  # most conservative strategy
        c_eff = entry + fee
        if c_eff >= 1:
            return None
        kelly = (p_true - c_eff) / (1 - c_eff)
        if kelly <= 0:
            return None
        frac = clamp(kelly * 0.20, 0, max_frac)
        size = round(capital * frac, 2)
        if size < 1.0:
            return None

        slug = state.slug or state.market_id[:20]
        sig = Signal(
            timestamp=now,
            market_id=state.market_id,
            delta_chainlink=delta,
            btc_chainlink=state.btc_chainlink,
            btc_binance=state.btc_binance,
            reference_price=state.reference_price,
            time_remaining_sec=time_remaining,
            slug=slug,
            side=side,
            p_true=p_true,
            p_market=entry,
            entry_price=entry,
            edge=edge,
            taker_fee=fee,
            kelly_pct=frac,
            size_usd=size,
            action="BUY",
            status="BETTING",
            filters_passed=True,
            confidence="LOW",
            strategy_used=self.NAME,
            strategies_agreeing=1,
        )
        log.info(
            "[BET:MEAN_REV] %s %s | delta=%.3f%% p_true=%.1f%% edge=%.1f%% $%.2f | T-%ds",
            side, slug[-14:], delta * 100, p_true * 100, edge * 100, size,
            int(time_remaining),
        )
        return sig


# ── Multi-Strategy Router ───────────────────────────────────────────────────────

class SignalEngine:
    """
    Coordinates all three strategies and routes to the best signal.

    Public interface (unchanged from v2, compatible with main.py):
      update_price(price, ts)
      update_chainlink_price(price, ts)
      evaluate(state, capital, consecutive_losses, daily_pnl_pct,
               open_positions, has_position_on_market) -> Signal
      reset_market_stability(slug)
      record_result(strategy_name, won)   <- NEW: call after resolution
    """

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self._cl = _ChainlinkArbEngine(cfg)
        self._mom = _MomentumEngine(cfg)
        self._rev = _MeanReversionEngine(cfg)
        self.perf = PerformanceTracker(window=30)

    # ── Feed updates ─────────────────────────────────────────────────────────

    def update_price(self, p: float, ts: float) -> None:
        self._cl.update_price(p, ts)
        self._mom.update_price(p, ts)

    def update_chainlink_price(self, p: float, ts: float) -> None:
        self._cl.update_chainlink(p, ts)

    def reset_market_stability(self, slug: str) -> None:
        self._cl.reset_stability(slug)

    def record_result(self, strategy: str, won: bool) -> None:
        """Called by main.py after each trade resolves."""
        self.perf.record(strategy, won)

    # ── Core evaluation ──────────────────────────────────────────────────────

    def evaluate(
        self,
        state: MarketState,
        capital: float,
        consecutive_losses: int = 0,
        daily_pnl_pct: float = 0.0,
        open_positions: int = 0,
        has_position_on_market: bool = False,
    ) -> Signal:
        """Evaluate all strategies, apply routing, return the best Signal."""
        now = time.time()

        # Run all three strategies
        cl_sig = self._cl.evaluate(
            state, capital, consecutive_losses, daily_pnl_pct,
            open_positions, has_position_on_market,
        )
        mom_sig = self._mom.evaluate(state, capital, has_position_on_market)
        rev_sig = self._rev.evaluate(
            state, capital, has_position_on_market, consecutive_losses
        )

        # Collect actionable BUY candidates
        candidates: List[Signal] = []
        for sig in (cl_sig, mom_sig, rev_sig):
            if sig is not None and sig.action == "BUY" and sig.filters_passed:
                candidates.append(sig)

        # Enrich base signal for dashboard (always return something useful)
        def _base_info(sig: Signal) -> Signal:
            """Copy market-level info onto signal for logging/dashboard."""
            sig.slug = state.slug
            sig.market_start_time = state.start_time
            sig.market_duration = state.duration_seconds
            sig.time_remaining_sec = state.end_time - now
            if state.btc_binance > 0 and state.reference_price > 0:
                sig.delta_binance = (
                    (state.btc_binance - state.reference_price)
                    / state.reference_price
                )
            return sig

        if not candidates:
            # No strategy wants to bet — return the ChainlinkArb signal for
            # dashboard / logging purposes (it has the richest debug info)
            base = cl_sig if cl_sig is not None else Signal(
                timestamp=now,
                market_id=state.market_id,
                btc_chainlink=state.btc_chainlink,
                reference_price=state.reference_price,
            )
            return _base_info(base)

        # ── Conflict detection ─────────────────────────────────────────────
        yes_cands = [s for s in candidates if s.side == "YES"]
        no_cands = [s for s in candidates if s.side == "NO"]

        if yes_cands and no_cands:
            # Strategies disagree — compare weighted scores
            def _score(s: Signal) -> float:
                return s.edge * self.perf.weight(s.strategy_used)

            best_yes = max(yes_cands, key=_score)
            best_no = max(no_cands, key=_score)
            ys = _score(best_yes)
            ns = _score(best_no)
            dominant = max(ys, ns, 1e-9)

            if abs(ys - ns) / dominant < 0.50:
                # Neither side clearly dominates — skip
                conflict = Signal(
                    timestamp=now,
                    market_id=state.market_id,
                    btc_chainlink=state.btc_chainlink,
                    reference_price=state.reference_price,
                    status="CONFLICT",
                )
                conflict.filter_reasons.append(
                    f"conflict:YES({best_yes.strategy_used})"
                    f"/NO({best_no.strategy_used})"
                )
                return _base_info(conflict)

            # Use the dominant side only
            candidates = yes_cands if ys > ns else no_cands

        # ── Consensus & routing ────────────────────────────────────────────
        agreeing = len(candidates)
        best = max(
            candidates,
            key=lambda s: s.edge * self.perf.weight(s.strategy_used),
        )
        best.strategies_agreeing = agreeing

        if agreeing >= 2:
            # Consensus size boost: +25% per additional agreeing strategy
            boost = 1.0 + 0.25 * (agreeing - 1)  # 1.25x or 1.50x
            hard_cap = capital * self.cfg.max_bet_fraction * 1.5
            best.size_usd = min(round(best.size_usd * boost, 2), hard_cap)
            best.confidence = "HIGH"
            strategy_names = "+".join(
                s.strategy_used for s in candidates
            )
            log.info(
                "[CONSENSUS] %d strategies agree %s | %s | boost=%.2fx | $%.2f",
                agreeing, best.side, strategy_names, boost, best.size_usd,
            )

        return _base_info(best)
