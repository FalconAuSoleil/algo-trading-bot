"""Signal Engine v2 - Market Microstructure Brain
================================================

4 components combined into a bayesian score in logit space.
Edge is computed as: P(our_side) - entry_price - fees.
We ONLY bet when our model AND the market roughly agree on direction.
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


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 500)))
    e = math.exp(max(x, -500))
    return e / (1.0 + e)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1.0 - p))

def shrink_logit(logit_val: float, factor: float) -> float:
    return logit_val * clamp(factor, 0.0, 1.0)

def calc_taker_fee(p_market: float, fee_rate: float = 0.25) -> float:
    if p_market <= 0 or p_market >= 1:
        return 0.0
    return fee_rate * (p_market * (1.0 - p_market)) ** 2


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


class ChainlinkArbModule:
    def __init__(self, cfg: SignalConfig):
        self._last_ts: float = 0.0
        self._updates: deque = deque(maxlen=20)
        self._period: float = cfg.chainlink_period
        self._window: float = cfg.chainlink_edge_window

    def on_chainlink_update(self, price: float, ts: float) -> None:
        if self._last_ts > 0:
            delta = ts - self._last_ts
            if 5.0 < delta < 120.0:
                self._updates.append(delta)
                if len(self._updates) >= 3:
                    w = [0.5**i for i in range(len(self._updates))]
                    tw = sum(w)
                    v = list(reversed(self._updates))
                    self._period = sum(a*b for a, b in zip(v, w)) / tw
        self._last_ts = ts

    def compute(self, binance_price: float, chainlink_price: float, now: float) -> tuple[float, float]:
        if self._last_ts <= 0 or chainlink_price <= 0:
            return 0.0, 0.0
        lag = now - self._last_ts
        ttn = self._period - (lag % self._period)
        gap = (binance_price - chainlink_price) / chainlink_price
        prox = clamp(1.0 - ttn / self._window, 0.0, 1.0)
        boost = clamp(prox * gap * 1500.0, -1.5, 1.5)
        return lag, boost


class OFIModule:
    def __init__(self, cfg: SignalConfig):
        self._hist: deque = deque(maxlen=30)
        self._w: float = cfg.ofi_weight

    def compute(self, bid_up, ask_up, bid_down, ask_down, depth_up, depth_down):
        tu = bid_up + ask_up
        td = bid_down + ask_down
        if tu <= 0:
            return 0.0, 0.0
        ofi_u = (bid_up - ask_up) / max(tu, 1e-6)
        ofi_d = (bid_down - ask_down) / max(td + 1e-6, 1e-6) if td > 0 else 0.0
        ofi_net = 0.5 * ofi_u - 0.5 * ofi_d
        tot = depth_up + depth_down
        di = (depth_up - depth_down) / tot if tot > 0 else 0.0
        combined = 0.6 * ofi_net + 0.4 * di
        self._hist.append((time.time(), combined))
        mom = 0.0
        if len(self._hist) >= 5:
            r = list(self._hist)[-10:]
            mom = r[-1][1] - r[0][1]
        sig = combined * self._w + mom * 0.1
        return combined, clamp(sig * 2.0, -0.8, 0.8)


class KyleModule:
    def __init__(self, cfg: SignalConfig):
        self._sh: deque = deque(maxlen=50)
        self._dh: deque = deque(maxlen=50)
        self._pen: float = cfg.kyle_spread_penalty

    def compute(self, spread_up, spread_down, depth_up, depth_down):
        avs = (spread_up + (spread_down or spread_up)) / 2.0
        avd = (depth_up + (depth_down or depth_up)) / 2.0
        self._sh.append(avs)
        self._dh.append(avd)
        if avd <= 0:
            return 0.0, 1.0
        kl = avs / (2.0 * math.sqrt(max(avd, 1.0)))
        if len(self._sh) >= 5:
            hs = sum(self._sh) / len(self._sh)
            hd = sum(self._dh) / len(self._dh)
            rs = avs / max(hs, 1e-6)
            rd = avd / max(hd, 1e-6)
        else:
            rs, rd = 1.0, 1.0
        sp = clamp((rs - 1.0) * self._pen, 0.0, self._pen)
        db = clamp((rd - 1.0) * 0.05, 0.0, 0.10)
        return kl, clamp(1.0 - sp + db, 0.3, 1.0)


class HawkesModule:
    def __init__(self, cfg: SignalConfig):
        self._ev: deque = deque(maxlen=cfg.hawkes_history)
        self._last_mid: float = 0.5
        self._mu = cfg.hawkes_mu
        self._a = cfg.hawkes_alpha
        self._b = cfg.hawkes_beta

    def on_mid_update(self, mid_up: float, ts: float) -> None:
        if self._last_mid > 0:
            ch = abs(mid_up - self._last_mid)
            if ch >= 0.005:
                self._ev.append((ts, min(ch / 0.005, 5.0)))
        self._last_mid = mid_up

    def regime_boost(self, t=None):
        t = t or time.time()
        lam = self._mu
        for ts, m in self._ev:
            dt = t - ts
            if dt >= 0:
                lam += self._a * m * math.exp(-self._b * dt)
        excess = max(0.0, lam - self._mu)
        return clamp(excess / (5.0 * self._a) * 0.3, 0.0, 0.3), lam


class StabilityFilter:
    def __init__(self, cfg: SignalConfig):
        self._h: dict[str, deque] = {}
        self._ws = cfg.stability_window_sec
        self._ms = cfg.stability_min_samples
        self._mr = cfg.stability_min_ratio
        self._mc = cfg.stability_edge_cv_max

    def _buf(self, s):
        if s not in self._h:
            self._h[s] = deque(maxlen=100)
        return self._h[s]

    def record(self, slug, side, edge, ts):
        self._buf(slug).append((ts, side, abs(edge)))

    def evaluate(self, slug, side):
        b = self._buf(slug)
        if not b:
            return False, 0.0, 999.0, 0
        ticks = list(b)
        n = len(ticks)
        if n < self._ms:
            return False, 0.0, 999.0, n
        dr = sum(1 for _, s, _ in ticks if s == side) / n
        now = time.time()
        recent = [e for ts, _, e in ticks if ts >= now - self._ws] or [e for _, _, e in ticks]
        me = sum(recent) / len(recent)
        if me < 1e-6:
            return False, dr, 999.0, n
        cv = math.sqrt(sum((e - me)**2 for e in recent) / len(recent)) / me
        return (dr >= self._mr and cv <= self._mc), dr, cv, n

    def reset_market(self, slug):
        self._h.pop(slug, None)


class SignalEngine:
    """Microstructure bayesian signal engine.

    KEY DESIGN PRINCIPLE:
    We only bet when our model's direction AGREES with where the
    market is leaning. If the market prices UP at 80% and our model
    says DOWN, we DO NOT bet -- the market has more information.

    We look for situations where the market is right about direction
    but UNDERPRICES the magnitude. E.g., market says 55% UP but we
    think 65% UP. That's a 10% edge on the YES side at ~55c entry.
    """

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self._ph: deque = deque(maxlen=120)
        self.cl_arb = ChainlinkArbModule(cfg)
        self.ofi = OFIModule(cfg)
        self.kyle = KyleModule(cfg)
        self.hawkes = HawkesModule(cfg)
        self.stability = StabilityFilter(cfg)
        self._bp: float = 0.0
        self._cp: float = 0.0

    def update_price(self, price, ts):
        self._bp = price
        self._ph.append((ts, price))

    def update_chainlink_price(self, price, ts):
        self._cp = price
        self.cl_arb.on_chainlink_update(price, ts)

    def _momentum(self, win=15.0):
        if len(self._ph) < 3:
            return 0.0
        now = time.time()
        r = [(t, p) for t, p in self._ph if t >= now - win]
        if len(r) < 2:
            return 0.0
        return clamp((r[-1][1] - r[0][1]) / r[0][1] / 0.002, -1.0, 1.0)

    def evaluate(self, state: MarketState, capital: float,
                 consecutive_losses=0, daily_pnl_pct=0.0,
                 open_positions=0, has_position_on_market=False) -> Signal:
        now = time.time()
        cfg = self.cfg

        sig = Signal(timestamp=now, market_id=state.market_id,
                     btc_chainlink=state.btc_chainlink,
                     btc_binance=state.btc_binance,
                     reference_price=state.reference_price)
        micro = sig.micro

        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            sig.filter_reasons.append("missing_price_data")
            return sig

        sig.delta_chainlink = (state.btc_chainlink - state.reference_price) / state.reference_price
        if state.btc_binance > 0:
            sig.delta_binance = (state.btc_binance - state.reference_price) / state.reference_price

        # Source coherence
        if state.btc_binance > 0 and state.btc_chainlink > 0:
            sd = abs(state.btc_binance - state.btc_chainlink) / state.btc_chainlink
            micro.source_divergence = sd
            if sd > cfg.source_coherence_max:
                sig.filter_reasons.append(f"source_divergence:{sd:.5f}")
                sig.status = f"SOURCE_DIV ({sd*100:.3f}%)"
                return sig

        sig.time_remaining_sec = state.end_time - now
        slug = state.slug or state.market_id[:20]
        is5 = "5m" in slug or sig.time_remaining_sec < 330

        min_t = cfg.time_min_5m if is5 else cfg.time_min_15m
        max_t = cfg.time_max_5m if is5 else cfg.time_max_15m
        max_a = cfg.time_max_5m_accum if is5 else cfg.time_max_15m

        if sig.time_remaining_sec < min_t:
            sig.filter_reasons.append(f"too_late:{sig.time_remaining_sec:.0f}s")
            sig.status = f"TOO_LATE ({sig.time_remaining_sec:.0f}s)"
            return sig
        if sig.time_remaining_sec > max_a:
            sig.filter_reasons.append(f"too_early:{sig.time_remaining_sec:.0f}s")
            sig.status = f"TOO_EARLY ({sig.time_remaining_sec:.0f}s)"
            return sig

        accum_only = sig.time_remaining_sec > max_t

        # Time decay
        tr = max_t - min_t
        tdf = (0.6 + 0.4 * clamp((sig.time_remaining_sec - min_t) / tr, 0, 1)) if tr > 0 else 1.0
        micro.time_decay_factor = tdf
        eff_edge_min = cfg.edge_min * tdf

        # ======== BAYESIAN PIPELINE ========

        pd = sig.delta_chainlink
        mom = self._momentum(15.0)
        z = cfg.momentum_factor * pd + 0.2 * mom
        pp = sigmoid(z)
        micro.base_prob_up = pp
        ls = logit(pp)

        bp = self._bp if self._bp > 0 else state.btc_chainlink
        cp = self._cp if self._cp > 0 else state.reference_price
        lag, cl_boost = self.cl_arb.compute(bp, cp, now)
        micro.chainlink_lag_seconds = lag
        micro.chainlink_edge_boost = cl_boost
        ls += cl_boost

        self.hawkes.on_mid_update(state.p_market_yes, now)
        ofi_raw, ofi_adj = self.ofi.compute(
            state.best_bid_yes, state.best_ask_yes,
            state.best_bid_no, state.best_ask_no,
            state.depth_yes, state.depth_no)
        micro.ofi_raw = ofi_raw
        micro.ofi_signal = ofi_adj
        ls += ofi_adj

        hb, hi = self.hawkes.regime_boost()
        micro.hawkes_intensity = hi
        micro.hawkes_boost = hb
        ls *= (1.0 + hb)

        kl, kq = self.kyle.compute(
            state.spread_yes, state.spread_no,
            state.depth_yes, state.depth_no)
        micro.kyle_lambda = kl
        micro.kyle_penalty = kq
        ls = shrink_logit(ls, kq)

        # ======== DECISION ========

        prob_up = sigmoid(ls)
        prob_down = 1.0 - prob_up
        micro.final_prob_up = prob_up

        mkt_up = state.p_market_yes
        mkt_down = 1.0 - mkt_up

        # CRITICAL: Determine side based on DELTA DIRECTION.
        # If BTC is above ref -> we lean YES. If below -> NO.
        # This ensures we bet WITH the price move, not against it.
        if sig.delta_chainlink > 0:
            our_side = "YES"
            prob_ours = prob_up
            entry = state.best_ask_yes if state.best_ask_yes > 0 else mkt_up
        elif sig.delta_chainlink < 0:
            our_side = "NO"
            prob_ours = prob_down
            entry = state.best_ask_no if state.best_ask_no > 0 else mkt_down
        else:
            sig.filter_reasons.append("delta_zero")
            sig.status = "WATCHING"
            return sig

        # EDGE = how much our model exceeds the entry cost
        # This is the REAL edge: what we pay vs what we think it's worth
        fee = calc_taker_fee(entry, cfg.fee_rate)
        edge = prob_ours - entry - fee

        sig.side = our_side
        sig.p_true = prob_ours
        sig.p_market = entry  # what we actually pay
        sig.entry_price = entry
        sig.edge = edge
        sig.taker_fee = fee
        micro.taker_fee = fee

        # ======== FILTERS ========

        candidate = our_side

        # Edge too low
        if edge < eff_edge_min:
            candidate = ""
            if edge > 0:
                sig.status = f"LOW_EDGE ({edge*100:.1f}% < {eff_edge_min*100:.1f}%)"
            else:
                sig.status = "WATCHING"

        # Conviction: our prob must be > threshold
        if candidate and prob_ours < cfg.min_true_prob:
            candidate = ""
            sig.filter_reasons.append(f"low_conviction:{prob_ours*100:.0f}%")

        # Entry price guard: don't buy longshots (< 30c) or near-certainties (> 75c)
        if candidate:
            if entry < cfg.min_market_prob_side:
                candidate = ""
                sig.filter_reasons.append(f"longshot_entry:{entry*100:.0f}c")
            elif entry > cfg.max_market_prob_side:
                candidate = ""
                sig.filter_reasons.append(f"expensive_entry:{entry*100:.0f}c")

        # Model vs Market disagreement guard
        # If our model says 60% but entry is 80%, we're fighting the market
        if candidate and prob_ours < entry:
            candidate = ""
            sig.filter_reasons.append(f"model_below_market:{prob_ours*100:.0f}%<{entry*100:.0f}c")

        # Liquidity
        min_depth = min(state.depth_yes or 0, state.depth_no or 0)
        if candidate and min_depth < cfg.min_market_liquidity:
            candidate = ""
            sig.filter_reasons.append(f"no_liquidity:{min_depth:.0f}")
            sig.status = f"NO_LIQ (${min_depth:.0f})"

        if not candidate:
            sig.side = ""
            # Still accumulate stability data
            ds = "YES" if pd > 0 else "NO"
            self.stability.record(slug, ds, abs(edge) if edge > 0 else abs(pd)*100, now)
            return sig

        # Stability
        self.stability.record(slug, candidate, edge, now)
        sok, dr, ecv, nt = self.stability.evaluate(slug, candidate)
        micro.stability_ratio = dr
        micro.stability_edge_cv = ecv
        micro.stability_ok = sok
        micro.stability_ticks = nt

        should_bet = sok and not accum_only

        # Risk filters
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
            if not sok:
                sig.status = (f"STABILIZING ({nt}/{cfg.stability_min_samples})"
                             if nt < cfg.stability_min_samples
                             else f"UNSTABLE ({dr*100:.0f}% {candidate})")
            return sig

        # ======== KELLY SIZING ========

        c_eff = entry + fee
        if c_eff >= 1.0 or c_eff <= 0:
            sig.filter_reasons.append("cost_exceeds_payout")
            sig.filters_passed = False
            return sig

        kelly = (prob_ours - c_eff) / (1.0 - c_eff)
        if kelly <= 0:
            sig.filter_reasons.append(f"neg_kelly:{kelly:.4f}")
            sig.filters_passed = False
            return sig

        frac = clamp(kelly * 0.25, 0.0, cfg.max_bet_fraction)
        if hi < cfg.hawkes_mu * 1.5:
            frac *= 0.7
        sb = clamp((dr - cfg.stability_min_ratio) * 2.0, 0.0, 0.3)
        frac = clamp(frac * (1.0 + sb), 0.0, cfg.max_bet_fraction)

        sig.size_usd = round(capital * frac, 2)
        sig.kelly_pct = frac

        # Hard cap
        sig.size_usd = min(sig.size_usd, capital * cfg.max_bet_fraction)

        # Depth limit
        depth = state.depth_yes if our_side == "YES" else state.depth_no
        if depth > 0:
            sig.size_usd = min(sig.size_usd, depth * 0.3)

        if sig.size_usd < 1.0:
            sig.filter_reasons.append("size_too_small")
            sig.filters_passed = False
            return sig

        cs = edge * 0.4 + kq * 0.2 + hb * 0.2 + (dr if sok else 0) * 0.2
        sig.confidence = "HIGH" if cs >= 0.12 else "MEDIUM" if cs >= 0.07 else "LOW"

        sig.action = "BUY"
        sig.status = "BETTING"
        sig.filters_passed = True

        log.info(
            "[BET] %s %s | edge=+%.1f%% P=%.0f%% entry=%.0f%% | "
            "$%.2f | T-%ds | d=%.3f%% | %s",
            sig.side, slug[-16:],
            edge * 100, prob_ours * 100, entry * 100,
            sig.size_usd, int(sig.time_remaining_sec),
            sig.delta_chainlink * 100, sig.confidence)

        return sig

    def reset_market_stability(self, slug):
        self.stability.reset_market(slug)
