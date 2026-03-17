"""Signal Engine v2 - Market Microstructure Brain
================================================

We ONLY bet when:
1. Price delta and our model agree on direction
2. Edge (P_ours - entry - fee) is between 6% and 15%
3. Entry price is 35c-70c (fair odds zone)
4. Our probability > entry price (we're not overpaying)
5. Signal is stable over multiple ticks
6. Risk limits are not breached
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


def sigmoid(x):
    if x >= 0: return 1.0 / (1.0 + math.exp(-min(x, 500)))
    e = math.exp(max(x, -500)); return e / (1.0 + e)

def clamp(x, lo, hi): return max(lo, min(hi, x))

def logit(p):
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1.0 - p))

def shrink_logit(v, f): return v * clamp(f, 0.0, 1.0)

def calc_fee(p, rate=0.25):
    if p <= 0 or p >= 1: return 0.0
    return rate * (p * (1.0 - p)) ** 2


@dataclass
class MarketState:
    market_id: str = ""; reference_price: float = 0.0
    end_time: float = 0.0; btc_chainlink: float = 0.0
    btc_binance: float = 0.0; p_market_yes: float = 0.5
    depth_yes: float = 0.0; depth_no: float = 0.0
    best_bid_yes: float = 0.0; best_ask_yes: float = 0.0
    best_bid_no: float = 0.0; best_ask_no: float = 0.0
    spread_yes: float = 0.01; spread_no: float = 0.01
    slug: str = ""; start_time: float = 0.0
    duration_seconds: int = 300

@dataclass
class MicrostructureState:
    chainlink_lag_seconds: float = 0.0; chainlink_edge_boost: float = 0.0
    ofi_raw: float = 0.0; ofi_signal: float = 0.0
    kyle_lambda: float = 0.0; kyle_penalty: float = 0.0
    hawkes_intensity: float = 0.0; hawkes_boost: float = 0.0
    base_prob_up: float = 0.5; final_prob_up: float = 0.5
    stability_ratio: float = 0.0; stability_edge_cv: float = 0.0
    stability_ok: bool = False; stability_ticks: int = 0
    taker_fee: float = 0.0; source_divergence: float = 0.0
    time_decay_factor: float = 1.0
    components: dict = field(default_factory=dict)

@dataclass
class Signal:
    timestamp: float = 0.0; market_id: str = ""
    delta_chainlink: float = 0.0; delta_binance: float = 0.0
    sigma: float = 0.0; time_remaining_sec: float = 0.0
    p_true: float = 0.5; p_market: float = 0.5
    edge: float = 0.0; taker_fee: float = 0.0; kelly_pct: float = 0.0
    side: str = ""; action: str = "NO_TRADE"
    entry_price: float = 0.0; size_usd: float = 0.0
    filters_passed: bool = False
    filter_reasons: list[str] = field(default_factory=list)
    btc_chainlink: float = 0.0; btc_binance: float = 0.0
    reference_price: float = 0.0; slug: str = ""
    market_start_time: float = 0.0; market_duration: int = 300
    micro: MicrostructureState = field(default_factory=MicrostructureState)
    confidence: str = "LOW"; status: str = "WATCHING"


class ChainlinkArbModule:
    def __init__(self, cfg):
        self._ts = 0.0; self._ups = deque(maxlen=20)
        self._per = cfg.chainlink_period; self._win = cfg.chainlink_edge_window
    def on_update(self, price, ts):
        if self._ts > 0:
            d = ts - self._ts
            if 5 < d < 120:
                self._ups.append(d)
                if len(self._ups) >= 3:
                    w = [0.5**i for i in range(len(self._ups))]
                    v = list(reversed(self._ups))
                    self._per = sum(a*b for a,b in zip(v,w)) / sum(w)
        self._ts = ts
    def compute(self, bp, cp, now):
        if self._ts <= 0 or cp <= 0: return 0.0, 0.0
        lag = now - self._ts
        ttn = self._per - (lag % self._per)
        gap = (bp - cp) / cp
        prox = clamp(1 - ttn / self._win, 0, 1)
        return lag, clamp(prox * gap * 1500, -1.5, 1.5)

class OFIModule:
    def __init__(self, cfg):
        self._h = deque(maxlen=30); self._w = cfg.ofi_weight
    def compute(self, bu, au, bd, ad, du, dd):
        tu = bu + au; td = bd + ad
        if tu <= 0: return 0.0, 0.0
        ou = (bu-au)/max(tu,1e-6)
        od = (bd-ad)/max(td+1e-6,1e-6) if td > 0 else 0.0
        on = 0.5*ou - 0.5*od
        tot = du+dd; di = (du-dd)/tot if tot > 0 else 0.0
        c = 0.6*on + 0.4*di; self._h.append((time.time(),c))
        m = 0.0
        if len(self._h) >= 5:
            r = list(self._h)[-10:]; m = r[-1][1]-r[0][1]
        return c, clamp((c*self._w + m*0.1)*2, -0.8, 0.8)

class KyleModule:
    def __init__(self, cfg):
        self._sh = deque(maxlen=50); self._dh = deque(maxlen=50)
        self._p = cfg.kyle_spread_penalty
    def compute(self, su, sd, du, dd):
        avs = (su+(sd or su))/2; avd = (du+(dd or du))/2
        self._sh.append(avs); self._dh.append(avd)
        if avd <= 0: return 0.0, 1.0
        kl = avs/(2*math.sqrt(max(avd,1)))
        if len(self._sh) >= 5:
            hs = sum(self._sh)/len(self._sh); hd = sum(self._dh)/len(self._dh)
            rs = avs/max(hs,1e-6); rd = avd/max(hd,1e-6)
        else: rs, rd = 1.0, 1.0
        sp = clamp((rs-1)*self._p, 0, self._p)
        db = clamp((rd-1)*0.05, 0, 0.1)
        return kl, clamp(1-sp+db, 0.3, 1.0)

class HawkesModule:
    def __init__(self, cfg):
        self._ev = deque(maxlen=cfg.hawkes_history)
        self._lm = 0.5; self._mu = cfg.hawkes_mu
        self._a = cfg.hawkes_alpha; self._b = cfg.hawkes_beta
    def on_mid(self, mid, ts):
        if self._lm > 0:
            ch = abs(mid - self._lm)
            if ch >= 0.005: self._ev.append((ts, min(ch/0.005, 5)))
        self._lm = mid
    def boost(self, t=None):
        t = t or time.time(); lam = self._mu
        for ts, m in self._ev:
            dt = t - ts
            if dt >= 0: lam += self._a * m * math.exp(-self._b * dt)
        ex = max(0, lam - self._mu)
        return clamp(ex/(5*self._a)*0.3, 0, 0.3), lam

class StabilityFilter:
    def __init__(self, cfg):
        self._h = {}; self._ws = cfg.stability_window_sec
        self._ms = cfg.stability_min_samples
        self._mr = cfg.stability_min_ratio; self._mc = cfg.stability_edge_cv_max
    def _b(self, s):
        if s not in self._h: self._h[s] = deque(maxlen=100)
        return self._h[s]
    def record(self, s, side, edge, ts): self._b(s).append((ts, side, abs(edge)))
    def evaluate(self, s, side):
        b = self._b(s)
        if not b: return False, 0, 999, 0
        t = list(b); n = len(t)
        if n < self._ms: return False, 0, 999, n
        dr = sum(1 for _,sd,_ in t if sd == side)/n
        now = time.time()
        r = [e for ts,_,e in t if ts >= now-self._ws] or [e for _,_,e in t]
        me = sum(r)/len(r)
        if me < 1e-6: return False, dr, 999, n
        cv = math.sqrt(sum((e-me)**2 for e in r)/len(r))/me
        return (dr >= self._mr and cv <= self._mc), dr, cv, n
    def reset(self, s): self._h.pop(s, None)


class SignalEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self._ph = deque(maxlen=120)
        self.cl = ChainlinkArbModule(cfg)
        self.ofi = OFIModule(cfg)
        self.kyle = KyleModule(cfg)
        self.hawkes = HawkesModule(cfg)
        self.stab = StabilityFilter(cfg)
        self._bp = 0.0; self._cp = 0.0

    def update_price(self, p, ts):
        self._bp = p; self._ph.append((ts, p))
    def update_chainlink_price(self, p, ts):
        self._cp = p; self.cl.on_update(p, ts)

    def _mom(self, w=15.0):
        if len(self._ph) < 3: return 0.0
        now = time.time()
        r = [(t,p) for t,p in self._ph if t >= now-w]
        if len(r) < 2: return 0.0
        return clamp((r[-1][1]-r[0][1])/r[0][1]/0.002, -1, 1)

    def evaluate(self, state, capital, consecutive_losses=0,
                 daily_pnl_pct=0.0, open_positions=0,
                 has_position_on_market=False):
        now = time.time(); cfg = self.cfg
        sig = Signal(timestamp=now, market_id=state.market_id,
                     btc_chainlink=state.btc_chainlink,
                     btc_binance=state.btc_binance,
                     reference_price=state.reference_price)
        micro = sig.micro

        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            sig.filter_reasons.append("no_price"); return sig

        sig.delta_chainlink = (state.btc_chainlink - state.reference_price) / state.reference_price
        if state.btc_binance > 0:
            sig.delta_binance = (state.btc_binance - state.reference_price) / state.reference_price

        # Source coherence
        if state.btc_binance > 0 and state.btc_chainlink > 0:
            sd = abs(state.btc_binance - state.btc_chainlink) / state.btc_chainlink
            micro.source_divergence = sd
            if sd > cfg.source_coherence_max:
                sig.filter_reasons.append(f"src_div:{sd:.5f}")
                sig.status = f"SRC_DIV ({sd*100:.3f}%)"; return sig

        sig.time_remaining_sec = state.end_time - now
        slug = state.slug or state.market_id[:20]
        is5 = "5m" in slug or sig.time_remaining_sec < 330
        min_t = cfg.time_min_5m if is5 else cfg.time_min_15m
        max_t = cfg.time_max_5m if is5 else cfg.time_max_15m
        max_a = cfg.time_max_5m_accum if is5 else cfg.time_max_15m

        if sig.time_remaining_sec < min_t:
            sig.filter_reasons.append(f"late:{sig.time_remaining_sec:.0f}s")
            sig.status = f"TOO_LATE"; return sig
        if sig.time_remaining_sec > max_a:
            sig.filter_reasons.append(f"early:{sig.time_remaining_sec:.0f}s")
            sig.status = f"TOO_EARLY"; return sig

        accum = sig.time_remaining_sec > max_t
        tr = max_t - min_t
        tdf = (0.6 + 0.4*clamp((sig.time_remaining_sec-min_t)/tr, 0, 1)) if tr > 0 else 1.0
        micro.time_decay_factor = tdf
        eff_min = cfg.edge_min * tdf

        # ====== BAYESIAN PIPELINE ======
        pd = sig.delta_chainlink
        z = cfg.momentum_factor * pd + 0.2 * self._mom(15)
        pp = sigmoid(z); micro.base_prob_up = pp
        ls = logit(pp)

        bp = self._bp if self._bp > 0 else state.btc_chainlink
        cp = self._cp if self._cp > 0 else state.reference_price
        lag, clb = self.cl.compute(bp, cp, now)
        micro.chainlink_lag_seconds = lag; micro.chainlink_edge_boost = clb
        ls += clb

        self.hawkes.on_mid(state.p_market_yes, now)
        oraw, oadj = self.ofi.compute(
            state.best_bid_yes, state.best_ask_yes,
            state.best_bid_no, state.best_ask_no,
            state.depth_yes, state.depth_no)
        micro.ofi_raw = oraw; micro.ofi_signal = oadj
        ls += oadj

        hb, hi = self.hawkes.boost()
        micro.hawkes_intensity = hi; micro.hawkes_boost = hb
        ls *= (1 + hb)

        kl, kq = self.kyle.compute(
            state.spread_yes, state.spread_no,
            state.depth_yes, state.depth_no)
        micro.kyle_lambda = kl; micro.kyle_penalty = kq
        ls = shrink_logit(ls, kq)

        # ====== DECISION ======
        pu = sigmoid(ls); pdn = 1 - pu
        micro.final_prob_up = pu

        # Side from delta direction
        if pd > 0:
            side = "YES"; prob = pu
            entry = state.best_ask_yes if state.best_ask_yes > 0 else state.p_market_yes
        elif pd < 0:
            side = "NO"; prob = pdn
            entry = state.best_ask_no if state.best_ask_no > 0 else (1 - state.p_market_yes)
        else:
            sig.status = "WATCHING"; return sig

        fee = calc_fee(entry, cfg.fee_rate)
        edge = prob - entry - fee

        sig.side = side; sig.p_true = prob; sig.p_market = entry
        sig.entry_price = entry; sig.edge = edge
        sig.taker_fee = fee; micro.taker_fee = fee

        cand = side

        # ====== FILTERS ======

        # Edge bounds
        if edge < eff_min:
            cand = ""
            sig.status = f"LOW_EDGE ({edge*100:.1f}%)" if edge > 0 else "WATCHING"
        elif edge > cfg.edge_max:
            cand = ""
            sig.filter_reasons.append(f"edge_suspicious:{edge*100:.1f}%")
            sig.status = f"SUSPICIOUS_EDGE ({edge*100:.1f}%)"

        # Prob vs entry: never pay more than we think it's worth
        if cand and prob < entry + 0.02:
            cand = ""
            sig.filter_reasons.append(f"overpaying:{prob*100:.0f}%<{entry*100:.0f}c+2%")

        # Conviction
        if cand and prob < cfg.min_true_prob:
            cand = ""
            sig.filter_reasons.append(f"low_conv:{prob*100:.0f}%")

        # Entry price range
        if cand and entry < cfg.min_market_prob_side:
            cand = ""
            sig.filter_reasons.append(f"longshot:{entry*100:.0f}c")
        if cand and entry > cfg.max_market_prob_side:
            cand = ""
            sig.filter_reasons.append(f"expensive:{entry*100:.0f}c")

        # Liquidity
        md = min(state.depth_yes or 0, state.depth_no or 0)
        if cand and md < cfg.min_market_liquidity:
            cand = ""
            sig.filter_reasons.append(f"no_liq:{md:.0f}")
            sig.status = f"NO_LIQ"

        if not cand:
            sig.side = ""
            ds = "YES" if pd > 0 else "NO"
            self.stab.record(slug, ds, max(edge, 0.001), now)
            return sig

        # Stability
        self.stab.record(slug, cand, edge, now)
        sok, dr, ecv, nt = self.stab.evaluate(slug, cand)
        micro.stability_ratio = dr; micro.stability_edge_cv = ecv
        micro.stability_ok = sok; micro.stability_ticks = nt

        if not sok or accum:
            sig.filters_passed = False
            sig.status = (f"STAB ({nt}/{cfg.stability_min_samples})" if nt < cfg.stability_min_samples
                         else f"UNSTABLE ({dr*100:.0f}%)")
            return sig

        # Risk filters
        reasons = []
        if has_position_on_market: reasons.append("in_market")
        from src.config import config
        risk = config.risk
        if consecutive_losses >= risk.max_consecutive_losses:
            reasons.append(f"circuit:{consecutive_losses}")
        if daily_pnl_pct < -risk.max_daily_drawdown:
            reasons.append(f"daily_loss:{daily_pnl_pct:.1%}")
        if open_positions >= risk.max_open_positions:
            reasons.append(f"max_pos:{open_positions}")
        sig.filter_reasons.extend(reasons)
        if reasons:
            sig.filters_passed = False; return sig

        # ====== SIZING ======
        c_eff = entry + fee
        if c_eff >= 1 or c_eff <= 0:
            sig.filters_passed = False; return sig

        kelly = (prob - c_eff) / (1 - c_eff)
        if kelly <= 0:
            sig.filter_reasons.append(f"neg_k:{kelly:.4f}")
            sig.filters_passed = False; return sig

        frac = clamp(kelly * 0.25, 0, cfg.max_bet_fraction)
        if hi < cfg.hawkes_mu * 1.5: frac *= 0.7
        frac = clamp(frac, 0, cfg.max_bet_fraction)

        size = round(capital * frac, 2)
        # ABSOLUTE HARD CAP
        hard_cap = capital * cfg.max_bet_fraction
        size = min(size, hard_cap)

        # Depth limit
        depth = state.depth_yes if side == "YES" else state.depth_no
        if depth > 0: size = min(size, depth * 0.3)

        if size < 1:
            sig.filter_reasons.append("tiny"); sig.filters_passed = False; return sig

        sig.size_usd = size; sig.kelly_pct = frac
        cs = edge*0.4 + kq*0.2 + hb*0.2 + (dr if sok else 0)*0.2
        sig.confidence = "HIGH" if cs >= 0.10 else "MEDIUM" if cs >= 0.06 else "LOW"
        sig.action = "BUY"; sig.status = "BETTING"; sig.filters_passed = True

        log.info(
            "[BET] %s %s | edge=+%.1f%% P=%.0f%% @%.0fc | $%.2f | T-%ds | %s",
            side, slug[-16:], edge*100, prob*100, entry*100,
            size, int(sig.time_remaining_sec), sig.confidence)
        return sig

    def reset_market_stability(self, s): self.stab.reset(s)
