"""Signal Engine v4.2 - Multi-Strategy Router + Diffusion Risk Model
======================================================================

v4.2 changes (2026-03-28):
  - Stale oracle diffusion penalty: t_eff = t_rem + oracle_age*0.5 inflates
    the diffusion threshold when CL data is stale (prevents "wait out" bug).
  - Off-peak adaptive mode: nights/weekends raise edge_min ×1.3 and shrink
    sizing ×0.6. Bot still trades 24/7 but more conservatively.
  - SUSPICIOUS_EDGE no longer pollutes stability buffer — edges >15% are
    not recorded, preventing marginal trades from riding stale stab data.
  - BTC 5m delta floor raised: delta_min_abs ×1.5 for 5m markets.
  - Multi-timeframe momentum: 240s window added, all 3 must agree.
  - BTCStab dynamic max_swing: volatility-adjusted instead of fixed 8¢.
  - BTCStab order book imbalance: boost/penalize based on depth asymmetry.

v4.1.1 hotfix (2026-03-26):
  - CRITICAL: BTC 15m routing no longer short-circuits to WATCHING
    when BTCStabilization doesn't fire. Falls through to ChainlinkArb
    multi-strategy router as fallback (restores v4.0 BTC 15m behavior).
  - Peak hours gate: now disabled by default (config change).
  - BTCStabilization relaxed: wider price zone (58-85¢), longer time
    window (T=45-300s), faster stability (20s/3obs), lower edge floor
    (2%). Fires much more often on the "one side at ~70¢, unlikely to
    reverse" pattern.

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
  If Chainlink has been silent for >55s when T_remaining < 90s,
  the final resolution price may capture a temporary dip/spike that
  occurred before BTC recovered. Three confirmed losses had oracle
  silence of 60-170s at bet time (19:03, 02:29, 21:59).

  Fix: ORACLE_STALE filter -- if (now - last_cl_update) > 55s AND
  T_remaining < 90s, bet is blocked. Threshold configurable via
  ORACLE_FRESHNESS_MAX_AGE_SEC env var (set to 0.0 to disable).
  oracle_age_sec added to MicrostructureState, Signal, and trade DB.

v3.4 fix (diffusion filter bypass closure):
  DELTA_WEAK was gated by `t_rem > 90`, creating a dead zone between
  T=45s and T=90s where delta < sigma*sqrt(t) bets could slip through.
  Confirmed culprit in the 21:59:05 losing trade (T=54s, delta=0.185%).

v3.3 addition (cross-market propagation exploit):
  After a 5m BTC market resolves, Polymarket takes 12-45 seconds to
  reprice correlated 15m markets. CrossMarketBooster tracks 5m closes
  and adds an additive p_true boost to 15m bets during this window.

Three independent strategies:
  1. ChainlinkArb  -- Bayesian oracle lag + OFI + Kyle + Hawkes (primary)
  2. PriceMomentum -- Sustained BTC price trend (60s + 120s windows)
  3. MeanReversion -- Contrarian fade of extreme delta moves
  4. BTCStabilization -- BTC 15m late-entry at stable 58-85¢ (v4.1.1)

Routing: consensus boost, conflict detection, performance weighting.
"""

from __future__ import annotations

import datetime
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import SignalConfig, config as _global_cfg
from src.engine.performance import PerformanceTracker
from src.engine.cross_market import CrossMarketBooster
from src.utils.logger import setup_logger

log = setup_logger("engine.signal")


# ---- math helpers ----------------------------------------------------------

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


def calc_fee(p: float, rate: float = 0.02) -> float:
    """Estimate Polymarket taker fee per token.

    v4.0: Replaced quadratic approximation with accurate linear model.
    Polymarket charges ~2% of potential winnings (notional taker fee).

    fee = rate x (1 - p)

    At p=0.50: fee = 0.010  (old formula: 0.0156 -- 56% overestimate)
    At p=0.60: fee = 0.008  (old formula: 0.0144)
    At p=0.40: fee = 0.012  (old formula: 0.0144)

    The old formula filtered out ~33% of valid signals unnecessarily.
    """
    if p <= 0 or p >= 1:
        return 0.0
    return rate * (1.0 - p)


def _erf_approx(x: float) -> float:
    """
    Fast, no-dependency approximation of erf(x).
    Abramowitz & Stegun 7.1.26 -- max error < 1.5e-7.
    """
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741,
        -1.453152027, 1.061405429,
    )
    p = 0.3275911
    t = 1.0 / (1.0 + p * abs(x))
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    result = 1.0 - poly * math.exp(-x * x)
    return result if x >= 0 else -result


def p_brownian(
    delta: float,
    time_remaining_sec: float,
    sigma_per_sec: float,
) -> float:
    """
    Probability that BTC stays on the correct side of the reference price
    until expiry, modeled as standard Brownian motion.

    P = N(|delta| / (sigma_per_sec * sqrt(t_remaining)))

    Returns:
        Float in [0.5, 1.0].
    """
    if time_remaining_sec <= 1:
        return 1.0
    if sigma_per_sec <= 0 or delta == 0:
        return 0.5
    z = abs(delta) / (sigma_per_sec * math.sqrt(time_remaining_sec))
    return 0.5 * (1.0 + _erf_approx(z / 1.41421356237))


# ---- peak hours gate -------------------------------------------------------

def is_peak_hours(now: Optional[float] = None) -> bool:
    """
    Returns True during peak Polymarket liquidity hours.

    Peak = Monday-Friday, 08:00-18:00 ET.
    ET offset: UTC-5 (EST, Nov-Mar) or UTC-4 (EDT, Mar-Nov).

    v4.1.1: disabled by default (peak_hours_enabled=False in config).
    Set PEAK_HOURS_ENABLED=true env var to restore.
    """
    cfg = _global_cfg.signal
    if not cfg.peak_hours_enabled:
        return True  # gate disabled — always active

    ts = now if now is not None else time.time()
    utc_dt = datetime.datetime.utcfromtimestamp(ts)

    # DST approximation: March–November = EDT (UTC-4), else EST (UTC-5).
    et_offset = -4 if 3 <= utc_dt.month <= 11 else -5
    et_dt = utc_dt + datetime.timedelta(hours=et_offset)

    # Weekends: Saturday(5) and Sunday(6)
    if et_dt.weekday() >= 5:
        return False

    return cfg.peak_start_hour_et <= et_dt.hour < cfg.peak_end_hour_et


def is_offpeak(now: Optional[float] = None) -> bool:
    """Returns True during nights and weekends (off-peak hours).

    v4.2: Used to apply conservative multipliers (higher edge floor,
    smaller sizing) rather than blocking trades entirely.

    Off-peak = weekends OR outside 08:00-20:00 ET on weekdays.
    Wider than peak_hours (08-18h) to give a buffer zone.
    """
    ts = now if now is not None else time.time()
    utc_dt = datetime.datetime.utcfromtimestamp(ts)
    et_offset = -4 if 3 <= utc_dt.month <= 11 else -5
    et_dt = utc_dt + datetime.timedelta(hours=et_offset)

    if et_dt.weekday() >= 5:
        return True  # weekends
    return et_dt.hour < 8 or et_dt.hour >= 20  # before 8am or after 8pm ET


# ---- data structures -------------------------------------------------------

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
    p_diffusion: float = 0.5
    realized_sigma_pct: float = 0.0
    min_viable_delta_pct: float = 0.0
    oracle_age_sec: float = 0.0
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
    strategy_used: str = "chainlink_arb"
    strategies_agreeing: int = 1
    token_id_yes: str = ""
    token_id_no: str = ""
    oracle_age_sec: float = 0.0


# ---- microstructure sub-modules --------------------------------------------

class _OFI:
    def __init__(self, w):
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
    def __init__(self, p):
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
    def __init__(self, mu, alpha, beta, history):
        self._ev: deque = deque(maxlen=history)
        self._lm = 0.5
        self._mu = mu
        self._a = alpha
        self._b = beta

    def on_mid(self, mid, ts):
        if self._lm > 0:
            ch = abs(mid - self._lm)
            if ch >= 0.005:
                self._ev.append((ts, min(ch / 0.005, 5)))
        self._lm = mid

    def boost(self, t=0):
        t = t or time.time()
        lam = self._mu
        for ts, m in self._ev:
            dt = t - ts
            if dt >= 0:
                lam += self._a * m * math.exp(-self._b * dt)
        ex = max(0, lam - self._mu)
        return clamp(ex / (5 * self._a) * 0.3, 0, 0.3), lam


class _Stability:
    def __init__(self, ws, ms, mr, mc):
        self._h: dict = {}
        self._ws = ws
        self._ms = ms
        self._mr = mr
        self._mc = mc

    def _buf(self, s):
        if s not in self._h:
            self._h[s] = deque(maxlen=100)
        return self._h[s]

    def record(self, s, side, edge, ts):
        self._buf(s).append((ts, side, abs(edge)))

    def evaluate(self, s, side):
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

    def reset(self, s):
        self._h.pop(s, None)


# ---- Strategy 1: Chainlink Arb + Diffusion Model ---------------------------

class _ChainlinkArbEngine:
    """
    Primary strategy: exploit Chainlink oracle lag relative to Binance spot.

    v4.0: calc_fee now uses accurate linear model (see module docstring).
    v3.7: sigma_fallback and delta_min_abs are per-asset.
    v3.5: oracle freshness filter.
    v3.4: diffusion filter bypass closed.
    v3.2: sigma floor + raised min_viable_delta multiplier + abs delta floor.
    v3.1: Brownian diffusion model.
    """
    NAME = "chainlink_arb"

    def __init__(
        self,
        cfg: SignalConfig,
        sigma_fallback: float = 0.005 / math.sqrt(300),
        delta_min_abs: float = 0.0010,
    ):
        self.cfg = cfg
        self.sigma_fallback = sigma_fallback
        self.sigma_floor = sigma_fallback
        self.delta_min_abs = delta_min_abs
        self._ph: deque = deque(maxlen=200)
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

    def _realized_vol_per_sec(self, window_sec: float = 300.0) -> float:
        now = time.time()
        r = [(t, p) for t, p in self._ph if t >= now - window_sec]
        if len(r) < 6:
            return self.sigma_floor
        sq_returns = []
        for i in range(1, len(r)):
            dt = r[i][0] - r[i - 1][0]
            if dt > 0.1 and r[i - 1][1] > 0:
                lr = math.log(r[i][1] / r[i - 1][1])
                sq_returns.append(lr * lr / dt)
        if not sq_returns:
            return self.sigma_floor
        variance_per_sec = sum(sq_returns) / len(sq_returns)
        sigma = math.sqrt(variance_per_sec)
        return clamp(sigma, self.sigma_floor, 0.02 / math.sqrt(300))

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

        # Source coherence
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

        # v3.5: Oracle freshness filter
        oracle_age = (now - self._cl_ts) if self._cl_ts > 0 else 0.0
        micro.oracle_age_sec = round(oracle_age, 1)
        if (
            cfg.oracle_freshness_max_age_sec > 0
            and sig.time_remaining_sec < 90.0
            and self._cl_ts > 0
            and oracle_age > cfg.oracle_freshness_max_age_sec
        ):
            sig.filter_reasons.append(f"oracle_stale:{oracle_age:.0f}s")
            sig.status = f"ORACLE_STALE ({oracle_age:.0f}s)"
            sig.oracle_age_sec = round(oracle_age, 1)
            sig.micro = micro
            return sig

        accum = sig.time_remaining_sec > max_t
        tr = max_t - min_t
        tdf = (
            (0.6 + 0.4 * clamp((sig.time_remaining_sec - min_t) / tr, 0, 1))
            if tr > 0 else 1.0
        )
        micro.time_decay_factor = tdf
        # v4.2: off-peak raises edge floor (nights/weekends = lower confidence)
        offpeak_mult = cfg.offpeak_edge_multiplier if is_offpeak(now) else 1.0
        eff_min = cfg.edge_min * tdf * offpeak_mult

        pd = sig.delta_chainlink

        # ---- DIFFUSION RISK CHECK (v3.1/v3.2/v3.4/v4.1.2) ------------------
        sigma_ps = self._realized_vol_per_sec()
        t_rem = sig.time_remaining_sec

        # v4.1.2: stale oracle data adds uncertainty — the real price may
        # already be closer to the strike than the frozen CL delta suggests.
        # Penalise by inflating effective time so the diffusion threshold
        # stays higher when data is old.
        t_eff = t_rem + oracle_age * 0.5
        min_viable_delta = 1.0 * sigma_ps * math.sqrt(max(t_eff, 1))
        micro.realized_sigma_pct = round(sigma_ps * math.sqrt(300) * 100, 4)
        micro.min_viable_delta_pct = round(min_viable_delta * 100, 4)

        if abs(pd) < min_viable_delta and t_rem > 45:
            sig.filter_reasons.append(
                f"delta_weak:{abs(pd)*100:.3f}%<{min_viable_delta*100:.3f}%"
            )
            sig.status = f"DELTA_WEAK (sigma*sqrt(t)={min_viable_delta*100:.3f}%)"
            sig.micro = micro
            return sig

        # v4.2: 5m markets require higher absolute delta (more noise in short window)
        eff_delta_min = self.delta_min_abs * (cfg.delta_min_abs_5m_mult if is5 else 1.0)
        if abs(pd) < eff_delta_min and t_rem > 30:
            sig.filter_reasons.append(
                f"delta_abs:{abs(pd)*100:.3f}%<{eff_delta_min*100:.3f}%"
            )
            sig.status = f"DELTA_ABS_FLOOR ({abs(pd)*100:.3f}%)"
            sig.micro = micro
            return sig

        # ---- Bayesian pipeline ---------------------------------------------
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

        # ---- Blend Bayesian + Diffusion ------------------------------------
        p_diff = p_brownian(pd, t_rem, sigma_ps)
        micro.p_diffusion = round(p_diff, 4)
        blend_w = clamp(t_rem / max(max_t, 1), 0.15, 0.55)

        if pd > 0:
            p_bayes = pu
            side = "YES"
            entry = state.best_ask_yes if state.best_ask_yes > 0 else state.p_market_yes
        elif pd < 0:
            p_bayes = pdn
            side = "NO"
            entry = (
                state.best_ask_no if state.best_ask_no > 0
                else (1 - state.p_market_yes)
            )
        else:
            sig.status = "WATCHING"
            sig.micro = micro
            return sig

        prob = blend_w * p_diff + (1 - blend_w) * p_bayes
        prob = clamp(prob, 0.50, 0.98)

        # v4.0: uses corrected linear fee model
        fee = calc_fee(entry, cfg.fee_rate)
        edge = prob - entry - fee
        sig.side = side
        sig.p_true = prob
        sig.p_market = entry
        sig.entry_price = entry
        sig.edge = edge
        sig.taker_fee = fee
        micro.taker_fee = fee

        # ---- Filters -------------------------------------------------------
        if edge < eff_min:
            self.stab.record(slug, side, max(edge, 0.001), now)
            sig.status = (
                f"LOW_EDGE ({edge * 100:.1f}%)" if edge > 0 else "WATCHING"
            )
            sig.micro = micro
            return sig
        if edge > cfg.edge_max:
            # v4.2: do NOT record in stability buffer — suspicious edges
            # polluted the stab counter, letting marginal trades fire when
            # edge briefly dipped into the acceptable range.
            sig.filter_reasons.append(f"edge_suspicious:{edge * 100:.1f}%")
            sig.status = f"SUSPICIOUS_EDGE ({edge * 100:.1f}%)"
            sig.micro = micro
            return sig
        if prob < entry + 0.02:
            self.stab.record(slug, side, max(edge, 0.001), now)
            sig.filter_reasons.append("overpaying")
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
        risk = _global_cfg.risk
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
        # v4.2: shrink sizing at night/weekends (lower signal quality)
        if is_offpeak(now):
            frac *= cfg.offpeak_sizing_multiplier
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
        sig.oracle_age_sec = micro.oracle_age_sec
        sig.micro = micro

        log.info(
            "[BET:CL_ARB] %s %s | edge=+%.1f%% P=%.0f%%(diff=%.0f%%) "
            "@%.0fc | $%.2f | T-%ds | sigma*sqrt(t)=%.3f%% | abs_delta=%.3f%% | "
            "CL_age=%.0fs | %s",
            side, slug[-14:], edge * 100, prob * 100, p_diff * 100,
            entry * 100, size, int(t_rem),
            min_viable_delta * 100, abs(pd) * 100,
            micro.oracle_age_sec, sig.confidence,
        )
        return sig

    def reset_stability(self, slug: str) -> None:
        self.stab.reset(slug)


# ---- Strategy 2: Price Momentum --------------------------------------------

class _MomentumEngine:
    """
    Bet in the direction of sustained BTC price momentum.
    v4.2: All three windows (60s, 120s, 240s) must agree.
    Window 60-150s. Applies diffusion check before committing.
    """
    NAME = "momentum"

    def __init__(self, cfg: SignalConfig, sigma_fallback: float = 0.005 / math.sqrt(300)):
        self.cfg = cfg
        self.sigma_fallback = sigma_fallback
        self.sigma_floor = sigma_fallback
        self._ph: deque = deque(maxlen=200)

    def update_price(self, p: float, ts: float) -> None:
        self._ph.append((ts, p))

    def _mom_window(self, seconds: float) -> float:
        now = time.time()
        r = [(t, p) for t, p in self._ph if t >= now - seconds]
        if len(r) < 4:
            return 0.0
        return (r[-1][1] - r[0][1]) / r[0][1]

    def _sigma_ps(self) -> float:
        now = time.time()
        r = [(t, p) for t, p in self._ph if t >= now - 300]
        if len(r) < 6:
            return self.sigma_floor
        sq = []
        for i in range(1, len(r)):
            dt = r[i][0] - r[i - 1][0]
            if dt > 0.1 and r[i - 1][1] > 0:
                lr = math.log(r[i][1] / r[i - 1][1])
                sq.append(lr * lr / dt)
        raw = math.sqrt(sum(sq) / len(sq)) if sq else self.sigma_floor
        return max(raw, self.sigma_floor)

    def evaluate(
        self,
        state: MarketState,
        capital: float,
        has_position_on_market: bool,
    ) -> Optional[Signal]:
        cfg = self.cfg
        now = time.time()
        time_remaining = state.end_time - now

        if time_remaining < 60 or time_remaining > 150:
            return None
        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            return None
        if has_position_on_market:
            return None

        m60 = self._mom_window(60)
        m120 = self._mom_window(120)
        m240 = self._mom_window(240)  # v4.2: longer confirmation window
        threshold = cfg.momentum_min_threshold

        if abs(m60) < threshold or abs(m120) < threshold:
            return None
        if (m60 > 0) != (m120 > 0):
            return None
        # v4.2: 240s window must agree if available (filters single-spike noise)
        if abs(m240) >= threshold * 0.5 and (m240 > 0) != (m60 > 0):
            return None

        going_up = m60 > 0
        delta = (
            (state.btc_chainlink - state.reference_price) / state.reference_price
        )

        if going_up and delta < -0.0005:
            return None
        if not going_up and delta > 0.0005:
            return None

        sigma_ps = self._sigma_ps()
        p_diff = p_brownian(delta, time_remaining, sigma_ps)
        if p_diff < 0.60:
            return None

        delta_abs_floor = getattr(cfg, "delta_min_abs", 0.0010)
        if abs(delta) < delta_abs_floor:
            return None

        side = "YES" if going_up else "NO"
        entry = (
            (state.best_ask_yes if state.best_ask_yes > 0 else state.p_market_yes)
            if going_up
            else (state.best_ask_no if state.best_ask_no > 0
                  else (1 - state.p_market_yes))
        )

        # v4.2: include 240s window in strength (triple-timeframe confirmation)
        mom_vals = [abs(m60), abs(m120)]
        if abs(m240) >= threshold * 0.5:
            mom_vals.append(abs(m240))
        momentum_strength = sum(mom_vals) / len(mom_vals)
        p_bayes = clamp(0.53 + momentum_strength * 20, 0.53, 0.67)
        blend_w = clamp(time_remaining / 150, 0.2, 0.45)
        prob = blend_w * p_diff + (1 - blend_w) * p_bayes
        prob = clamp(prob, 0.50, 0.98)

        fee = calc_fee(entry)  # uses new linear model
        edge = prob - entry - fee

        if edge < cfg.edge_min * 0.9:
            return None
        if entry < cfg.min_market_prob_side or entry > cfg.max_market_prob_side:
            return None

        max_frac = cfg.max_bet_fraction * 0.5
        c_eff = entry + fee
        if c_eff >= 1:
            return None
        kelly = (prob - c_eff) / (1 - c_eff)
        if kelly <= 0:
            return None
        frac = clamp(kelly * 0.25, 0, max_frac)
        size = round(capital * frac, 2)
        if size < 1.0:
            return None

        slug = state.slug or state.market_id[:20]
        sig = Signal(
            timestamp=now, market_id=state.market_id,
            delta_chainlink=delta, btc_chainlink=state.btc_chainlink,
            btc_binance=state.btc_binance, reference_price=state.reference_price,
            time_remaining_sec=time_remaining, slug=slug,
            side=side, p_true=prob, p_market=entry, entry_price=entry,
            edge=edge, taker_fee=fee, kelly_pct=frac, size_usd=size,
            action="BUY", status="BETTING", filters_passed=True,
            confidence="MEDIUM", strategy_used=self.NAME, strategies_agreeing=1,
        )
        sig.micro.p_diffusion = round(p_diff, 4)
        log.info(
            "[BET:MOMENTUM] %s %s | m60=%.3f%% m120=%.3f%% "
            "p_diff=%.0f%% edge=%.1f%% $%.2f | T-%ds",
            side, slug[-14:], m60 * 100, m120 * 100,
            p_diff * 100, edge * 100, size, int(time_remaining),
        )
        return sig


# ---- Strategy 3: Mean Reversion --------------------------------------------

class _MeanReversionEngine:
    """
    Contrarian: fade extreme delta moves (>0.20%). Window 90-180s.
    Requires p_diffusion of the contrarian move to be favourable.
    NOTE (v4.0 team audit): p_true formula is a heuristic without calibration.
    Monitor win rate closely; consider disabling if WR < 50% over 20+ trades.
    """
    NAME = "mean_rev"

    def __init__(self, cfg: SignalConfig, sigma_fallback: float = 0.005 / math.sqrt(300)):
        self.cfg = cfg
        self.sigma_fallback = sigma_fallback

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
        if time_remaining < 90 or time_remaining > 180:
            return None
        if consecutive_losses >= 2:
            return None
        if has_position_on_market:
            return None

        sigma_ps = self.sigma_fallback * 1.5
        one_sigma = sigma_ps * math.sqrt(max(time_remaining, 1))
        if abs_delta < 1.5 * one_sigma:
            return None

        if delta > 0:
            side = "NO"
            entry = (
                state.best_ask_no if state.best_ask_no > 0
                else (1 - state.p_market_yes)
            )
        else:
            side = "YES"
            entry = state.best_ask_yes if state.best_ask_yes > 0 else state.p_market_yes

        p_true = clamp(0.52 + (abs_delta - thresh) * 8, 0.52, 0.62)
        fee = calc_fee(entry)  # uses new linear model
        edge = p_true - entry - fee

        if edge < cfg.edge_min:
            return None
        if entry < cfg.min_market_prob_side or entry > cfg.max_market_prob_side:
            return None

        max_frac = cfg.max_bet_fraction * 0.4
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
            timestamp=now, market_id=state.market_id,
            delta_chainlink=delta, btc_chainlink=state.btc_chainlink,
            btc_binance=state.btc_binance, reference_price=state.reference_price,
            time_remaining_sec=time_remaining, slug=slug,
            side=side, p_true=p_true, p_market=entry, entry_price=entry,
            edge=edge, taker_fee=fee, kelly_pct=frac, size_usd=size,
            action="BUY", status="BETTING", filters_passed=True,
            confidence="LOW", strategy_used=self.NAME, strategies_agreeing=1,
        )
        log.info(
            "[BET:MEAN_REV] %s %s | delta=%.3f%% p_true=%.1f%% "
            "edge=%.1f%% $%.2f | T-%ds",
            side, slug[-14:], delta * 100, p_true * 100,
            edge * 100, size, int(time_remaining),
        )
        return sig


# ---- Strategy 4: BTC 15m Stabilization (v4.1 / v4.1.1) --------------------

class _BTCStabilizationEngine:
    """
    BTC 15m high-confidence late-entry strategy (v4.1, relaxed v4.1.1).

    Targets the window T=45-300s (last 5 minutes) when one side has
    stabilized at 58-85¢, indicating strong directional consensus that
    the market hasn't yet fully priced to 1.0.

    v4.1.1 changes (relaxed to trade more often):
      - Price zone: 63-80¢ → 58-85¢ (wider catch zone)
      - Time window: T=60-180s → T=45-300s (fires earlier)
      - Stability: 30s/5obs → 20s/3obs (triggers faster)
      - Max swing: 6¢ → 8¢ (tolerates more noise)
      - Edge floor: 3% → 2% (relies on diffusion as guard)
      - Sizing: Kelly×0.22 cap 3.5% (slightly more aggressive)

    Core logic:
    1. Market price in [58¢, 85¢] on one side (stability zone)
    2. That price stable for ≥20s with ≥3 observations
    3. Delta direction matches the winning side
    4. Brownian diffusion: p_true > market_price + fee (real edge)
    5. Chainlink oracle fresh (<55s)
    6. Standard risk filters (circuit breaker, drawdown, position cap)
    """
    NAME = "btc_stabilization"

    def __init__(
        self,
        cfg: SignalConfig,
        sigma_fallback: float = 0.005 / math.sqrt(900),
    ):
        self.cfg = cfg
        self.sigma_fallback = sigma_fallback
        self.sigma_floor = sigma_fallback
        # BTC price history — 10-minute window for 15m vol estimation
        self._ph: deque = deque(maxlen=600)
        # Market price history — tracks YES price stability over ~3min
        self._market_ph: deque = deque(maxlen=150)
        self._cl_ts: float = 0.0
        self._cp: float = 0.0

    def update_price(self, p: float, ts: float) -> None:
        self._ph.append((ts, p))

    def update_chainlink(self, p: float, ts: float) -> None:
        self._cp = p
        self._cl_ts = ts

    def _sigma_ps(self) -> float:
        """Realized vol per second from last 10 minutes of BTC price data."""
        now = time.time()
        r = [(t, p) for t, p in self._ph if t >= now - 600]
        if len(r) < 6:
            return self.sigma_floor
        sq = []
        for i in range(1, len(r)):
            dt = r[i][0] - r[i - 1][0]
            if dt > 0.1 and r[i - 1][1] > 0:
                lr = math.log(r[i][1] / r[i - 1][1])
                sq.append(lr * lr / dt)
        if not sq:
            return self.sigma_floor
        return clamp(
            math.sqrt(sum(sq) / len(sq)),
            self.sigma_floor,
            0.02 / math.sqrt(300),
        )

    def _is_market_stable(self, current_price: float) -> bool:
        """
        Returns True if the market price has been in a tight range
        for the stability window with enough observations.

        v4.2: dynamic max_swing based on realized vol instead of fixed 8¢.
        In low-vol regimes, 8¢ is huge (allows too much noise).
        In high-vol regimes, 8¢ is tiny (rejects genuine stability).
        Dynamic: max_swing = max(cfg.btc_stab_max_swing, 2σ·√(window_sec)·100¢)
        """
        cfg = self.cfg
        now = time.time()
        recent = [
            (t, p) for t, p in self._market_ph
            if t >= now - cfg.btc_stab_window_sec
        ]
        if len(recent) < cfg.btc_stab_min_obs:
            return False
        prices = [p for _, p in recent]
        swing = max(prices) - min(prices)
        # v4.2: volatility-adaptive swing tolerance
        sigma_ps = self._sigma_ps()
        # 2-sigma band in "market price cents" (sigma_ps is in price-fraction/sec)
        # Market price moves ~proportionally to BTC, so scale by 100¢
        dynamic_swing = 2.0 * sigma_ps * math.sqrt(cfg.btc_stab_window_sec) * 100
        max_swing = clamp(dynamic_swing, 0.03, cfg.btc_stab_max_swing)
        return swing <= max_swing

    def evaluate(
        self,
        state: MarketState,
        capital: float,
        consecutive_losses: int,
        daily_pnl_pct: float,
        open_positions: int,
        has_position_on_market: bool,
    ) -> Optional[Signal]:
        now = time.time()
        cfg = self.cfg
        t_rem = state.end_time - now

        # Window: T=45-300s (last 5 minutes of 15m market)
        if t_rem < cfg.btc_stab_time_min or t_rem > cfg.btc_stab_time_max:
            return None
        if state.reference_price <= 0 or state.btc_chainlink <= 0:
            return None
        if has_position_on_market:
            return None

        # Identify which side is in the stabilization zone
        p_yes = state.p_market_yes
        p_no = 1.0 - p_yes

        if cfg.btc_stab_price_min <= p_yes <= cfg.btc_stab_price_max:
            side = "YES"
            market_price = p_yes
            entry = state.best_ask_yes if state.best_ask_yes > 0 else p_yes
            # BTC must be above reference for YES to be the winning side
            delta = (state.btc_chainlink - state.reference_price) / state.reference_price
            if delta <= 0:
                return None
        elif cfg.btc_stab_price_min <= p_no <= cfg.btc_stab_price_max:
            side = "NO"
            market_price = p_no
            entry = state.best_ask_no if state.best_ask_no > 0 else p_no
            # BTC must be below reference for NO to be the winning side
            delta = (state.reference_price - state.btc_chainlink) / state.reference_price
            if delta <= 0:
                return None
        else:
            # Neither side in the target zone
            return None

        # Record market price every loop iteration for stability tracking
        self._market_ph.append((now, market_price))

        # Stability: price must have been in zone continuously
        if not self._is_market_stable(market_price):
            return None

        # Oracle freshness: Chainlink data must not be stale
        oracle_age = (now - self._cl_ts) if self._cl_ts > 0 else 999.0
        if oracle_age > cfg.oracle_freshness_max_age_sec:
            return None

        # Absolute delta floor: move must be meaningful
        if abs(delta) < cfg.delta_min_abs:
            return None

        # Brownian reversal probability
        sigma_ps = self._sigma_ps()
        p_diff = p_brownian(delta, t_rem, sigma_ps)

        # v4.2: order book imbalance — penalize if opposing side has deeper book
        total_depth = (state.depth_yes or 0) + (state.depth_no or 0)
        if total_depth > 0:
            own_depth = state.depth_yes if side == "YES" else state.depth_no
            opp_depth = state.depth_no if side == "YES" else state.depth_yes
            imbalance = (opp_depth - own_depth) / total_depth  # >0 means opposition deeper
            if imbalance > 0.5:
                # Heavy opposing book = market expects reversal, penalize
                p_diff -= 0.02 * imbalance  # up to -2% at extreme imbalance
            elif imbalance < -0.3:
                # Opposing side thin = exhausted sellers, slight boost
                p_diff += 0.01
            p_diff = clamp(p_diff, 0.50, 0.98)

        # v4.0 linear fee model
        fee = calc_fee(entry, cfg.fee_rate)
        edge = p_diff - entry - fee

        # v4.1.1: relaxed edge floor (2% vs original 3%)
        if edge < cfg.btc_stab_edge_min:
            return None
        if p_diff <= entry:  # no true edge: market already fairly priced
            return None

        # Risk filters
        risk = _global_cfg.risk
        if consecutive_losses >= risk.max_consecutive_losses:
            return None
        if daily_pnl_pct < -risk.max_daily_drawdown:
            return None
        if open_positions >= risk.max_open_positions:
            return None

        # Liquidity check
        md = min(state.depth_yes or 0, state.depth_no or 0)
        if md < cfg.min_market_liquidity:
            return None

        # Conservative Kelly sizing
        c_eff = entry + fee
        if c_eff >= 1:
            return None
        kelly = (p_diff - c_eff) / (1 - c_eff)
        if kelly <= 0:
            return None
        frac = clamp(kelly * cfg.btc_stab_kelly_fraction, 0, cfg.btc_stab_max_bet_fraction)
        # v4.2: conservative sizing at night/weekends
        if is_offpeak(now):
            frac *= cfg.offpeak_sizing_multiplier
        size = round(capital * frac, 2)
        depth = state.depth_yes if side == "YES" else state.depth_no
        if depth > 0:
            size = min(size, depth * 0.25)
        if size < 1.0:
            return None

        slug = state.slug or state.market_id[:20]
        micro = MicrostructureState()
        micro.p_diffusion = round(p_diff, 4)
        micro.oracle_age_sec = round(oracle_age, 1)
        micro.realized_sigma_pct = round(sigma_ps * math.sqrt(300) * 100, 4)
        micro.min_viable_delta_pct = round(sigma_ps * math.sqrt(t_rem) * 100, 4)
        micro.taker_fee = fee

        delta_binance = 0.0
        if state.btc_binance > 0 and state.reference_price > 0:
            delta_binance = (state.btc_binance - state.reference_price) / state.reference_price

        sig = Signal(
            timestamp=now,
            market_id=state.market_id,
            delta_chainlink=delta if side == "YES" else -delta,
            delta_binance=delta_binance,
            btc_chainlink=state.btc_chainlink,
            btc_binance=state.btc_binance,
            reference_price=state.reference_price,
            time_remaining_sec=t_rem,
            slug=slug,
            side=side,
            p_true=p_diff,
            p_market=market_price,
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
            oracle_age_sec=micro.oracle_age_sec,
            micro=micro,
        )

        log.info(
            "[BET:BTC_STAB] %s %s | mkt=%.0f¢ p_diff=%.0f%% "
            "delta=%.3f%% edge=+%.1f%% $%.2f | T-%ds | "
            "sigma*sqrt(t)=%.3f%% | CL_age=%.0fs",
            side, slug[-14:],
            market_price * 100, p_diff * 100,
            delta * 100, edge * 100, size, int(t_rem),
            sigma_ps * math.sqrt(t_rem) * 100,
            oracle_age,
        )
        return sig


# ---- Multi-Strategy Router -------------------------------------------------

class SignalEngine:
    """
    Coordinates all strategies and routes to the correct engine per asset/interval.

    v4.1.1 routing (fixed):
      - BTC 15m  → Try _BTCStabilizationEngine first (24/7, T=45-300s).
                   If it doesn't fire, fall through to ChainlinkArb
                   multi-strategy router (restores v4.0 behavior as fallback).
      - BTC 5m   → ChainlinkArb multi-strategy (24/7, peak hours disabled)
      - ETH/SOL/XRP → ChainlinkArb multi-strategy (24/7, peak hours disabled)

    v4.0: Uses corrected fee model across all strategies.
    v3.7: Accepts optional per-asset parameters (sigma_fallback, delta_min_abs).
    """

    def __init__(
        self,
        cfg: SignalConfig,
        sigma_fallback: float = 0.005 / math.sqrt(300),
        delta_min_abs: float = 0.0010,
        asset_symbol: str = "BTC",
    ):
        self.cfg = cfg
        self.asset_symbol = asset_symbol.upper()
        self._cl = _ChainlinkArbEngine(cfg, sigma_fallback, delta_min_abs)
        self._mom = _MomentumEngine(cfg, sigma_fallback)
        self._rev = _MeanReversionEngine(cfg, sigma_fallback)
        # v4.1: BTC 15m stabilization engine
        self._btc_stab = _BTCStabilizationEngine(cfg, sigma_fallback)
        self.perf = PerformanceTracker(window=30)
        self.cross_market = CrossMarketBooster()

    def update_price(self, p: float, ts: float) -> None:
        self._cl.update_price(p, ts)
        self._mom.update_price(p, ts)
        self._btc_stab.update_price(p, ts)

    def update_chainlink_price(self, p: float, ts: float) -> None:
        self._cl.update_chainlink(p, ts)
        self._btc_stab.update_chainlink(p, ts)

    def reset_market_stability(self, slug: str) -> None:
        self._cl.reset_stability(slug)

    def record_result(self, strategy: str, won: bool) -> None:
        self.perf.record(strategy, won)

    def record_5m_resolution(
        self, chainlink_price: float, reference_price: float, direction: str
    ) -> None:
        self.cross_market.record_5m_close(
            chainlink_price=chainlink_price,
            reference_price_5m=reference_price,
            direction=direction,
        )

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

        def _blank(status: str) -> Signal:
            """Return a no-trade signal with metadata populated."""
            sig = Signal(
                timestamp=now,
                market_id=state.market_id,
                btc_chainlink=state.btc_chainlink,
                btc_binance=state.btc_binance,
                reference_price=state.reference_price,
                status=status,
            )
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

        # ── v4.1.1: ROUTING (fixed) ──────────────────────────────────────────
        is_15m = state.duration_seconds >= 900 or "15m" in (state.slug or "")

        # BTC 15m → Try BTCStabilization first, then fall through to
        # ChainlinkArb multi-strategy if it doesn't fire.
        # v4.1 BUG FIX: was "return _blank('WATCHING')" which killed
        # ChainlinkArb entirely for BTC 15m. Now falls through.
        if self.asset_symbol == "BTC" and is_15m:
            stab_sig = self._btc_stab.evaluate(
                state, capital, consecutive_losses, daily_pnl_pct,
                open_positions, has_position_on_market,
            )
            if stab_sig is not None:
                stab_sig.slug = state.slug
                stab_sig.market_start_time = state.start_time
                stab_sig.market_duration = state.duration_seconds
                stab_sig.time_remaining_sec = state.end_time - now
                return stab_sig
            # v4.1.1: fall through to ChainlinkArb multi-strategy below
            # (BTCStab only fires under specific conditions; ChainlinkArb
            # covers the rest of the BTC 15m opportunity space)

        # All assets: ChainlinkArb multi-strategy router
        # v4.1.1: peak hours gate disabled by default (trades 24/7)
        if not is_peak_hours(now):
            return _blank("OFF_PEAK")

        # ── Existing multi-strategy router (unchanged from v4.0) ─────────────
        cl_sig = self._cl.evaluate(
            state, capital, consecutive_losses, daily_pnl_pct,
            open_positions, has_position_on_market,
        )
        mom_sig = self._mom.evaluate(state, capital, has_position_on_market)
        rev_sig = self._rev.evaluate(
            state, capital, has_position_on_market, consecutive_losses
        )

        candidates: List[Signal] = [
            s for s in (cl_sig, mom_sig, rev_sig)
            if s is not None and s.action == "BUY" and s.filters_passed
        ]

        def _base_info(sig: Signal) -> Signal:
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
            base = cl_sig if cl_sig is not None else Signal(
                timestamp=now, market_id=state.market_id,
                btc_chainlink=state.btc_chainlink,
                reference_price=state.reference_price,
            )
            return _base_info(base)

        yes_cands = [s for s in candidates if s.side == "YES"]
        no_cands = [s for s in candidates if s.side == "NO"]

        if yes_cands and no_cands:
            def _score(s: Signal) -> float:
                return s.edge * self.perf.weight(s.strategy_used)
            best_yes = max(yes_cands, key=_score)
            best_no = max(no_cands, key=_score)
            ys, ns = _score(best_yes), _score(best_no)
            if abs(ys - ns) / max(ys, ns, 1e-9) < 0.50:
                conflict = Signal(
                    timestamp=now, market_id=state.market_id,
                    btc_chainlink=state.btc_chainlink,
                    reference_price=state.reference_price,
                    status="CONFLICT",
                )
                conflict.filter_reasons.append(
                    f"conflict:YES({best_yes.strategy_used})"
                    f"/NO({best_no.strategy_used})"
                )
                return _base_info(conflict)
            candidates = yes_cands if ys > ns else no_cands

        agreeing = len(candidates)
        best = max(candidates, key=lambda s: s.edge * self.perf.weight(s.strategy_used))
        best.strategies_agreeing = agreeing

        if agreeing >= 2:
            boost = 1.0 + 0.25 * (agreeing - 1)
            hard_cap = capital * self.cfg.max_bet_fraction * 1.5
            best.size_usd = min(round(best.size_usd * boost, 2), hard_cap)
            best.confidence = "HIGH"
            log.info(
                "[CONSENSUS] %d strategies agree %s | boost=%.2fx | $%.2f",
                agreeing, best.side, boost, best.size_usd,
            )

        # ---- Cross-market propagation boost --------------------------------
        if best.filters_passed and best.action == "BUY":
            is_15m_mkt = (
                state.duration_seconds >= 900
                or "15m" in (state.slug or "")
            )
            if is_15m_mkt:
                cm_boost = self.cross_market.get_boost(
                    best.side, state.btc_chainlink, state.reference_price
                )
                if cm_boost > 0:
                    best.p_true = min(best.p_true + cm_boost, 0.97)
                    best.edge = best.p_true - best.entry_price - best.taker_fee
                    if best.edge > self.cfg.edge_max:
                        best.edge = self.cfg.edge_max
                    log.info(
                        "[CrossMarket] 15m boost +%.1f%% -> p_true=%.1f%% "
                        "edge=%.1f%% | %s %s",
                        cm_boost * 100, best.p_true * 100, best.edge * 100,
                        best.side, state.slug or state.market_id[:14],
                    )

        # ---- v4.2: Trend exhaustion penalty ----------------------------------
        # After 3+ consecutive resolutions in the same direction, the next
        # bet continuing that direction faces higher pullback risk. Reduce
        # sizing progressively (not p_true — the model may be right on
        # direction, but the risk of a brief dip is elevated).
        if best.filters_passed and best.action == "BUY":
            streak_n, streak_dir = self.cross_market.consecutive_same_direction()
            if streak_n >= 3:
                # Check if we're betting WITH the streak
                bet_with_streak = (
                    (best.side == "YES" and streak_dir == "up")
                    or (best.side == "NO" and streak_dir == "down")
                )
                if bet_with_streak:
                    # Progressive penalty: 3→0.7x, 4→0.5x, 5+→0.35x sizing
                    exhaust_mult = max(0.35, 1.0 - 0.15 * streak_n)
                    best.size_usd = round(best.size_usd * exhaust_mult, 2)
                    if best.size_usd < 1.0:
                        best.action = "HOLD"
                        best.filters_passed = False
                        best.filter_reasons.append(
                            f"trend_exhaust:{streak_n}x{streak_dir}"
                        )
                        best.status = f"TREND_EXHAUST ({streak_n}x {streak_dir})"
                    else:
                        log.info(
                            "[TrendExhaust] %dx %s streak | sizing %.0f%% | "
                            "$%.2f | %s %s",
                            streak_n, streak_dir, exhaust_mult * 100,
                            best.size_usd, best.side,
                            state.slug or state.market_id[:14],
                        )

        return _base_info(best)
