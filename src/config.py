"""Configuration management with environment variable support."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

def _env(key, default=""): return os.getenv(key, default)
def _envf(key, default=0.0): return float(os.getenv(key, str(default)))
def _envi(key, default=0): return int(os.getenv(key, str(default)))


@dataclass(frozen=True)
class AssetConfig:
    """Per-asset configuration for multi-asset support (v3.7).

    Each asset has its own oracle update frequency, volatility profile,
    delta thresholds, and supported market intervals.

    IMPORTANT — Polymarket API constraint (verified 2026-03-21):
      The past-results API (used to fetch reference prices) only supports
      5-minute markets for BTC. All other assets (ETH, SOL, XRP) must use
      supported_intervals=(900,) — 15-minute markets only.
    """
    symbol: str                                   # "BTC", "ETH", "XRP", "SOL"
    chainlink_address: str                        # Polygon contract address
    binance_symbol: str                           # "BTCUSDT", "ETHUSDT", etc.
    polymarket_prefix: str                        # "btc", "eth", "xrp", "sol"
    chainlink_period: float = 27.0                # expected update interval (seconds)
    oracle_freshness_max_age_sec: float = 55.0    # staleness threshold
    delta_min_abs: float = 0.0010                 # minimum |delta| to bet
    sigma_fallback: float = 0.0                   # fallback sigma for diffusion model
    # Polymarket market intervals this asset supports (seconds).
    # ETH/SOL/XRP: (900,) only — 5m past-results API not supported.
    # BTC: (300, 900) — both 5m and 15m.
    supported_intervals: tuple = (300, 900)
    enabled: bool = True


@dataclass(frozen=True)
class SignalConfig:
    # ── Edge filters ───────────────────────────────────────────────────────────────────────────────
    edge_min: float = _envf("EDGE_MIN", 0.06)
    edge_max: float = _envf("EDGE_MAX", 0.15)
    min_true_prob: float = _envf("MIN_TRUE_PROB", 0.56)
    max_bet_fraction: float = _envf("MAX_BET_FRACTION", 0.04)
    min_market_prob_side: float = _envf("MIN_MARKET_PROB_SIDE", 0.35)
    max_market_prob_side: float = _envf("MAX_MARKET_PROB_SIDE", 0.70)
    min_market_liquidity: float = _envf("MIN_MARKET_LIQUIDITY", 15.0)

    # ── Timing windows ────────────────────────────────────────────────────────────────────────────
    # v4.0: raised from 45s → 65s.
    # Analysis of paper trade screens shows 3 consecutive losses all
    # occurring at T=58-68s with small deltas (0.25-0.27%).
    # Late-window bets on small moves are systematically unprofitable:
    # the signal-to-noise ratio at T<65s is insufficient to overcome
    # the diffusion uncertainty on a small delta.
    time_min_5m: float = _envf("TIME_MIN_5M", 65.0)
    time_max_5m: float = _envf("TIME_MAX_5M", 180.0)
    time_max_5m_accum: float = _envf("TIME_MAX_5M_ACCUM", 290.0)
    time_min_15m: float = _envf("TIME_MIN_15M", 60.0)
    # v3.8: reduced from 780s → 550s.
    time_max_15m: float = _envf("TIME_MAX_15M", 550.0)

    # ── Chainlink arb ─────────────────────────────────────────────────────────────────────────────────
    chainlink_period: float = _envf("CHAINLINK_PERIOD", 27.0)
    chainlink_edge_window: float = _envf("CHAINLINK_EDGE_WINDOW", 8.0)

    # ── OFI ─────────────────────────────────────────────────────────────────────────────────────────
    ofi_weight: float = _envf("OFI_WEIGHT", 0.20)

    # ── Kyle ───────────────────────────────────────────────────────────────────────────────────────────────
    kyle_spread_penalty: float = _envf("KYLE_SPREAD_PENALTY", 0.15)

    # ── Hawkes ───────────────────────────────────────────────────────────────────────────────────────────
    hawkes_mu: float = _envf("HAWKES_MU", 0.1)
    hawkes_alpha: float = _envf("HAWKES_ALPHA", 0.8)
    hawkes_beta: float = _envf("HAWKES_BETA", 2.0)
    hawkes_history: int = _envi("HAWKES_HISTORY", 200)

    # ── Momentum ───────────────────────────────────────────────────────────────────────────────────────────
    momentum_factor: float = _envf("MOMENTUM_FACTOR", 80.0)
    momentum_min_threshold: float = _envf("MOMENTUM_MIN_THRESHOLD", 0.0008)

    # ── Mean Reversion ────────────────────────────────────────────────────────────────────────────────────────
    mean_reversion_delta_threshold: float = _envf("MEAN_REV_DELTA_THRESHOLD", 0.0020)

    # ── Diffusion model ─────────────────────────────────────────────────────────────────────────────────────
    delta_min_abs: float = _envf("DELTA_MIN_ABS", 0.0010)

    # ── Oracle freshness filter ─────────────────────────────────────────────────────────────────────────
    oracle_freshness_max_age_sec: float = _envf("ORACLE_FRESHNESS_MAX_AGE_SEC", 55.0)

    # ── Stability filter ──────────────────────────────────────────────────────────────────────────────────────
    stability_window_sec: float = _envf("STABILITY_WINDOW_SEC", 45.0)
    # v4.0: reduced from 8 → 5.
    # v3.8 raised this from 3→8 without impact analysis. At 2s loop
    # interval, 8 samples = 16s minimum wait, causing the bot to likely
    # under-trade and lose opportunities. 5 samples (10s accumulation)
    # provides solid evidence while maintaining trade frequency.
    stability_min_samples: int = _envi("STABILITY_MIN_SAMPLES", 5)
    stability_min_ratio: float = _envf("STABILITY_MIN_RATIO", 0.75)
    # v3.8: tightened from 0.60 → 0.40. CV 60% accepted very dispersed
    # edges. 40% requires more consistent edge magnitude before betting.
    stability_edge_cv_max: float = _envf("STABILITY_EDGE_CV_MAX", 0.40)

    # ── Source coherence ────────────────────────────────────────────────────────────────────────────────────
    # v3.8: raised from 0.003 → 0.005 (correct, kept in v4.0).
    source_coherence_max: float = _envf("SOURCE_COHERENCE_MAX", 0.005)

    # ── Fee model ──────────────────────────────────────────────────────────────────────────────────────────
    delta_min: float = _envf("DELTA_MIN", 0.0012)
    volatility_max: float = _envf("VOLATILITY_MAX", 0.0015)
    volatility_window_minutes: int = 30
    fee_rate: float = 0.02   # v4.0: corrected from 0.25 to match linear fee model
    fee_exponent: int = 2


@dataclass(frozen=True)
class RiskConfig:
    kelly_fraction: float = _envf("KELLY_FRACTION", 0.25)
    max_position_pct: float = _envf("MAX_POSITION_PCT", 0.05)
    max_daily_drawdown: float = _envf("MAX_DAILY_DRAWDOWN", 0.06)
    # v4.0: reduced from 4 → 3.
    # 3 consecutive losses on a ~60% win-rate strategy is a 1-in-15
    # event (≈ (0.40)^3 = 6.4%). This is a strong signal to pause and
    # reassess. The old threshold of 4 allowed too much capital erosion.
    max_consecutive_losses: int = _envi("MAX_CONSECUTIVE_LOSSES", 3)
    max_open_positions: int = _envi("MAX_OPEN_POSITIONS", 2)
    max_daily_risk: float = _envf("MAX_DAILY_RISK", 0.12)


@dataclass(frozen=True)
class PolymarketConfig:
    api_key: str = _env("POLYMARKET_API_KEY")
    api_secret: str = _env("POLYMARKET_API_SECRET")
    api_passphrase: str = _env("POLYMARKET_API_PASSPHRASE")
    wallet_address: str = _env("POLYMARKET_WALLET_ADDRESS")
    private_key: str = _env("POLYMARKET_PRIVATE_KEY")
    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    rtds_url: str = "wss://ws-live-data.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/"


@dataclass(frozen=True)
class BinanceConfig:
    ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    rest_url: str = "https://api.binance.com"


@dataclass(frozen=True)
class DashboardConfig:
    host: str = _env("DASHBOARD_HOST", "0.0.0.0")
    port: int = _envi("DASHBOARD_PORT", 8080)


@dataclass
class AppConfig:
    trading_mode: str = _env("TRADING_MODE", "paper")
    paper_initial_balance: float = _envf("PAPER_INITIAL_BALANCE", 10000)
    db_path: Path = BASE_DIR / _env("DB_PATH", "data/trades.db")
    log_level: str = _env("LOG_LEVEL", "INFO")

    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # v3.7: Per-asset configurations
    # v3.8: supported_intervals added — ETH/SOL/XRP restricted to 15m only.
    #   Polymarket past-results API does not support 5m for non-BTC assets.
    #   Verified 2026-03-21 via live API: returns
    #   {"error": "5-minute markets currently supported only for BTC"}
    assets: list = field(default_factory=lambda: [
        AssetConfig(
            symbol="BTC",
            chainlink_address="0xc907E116054Ad103354f2D350FD2514433D57F6f",
            binance_symbol="BTCUSDT",
            polymarket_prefix="btc",
            chainlink_period=27.0,
            oracle_freshness_max_age_sec=55.0,
            delta_min_abs=0.0010,
            sigma_fallback=0.005 / (300 ** 0.5),  # ~2.89e-4
            supported_intervals=(300, 900),  # 5m + 15m
            enabled=True,
        ),
        AssetConfig(
            symbol="ETH",
            chainlink_address="0xF9680D99D6C9589e2a93a78A04A279e509205945",
            binance_symbol="ETHUSDT",
            polymarket_prefix="eth",
            chainlink_period=45.0,
            oracle_freshness_max_age_sec=55.0,
            delta_min_abs=0.0015,
            sigma_fallback=0.007 / (300 ** 0.5),  # ~4.04e-4
            supported_intervals=(900,),  # 15m only — 5m API not supported for ETH
            enabled=True,
        ),
        AssetConfig(
            symbol="SOL",
            chainlink_address="0x10C8264C0935b3B9870013e057f330Ff3e9C56dC",
            binance_symbol="SOLUSDT",
            polymarket_prefix="sol",
            chainlink_period=90.0,
            oracle_freshness_max_age_sec=90.0,
            delta_min_abs=0.0020,
            sigma_fallback=0.009 / (300 ** 0.5),  # ~5.20e-4
            supported_intervals=(900,),  # 15m only — 5m API not supported for SOL
            enabled=True,
        ),
        AssetConfig(
            symbol="XRP",
            chainlink_address="0x785ba89291f676b5386652eB12b30cF361020694",
            binance_symbol="XRPUSDT",
            polymarket_prefix="xrp",
            chainlink_period=120.0,
            oracle_freshness_max_age_sec=90.0,
            delta_min_abs=0.0015,
            sigma_fallback=0.008 / (300 ** 0.5),  # ~4.62e-4
            supported_intervals=(900,),  # 15m only — 5m API not supported for XRP
            enabled=True,
        ),
    ])

    @property
    def is_paper(self): return self.trading_mode == "paper"
    @property
    def is_live(self): return self.trading_mode == "live"

    def get_asset_config(self, symbol: str):
        """Look up asset config by symbol."""
        for ac in self.assets:
            if ac.symbol == symbol:
                return ac
        return None


config = AppConfig()
