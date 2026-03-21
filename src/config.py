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
    
    Each asset can have its own oracle update frequency, volatility profile,
    and delta thresholds. The signal engine will use these overrides when
    evaluating bets on that asset's markets.
    """
    symbol: str                              # "BTC", "ETH", "XRP", "SOL"
    chainlink_address: str                   # Polygon contract address
    binance_symbol: str                      # "BTCUSDT", "ETHUSDT", etc.
    polymarket_prefix: str                   # "btc", "eth", "xrp", "sol"
    chainlink_period: float = 27.0           # expected update interval (seconds)
    oracle_freshness_max_age_sec: float = 55.0  # staleness threshold
    delta_min_abs: float = 0.0010            # minimum |delta| to bet (%)
    sigma_fallback: float = 0.0              # fallback sigma for diffusion model
    enabled: bool = True


@dataclass(frozen=True)
class SignalConfig:
    # ── Edge filters ─────────────────────────────────────────────────────
    edge_min: float = _envf("EDGE_MIN", 0.06)
    edge_max: float = _envf("EDGE_MAX", 0.15)
    min_true_prob: float = _envf("MIN_TRUE_PROB", 0.56)
    max_bet_fraction: float = _envf("MAX_BET_FRACTION", 0.04)
    min_market_prob_side: float = _envf("MIN_MARKET_PROB_SIDE", 0.35)
    max_market_prob_side: float = _envf("MAX_MARKET_PROB_SIDE", 0.70)
    min_market_liquidity: float = _envf("MIN_MARKET_LIQUIDITY", 15.0)

    # ── Timing windows ───────────────────────────────────────────────────────
    time_min_5m: float = _envf("TIME_MIN_5M", 45.0)
    # WARROOM FIX: reduced from 240s → 180s.
    # Hard cap on how early we bet in a 5-min window.
    # The diffusion model handles fine-grained rejection within this window.
    # Evidence: all screenshot losses occurred at T=248-269s (well above old cap).
    time_max_5m: float = _envf("TIME_MAX_5M", 180.0)
    time_max_5m_accum: float = _envf("TIME_MAX_5M_ACCUM", 290.0)
    time_min_15m: float = _envf("TIME_MIN_15M", 60.0)
    time_max_15m: float = _envf("TIME_MAX_15M", 780.0)

    # ── Chainlink arb ───────────────────────────────────────────────────────
    chainlink_period: float = _envf("CHAINLINK_PERIOD", 27.0)
    chainlink_edge_window: float = _envf("CHAINLINK_EDGE_WINDOW", 8.0)

    # ── OFI ──────────────────────────────────────────────────────────────────
    ofi_weight: float = _envf("OFI_WEIGHT", 0.20)

    # ── Kyle ─────────────────────────────────────────────────────────────────────
    kyle_spread_penalty: float = _envf("KYLE_SPREAD_PENALTY", 0.15)

    # ── Hawkes ───────────────────────────────────────────────────────────────────
    hawkes_mu: float = _envf("HAWKES_MU", 0.1)
    hawkes_alpha: float = _envf("HAWKES_ALPHA", 0.8)
    hawkes_beta: float = _envf("HAWKES_BETA", 2.0)
    hawkes_history: int = _envi("HAWKES_HISTORY", 200)

    # ── Momentum ───────────────────────────────────────────────────────────────────
    momentum_factor: float = _envf("MOMENTUM_FACTOR", 80.0)
    momentum_min_threshold: float = _envf("MOMENTUM_MIN_THRESHOLD", 0.0008)

    # ── Mean Reversion ─────────────────────────────────────────────────────────────────
    mean_reversion_delta_threshold: float = _envf("MEAN_REV_DELTA_THRESHOLD", 0.0020)

    # ── Diffusion model (v3.2) ────────────────────────────────────────────────────────────
    # Hard absolute floor on |delta| regardless of realized volatility.
    # 0.10% = ~$74 on a $74k BTC. Any smaller and 30s of random walk can
    # wipe the edge before close. Override via DELTA_MIN_ABS env var.
    delta_min_abs: float = _envf("DELTA_MIN_ABS", 0.0010)

    # ── Oracle freshness filter (v3.5) ───────────────────────────────────────────────────────
    # If Chainlink's last update is older than this (seconds) when
    # T_remaining < 90s, the bet is blocked (ORACLE_STALE). Resolution
    # uses the last oracle update BEFORE expiry; a stale oracle can
    # capture a temporary dip/spike even when BTC is correct at expiry.
    # Confirmed mechanism: 3 losses had oracle silence 60-170s at bet time.
    # Set to 0.0 to disable. Default: 55s (~2× median Chainlink period).
    oracle_freshness_max_age_sec: float = _envf("ORACLE_FRESHNESS_MAX_AGE_SEC", 55.0)

    # ── Stability filter ─────────────────────────────────────────────────────────────────
    stability_window_sec: float = _envf("STABILITY_WINDOW_SEC", 45.0)
    stability_min_samples: int = _envi("STABILITY_MIN_SAMPLES", 3)
    stability_min_ratio: float = _envf("STABILITY_MIN_RATIO", 0.75)
    stability_edge_cv_max: float = _envf("STABILITY_EDGE_CV_MAX", 0.60)

    # ── Source coherence ─────────────────────────────────────────────────────────────────
    source_coherence_max: float = _envf("SOURCE_COHERENCE_MAX", 0.003)

    # ── Fee model ────────────────────────────────────────────────────────────────────
    delta_min: float = _envf("DELTA_MIN", 0.0012)
    volatility_max: float = _envf("VOLATILITY_MAX", 0.0015)
    volatility_window_minutes: int = 30
    fee_rate: float = 0.25
    fee_exponent: int = 2


@dataclass(frozen=True)
class RiskConfig:
    kelly_fraction: float = _envf("KELLY_FRACTION", 0.25)
    max_position_pct: float = _envf("MAX_POSITION_PCT", 0.05)
    max_daily_drawdown: float = _envf("MAX_DAILY_DRAWDOWN", 0.06)
    max_consecutive_losses: int = _envi("MAX_CONSECUTIVE_LOSSES", 4)
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
    # Only assets with enabled=True will be traded.
    # Asset parameters can be overridden via env vars:
    #   ASSET_<SYMBOL>_ENABLED=true/false
    #   ASSET_<SYMBOL>_CHAINLINK_PERIOD=<seconds>
    #   ASSET_<SYMBOL>_ORACLE_FRESHNESS=<seconds>
    #   ASSET_<SYMBOL>_DELTA_MIN_ABS=<percent>
    assets: list[AssetConfig] = field(default_factory=lambda: [
        AssetConfig(
            symbol="BTC",
            chainlink_address="0xc907E116054Ad103354f2D350FD2514433D57F6f",
            binance_symbol="BTCUSDT",
            polymarket_prefix="btc",
            chainlink_period=27.0,
            oracle_freshness_max_age_sec=55.0,
            delta_min_abs=0.0010,
            sigma_fallback=0.005 / (300 ** 0.5),  # ~2.89e-4
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
            sigma_fallback=0.009 / (300 ** 0.5),  # ~5.20e-4
            enabled=True,
        ),
    ])

    @property
    def is_paper(self): return self.trading_mode == "paper"
    @property
    def is_live(self): return self.trading_mode == "live"

    def get_asset_config(self, symbol: str) -> AssetConfig | None:
        """Look up asset config by symbol."""
        for ac in self.assets:
            if ac.symbol == symbol:
                return ac
        return None


config = AppConfig()
