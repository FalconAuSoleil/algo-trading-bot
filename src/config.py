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
class SignalConfig:
    # ── Edge filters ─────────────────────────────────────────────────────────
    # P(ours) - entry_price - fee must exceed this to place a bet
    edge_min: float = _envf("EDGE_MIN", 0.06)
    # Maximum believable edge — anything above is likely model error
    edge_max: float = _envf("EDGE_MAX", 0.15)
    # Our probability must be at least this to bet
    min_true_prob: float = _envf("MIN_TRUE_PROB", 0.56)
    # Max Kelly fraction of bankroll per single bet
    # Raised from 0.02 → 0.04 to allow larger bets on HIGH-confidence signals
    max_bet_fraction: float = _envf("MAX_BET_FRACTION", 0.04)
    # Entry price must be between these (avoid longshots and near-certainties)
    min_market_prob_side: float = _envf("MIN_MARKET_PROB_SIDE", 0.35)
    max_market_prob_side: float = _envf("MAX_MARKET_PROB_SIDE", 0.70)
    # Min combined orderbook depth (USD) for acceptable liquidity
    min_market_liquidity: float = _envf("MIN_MARKET_LIQUIDITY", 15.0)

    # ── Timing windows ───────────────────────────────────────────────────────
    time_min_5m: float = _envf("TIME_MIN_5M", 45.0)
    time_max_5m: float = _envf("TIME_MAX_5M", 240.0)
    time_max_5m_accum: float = _envf("TIME_MAX_5M_ACCUM", 350.0)
    time_min_15m: float = _envf("TIME_MIN_15M", 60.0)
    time_max_15m: float = _envf("TIME_MAX_15M", 780.0)
    time_min_seconds: int = _envi("TIME_MIN_SECONDS", 45)
    time_max_seconds: int = _envi("TIME_MAX_SECONDS", 240)

    # ── Chainlink arb ────────────────────────────────────────────────────────
    chainlink_period: float = _envf("CHAINLINK_PERIOD", 27.0)
    chainlink_edge_window: float = _envf("CHAINLINK_EDGE_WINDOW", 8.0)

    # ── OFI (Order Flow Imbalance) ───────────────────────────────────────────
    ofi_weight: float = _envf("OFI_WEIGHT", 0.20)

    # ── Kyle lambda (market impact) ──────────────────────────────────────────
    kyle_spread_penalty: float = _envf("KYLE_SPREAD_PENALTY", 0.15)

    # ── Hawkes process (microstructure activity) ─────────────────────────────
    hawkes_mu: float = _envf("HAWKES_MU", 0.1)
    hawkes_alpha: float = _envf("HAWKES_ALPHA", 0.8)
    hawkes_beta: float = _envf("HAWKES_BETA", 2.0)
    hawkes_history: int = _envi("HAWKES_HISTORY", 200)

    # ── Momentum ─────────────────────────────────────────────────────────────
    # Bayesian momentum scaling factor for Chainlink delta
    momentum_factor: float = _envf("MOMENTUM_FACTOR", 80.0)
    # PriceMomentum strategy: minimum fractional BTC price change required
    # over BOTH 60s and 120s windows to trigger a momentum bet.
    # 0.0008 = 0.08% move on BTC (~$32 on a $40k BTC)
    momentum_min_threshold: float = _envf("MOMENTUM_MIN_THRESHOLD", 0.0008)

    # ── Mean Reversion strategy ───────────────────────────────────────────────
    # |delta| must exceed this to activate the MeanReversion contrarian bet.
    # 0.0020 = 0.20% — BTC has moved significantly from reference price
    mean_reversion_delta_threshold: float = _envf("MEAN_REV_DELTA_THRESHOLD", 0.0020)

    # ── Stability filter ─────────────────────────────────────────────────────
    stability_window_sec: float = _envf("STABILITY_WINDOW_SEC", 45.0)
    # Reduced from 5 → 3 to allow faster signal confirmation in short windows
    stability_min_samples: int = _envi("STABILITY_MIN_SAMPLES", 3)
    stability_min_ratio: float = _envf("STABILITY_MIN_RATIO", 0.75)
    stability_edge_cv_max: float = _envf("STABILITY_EDGE_CV_MAX", 0.60)

    # ── Source coherence ─────────────────────────────────────────────────────
    # Max fractional divergence between Binance and Chainlink before skipping.
    # FIXED: raised from 0.0008 → 0.003 (0.08% → 0.30%).
    # The original value was too aggressive: on a $40k BTC a mere $32 gap
    # would abort the signal even when both sources agree on direction.
    source_coherence_max: float = _envf("SOURCE_COHERENCE_MAX", 0.003)

    # ── Fee model ────────────────────────────────────────────────────────────
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
    # Raised from 3 → 4: give strategies more breathing room before stopping
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

    @property
    def is_paper(self): return self.trading_mode == "paper"
    @property
    def is_live(self): return self.trading_mode == "live"

config = AppConfig()
