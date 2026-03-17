"""Configuration management with environment variable support.

Includes microstructure strategy hyperparameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


@dataclass(frozen=True)
class SignalConfig:
    # -- Microstructure thresholds --
    edge_min: float = _env_float("EDGE_MIN", 0.06)
    min_true_prob: float = _env_float("MIN_TRUE_PROB", 0.56)
    max_bet_fraction: float = _env_float("MAX_BET_FRACTION", 0.03)
    min_market_prob_side: float = _env_float("MIN_MARKET_PROB_SIDE", 0.25)
    max_market_prob_side: float = _env_float("MAX_MARKET_PROB_SIDE", 0.78)
    min_market_liquidity: float = _env_float("MIN_MARKET_LIQUIDITY", 15.0)

    # -- Timing --
    time_min_5m: float = _env_float("TIME_MIN_5M", 45.0)
    time_max_5m: float = _env_float("TIME_MAX_5M", 240.0)
    time_max_5m_accum: float = _env_float("TIME_MAX_5M_ACCUM", 350.0)
    time_min_15m: float = _env_float("TIME_MIN_15M", 60.0)
    time_max_15m: float = _env_float("TIME_MAX_15M", 780.0)
    time_min_seconds: int = _env_int("TIME_MIN_SECONDS", 45)
    time_max_seconds: int = _env_int("TIME_MAX_SECONDS", 240)

    # -- Chainlink arb --
    chainlink_period: float = _env_float("CHAINLINK_PERIOD", 27.0)
    chainlink_edge_window: float = _env_float("CHAINLINK_EDGE_WINDOW", 8.0)

    # -- OFI --
    ofi_weight: float = _env_float("OFI_WEIGHT", 0.25)

    # -- Kyle --
    kyle_spread_penalty: float = _env_float("KYLE_SPREAD_PENALTY", 0.15)

    # -- Hawkes --
    hawkes_mu: float = _env_float("HAWKES_MU", 0.1)
    hawkes_alpha: float = _env_float("HAWKES_ALPHA", 0.8)
    hawkes_beta: float = _env_float("HAWKES_BETA", 2.0)
    hawkes_history: int = _env_int("HAWKES_HISTORY", 200)

    # -- Momentum --
    momentum_factor: float = _env_float("MOMENTUM_FACTOR", 120.0)

    # -- Stability filter --
    stability_window_sec: float = _env_float("STABILITY_WINDOW_SEC", 45.0)
    stability_min_samples: int = _env_int("STABILITY_MIN_SAMPLES", 4)
    stability_min_ratio: float = _env_float("STABILITY_MIN_RATIO", 0.70)
    stability_edge_cv_max: float = _env_float("STABILITY_EDGE_CV_MAX", 0.70)

    # -- Legacy compat --
    delta_min: float = _env_float("DELTA_MIN", 0.0012)
    volatility_max: float = _env_float("VOLATILITY_MAX", 0.0015)
    source_coherence_max: float = _env_float("SOURCE_COHERENCE_MAX", 0.0008)
    volatility_window_minutes: int = 30
    fee_rate: float = 0.25
    fee_exponent: int = 2


@dataclass(frozen=True)
class RiskConfig:
    kelly_fraction: float = _env_float("KELLY_FRACTION", 0.25)
    max_position_pct: float = _env_float("MAX_POSITION_PCT", 0.08)
    max_daily_drawdown: float = _env_float("MAX_DAILY_DRAWDOWN", 0.10)
    max_consecutive_losses: int = _env_int("MAX_CONSECUTIVE_LOSSES", 4)
    max_open_positions: int = _env_int("MAX_OPEN_POSITIONS", 2)
    max_daily_risk: float = _env_float("MAX_DAILY_RISK", 0.20)


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
    port: int = _env_int("DASHBOARD_PORT", 8080)


@dataclass
class AppConfig:
    trading_mode: str = _env("TRADING_MODE", "paper")
    paper_initial_balance: float = _env_float("PAPER_INITIAL_BALANCE", 10000)
    db_path: Path = BASE_DIR / _env("DB_PATH", "data/trades.db")
    log_level: str = _env("LOG_LEVEL", "INFO")

    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"

    @property
    def is_live(self) -> bool:
        return self.trading_mode == "live"


config = AppConfig()
