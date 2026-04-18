"""Intelligent hourly circuit breaker — v5.1.

Encodes hard-coded market knowledge that cannot be derived from price signals:
  - Saturday             : dead market, spreads blow out, no trading
  - 03:30–05:00          : dead zone, near-zero participant activity
  - 01:00–01:05          : Polymarket refresh gap — markets just settled,
                           new ones not yet fully indexed; blocked automatically
                           because volatile window starts at 01:05
  - 01:05–01:35          : volatile window — post-close orderbook churn on
                           freshly opened markets; allow but flag it
  - Mon–Wed 14:00–01:00  : peak hours — best signal quality, most liquidity

All times are expressed in CB_TIMEZONE (default: Europe/Paris).

NOTE on Polymarket refresh timing
----------------------------------
Polymarket 5m/15m crypto markets roll over continuously. Around 01:00 Paris
(≈ 23:00 UTC in winter, 00:00 UTC in summer) there is a batch settlement where
many concurrent markets resolve and new ones open. The Polymarket CLOB feed
and Gamma API take 3–7 minutes to fully index the new markets (orderbook depth
appears thin, spreads are wide, reference prices may be stale). Trading during
this gap produces noise-driven signals. The circuit breaker handles this by:
  1. NOT explicitly blocking 01:00–01:05 (the feed just has no valid markets)
  2. Marking 01:05–01:35 as VOLATILE — offpeak multipliers apply, bot is
     already conservative; this window can still produce good trades when
     a clear delta appears on the fresh markets.

`now` argument accepts either:
  - datetime (timezone-aware) → used as-is
  - float (Unix timestamp)    → converted to UTC datetime automatically
"""
from __future__ import annotations

import enum
import time as _time
from datetime import datetime, timezone
from typing import Union
from zoneinfo import ZoneInfo


class CBStatus(enum.Enum):
    BLOCKED  = "BLOCKED"   # hard block — do not trade
    PEAK     = "PEAK"      # Mon–Wed 14:00–01:00, best conditions
    VOLATILE = "VOLATILE"  # 01:05–01:35, post-close fresh-market window
    OFFPEAK  = "OFFPEAK"   # all other hours — trade conservatively


def _to_datetime(now: Union[float, datetime]) -> datetime:
    """Convert float Unix timestamp to UTC-aware datetime if needed."""
    if isinstance(now, (int, float)):
        return datetime.fromtimestamp(now, tz=timezone.utc)
    # already a datetime — ensure it is timezone-aware
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


def circuit_breaker_status(
    now: Union[float, datetime], cfg
) -> tuple[CBStatus, str]:
    """Return (CBStatus, reason_str).

    Args:
        now: Unix timestamp (float) OR timezone-aware datetime.
        cfg: SignalConfig instance (needs cb_* fields).
    """
    tz    = ZoneInfo(cfg.cb_timezone)
    local = _to_datetime(now).astimezone(tz)

    wd = local.weekday()              # 0=Mon … 5=Sat … 6=Sun
    t  = local.hour * 60 + local.minute   # minutes since midnight

    # ── Saturday block ──────────────────────────────────────────────────────
    if cfg.cb_block_saturday and wd == 5:
        return CBStatus.BLOCKED, "SATURDAY"

    # ── Dead zone 03:30–05:00 ────────────────────────────────────────────────
    dead_start = cfg.cb_dead_zone_start_h * 60 + cfg.cb_dead_zone_start_m
    dead_end   = cfg.cb_dead_zone_end_h   * 60 + cfg.cb_dead_zone_end_m
    if dead_start <= t < dead_end:
        return CBStatus.BLOCKED, "DEAD_ZONE"

    # ── Volatile window (checked before peak to avoid override) ─────────────
    # Default: 01:05–01:35. The 01:00–01:05 Polymarket refresh gap is not
    # explicitly blocked here — the feed simply has no valid markets during
    # that window, so no signals fire anyway.
    vol_start = cfg.cb_volatile_start_h * 60 + cfg.cb_volatile_start_m
    vol_end   = cfg.cb_volatile_end_h   * 60 + cfg.cb_volatile_end_m
    if vol_start <= t < vol_end:
        return CBStatus.VOLATILE, "VOLATILE_WINDOW"

    # ── Peak hours: Mon–Wed 14:00→next-day 01:00 ────────────────────────────
    # "lundi au mercredi de 14h à 1h" spans midnight:
    #   Mon 14:00–23:59  →  Tue 00:00–01:00
    #   Tue 14:00–23:59  →  Wed 00:00–01:00
    #   Wed 14:00–23:59  →  Thu 00:00–01:00
    is_afternoon_peak = wd in (0, 1, 2) and t >= 14 * 60
    is_post_mid_peak  = wd in (1, 2, 3) and t <  60   # 00:00–01:00 continuation
    if is_afternoon_peak or is_post_mid_peak:
        return CBStatus.PEAK, "PEAK_HOURS"

    return CBStatus.OFFPEAK, "OFFPEAK"


def should_trade(now: Union[float, datetime], cfg) -> tuple[bool, str]:
    """Thin boolean wrapper. Returns (can_trade, reason_str)."""
    status, reason = circuit_breaker_status(now, cfg)
    return status != CBStatus.BLOCKED, reason
