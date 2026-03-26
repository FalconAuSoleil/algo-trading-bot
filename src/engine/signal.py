"""Signal Engine v4.1 - Multi-Strategy Router + Diffusion Risk Model
======================================================================

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