"""Data collector — records Polymarket + Binance + Chainlink in shadow mode.

Runs 24/7 to build a replayable dataset for offline grid-search on REAL
orderbook dynamics (not a simulated p_market=0.50 like the synthetic
backtest). After 1 week of collection, replay_backtest.py can replay
every tick to compute actual WR/ROI with any signal-engine config.
"""
