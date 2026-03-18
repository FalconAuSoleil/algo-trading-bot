# BTC Sniper v3 — Multi-Strategy Polymarket Bot

A high-frequency algorithmic trading bot for Polymarket BTC Up/Down binary markets (5-minute and 15-minute windows).

---

## Architecture overview

```
Binance WebSocket  →  BinanceFeed
Chainlink (Polygon) →  ChainlinkFeed      →  SignalEngine (Multi-Strategy)
Polymarket CLOB    →  PolymarketFeed                  ↓
                                          StrategyRouter
                                    ┌─────┴─────────────┐
                              ChainlinkArb  Momentum  MeanReversion
                                    └─────┬─────────────┘
                                          ↓
                              PaperTrader / LiveTrader
                                          ↓
                              Portfolio → SQLite DB → Dashboard
```

---

## Strategies

### 1. ChainlinkArb (primary)
Exploits the lag between Chainlink on-chain oracle updates (~27s) and real-time Binance price. When Binance has moved but Chainlink hasn’t updated yet, we bet in the direction Chainlink will move. Enhanced by:
- **OFI** (Order Flow Imbalance): detects directional pressure in the orderbook
- **Kyle λ**: penalises high-spread / low-liquidity markets
- **Hawkes process**: boosts signal during microstructure activity bursts
- **Stability filter**: only bet when signal has been consistent for ≥3 ticks

### 2. PriceMomentum (secondary)
Bets in the direction of sustained BTC price momentum. Requires consistent momentum over both 60s and 120s windows AND alignment with the current market delta. Conservative sizing (half of ChainlinkArb cap).

### 3. MeanReversion (contrarian)
Fades extreme delta moves (>0.20%). When BTC has moved far from the reference price, the market tends to overprice continuation. Contrarian bet with conservative sizing. Deactivates during losing streaks (≥2 consecutive losses).

### Consensus routing
When 2 or more strategies agree on the same direction, the bet receives a **+25% size boost per additional strategy** (capped at 1.5x). Conflicting signals are skipped unless one side has a >50% better weighted score.

### Performance weighting
Each strategy tracks its rolling win rate (last 30 trades). The router weights each signal by `0.5 + win_rate`, so strategies that are working get more influence.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

**Required for live trading:**
```env
POLYMARKET_PRIVATE_KEY=0x...     # Polygon wallet private key
POLYMARKET_WALLET_ADDRESS=0x...  # Polygon wallet address

# Optional: pre-configured API credentials (else derived from private key)
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
```

### 3. Run in paper mode (default)
```bash
python main.py
python main.py --mode paper --balance 100
```

### 4. Run in live mode
```bash
python main.py --mode live
```

### 5. Open dashboard
Visit `http://localhost:8080` after starting.

---

## Key parameters (`.env`)

| Variable | Default | Description |
|---|---|---|
| `EDGE_MIN` | `0.06` | Minimum edge (P_true − entry − fee) to place a bet |
| `MAX_BET_FRACTION` | `0.04` | Max Kelly fraction per bet (4% of bankroll) |
| `SOURCE_COHERENCE_MAX` | `0.003` | Max Binance–Chainlink divergence before skipping (0.3%) |
| `STABILITY_MIN_SAMPLES` | `3` | Min ticks of consistent signal before betting |
| `MAX_CONSECUTIVE_LOSSES` | `4` | Circuit breaker: pause after N losses in a row |
| `MOMENTUM_MIN_THRESHOLD` | `0.0008` | Min BTC move (0.08%) to activate Momentum strategy |
| `MEAN_REV_DELTA_THRESHOLD` | `0.002` | Min delta (0.20%) to activate MeanReversion strategy |

---

## Files

```
main.py                      Orchestrator
src/
  config.py                  All configuration
  feeds/
    binance.py               BTC real-time price (WebSocket)
    chainlink.py             On-chain Chainlink oracle (Polygon)
    polymarket.py            Market discovery + orderbook polling
  engine/
    signal.py                Multi-strategy engine + router
    performance.py           Per-strategy rolling win rate
    trend.py                 Market outcome ring buffer
  trading/
    portfolio.py             Capital + position tracking
    paper.py                 Paper trading (simulation)
    live.py                  Live trading (py-clob-client)
  utils/
    db.py                    SQLite persistence
    logger.py                Logging setup
  dashboard/
    app.py                   FastAPI dashboard
```

---

## Bug fixes in v3

- **`live.py` token ID**: was sending `conditionId` as the CLOB `tokenID`, which caused all orders to be rejected. Now uses the correct YES/NO outcome token IDs from `MarketInfo`.
- **`live.py` auth**: incomplete header-based auth replaced with proper `py-clob-client` EIP-712 signing.
- **`source_coherence_max`**: raised from `0.0008` to `0.003`. The previous value (0.08%) was triggering on any minor Binance–Chainlink discrepancy during volatile periods, silently killing most signals.
- **`stability_min_samples`**: reduced from `5` to `3` to allow faster signal confirmation within the short betting windows.
