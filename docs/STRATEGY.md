# Polymarket BTC Sniper — Strategy Document

## Executive Summary

This system exploits a structural inefficiency in Polymarket's short-duration BTC binary
markets (5-min and 15-min "Up/Down"). Near market expiry (~30-90 seconds remaining), the
outcome becomes quasi-certain based on BTC price physics, but Polymarket odds fail to
converge to $1.00/$0.00 fast enough. We buy the near-certain outcome at a discount.

**This is NOT prediction. This is probability calculation.**

---

## 1. Market Mechanics

### 1.1 How Polymarket BTC Up/Down Markets Work

- **Market type**: Binary (YES/NO) — "Will BTC be higher than $X in 15 minutes?"
- **Reference price**: Chainlink BTC/USD Data Stream snapshot at market open
- **Resolution price**: Chainlink BTC/USD Data Stream snapshot at market close
- **Resolution**: Automatic via Chainlink Automation, within seconds of close
- **Blockchain**: Polygon (block time ~2s)
- **Durations available**: 5 minutes, 15 minutes

### 1.2 Fee Structure (Dynamic Taker Fees — January 2026)

```
fee = fee_rate × (p × (1 - p))^exponent

Where:
  fee_rate = 0.25 (crypto markets)
  exponent = 2
  p = share price (probability)
```

| Share Price (p) | Taker Fee  |
|-----------------|------------|
| 0.50            | 1.56%      |
| 0.60            | 1.44%      |
| 0.70            | 1.10%      |
| 0.78            | 0.74%      |
| 0.85            | 0.41%      |
| 0.90            | 0.20%      |
| 0.95            | 0.06%      |

**Maker fees: ZERO + daily USDC rebates from taker fees.**

### 1.3 Key Infrastructure

- **Oracle**: Chainlink Data Streams — updates every 10-30s or on 0.5% deviation
- **CLOB**: Hybrid off-chain matching, on-chain settlement
- **API**: REST + WebSocket, no position limits, 3500 orders/10s burst
- **RTDS WebSocket**: `wss://ws-live-data.polymarket.com` streams Chainlink + Binance

---

## 2. The Strategy — "Last-Minute Endgame"

### 2.1 Core Thesis

With T minutes remaining in a market and BTC at δ% from the reference price:

```
P(reversal) = Φ(-|δ| / (σ₁ × √T))

Where:
  δ     = (current_price - reference_price) / reference_price
  σ₁    = BTC 1-minute realized volatility
  T     = time remaining in minutes
  Φ     = standard normal CDF
```

When T < 2 minutes and |δ| > 0.12%, the probability of reversal drops to near zero,
but the market price remains significantly below $1.00.

### 2.2 Decision Logic

```
P_true = 1 - Φ(-|δ_chainlink| / (σ × √T))
fee = 0.25 × (p_market × (1 - p_market))²
edge = P_true - p_market - fee

IF all filters pass AND edge > threshold:
  → BUY the near-certain side (YES if δ > 0, NO if δ < 0)
ELSE:
  → NO TRADE
```

### 2.3 Entry Window

- **Optimal entry**: 30s to 90s before market close
- **Too early** (>2 min): BTC can still reverse meaningfully
- **Too late** (<15s): Polygon block time + execution latency risk

### 2.4 Expected Profile

| Metric                    | Conservative | Optimistic |
|---------------------------|-------------|------------|
| Win rate                  | 92%         | 97%        |
| Avg profit per winner     | 15%         | 25%        |
| Loss per loser            | 100%        | 100%       |
| Trades per day            | 5           | 20         |
| Monthly return            | 15%         | 40%        |
| Max drawdown              | -20%        | -10%       |

---

## 3. Data Feeds Required

### 3.1 Primary: Polymarket RTDS (Real-Time Data Socket)

```
WebSocket: wss://ws-live-data.polymarket.com
Streams:   btcusdt (Binance), btc/usd (Chainlink)
Format:    { "asset": "btc/usd", "price": 84250.32, "timestamp": 1709395200000 }
```

### 3.2 Secondary: Binance BTC/USDT

```
WebSocket: wss://stream.binance.com:9443/ws/btcusdt@trade
Purpose:   Cross-reference, early detection, volatility calculation
```

### 3.3 Market Data: Polymarket Gamma API

```
REST: https://gamma-api.polymarket.com
Purpose: Discover active BTC Up/Down markets, get condition IDs, token IDs
```

### 3.4 Orderbook: Polymarket CLOB API

```
REST: https://clob.polymarket.com
WebSocket: wss://ws-subscriptions-clob.polymarket.com/ws/
Purpose: Real-time orderbook depth, best bid/ask, place orders
```

---

## 4. Signal Engine

### 4.1 Volatility Calculation

Rolling realized volatility on 1-minute BTC returns:

```
σ₁ = std(log_returns_1min, window=30)
```

Using a 30-minute rolling window of 1-minute log returns.
Updated every tick from the Binance feed.

### 4.2 True Probability Calculation

```python
from scipy.stats import norm

def calc_true_probability(delta: float, sigma_1min: float, T_minutes: float) -> float:
    if T_minutes <= 0 or sigma_1min <= 0:
        return 0.5
    sigma_T = sigma_1min * math.sqrt(T_minutes)
    z = abs(delta) / sigma_T
    p_reversal = norm.cdf(-z)
    return 1.0 - p_reversal  # probability current side holds
```

### 4.3 Edge Calculation

```python
def calc_edge(p_true: float, p_market: float) -> float:
    fee = 0.25 * (p_market * (1 - p_market)) ** 2
    return p_true - p_market - fee
```

### 4.4 Kelly Sizing

```python
def kelly_fraction(edge: float, p_true: float, fraction: float = 0.25) -> float:
    # Binary outcome: win = (1/p_market - 1), lose = 1
    # Simplified Kelly for binary: f = edge / (1 - p_market)
    if p_true >= 1.0 or p_true <= 0.0:
        return 0.0
    f = edge / (1 - p_true)
    return max(0.0, min(fraction * f, 0.05))  # cap at 5% of bankroll
```

---

## 5. Trade Filters

ALL filters must pass for a trade to execute:

| Filter                    | Condition                                  | Rationale                                |
|---------------------------|--------------------------------------------|------------------------------------------|
| Time window               | 15s < T < 120s                             | Sweet spot: enough certainty, enough time |
| Delta minimum             | \|δ_chainlink\| > 0.12%                    | Avoid coin-flip near δ=0                 |
| Source coherence           | \|δ_binance - δ_chainlink\| < 0.08%        | Detect oracle divergence                 |
| Minimum edge              | edge > 0.08 (8%)                           | Sufficient margin after fees             |
| Orderbook depth           | depth at target price > $2,000             | Ensure execution possible                |
| Volatility cap            | σ₁ < 0.15%                                | Avoid extreme volatility regimes         |
| Macro blackout            | Not within 30min of CPI/FOMC/NFP           | Avoid macro event risk                   |
| Circuit breaker           | consecutive_losses < 3                      | Stop-loss on regime change               |
| Daily loss limit          | daily_pnl > -15% of capital               | Hard daily stop                          |
| Max concurrent            | open_positions < 3                          | Limit exposure                           |

---

## 6. Risk Management

### 6.1 Position Sizing

```
size = min(
    kelly_fraction × capital,
    orderbook_depth × 0.30,     # max 30% of available depth
    capital × 0.05              # hard cap 5% per trade
)
```

### 6.2 Portfolio Limits

- **Max daily drawdown**: -15% → stop trading for 24h
- **Max consecutive losses**: 3 → stop trading for 2h
- **Max open positions**: 3 simultaneously
- **Max daily capital at risk**: 25% of total capital

### 6.3 Loss Analysis

A losing trade = 100% of the position. With 95% win rate and 20% avg profit:

```
EV = 0.95 × 0.20 - 0.05 × 1.00 = 0.19 - 0.05 = +0.14 (14% per trade)
```

Even with 90% win rate and 15% profit:
```
EV = 0.90 × 0.15 - 0.10 × 1.00 = 0.135 - 0.10 = +0.035 (3.5% per trade)
```

---

## 7. Edge Cases and Mitigations

### 7.1 Chainlink ≠ Binance at Resolution

**Risk**: Chainlink snapshot diverges from Binance spot at T=0.
**Mitigation**: Base all decisions on Chainlink RTDS feed, NOT Binance. Binance is
secondary/cross-reference only. Add source coherence filter.

### 7.2 Flash Crash in Last 30 Seconds

**Risk**: BTC drops 0.5% in seconds due to liquidation cascade after entry.
**Mitigation**: Only enter when δ > 0.12%. Accept that ~2-5% of trades will lose
due to this. Kelly sizing ensures no single loss is catastrophic.

### 7.3 Empty Orderbook Near Expiry

**Risk**: Market makers pull orders as resolution approaches.
**Mitigation**: Check depth before sending order. Use FOK (Fill-or-Kill) to avoid
partial fills at bad prices. Skip if insufficient depth.

### 7.4 API Latency Spike

**Risk**: Network delay causes order to arrive after resolution.
**Mitigation**: Don't enter after T < 15s. Monitor round-trip latency; auto-adjust
minimum T based on measured latency.

### 7.5 Strategy Competition / Edge Compression

**Risk**: More bots = price converges faster = less edge.
**Mitigation**: Monitor avg edge over time. If avg edge < 5% over 50 trades, reduce
sizing. If < 3%, halt and reassess.

### 7.6 Polymarket Rule Changes

**Risk**: Fee change, market structure change, or market type removal.
**Mitigation**: Config-driven parameters. Alert on any unexpected API response changes.
Maintain ability to shut down instantly.

---

## 8. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     MAIN ORCHESTRATOR                     │
│  Coordinates all components, handles lifecycle            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Binance WS  │  │ Polymarket  │  │  Polymarket     │  │
│  │  BTC/USDT    │  │ RTDS (CL)   │  │  CLOB/Gamma    │  │
│  │  Feed        │  │ Feed        │  │  Feed          │  │
│  └──────┬───────┘  └──────┬──────┘  └───────┬────────┘  │
│         │                 │                  │           │
│         └────────┬────────┘──────────────────┘           │
│                  │                                       │
│         ┌────────▼────────┐                              │
│         │  SIGNAL ENGINE   │                              │
│         │  δ, σ, P_true   │                              │
│         │  edge, filters   │                              │
│         └────────┬────────┘                              │
│                  │                                       │
│         ┌────────▼────────┐                              │
│         │  TRADING ENGINE  │                              │
│         │  Paper / Live    │                              │
│         │  Portfolio       │                              │
│         │  Risk Manager    │                              │
│         └────────┬────────┘                              │
│                  │                                       │
│         ┌────────▼────────┐                              │
│         │  DATABASE        │                              │
│         │  SQLite          │                              │
│         │  Trades, PnL     │                              │
│         └────────┬────────┘                              │
│                  │                                       │
│         ┌────────▼────────┐                              │
│         │  DASHBOARD       │                              │
│         │  FastAPI + HTML  │                              │
│         │  Chart.js        │                              │
│         │  WebSocket RT    │                              │
│         └─────────────────┘                              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 9. Operating Modes

### 9.1 Paper Trading (Default)

- All data feeds are LIVE (real Binance + Polymarket data)
- Orders are SIMULATED against the live orderbook
- Fake portfolio tracks positions and PnL
- Full logging and dashboard
- Purpose: Validate strategy with zero financial risk

### 9.2 Live Trading

- All data feeds are LIVE
- Orders are EXECUTED via Polymarket CLOB API
- Real portfolio with real USDC
- Same filters and risk management as paper
- Requires: Polymarket API credentials, funded wallet

### 9.3 Data Collection (Passive)

- All data feeds are LIVE
- No orders placed
- Records all market data, signals, and hypothetical trades
- Purpose: Build historical dataset, measure theoretical edge

---

## 10. Dashboard Specifications

### 10.1 Real-Time Panel

- Current BTC price (Binance + Chainlink)
- Active market info (reference price, time remaining)
- Current δ, σ, P_true, edge
- Filter status (which pass, which block)
- Signal: TRADE / NO TRADE with confidence

### 10.2 Portfolio Panel

- Current balance (paper or real)
- Open positions
- Today's PnL (absolute + %)
- Total PnL since inception

### 10.3 Performance Charts

- Equity curve over time
- PnL distribution histogram
- Win rate rolling average
- Edge distribution
- Trade log table with all details

### 10.4 Risk Panel

- Consecutive losses counter
- Daily drawdown meter
- Max drawdown tracker
- Average edge trend (detect compression)

---

## 11. Deployment Checklist

### Phase 1: Data Collection (Week 1-2)
- [ ] Deploy feeds + database only
- [ ] Record 200+ market cycles
- [ ] Measure real edge distribution
- [ ] Validate volatility model against actual outcomes

### Phase 2: Paper Trading (Week 3-4)
- [ ] Enable paper trading engine
- [ ] Run for 100+ simulated trades
- [ ] Verify win rate > 90%
- [ ] Verify average edge > 8%
- [ ] Check drawdown behavior

### Phase 3: Live Trading (Week 5+)
- [ ] Start with $200-500
- [ ] Max $50 per trade
- [ ] Compare live vs paper results
- [ ] Scale progressively if consistent
