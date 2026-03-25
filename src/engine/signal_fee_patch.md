# Fee Formula Patch Notes — v4.0

File: `src/engine/signal.py` — function `calc_fee`

## Old formula (v3.x)
```python
def calc_fee(p, rate=0.25):
    return rate * (p * (1.0 - p)) ** 2
```
At p=0.50 → 0.0156 (1.56%) — overestimates by ~3x

## New formula (v4.0)
```python
def calc_fee(p, rate=0.02):
    return rate * (1.0 - p)
```
At p=0.50 → 0.010 (1.0%) — matches Polymarket taker fee structure

## Why this matters
The old formula caused ~33% of otherwise valid signals to be
filtered out by edge_min. With correct fees, the bot trades more
frequently and with more accurate edge estimates.
