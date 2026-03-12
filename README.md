# BTC Sniper — Microstructure Bayesian Strategy

Bot de trading automatisé pour les marchés **BTC Up/Down 5min et 15min** sur [Polymarket](https://polymarket.com).

## Stratégie

Ce bot utilise une approche **bayésienne microstructure** qui combine 4 signaux indépendants dans l'espace logit pour calculer la probabilité réelle que le BTC finisse au-dessus ou en-dessous du prix de référence ("price to beat") Chainlink.

### Les 4 composantes

#### 1. Chainlink Lag Arbitrage
Chainlink met à jour son prix on-chain toutes les ~27 secondes sur Polygon. Dans les secondes précédant un update, le prix Binance (temps réel) prédit le prochain prix Chainlink — créant une fenêtre d'edge exploitable.

- Estimation dynamique de la période d'update via moyenne pondérée
- Edge proportionnel au gap Binance-Chainlink × proximité du prochain update
- Fenêtre d'edge : 8 secondes avant l'update estimé

#### 2. Order Flow Imbalance (OFI)
Basé sur le modèle de **Cont, Kukanov & Stoikov (2014)**. L'asymétrie du carnet d'ordres Polymarket révèle la pression d'achat/vente.

- OFI net = différence normalisée entre bids et asks des deux côtés
- Depth imbalance : ratio de profondeur UP vs DOWN
- Momentum OFI : taux de changement sur les 10 derniers snapshots
- Conversion en ajustement logit borné \[-0.8, +0.8\]

#### 3. Kyle Lambda — Price Impact Filter
Modèle de **Kyle (1985)** : le spread révèle l'information privée des market makers.

- λ = spread / (2 × √depth) — mesure l'impact prix
- Si λ est élevé → market makers incertains → signal peu fiable → pénalité
- Quality factor ∈ \[0.3, 1.0\] agit comme shrinkage vers 0.5 dans l'espace logit

#### 4. Hawkes Process — Regime Detection
Processus auto-excitateur de **Hawkes (1971)** : les mouvements de prix arrivent en clusters.

- Intensité conditionnelle : λ(t) = μ + Σ α·m·exp(-β·(t - tᵢ))
- Détecte les régimes de forte activité (marché en train de pricer une info)
- Boost de confiance ∈ \[0, 0.3\] qui amplifie le signal existant

### Combinaison bayésienne

Tous les signaux sont combinés dans l'**espace logit** (log-odds) :

```
logit(P) = logit(prior)           ← delta prix Chainlink + momentum
         + chainlink_boost        ← lag arbitrage
         + OFI_adjustment         ← order flow
         × kyle_quality           ← filtre de fiabilité
         × (1 + hawkes_boost)     ← amplification régime actif
```

### Filtre de stabilité

Un signal n'est validé que si :
- **Direction ratio ≥ 65%** : au moins 65% des ticks récents pointent dans la même direction
- **Edge CV ≤ 0.80** : coefficient de variation de l'edge suffisamment faible
- **Minimum 3 ticks** accumulés dans la fenêtre de 60 secondes

### Sizing (Kelly fractionnel)

- Quarter-Kelly avec cap à 4% de bankroll
- Réduction ×0.7 en régime calme (Hawkes < 1.5×μ)
- Bonus de taille si signal très stable (ratio élevé)

## Architecture

```
main.py                    ← Orchestrateur principal
src/
├── config.py              ← Configuration + hyperparamètres .env
├── engine/
│   └── signal.py          ← Signal Engine v2 (4 modules microstructure)
├── feeds/
│   ├── binance.py         ← WebSocket BTC/USDT temps réel
│   ├── chainlink.py       ← RPC Polygon + RTDS + Binance fallback
│   └── polymarket.py      ← Découverte marchés + orderbook + résolution
├── trading/
│   ├── portfolio.py       ← Gestion du portefeuille
│   ├── paper.py           ← Paper trading (simulation)
│   └── live.py            ← Live trading (CLOB API)
├── dashboard/
│   ├── app.py             ← FastAPI + WebSocket dashboard
│   └── templates/
│       └── index.html     ← UI temps réel
└── utils/
    ├── logger.py          ← Logging Rich
    ├── db.py              ← SQLite async
    └── math_utils.py      ← Fonctions mathématiques
```

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Ajuster les paramètres dans .env
python main.py
```

## Modes

```bash
# Paper trading (par défaut)
python main.py --mode paper --balance 10000

# Live trading (nécessite clés API Polymarket)
python main.py --mode live

# Dashboard accessible sur http://localhost:8080
```

## Sources de prix

1. **Chainlink RPC Polygon** (source principale) : poll `latestRoundData()` toutes les 3s avec rotation sur 3 endpoints
2. **RTDS Polymarket** (bonus) : stream WebSocket push pour prix inter-ticks
3. **Binance WebSocket** : prix temps réel pour le momentum et cross-ref
4. **Binance REST** (fallback) : si RPC et RTDS sont down

## Risk Management

- **Stop-loss journalier** : -15% de drawdown max
- **Circuit breaker** : pause après 3 pertes consécutives
- **Max positions** : 3 simultanées
- **Kelly fractionnel** : quarter-Kelly avec cap à 4%
- **Guards liquidité** : $15 minimum de profondeur
- **Guards payout** : prix marché entre 20¢ et 80¢ (évite longshots et mauvais odds)
