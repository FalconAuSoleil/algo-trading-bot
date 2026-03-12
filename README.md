# BTC Sniper — Polymarket Binary Options Trading Bot

Bot de trading automatise pour les marches binaires BTC Up/Down de Polymarket (5 minutes et 15 minutes).

## La Strategie : "Last-Minute Endgame"

### Principe

Polymarket propose des paris binaires : "Le BTC sera-t-il au-dessus de $X dans 5 minutes ?".
Chaque fenetre a un **price to beat** (prix Chainlink a l'ouverture). A l'expiration :

- Si BTC >= price to beat → le token **UP** vaut **$1.00**, DOWN vaut $0.00
- Si BTC < price to beat → le token **DOWN** vaut **$1.00**, UP vaut $0.00

**Le bot achete le token gagnant dans les 15 a 120 dernieres secondes**, quand le resultat est quasi-certain mais que le prix n'a pas encore converge vers $1.00.

### Pourquoi ca marche

Quand il reste 30 secondes et que le BTC est 0.25% au-dessus du price to beat, la probabilite que ca s'inverse est inferieure a 1%. Pourtant le token UP se trade encore a ~$0.85-$0.95, laissant un **edge de 5 a 15%**.

### Le modele mathematique

```
P_true = 1 - Phi(-|delta| / (sigma * sqrt(T)))

delta = (BTC_actuel - price_to_beat) / price_to_beat
sigma = volatilite realisee 1-minute du BTC (rolling 30 min)
T     = temps restant en minutes
Phi   = fonction de repartition normale

Edge  = P_true - P_market - fees
```

Le bot achete uniquement quand `edge > 8%` et que tous les filtres de securite passent.

### Position sizing

Quarter Kelly avec cap a 5% du capital et 30% de la profondeur du carnet :

```
size = min(0.25 * Kelly * capital, 0.05 * capital, 0.30 * depth)
```

### Filtres de protection

| Filtre | Seuil | Description |
|--------|-------|-------------|
| Temps | 15s - 120s | Fenetre d'entree optimale |
| Delta minimum | 0.12% | Eviter les paris a 50/50 |
| Edge minimum | 8% | Marge suffisante apres fees |
| Coherence sources | 0.08% | Binance et Chainlink concordent |
| Profondeur carnet | > $2,000 | Assurer l'execution |
| Volatilite max | 0.15% | Eviter les regimes extremes |
| Pertes consecutives | < 3 | Circuit breaker |
| Drawdown journalier | < 15% | Stop du jour |
| Positions ouvertes | < 3 | Limiter l'exposition |
| 1 par marche | Unique | Pas de double position |

### Profil attendu

| Metrique | Conservateur | Optimiste |
|----------|-------------|-----------|
| Win rate | 92% | 97% |
| Profit moyen/trade gagnant | 15% | 25% |
| Perte par trade perdant | 100% | 100% |
| Trades/jour | 5 | 20 |
| Rendement mensuel | 15% | 40% |

---

## Installation

### Prerequis

- Python 3.12+
- Connexion internet stable (WebSocket Binance + API Polymarket)

### Setup

```bash
# Cloner le repo
git clone <repo-url>
cd polymarket-btc-sniper

# Creer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dependances
pip install -r requirements.txt
```

### Configuration

Copier le fichier d'exemple et editer :

```bash
cp .env.example .env
```

Editer `.env` avec vos parametres :

```env
# Mode de trading : "paper" (simulation) ou "live" (reel)
TRADING_MODE=paper

# Capital initial en paper trading (USDC)
PAPER_INITIAL_BALANCE=10000

# Polymarket API (requis pour le mode live uniquement)
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_API_PASSPHRASE=
POLYMARKET_WALLET_ADDRESS=
POLYMARKET_PRIVATE_KEY=

# Parametres de signal (valeurs par defaut recommandees)
DELTA_MIN=0.0012          # delta minimum (0.12%)
EDGE_MIN=0.08             # edge minimum (8%)
TIME_MIN_SECONDS=15       # temps min avant expiration
TIME_MAX_SECONDS=120      # temps max avant expiration
VOLATILITY_MAX=0.0015     # volatilite max (0.15%)
SOURCE_COHERENCE_MAX=0.0008  # divergence max Binance/Chainlink

# Parametres de risque
KELLY_FRACTION=0.25       # fraction de Kelly (1/4)
MAX_POSITION_PCT=0.05     # max 5% du capital par trade
MAX_DAILY_DRAWDOWN=0.15   # stop si -15% du capital
MAX_CONSECUTIVE_LOSSES=3  # stop apres 3 pertes d'affilee
MAX_OPEN_POSITIONS=3      # max 3 trades simultanement

# Dashboard
DASHBOARD_PORT=8080
DASHBOARD_HOST=0.0.0.0

# Base de donnees
DB_PATH=data/trades.db

# Logging
LOG_LEVEL=INFO
```

---

## Lancement

### Paper trading (simulation avec donnees live)

```bash
source .venv/bin/activate
python3 main.py
```

Ou avec des options CLI :

```bash
# Changer le mode
python3 main.py --mode paper

# Changer le capital initial
python3 main.py --balance 5000

# Changer le port du dashboard
python3 main.py --port 9090
```

### Live trading (argent reel)

```bash
# 1. Remplir les credentials Polymarket dans .env
# 2. Lancer en mode live
python3 main.py --mode live
```

**Attention** : Le mode live execute de vrais ordres sur Polymarket avec de l'argent reel.

### Dashboard

Une fois le bot lance, ouvrir dans le navigateur :

```
http://localhost:8080
```

Le dashboard affiche en temps reel :
- Prix BTC (Binance + Chainlink)
- Signal actuel (delta, P_true, edge, filtres)
- Portfolio (balance, PnL, win rate)
- Historique des trades (5m/15m, UP/DOWN, WON/LOST)
- Courbes d'equity et PnL journalier
- Statut des feeds (Binance, Chainlink, Polymarket)

---

## Architecture

```
polymarket-btc-sniper/
├── main.py                    # Orchestrateur principal
├── .env                       # Configuration (copier .env.example)
├── requirements.txt           # Dependances Python
├── data/
│   └── trades.db              # Base SQLite (auto-creee)
├── docs/
│   └── STRATEGY.md            # Document de strategie detaille
└── src/
    ├── config.py              # Gestion de la configuration
    ├── feeds/
    │   ├── binance.py         # WebSocket Binance BTC/USDT
    │   ├── chainlink.py       # Feed prix Chainlink (proxy Binance REST)
    │   └── polymarket.py      # Discovery marches + orderbook + price to beat
    ├── engine/
    │   └── signal.py          # Moteur de signal (P_true, edge, filtres)
    ├── trading/
    │   ├── paper.py           # Trading simule
    │   ├── live.py            # Trading reel (CLOB API)
    │   └── portfolio.py       # Gestion du portefeuille et PnL
    ├── dashboard/
    │   ├── app.py             # API FastAPI + WebSocket
    │   └── templates/
    │       └── index.html     # Interface web
    └── utils/
        ├── db.py              # Couche SQLite (trades, snapshots, signaux)
        ├── math_utils.py      # Probabilite, Kelly, volatilite
        └── logger.py          # Logging configure
```

## Flux du systeme

```
Binance WebSocket ──→ Prix BTC tick-by-tick ──→ Volatilite rolling (sigma)
                                                       │
Chainlink Feed ────→ Prix BTC (poll /3s) ──────────────┤
                                                       │
Polymarket Feed ───→ Discovery marches (slug) ─────────┤
                  ├→ Price to beat (past-results API)   │
                  └→ Orderbook (CLOB /book)             │
                                                       ▼
                                              Signal Engine (/2s)
                                              ├── delta, sigma, P_true
                                              ├── edge = P_true - P_market - fee
                                              ├── 10 filtres de securite
                                              └── Kelly sizing
                                                       │
                                              si edge > 8% et filtres OK
                                                       ▼
                                              Paper/Live Trader → Execute
                                                       │
                                              Resolution (/3s)
                                              BTC >= ref → UP gagne
                                              BTC <  ref → DOWN gagne
                                                       │
                                                       ▼
                                              Portfolio → PnL → Dashboard
```

---

## Commandes utiles

```bash
# Voir les logs en temps reel
tail -f logs/btc_sniper.log

# Reset la base de donnees (repart de zero)
rm -f data/trades.db

# Lancer avec plus de logs
LOG_LEVEL=DEBUG python3 main.py
```

---

## Phases de deploiement recommandees

### Phase 1 — Observer (1-2 semaines)
Lancer en paper trading, ne rien toucher. Observer les signaux, le win rate, le comportement du marche. Verifier que le bot detecte bien les opportunites et que les filtres fonctionnent.

### Phase 2 — Paper trading actif (1-2 semaines)
Analyser les resultats : win rate > 90% ? edge moyen > 8% ? drawdown acceptable ? Ajuster les parametres si necessaire (`DELTA_MIN`, `EDGE_MIN`, `TIME_MIN_SECONDS`).

### Phase 3 — Live trading (progressif)
Commencer avec $200-500. Max $50 par trade. Comparer les resultats live vs paper. Scaler progressivement si les resultats sont coherents.

---

## Avertissement

Ce bot est un outil experimental. Le trading comporte des risques de perte en capital. Les performances passees (paper ou live) ne garantissent pas les resultats futurs. Utilisez a vos propres risques.
