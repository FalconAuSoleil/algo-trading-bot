# Stratégie Microstructure Bayésienne — Documentation technique

## Vue d'ensemble

Ce bot exploite les inefficiences microstructurelles des marchés BTC Up/Down sur Polymarket. Chaque marché dure 5 ou 15 minutes et résout selon que le prix Chainlink BTC/USD est au-dessus ou en-dessous d'un prix de référence fixé à l'ouverture.

## Modèle probabiliste

### Prior : delta prix

Le prior est construit à partir de l'écart entre le prix BTC live et le prix de référence :

```
price_delta = (current_price - reference_price) / reference_price
z_prior = MOMENTUM_FACTOR × price_delta + 0.2 × momentum_15s
prob_prior = sigmoid(z_prior)
```

Avec `MOMENTUM_FACTOR = 150`, un écart de +0.1% donne un prior d'environ 0.57 pour UP.

### Update 1 : Chainlink Lag Arbitrage

Chainlink met à jour son prix on-chain avec une période estimée dynamiquement (~27s sur Polygon). Le module estime le temps restant avant le prochain update et calcule un boost proportionnel au gap Binance-Chainlink.

```
proximity = clamp(1 - time_to_next / EDGE_WINDOW, 0, 1)
edge_boost = proximity × price_gap × 1500
logit_score += edge_boost
```

Quand proximity → 1 (update imminent) et price_gap est significatif, le boost peut atteindre ±0.5 en logit, déplaçant la probabilité de manière significative.

### Update 2 : Order Flow Imbalance

Le module OFI combine :
- **OFI net** : (bids_up - asks_up) normalisé, pondéré 60%
- **Depth imbalance** : (depth_up - depth_down) / total, pondéré 40%
- **Momentum OFI** : variation sur les 10 derniers snapshots

Le signal est converti en ajustement logit borné à \[-0.8, +0.8\].

### Filtre 3 : Kyle Quality

λ de Kyle mesure l'information dans le spread. Le quality factor est calculé par rapport à l'historique :

- Spread élevé vs moyenne → pénalité (max 15%)
- Depth élevée vs moyenne → bonus (max 10%)
- Quality ∈ \[0.3, 1.0\] : multiplie le logit_score

Un quality de 0.3 ramène le signal vers 0.5 (incertitude), tandis que 1.0 le laisse intact.

### Scale 4 : Hawkes Regime

Le processus de Hawkes détecte les régimes de forte activité :

```
λ(t) = μ + Σᵢ α × mᵢ × exp(-β × (t - tᵢ))
```

Où les événements sont des changements de mid > 0.5%. Le boost ∈ \[0, 0.3\] amplifie le signal existant sans changer sa direction.

## Filtre de stabilité

Problème résolu : le signal flip-flop entre UP/DOWN toutes les 5-10s à cause du bruit OFI.

Solution : fenêtre glissante de 60s, minimum 3 ticks, exigeant :
- 65% de cohérence directionnelle
- CV de l'edge < 0.80

## Sizing Kelly

Formule Kelly pour option binaire :
```
f* = (p × b - (1-p)) / b
```
Où b = (1/market_price - 1) est l'odds net.

Appliqué à 25% (quarter Kelly) avec cap à 4% de bankroll. Ajustements :
- ×0.7 en régime calme (Hawkes < 1.5μ)
- Bonus jusqu'à +30% si direction_ratio élevé

## Guards de sécurité

| Guard | Seuil | Raison |
|-------|-------|--------|
| Edge minimum | 8% | Éviter le bruit |
| Prob vraie minimum | 58% | Conviction requise |
| Market prob côté | 20-80¢ | Éviter longshots et mauvais odds |
| Liquidité minimum | $15 | Profondeur suffisante |
| Timing 5M | 45s - 270s | Fenêtre de pari optimale |
| Timing 15M | 60s - 850s | Fenêtre de pari étendue |
| Stop-loss | -15% daily | Protection du capital |
| Circuit breaker | 3 pertes | Pause automatique |
