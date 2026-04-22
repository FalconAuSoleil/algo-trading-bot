"""Gridsearch par replay sur les vraies données collectées.

Principe :
  1. Charge tous les marchés RÉSOLUS de collector.db.
  2. Pour chaque marché, charge sa timeline d'orderbook_snapshots.
  3. Simule le parcours d'un pari pour chaque combinaison (entry x exit) :
       - Entry  : à T-entry_trem, achète le côté dont le best_ask in [0.55, 0.82]
                  (évite les paris "sûrs" à 90¢ et les "impossibles" à 10¢).
                  Si les deux côtés sont dans la fenêtre, prend le moins cher.
                  Sinon skip le marché.
       - Exit   : walk forward les snapshots et teste (dans l'ordre) :
                   * Mode F (force_sell) : bid < force_sell_price ET t_rem < force_sell_trem
                   * Mode E (late stage) : bid < entry x late_ratio ET
                                           (t_rem < late_trem OU elapsed > late_elapsed_pct)
                                           avec perte >= late_min_loss_pct
                  Sinon hold jusqu'à expiration.
  4. PnL par trade :
       - Vendu (force/late) : shares x exit_bid − shares x entry_ask
       - Held win          : shares x 1.00 − shares x entry_ask  (fees ignorés)
       - Held loss         : − shares x entry_ask
     Le size nominal est fixé à $1 (on compare les stratégies en ROI relatif).

  5. Agrège par (entry x exit) : n_trades, win_rate, roi, pnl_total,
     n_force_sells, loss_avoided_avg, etc.

  6. Écrit `reports/replay_grid_<date>.json` + un TOP 20 lisible en console.

Usage :
    python gridsearch_replay.py                       # grid par défaut
    python gridsearch_replay.py --interval 5m         # BTC 5m seulement
    python gridsearch_replay.py --asset BTC           # filtre asset
    python gridsearch_replay.py --small               # grid réduit (debug)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from itertools import product
from pathlib import Path
from typing import Optional

BASE = Path(__file__).parent
DB_PATH = BASE / "data" / "collector.db"
REPORTS_DIR = BASE / "reports"


# ─────────────────────────────────────────────────────────────────────────────
# GRID : modifiable en haut du fichier pour itérer rapidement
# ─────────────────────────────────────────────────────────────────────────────
GRID = {
    # ── Entry ─────────────────────────────────────────────
    # Seconds avant expiration où on entre. 5m: 60-180. 15m: 120-300.
    "entry_trem":        [60, 120, 180, 240],
    # Fenêtre de prix d'entrée acceptée (best_ask du côté choisi).
    "entry_min_ask":     [0.55, 0.60],
    "entry_max_ask":     [0.80, 0.85],
    # ── Exit Mode F (force sell brutal) ───────────────────
    "force_sell_price":  [0.25, 0.30, 0.35, 0.40],
    "force_sell_trem":   [60, 90, 120],
    # ── Exit Mode E (late stage losing) ───────────────────
    "late_ratio":        [0.85, 0.92, 1.00],   # 1.00 = désactivé
    "late_trem":         [120, 180],
    "late_min_loss_pct": [0.00, 0.08],
}  # 4*2*2*4*3*3*2*2 = 1152 combos

SMALL_GRID = {
    "entry_trem":        [120, 180],
    "entry_min_ask":     [0.55],
    "entry_max_ask":     [0.82],
    "force_sell_price":  [0.30, 0.40],
    "force_sell_trem":   [90, 120],
    "late_ratio":        [0.92, 1.00],
    "late_trem":         [120],
    "late_min_loss_pct": [0.08],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Snapshot:
    ts: float
    bid_yes: float
    ask_yes: float
    bid_no: float
    ask_no: float


@dataclass
class MarketRecord:
    market_id: str
    slug: str
    asset: str
    interval_sec: int
    start_time: float
    end_time: float
    outcome: str        # "up" or "down"
    snapshots: list[Snapshot] = field(default_factory=list)


def load_markets(
    interval_filter: Optional[int] = None,
    asset_filter: Optional[str] = None,
) -> list[MarketRecord]:
    """Charge tous les marchés résolus avec leurs snapshots."""
    if not DB_PATH.exists():
        sys.exit(f"ERROR: {DB_PATH} not found")

    c = sqlite3.connect(str(DB_PATH), timeout=60)
    c.execute("PRAGMA busy_timeout = 60000")

    q = """
        SELECT market_id, slug, asset, interval_sec, start_time, end_time,
               resolved_outcome
          FROM markets
         WHERE resolved_outcome IN ('up', 'down')
    """
    params = []
    if interval_filter:
        q += " AND interval_sec = ?"
        params.append(interval_filter)
    if asset_filter:
        q += " AND asset = ?"
        params.append(asset_filter.upper())

    rows = c.execute(q, params).fetchall()
    print(f"[replay] {len(rows)} marchés résolus à charger...")

    markets: list[MarketRecord] = []
    for mid, slug, asset, isec, ts_start, ts_end, out in rows:
        snaps_rows = c.execute(
            """SELECT timestamp, best_bid_yes, best_ask_yes,
                      best_bid_no, best_ask_no
                 FROM orderbook_snapshots
                WHERE market_id = ?
                ORDER BY timestamp ASC""",
            (mid,),
        ).fetchall()
        snaps = [
            Snapshot(ts=float(r[0]),
                     bid_yes=float(r[1] or 0),
                     ask_yes=float(r[2] or 0),
                     bid_no=float(r[3] or 0),
                     ask_no=float(r[4] or 0))
            for r in snaps_rows
        ]
        if len(snaps) < 3:
            continue  # pas assez de data pour replay
        markets.append(MarketRecord(
            market_id=mid, slug=slug, asset=asset,
            interval_sec=int(isec),
            start_time=float(ts_start), end_time=float(ts_end),
            outcome=out.lower(),
            snapshots=snaps,
        ))
    c.close()
    print(f"[replay] {len(markets)} marchés avec >=3 snapshots utilisables.")
    return markets


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TradeSim:
    market_id: str
    asset: str
    interval_sec: int
    side: str           # "YES" or "NO"
    entry_ask: float
    entry_time: float
    exit_price: float   # bid au moment de la vente OU 1.0/0.0 à resolution
    exit_time: float
    exit_reason: str    # "held_win", "held_loss", "force_sell", "late_stage"
    loss_avoided: float = 0.0  # différence vs hold-to-expiry (positif si on a économisé)
    won: bool = False


def simulate_market(
    market: MarketRecord,
    entry_trem: float,
    entry_min_ask: float,
    entry_max_ask: float,
    force_sell_price: float,
    force_sell_trem: float,
    late_ratio: float,
    late_trem: float,
    late_min_loss_pct: float,
) -> Optional[TradeSim]:
    """Simule un trade sur ce marché avec les params donnés.

    Retourne None si le marché ne match pas les critères d'entrée.
    """
    entry_ts_target = market.end_time - entry_trem

    # Trouver le snapshot le plus proche de entry_ts_target (>=)
    entry_snap = None
    for s in market.snapshots:
        if s.ts >= entry_ts_target:
            entry_snap = s
            break
    if entry_snap is None:
        return None

    # Choix du côté : prendre le moins cher DANS la fenêtre [min, max]
    candidates = []
    if entry_min_ask <= entry_snap.ask_yes <= entry_max_ask:
        candidates.append(("YES", entry_snap.ask_yes))
    if entry_min_ask <= entry_snap.ask_no <= entry_max_ask:
        candidates.append(("NO", entry_snap.ask_no))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    side, entry_ask = candidates[0]

    # Walk forward, test exits
    exit_price = None
    exit_ts = None
    exit_reason = None

    for s in market.snapshots:
        if s.ts <= entry_snap.ts:
            continue
        t_rem = market.end_time - s.ts
        if t_rem <= 0:
            break
        bid = s.bid_yes if side == "YES" else s.bid_no
        if bid <= 0:
            continue
        elapsed_pct = 1.0 - (t_rem / market.interval_sec)
        loss_pct = 1.0 - (bid / entry_ask) if entry_ask > 0 else 0.0

        # Mode F : force sell brutal
        if bid < force_sell_price and t_rem < force_sell_trem:
            exit_price = bid
            exit_ts = s.ts
            exit_reason = "force_sell"
            break
        # Mode E : late stage
        if (late_ratio < 1.0
                and bid < entry_ask * late_ratio
                and t_rem < late_trem
                and loss_pct >= late_min_loss_pct):
            exit_price = bid
            exit_ts = s.ts
            exit_reason = "late_stage"
            break

    won_held = (side == "YES" and market.outcome == "up") or \
               (side == "NO" and market.outcome == "down")

    if exit_price is None:
        # Held to expiry
        exit_price = 1.0 if won_held else 0.0
        exit_ts = market.end_time
        exit_reason = "held_win" if won_held else "held_loss"
        loss_avoided = 0.0
    else:
        # Gain évité = ce qu'on aurait perdu en tenant si on était perdant ;
        # coût évitable = ce qu'on a manqué si on était gagnant.
        would_be = 1.0 if won_held else 0.0
        loss_avoided = exit_price - would_be   # >0 si on a économisé une perte

    won = (exit_reason in ("held_win",)) or (exit_price > entry_ask)
    return TradeSim(
        market_id=market.market_id,
        asset=market.asset,
        interval_sec=market.interval_sec,
        side=side,
        entry_ask=entry_ask,
        entry_time=entry_snap.ts,
        exit_price=exit_price,
        exit_time=exit_ts,
        exit_reason=exit_reason,
        loss_avoided=loss_avoided,
        won=won,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grid run
# ─────────────────────────────────────────────────────────────────────────────
def run_combo(markets: list[MarketRecord], params: dict) -> dict:
    trades: list[TradeSim] = []
    for m in markets:
        t = simulate_market(m, **params)
        if t is not None:
            trades.append(t)

    if not trades:
        return {"params": params, "trades": 0}

    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    pnl_total = sum(t.exit_price - t.entry_ask for t in trades)  # size nominal = 1

    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    # Loss avoided : moyen sur les exits early
    early_exits = [t for t in trades if t.exit_reason in ("force_sell", "late_stage")]
    loss_avoided_avg = (
        sum(t.loss_avoided for t in early_exits) / len(early_exits)
        if early_exits else 0.0
    )
    # Coût d'erreurs d'exit (on a vendu à 20¢ puis le marché résout vers nous)
    early_exit_mistakes = sum(1 for t in early_exits if t.loss_avoided < 0)

    return {
        "params": params,
        "trades": n,
        "wins": wins,
        "win_rate": wins / n,
        "pnl_total": round(pnl_total, 3),
        "roi_pct": round(pnl_total / n * 100, 2),
        "exit_reasons": by_reason,
        "n_early_exits": len(early_exits),
        "early_exit_mistakes": early_exit_mistakes,
        "loss_avoided_avg": round(loss_avoided_avg, 3),
    }


def score(r: dict) -> float:
    """Priorise ROI élevé + volume de trades décent."""
    if r["trades"] < 10:
        return -999
    return r["roi_pct"] * min(1.0, r["trades"] / 50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", choices=["5m", "15m", "both"], default="both")
    ap.add_argument("--asset", default=None, help="BTC/ETH/SOL/XRP (default: tous)")
    ap.add_argument("--small", action="store_true",
                    help="Grid réduit (~24 combos) pour debug rapide")
    ap.add_argument("--top", type=int, default=20, help="Top N affiché en console")
    args = ap.parse_args()

    grid = SMALL_GRID if args.small else GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*grid.values())]
    print(f"[replay] Grid = {len(combos)} combos")

    interval_sec = {"5m": 300, "15m": 900, "both": None}[args.interval]
    markets = load_markets(interval_filter=interval_sec, asset_filter=args.asset)
    if not markets:
        sys.exit("Aucun marché résolu — vérifier le collector")

    # Pré-filtre : si param entry_trem > interval_sec, skip combo
    # (sinon on génère des combos inutiles pour les 5m)
    t0 = time.time()
    results = []
    for i, params in enumerate(combos, 1):
        r = run_combo(markets, params)
        results.append(r)
        if i % 50 == 0 or i == len(combos):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(combos) - i)
            print(f"  {i}/{len(combos)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    # Tri par score
    results.sort(key=score, reverse=True)

    # Output JSON
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"replay_grid_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_markets": len(markets),
        "n_combos": len(combos),
        "interval_filter": args.interval,
        "asset_filter": args.asset,
        "grid": {k: list(v) for k, v in grid.items()},
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[replay] JSON: {out_path}")

    # Top N console
    print()
    print(f"=== TOP {args.top} ===")
    print(
        f"{'roi%':>7} {'trd':>4} {'win%':>5} {'forc':>4} {'late':>4} "
        f"{'held_w':>6} {'held_l':>6} {'avoid':>6}  params"
    )
    for r in results[:args.top]:
        if r["trades"] == 0:
            continue
        er = r["exit_reasons"]
        p = r["params"]
        short_params = (
            f"eT{p['entry_trem']} ask[{p['entry_min_ask']:.2f}-{p['entry_max_ask']:.2f}] "
            f"F@{p['force_sell_price']:.2f}/{p['force_sell_trem']}s "
            f"L@{p['late_ratio']:.2f}/{p['late_trem']}s/"
            f"L>{int(p['late_min_loss_pct']*100)}%"
        )
        print(
            f"{r['roi_pct']:>7.2f} {r['trades']:>4} {r['win_rate']*100:>5.1f} "
            f"{er.get('force_sell',0):>4} {er.get('late_stage',0):>4} "
            f"{er.get('held_win',0):>6} {er.get('held_loss',0):>6} "
            f"{r['loss_avoided_avg']:>6.2f}  {short_params}"
        )

    # Summary stats globaux
    all_trades = sum(r["trades"] for r in results)
    print()
    print(f"[replay] Total combo runs    : {len(results)}")
    print(f"[replay] Total trade samples : {all_trades:,}")
    print(f"[replay] Best score          : {results[0].get('roi_pct', 0):.2f}%")
    print(f"[replay] Report -> {out_path}")
    print()
    print("Pour l'analyse Claude : cat reports/replay_grid_*.json | claude")


if __name__ == "__main__":
    main()
