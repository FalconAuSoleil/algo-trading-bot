"""Weekly performance report + Claude prompt generator — weekly_report.py

Reads the last 7 days of trades from trades.db, loads the latest grid-search
results, and writes two files to reports/:

  1. weekly_YYYY-MM-DD.json          — raw stats (for auditing)
  2. claude_prompt_YYYY-MM-DD.md     — ready-to-send prompt for Claude

Usage:
    # Step 1 (run first, takes ~5 min):
    python optimize.py

    # Step 2 (run after, takes ~10s):
    python weekly_report.py

    # Or override which grid-search file to use:
    python weekly_report.py --grid reports/grid_search_2026-04-14.json

Then open reports/claude_prompt_YYYY-MM-DD.md and send it to Claude.
Claude will respond with specific env var values to set for the coming week.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "data" / "trades.db"


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_trades(db_path: Path, days: int = 7) -> list[dict]:
    if not db_path.exists():
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    conn   = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT id, timestamp, slug, side, entry_price, size_usd, delta, sigma,
               p_true, p_market, edge, time_remaining_sec, oracle_age_sec,
               outcome, pnl, mode, resolved_at,
               COALESCE(exit_reason, 'normal') as exit_reason
        FROM trades
        WHERE timestamp >= ? AND outcome != 'pending'
        ORDER BY timestamp
        """,
        (cutoff,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _asset(slug: str) -> str:
    if not slug:
        return "unknown"
    for a in ("btc", "eth", "sol", "xrp"):
        if a in slug.lower():
            return a.upper()
    return "unknown"


def _interval(slug: str) -> str:
    if "15m" in slug:
        return "15m"
    if "5m" in slug:
        return "5m"
    return "?"


def _hour_paris(ts: float) -> int:
    """Return Paris local hour (0-23) for a UTC timestamp."""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ZoneInfo("Europe/Paris"))
        return dt.hour
    except Exception:
        return datetime.utcfromtimestamp(ts).hour


def _weekday_name(ts: float) -> str:
    names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    dt = datetime.utcfromtimestamp(ts)
    return names[dt.weekday()]


# ── Stats computation ─────────────────────────────────────────────────────────

def _compute_stats(trades: list[dict]) -> dict:
    if not trades:
        return {}

    # Defensive None coercion — trades.db may contain NULL pnl/size/edge
    # for rare edge cases (crash during resolution, partial fills, etc.)
    def _f(x): return float(x) if x is not None else 0.0

    total   = len(trades)
    won     = sum(1 for t in trades if t["outcome"] == "won")
    lost    = sum(1 for t in trades if t["outcome"] == "lost")
    early   = sum(1 for t in trades if t["outcome"] == "early_exit")
    total_pnl    = sum(_f(t["pnl"])      for t in trades)
    total_staked = sum(_f(t["size_usd"]) for t in trades)

    # by hour bucket (Paris time)
    by_hour: dict[int, dict] = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        h = _hour_paris(t["timestamp"])
        by_hour[h]["bets"] += 1
        by_hour[h]["wins"] += int(t["outcome"] == "won")
        by_hour[h]["pnl"]  += _f(t["pnl"])

    # by weekday
    by_day: dict[str, dict] = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        d = _weekday_name(t["timestamp"])
        by_day[d]["bets"] += 1
        by_day[d]["wins"] += int(t["outcome"] == "won")
        by_day[d]["pnl"]  += _f(t["pnl"])

    # by asset
    by_asset: dict[str, dict] = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        a = _asset(t.get("slug", ""))
        by_asset[a]["bets"] += 1
        by_asset[a]["wins"] += int(t["outcome"] == "won")
        by_asset[a]["pnl"]  += _f(t["pnl"])

    # by interval
    by_interval: dict[str, dict] = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        i = _interval(t.get("slug", ""))
        by_interval[i]["bets"] += 1
        by_interval[i]["wins"] += int(t["outcome"] == "won")
        by_interval[i]["pnl"]  += _f(t["pnl"])

    # worst and best hours (min 3 bets)
    def _wr(d): return d["wins"] / d["bets"] if d["bets"] else 0
    sorted_hours = sorted(
        [(h, d) for h, d in by_hour.items() if d["bets"] >= 3],
        key=lambda x: _wr(x[1])
    )
    worst_hours = [(h, d) for h, d in sorted_hours[:3]]
    best_hours  = [(h, d) for h, d in sorted_hours[-3:]][::-1]

    return {
        "period_days":   7,
        "total_trades":  total,
        "won":           won,
        "lost":          lost,
        "early_exit":    early,
        "win_rate":      won / total if total else 0,
        "total_pnl":     round(total_pnl, 2),
        "total_staked":  round(total_staked, 2),
        "roi":           total_pnl / total_staked if total_staked else 0,
        "avg_edge":      sum(_f(t["edge"]) for t in trades) / total if total else 0,
        "avg_size":      total_staked / total if total else 0,
        "by_hour":       {str(h): {"bets": d["bets"], "wins": d["wins"],
                                    "win_rate": round(_wr(d), 3),
                                    "pnl": round(d["pnl"], 2)}
                          for h, d in sorted(by_hour.items())},
        "by_day":        {d: {"bets": v["bets"], "wins": v["wins"],
                               "win_rate": round(_wr(v), 3),
                               "pnl": round(v["pnl"], 2)}
                          for d, v in by_day.items()},
        "by_asset":      {a: {"bets": v["bets"], "wins": v["wins"],
                               "win_rate": round(_wr(v), 3),
                               "pnl": round(v["pnl"], 2)}
                          for a, v in by_asset.items()},
        "by_interval":   {i: {"bets": v["bets"], "wins": v["wins"],
                               "win_rate": round(_wr(v), 3),
                               "pnl": round(v["pnl"], 2)}
                          for i, v in by_interval.items()},
        "worst_hours":   [(h, round(_wr(d), 2), d["bets"]) for h, d in worst_hours],
        "best_hours":    [(h, round(_wr(d), 2), d["bets"]) for h, d in best_hours],
    }


# ── Current .env reader ───────────────────────────────────────────────────────

_TRACKED_PARAMS = [
    # ── Global (ETH/SOL/XRP + BTC fallback) — ne pas changer sans validation ──
    "EDGE_MIN", "EDGE_MAX", "OFFPEAK_EDGE_MULT", "OFFPEAK_SIZING_MULT",
    "STABILITY_MIN_SAMPLES", "STABILITY_MIN_RATIO", "STABILITY_EDGE_CV_MAX",
    "VOLATILITY_MAX", "MIN_TRUE_PROB", "MIN_MARKET_PROB_SIDE",
    "MAX_MARKET_PROB_SIDE", "MOMENTUM_SIZE_MULT", "MEANREV_SIZE_MULT",
    "TIME_MIN_5M", "TIME_MAX_5M", "TIME_MIN_15M", "TIME_MAX_15M",
    # ── BTC-specific overrides (grid search output) — safe à modifier ─────────
    "BTC_EDGE_MIN", "BTC_OFFPEAK_EDGE_MULT",
    "BTC_STABILITY_MIN_SAMPLES", "BTC_VOLATILITY_MAX",
    # ── Risk management — NE JAMAIS TOUCHER ──────────────────────────────────
    "KELLY_FRACTION", "MAX_BET_FRACTION", "MAX_CONSECUTIVE_LOSSES",
    "PEAK_HOURS_ENABLED", "CB_BLOCK_SATURDAY",
]

def _read_env_params() -> dict[str, str]:
    env_path = BASE_DIR / ".env"
    params: dict[str, str] = {}
    if not env_path.exists():
        return params
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            k = k.strip()
            if k in _TRACKED_PARAMS:
                params[k] = v.strip().split("#")[0].strip()
    return params


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(stats: dict, grid_results: dict | None, env_params: dict) -> str:
    today = str(date.today())

    def _pct(v): return f"{v*100:.1f}%"
    def _sgn(v): return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
    # Escape pipes & newlines so values never break markdown table formatting
    def _md(v): return str(v).replace("|", "\\|").replace("\n", " ")

    lines = [
        f"# RefundEpitaDebt — Weekly Parameter Optimization",
        f"**Date:** {today}",
        "",
        "---",
        "",
        "## Contexte marché (règles fixes, ne pas modifier)",
        "",
        "- **Samedi** : marché mort, spread énorme → bot bloqué",
        "- **03h30–05h00** (Paris) : zone morte, quasi zéro activité → bot bloqué",
        "- **00h45–01h30** (Paris) : fermeture des marchés Polymarket, volatilité "
        "intéressante → trading autorisé (mode off-peak)",
        "- **Lundi–Mercredi 14h–01h** : peak hours, meilleure qualité de signal",
        "- **Jeudi–Vendredi, Dimanche** : activité modérée, bots conservateurs",
        "",
        "---",
        "",
        "## Performance de la semaine écoulée",
        "",
    ]

    if not stats:
        lines += ["*(Aucun trade résolu dans la base de données.)*", ""]
    else:
        wr  = stats["win_rate"]
        pnl = stats["total_pnl"]
        roi = stats["roi"]
        lines += [
            f"| Métrique | Valeur |",
            f"|---|---|",
            f"| Trades résolus | {stats['total_trades']} "
            f"({stats['won']}W / {stats['lost']}L / {stats['early_exit']} exits) |",
            f"| Win rate global | **{_pct(wr)}** |",
            f"| PnL total | **{_sgn(pnl)} USD** |",
            f"| ROI sur capital engagé | {_pct(roi)} |",
            f"| Edge moyen | {_pct(stats['avg_edge'])} |",
            f"| Mise moyenne | {stats['avg_size']:.2f} USD |",
            "",
        ]

        # by hour table
        lines += ["### Win rate par tranche horaire (Paris)", ""]
        lines += ["| Heure | Bets | WR | PnL |", "|---|---|---|---|"]
        for h in range(24):
            d = stats["by_hour"].get(str(h))
            if d and d["bets"] >= 2:
                flag = ""
                if 3 <= h < 5:
                    flag = " 🔴 zone morte"
                elif h == 0 or h == 1:
                    flag = " 🟡 volatile"
                elif 14 <= h or h < 1:
                    flag = " 🟢 peak"
                lines.append(
                    f"| {h:02d}h | {d['bets']} | {_pct(d['win_rate'])} | "
                    f"{_sgn(d['pnl'])} |{flag}"
                )
        lines.append("")

        # by weekday
        lines += ["### Win rate par jour", ""]
        lines += ["| Jour | Bets | WR | PnL |", "|---|---|---|---|"]
        day_order = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
        for d in day_order:
            v = stats["by_day"].get(d)
            if v and v["bets"] >= 2:
                lines.append(
                    f"| {d} | {v['bets']} | {_pct(v['win_rate'])} | {_sgn(v['pnl'])} |"
                )
        lines.append("")

        # by asset
        lines += ["### Win rate par asset", ""]
        lines += ["| Asset | Bets | WR | PnL |", "|---|---|---|---|"]
        for a, v in stats["by_asset"].items():
            lines.append(
                f"| {a} | {v['bets']} | {_pct(v['win_rate'])} | {_sgn(v['pnl'])} |"
            )
        lines.append("")

        # worst hours alert
        if stats["worst_hours"]:
            lines += ["### ⚠ Heures les plus mauvaises (≥3 bets)", ""]
            for h, wr_h, n in stats["worst_hours"]:
                lines.append(f"- **{h:02d}h** — WR {_pct(wr_h)} sur {n} trades")
            lines.append("")

    # ── Grid search results ──────────────────────────────────────────────────
    lines += ["---", "", "## Résultats grid search (108 configurations testées)", ""]

    if not grid_results or not grid_results.get("top"):
        lines += [
            "*(Pas de fichier grid_search disponible — lance d'abord `python optimize.py`)*",
            "",
        ]
    else:
        top = grid_results["top"][:10]
        lines += [
            f"Backtest sur **{grid_results.get('days', '?')} jours** de données BTC.",
            f"Flat bet $100 pour comparaison propre.",
            f"⚠️ **Ces params sont BTC-spécifiques** — ETH/SOL/XRP gardent leurs valeurs globales.",
            "",
            "| # | BTC_EDGE_MIN | BTC_OFFPEAK_MULT | BTC_STAB_SAMPLES | BTC_VOL_MAX | Interval | WR | ROI | Bets |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(top, 1):
            p = r["params"]
            lines.append(
                f"| {i} | {_md(p['BTC_EDGE_MIN'])} | {_md(p['BTC_OFFPEAK_EDGE_MULT'])} | "
                f"{_md(p['BTC_STABILITY_MIN_SAMPLES'])} | {_md(p['BTC_VOLATILITY_MAX'])} | "
                f"{_md(r['interval'])} | **{_pct(r['win_rate'])}** | "
                f"{_pct(r['roi'])} | {r['bets']} |"
            )
        lines.append("")

    # ── Current params ───────────────────────────────────────────────────────
    lines += ["---", "", "## Paramètres actuels (.env)", ""]
    if env_params:
        lines += ["| Paramètre | Valeur actuelle |", "|---|---|"]
        for k, v in env_params.items():
            lines.append(f"| `{_md(k)}` | `{_md(v)}` |")
    else:
        lines += ["*(Fichier .env non trouvé ou vide.)*"]
    lines.append("")

    # ── Request to Claude ────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Demande",
        "",
        "En te basant sur les données ci-dessus :",
        "",
        "1. **Quel set de paramètres BTC recommandes-tu pour la semaine prochaine ?**  ",
        "   Utilise UNIQUEMENT les variables `BTC_EDGE_MIN`, `BTC_OFFPEAK_EDGE_MULT`, "
        "   `BTC_STABILITY_MIN_SAMPLES`, `BTC_VOLATILITY_MAX` (BTC-spécifiques).  ",
        "   **Ne propose PAS de modifier** `EDGE_MIN`, `OFFPEAK_EDGE_MULT` etc. (globaux) — "
        "   ETH/SOL/XRP fonctionnent bien avec leurs valeurs actuelles.  ",
        "   Format : `NOM_PARAM=valeur`. Justifie chaque changement en 1 ligne.",
        "",
        "2. **Y a-t-il des patterns préoccupants dans les stats de performance ?**  ",
        "   Ex : une heure systématiquement perdante, un asset à couper, etc.",
        "",
        "3. **Recommandes-tu de modifier les paramètres du circuit breaker ?**  ",
        "   (heures bloquées, fenêtre volatile, etc.)",
        "",
        "Contraintes à respecter absolument :",
        "- Ne jamais modifier `KELLY_FRACTION`, `MAX_DAILY_DRAWDOWN`, "
        "`MAX_CONSECUTIVE_LOSSES` (paramètres de risk management)",
        "- Ne pas changer plus de 4 paramètres à la fois",
        "- Justifier chaque changement par les données ci-dessus, "
        "pas par une intuition générale",
        "",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate weekly performance report + Claude prompt")
    parser.add_argument("--db",   default=str(DB_PATH), help="Path to trades.db")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--grid", default=None, help="Path to grid_search JSON (auto-detected if omitted)")
    args = parser.parse_args()

    db_path = Path(args.db)
    today   = str(date.today())

    # Load trades
    print(f"[report] Loading trades from {db_path}…")
    trades = _load_trades(db_path, args.days)
    print(f"[report] {len(trades)} resolved trades in last {args.days} days")
    stats = _compute_stats(trades)

    # Load grid search results
    grid_results = None
    grid_path = args.grid
    if not grid_path:
        reports_dir = BASE_DIR / "reports"
        if reports_dir.exists():
            candidates = sorted(reports_dir.glob("grid_search_*.json"), reverse=True)
            if candidates:
                grid_path = str(candidates[0])
    if grid_path and Path(grid_path).exists():
        print(f"[report] Loading grid search results from {grid_path}…")
        grid_results = json.loads(Path(grid_path).read_text())
    else:
        print("[report] No grid search file found — run `python optimize.py` first")

    # Read current params
    env_params = _read_env_params()

    # Build prompt
    prompt = _build_prompt(stats, grid_results, env_params)

    # Write outputs
    out_dir = BASE_DIR / "reports"
    out_dir.mkdir(exist_ok=True)

    json_path   = out_dir / f"weekly_{today}.json"
    prompt_path = out_dir / f"claude_prompt_{today}.md"

    json_path.write_text(json.dumps(
        {"generated": today, "stats": stats, "env_params": env_params},
        indent=2
    ))
    prompt_path.write_text(prompt)

    print(f"\n[report] Files written:")
    print(f"  {json_path}")
    print(f"  {prompt_path}  ← envoie ce fichier à Claude")
    print(f"\n[report] Stats: {stats.get('total_trades', 0)} trades | "
          f"WR {stats.get('win_rate', 0)*100:.1f}% | "
          f"PnL {stats.get('total_pnl', 0):+.2f} USD")


if __name__ == "__main__":
    main()
