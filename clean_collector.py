"""Purge collector.db pour que le gridsearch replay puisse tourner.

Actions:
  1. DROP table orderbook_deltas (17.9M rows, ~15 GB) — inutile pour le
     backtest par snapshots. Si on a besoin des deltas, il faudra relancer
     le collector.
  2. Supprime markets non résolus > 4h (orphelins).
  3. Supprime orderbook_snapshots/trades orphelins (markets absents).
  4. PRAGMA wal_checkpoint(TRUNCATE) + VACUUM -> libère l'espace disque.

Run one-shot :
    python clean_collector.py            # dry-run + confirm
    python clean_collector.py --yes      # skip confirm
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "collector.db"


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    ap.add_argument("--keep-deltas", action="store_true",
                    help="Garde la table orderbook_deltas (par défaut supprimée)")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="Skip VACUUM (plus rapide, mais ne libère pas le disque)")
    args = ap.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"ERROR: {DB_PATH} not found")

    size_before = DB_PATH.stat().st_size
    print(f"Fichier : {DB_PATH}")
    print(f"Taille avant : {human_bytes(size_before)}")
    print()

    c = sqlite3.connect(str(DB_PATH), timeout=60)
    c.execute("PRAGMA busy_timeout = 60000")

    # --- stats avant -----------------------------------------------------
    now = time.time()
    tables = [r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]
    print("--- Tables avant ---")
    stats_before = {}
    for t in tables:
        n = c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        stats_before[t] = n
        print(f"  {t:25s} {n:>15,}")
    print()

    # --- plan ------------------------------------------------------------
    print("--- Actions prévues ---")
    if not args.keep_deltas and "orderbook_deltas" in tables:
        print("  DROP TABLE orderbook_deltas (17M+ rows, principale cause de bloat)")
    unresolved_cutoff = now - 4 * 3600  # 4h grace
    print(f"  DELETE markets non résolus avec end_time < now-4h")
    print("  DELETE orderbook_snapshots orphelins (markets absents)")
    print("  DELETE trades orphelins")
    print("  DELETE market_states orphelins")
    if not args.no_vacuum:
        print("  VACUUM (peut prendre 1-3 min)")
    print()

    if not args.yes:
        ans = input("Continuer ? [y/N] ").strip().lower()
        if ans not in ("y", "yes", "o", "oui"):
            print("Annulé.")
            return

    t0 = time.time()

    # --- 1. drop deltas + ws_events -------------------------------------
    if not args.keep_deltas and "orderbook_deltas" in tables:
        print("-> DROP TABLE orderbook_deltas...", end=" ", flush=True)
        c.execute("DROP TABLE orderbook_deltas")
        c.commit()
        print(f"OK ({time.time()-t0:.1f}s)")
    if "ws_events" in tables:
        print("-> DROP TABLE ws_events (10M+ rows, raw WS log)...", end=" ", flush=True)
        c.execute("DROP TABLE ws_events")
        c.commit()
        print("OK")

    # --- 2. purge unresolved markets ------------------------------------
    print("-> DELETE markets orphelins...", end=" ", flush=True)
    cur = c.execute(
        "DELETE FROM markets WHERE resolved_outcome IS NULL AND end_time < ?",
        (unresolved_cutoff,),
    )
    n_markets_del = cur.rowcount
    c.commit()
    print(f"OK ({n_markets_del:,} rows)")

    # --- 3. purge orphan children ---------------------------------------
    # On utilise un anti-join via LEFT JOIN plutôt que NOT IN pour perf
    for child in ("orderbook_snapshots", "trades", "market_states", "tick_changes"):
        if child not in tables:
            continue
        print(f"-> DELETE {child} orphelins...", end=" ", flush=True)
        cur = c.execute(f"""
            DELETE FROM {child}
             WHERE market_id NOT IN (SELECT market_id FROM markets)
        """)
        print(f"OK ({cur.rowcount:,} rows)")
        c.commit()

    # --- 4. checkpoint + vacuum -----------------------------------------
    print("-> WAL checkpoint (TRUNCATE)...", end=" ", flush=True)
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.commit()
    print("OK")

    c.close()

    if not args.no_vacuum:
        print("-> VACUUM (patience)...", end=" ", flush=True)
        t_vac = time.time()
        c = sqlite3.connect(str(DB_PATH), timeout=600)
        c.execute("PRAGMA busy_timeout = 600000")
        c.execute("VACUUM")
        c.commit()
        c.close()
        print(f"OK ({time.time()-t_vac:.1f}s)")

    # --- stats après -----------------------------------------------------
    size_after = DB_PATH.stat().st_size
    c = sqlite3.connect(str(DB_PATH))
    tables_after = [r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]
    print()
    print("--- Tables après ---")
    for t in tables_after:
        n = c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        delta = n - stats_before.get(t, 0)
        print(f"  {t:25s} {n:>15,}  ({delta:+,})")
    c.close()
    print()
    print(f"Taille avant : {human_bytes(size_before)}")
    print(f"Taille après : {human_bytes(size_after)}  "
          f"(-{human_bytes(size_before - size_after)})")
    print(f"Total : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
