"""Test d'early exit — reproduit le cas reel de la paire BTC 15m qui a perdu.

Scenario :
- Entree a 06:25:04, entry_price=0.63, p_true=0.705, t_rem=296s
- Puis BTC chute : a 06:25:50 (46s plus tard), ask=0.53, p_true_now estime a ~22%
- On verifie que early-exit declencherait bien la vente.

Valide aussi les conditions :
- pas trop tard (t_rem > 30s)
- pas profitable (current_value < size_usd)
- liquidite OK (best_bid > 0.05)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config


def p_brownian(delta: float, t_rem: float, sigma_ps: float) -> float:
    """Copie locale de la formule pour le test."""
    if t_rem <= 0 or sigma_ps <= 0:
        return 1.0 if delta > 0 else 0.0
    from math import erf
    z = delta / (sigma_ps * math.sqrt(t_rem))
    return 0.5 * (1 + erf(z / math.sqrt(2)))


def simulate_early_exit(
    entry_price: float,
    p_true_at_entry: float,
    delta_at_entry: float,
    duration: float,
    elapsed: float,
    ask_now: float,
    delta_now: float,
    sigma: float,
    best_bid: float,
    shares: float,
    size_usd: float,
    side: str = "YES",
) -> tuple[bool, str]:
    """Reproduit la logique de main._check_early_exits."""
    cfg = config.signal
    t_rem = duration - elapsed

    # Too late
    if t_rem < cfg.early_exit_min_t_rem:
        return False, f"t_rem={t_rem:.0f}s < {cfg.early_exit_min_t_rem}s"

    # p_true maintenant
    p_true_now = p_brownian(abs(delta_now), t_rem, sigma)
    if delta_now < 0:
        p_true_now = 1.0 - p_true_now
    # si side==NO on inverse le delta
    if side == "NO":
        # delta_now est deja cote bet direction
        pass

    sell_reason = None

    # Mode A : p_true collapse
    # Mode C : delta flip
    if (
        delta_at_entry != 0
        and delta_now * delta_at_entry < 0
        and elapsed > duration * 0.15
    ):
        sell_reason = "delta_flip"

    mode_a_cond = (
        sell_reason is None
        and elapsed > duration * 0.30
        and p_true_now < cfg.early_exit_p_true_floor
        and (
            p_true_at_entry <= 0
            or p_true_now < p_true_at_entry * (1.0 - cfg.early_exit_p_true_drop_pct)
        )
    )
    if mode_a_cond:
        sell_reason = "p_true_collapse"

    # Mode B : delta erosion
    if sell_reason is None and delta_at_entry != 0:
        delta_entry_abs = abs(delta_at_entry)
        delta_now_abs = abs(delta_now)
        erosion = (
            1.0 - (delta_now_abs / delta_entry_abs)
            if delta_entry_abs > 0 else 0.0
        )
        elapsed_pct = elapsed / duration if duration > 0 else 0.0

        if (
            elapsed_pct >= cfg.early_exit_erosion_min_elapsed_pct
            and erosion >= cfg.early_exit_delta_erosion_pct
            and delta_now_abs < cfg.early_exit_delta_abs_floor
        ):
            sell_reason = "delta_erosion"

    if sell_reason is None:
        return False, f"aucune condition (p_true_now={p_true_now*100:.1f}%)"

    # Shared conditions
    if best_bid < cfg.early_exit_min_bid:
        return False, f"best_bid={best_bid} < {cfg.early_exit_min_bid}"

    current_value = best_bid * shares
    if current_value >= size_usd:
        return False, f"profitable (value ${current_value:.2f} >= size ${size_usd:.2f})"

    return True, sell_reason


def main() -> None:
    print("=" * 70)
    print("  TEST : early exit sur le cas reel du 06:25")
    print("=" * 70)
    print()
    print(f"Config early-exit chargee depuis .env :")
    print(f"  enabled              : {config.signal.early_exit_enabled}")
    print(f"  min_t_rem            : {config.signal.early_exit_min_t_rem}s")
    print(f"  p_true_floor (A)     : {config.signal.early_exit_p_true_floor}")
    print(f"  p_true_drop_pct (A)  : {config.signal.early_exit_p_true_drop_pct}")
    print(f"  delta_erosion (B)    : {config.signal.early_exit_delta_erosion_pct}")
    print(f"  delta_floor (B)      : {config.signal.early_exit_delta_abs_floor}")
    print(f"  elapsed_pct (B)      : {config.signal.early_exit_erosion_min_elapsed_pct}")
    print(f"  min_bid              : {config.signal.early_exit_min_bid}")
    print()

    # Cas reel :
    # - Entree 06:25:04 : entry=$0.63, delta=0.271%, p_true=70.5%, t_rem=296s
    # - Ask a 06:25:50  : 0.53 (delta 0.278%, p_true 72.9%). MAIS l'ask DROP
    #   de 0.63 a 0.53 => le marche price la baisse. Le bot a DOUBLE au lieu
    #   de couper. Early-exit aurait-il coupe a ce moment-la ?

    # Pour early-exit, on regarde si le p_true recalcule tombe sous 35%.
    # Si BTC a chute brutalement, delta peut s'inverser (devenir negatif
    # pour un bet UP). Testons plusieurs scenarios.

    size_usd = 282.34
    shares = size_usd / 0.63  # ~448 shares
    duration = 900.0  # 15m

    scenarios = [
        # (titre, elapsed, delta_now, ask_now, best_bid, sigma)
        ("Scenario A: BTC chute, delta inverse a -0.05% (46s, 5% ecoule)",
         46.0, -0.0005, 0.53, 0.52, 0.005 / math.sqrt(300)),
        ("Scenario A2: delta flip a 150s (17% ecoule) — Mode C actif",
         150.0, -0.0005, 0.45, 0.44, 0.005 / math.sqrt(300)),
        ("Scenario B: delta s'erode mais reste positif +0.10% (tot)",
         46.0, 0.0010, 0.53, 0.52, 0.005 / math.sqrt(300)),
        ("Scenario B2: p_true tombe sous 45% a 35% ecoule",
         315.0, -0.00005, 0.35, 0.34, 0.005 / math.sqrt(300)),
        ("Scenario C: tard dans le market (70% ecoule, delta flip)",
         630.0, -0.0002, 0.15, 0.14, 0.005 / math.sqrt(300)),
        ("Scenario D: delta_erosion pur (92% erode, 75% ecoule)",
         675.0, 0.0002, 0.40, 0.38, 0.005 / math.sqrt(300)),
    ]

    for title, elapsed, delta_now, ask_now, best_bid, sigma in scenarios:
        print(f"--- {title}")
        ok, reason = simulate_early_exit(
            entry_price=0.63,
            p_true_at_entry=0.705,
            delta_at_entry=0.00271,
            duration=duration,
            elapsed=elapsed,
            ask_now=ask_now,
            delta_now=delta_now,
            sigma=sigma,
            best_bid=best_bid,
            shares=shares,
            size_usd=size_usd,
        )
        verdict = "VEND" if ok else "GARDE"
        print(f"  -> {verdict} ({reason})")
        print()


if __name__ == "__main__":
    main()
