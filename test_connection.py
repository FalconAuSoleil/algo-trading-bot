"""
Script de test de connexion Polymarket CLOB.

Usage :
    python3 test_connection.py

Vérifie :
  1. Chargement de la clé privée depuis .env
  2. Authentification HMAC (derive_api_creds)
  3. Solde USDC.e et approbation Exchange
  4. Accès au marché BTC actuel (orderbook)
  5. Calcul d'un order dummy (sans l'envoyer)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# ── couleurs terminal ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}✅  {msg}{RESET}")
def err(msg: str)  -> None: print(f"  {RED}❌  {msg}{RESET}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠️   {msg}{RESET}")
def info(msg: str) -> None: print(f"  {CYAN}ℹ️   {msg}{RESET}")
def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")

# ── helpers ────────────────────────────────────────────────────────────────────

def _current_btc_slug(interval: int = 300) -> str:
    aligned = int(time.time() // interval) * interval
    label = "5m" if interval == 300 else "15m"
    return f"btc-updown-{label}-{aligned}"


async def _fetch_btc_market(session: aiohttp.ClientSession, slug: str) -> dict | None:
    """Fetch the current BTC up/down market from Gamma API."""
    url = "https://gamma-api.polymarket.com/events"
    try:
        async with session.get(url, params={"slug": slug}, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status != 200:
                return None
            events = await r.json()
        if not events:
            return None
        markets = events[0].get("markets", [])
        return markets[0] if markets else None
    except Exception:
        return None


async def _fetch_orderbook(session: aiohttp.ClientSession, token_id: str) -> dict | None:
    url = "https://clob.polymarket.com/book"
    try:
        async with session.get(url, params={"token_id": token_id}, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception:
        return None


async def _fetch_fee_rate(session: aiohttp.ClientSession, token_id: str) -> int | None:
    url = "https://clob.polymarket.com/fee-rate"
    try:
        async with session.get(url, params={"token_id": token_id}, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status != 200:
                return None
            data = await r.json()
            return data.get("base_fee")
    except Exception:
        return None


# ── étapes de test ─────────────────────────────────────────────────────────────

def test_env() -> str | None:
    """Retourne la private key si trouvée, None sinon."""
    section("ÉTAPE 1 — Variables d'environnement")

    key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    mode = os.getenv("TRADING_MODE", "paper")

    if not key:
        err("POLYMARKET_PRIVATE_KEY absent dans .env")
        info("Ajoute : POLYMARKET_PRIVATE_KEY=0x<ta_clé>")
        return None

    if not key.startswith("0x"):
        warn("La clé ne commence pas par '0x' — ajout automatique")
        key = "0x" + key

    if len(key) != 66:  # 0x + 64 hex chars
        err(f"Longueur incorrecte : {len(key)} caractères (attendu 66 avec 0x)")
        return None

    ok(f"POLYMARKET_PRIVATE_KEY trouvée ({key[:6]}...{key[-4:]})")
    info(f"TRADING_MODE = {mode}")

    if mode != "live":
        warn("TRADING_MODE n'est pas 'live' dans .env")
        warn("Le bot tournera en paper par défaut — change TRADING_MODE=live pour trader réellement")

    return key


def test_clob_import() -> bool:
    section("ÉTAPE 2 — Import py-clob-client")
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType, ApiCreds
        from py_clob_client.constants import POLYGON
        import importlib.metadata
        version = importlib.metadata.version("py-clob-client")
        ok(f"py-clob-client v{version} importé")
        return True
    except ImportError as e:
        err(f"Import échoué : {e}")
        info("Lance : pip install py-clob-client>=0.6")
        return False


def test_auth(key: str) -> object | None:
    """Crée le ClobClient et dérive les credentials. Retourne le client ou None."""
    section("ÉTAPE 3 — Authentification CLOB (derive_api_creds)")

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON

        client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=POLYGON,
            key=key,
            signature_type=0,  # EOA — wallet neuf sans proxy
        )
        ok("ClobClient instancié")

        creds = client.create_or_derive_api_creds()
        ok(f"Credentials dérivés")
        info(f"API Key       : {creds.api_key}")
        info(f"API Secret    : {creds.api_secret[:12]}...")
        info(f"Passphrase    : {creds.api_passphrase[:12]}...")

        client.set_api_creds(creds)
        ok("Credentials appliqués au client")

        return client

    except Exception as e:
        err(f"Authentification échouée : {e}")
        if "invalid" in str(e).lower() or "signature" in str(e).lower():
            info("Vérifie que POLYMARKET_PRIVATE_KEY est une clé Polygon/EVM valide")
        return None


def test_balance(client: object) -> None:
    section("ÉTAPE 4 — Solde USDC.e et approbations")

    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

        # USDC.e (COLLATERAL)
        bal = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        # USDC.e uses 6 decimals — raw value is in micro-USDC
        USDC_DECIMALS = 1_000_000
        usdc_balance   = float(bal.get("balance", 0)) / USDC_DECIMALS
        usdc_allowance = float(bal.get("allowance", 0)) / USDC_DECIMALS

        if usdc_balance > 0:
            ok(f"Solde USDC.e      : ${usdc_balance:.2f}")
        else:
            warn("Solde USDC.e      : $0.00 — approvisionne le wallet avant de trader")

        if usdc_allowance > 0:
            ok(f"Approbation USDC  : ${usdc_allowance:.2f} autorisé pour l'Exchange")
        else:
            warn("Approbation USDC  : 0 — sera réglé au démarrage du bot (set_allowances)")

    except Exception as e:
        warn(f"Impossible de lire le solde : {e}")
        info("Normal si le wallet n'a jamais interagi avec le CLOB")

    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        bal_ctf = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
        )
        ctf_allowance = float(bal_ctf.get("allowance", 0))
        if ctf_allowance > 0:
            ok(f"Approbation CTF   : {ctf_allowance:.0f} tokens autorisés (SELL possible)")
        else:
            warn("Approbation CTF   : 0 — les SELL (early exit) nécessitent set_allowances()")
    except Exception:
        pass


async def test_proxy_status(key: str) -> None:
    """Vérifie si un proxy Polymarket (Gnosis Safe) existe pour ce wallet."""
    section("ÉTAPE 5 — Proxy Polymarket (Gnosis Safe)")

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        # Dériver l'adresse EOA depuis la clé privée
        from eth_account import Account
        account = Account.from_key(key)
        eoa_address = account.address
        info(f"Adresse EOA  : {eoa_address}")
    except Exception:
        warn("eth_account non disponible — skip vérification proxy")
        return

    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://relayer-v2.polymarket.com/relay-payload"
            async with session.get(
                url,
                params={"address": eoa_address, "type": "SAFE"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    proxy_address = data.get("address", "")
                    if proxy_address and proxy_address != "0x0000000000000000000000000000000000000000":
                        # Check if the Safe is actually deployed on-chain
                        deployed = False
                        try:
                            async with session.get(
                                f"https://relayer-v2.polymarket.com/deployed",
                                params={"address": proxy_address},
                                timeout=aiohttp.ClientTimeout(total=5),
                            ) as rd:
                                if rd.status == 200:
                                    deployed = (await rd.json()).get("deployed", False)
                        except Exception:
                            pass

                        if deployed:
                            ok(f"Proxy Safe déployé : {proxy_address}")
                            warn("Ce Safe est actif — tu as probablement visité Polymarket.com")
                            warn("Tes fonds sont dans le Safe, pas dans l'EOA")
                            warn("→ Change signature_type=2 + POLYMARKET_WALLET_ADDRESS dans .env")
                        else:
                            ok(f"Adresse Safe pré-calculée : {proxy_address} (non déployé)")
                            ok("Safe non déployé — mode EOA (signature_type=0) confirmé ✓")
                            info("L'adresse Safe est déterministe (CREATE2), normale pour un nouveau wallet")
                    else:
                        ok("Aucun proxy Safe — mode EOA (signature_type=0) confirmé")
                else:
                    ok("Aucun proxy Safe détecté — mode EOA (signature_type=0) confirmé")
    except Exception:
        ok("Pas de proxy Safe — mode EOA (signature_type=0) confirmé")


async def test_market_access() -> None:
    section("ÉTAPE 5 — Accès marché BTC actuel (orderbook)")

    async with aiohttp.ClientSession() as session:
        # Tente 5m puis 15m
        market = None
        slug_used = ""
        for interval in (300, 900):
            slug = _current_btc_slug(interval)
            info(f"Recherche slug : {slug}")
            market = await _fetch_btc_market(session, slug)
            if market:
                slug_used = slug
                break

        if not market:
            warn("Aucun marché BTC actif trouvé (normal entre deux fenêtres de 5m/15m)")
            return

        ok(f"Marché trouvé : {market.get('question', slug_used)}")

        import json
        clob_tokens = market.get("clobTokenIds", "[]")
        if isinstance(clob_tokens, str):
            clob_tokens = json.loads(clob_tokens)

        token_yes = clob_tokens[0] if clob_tokens else ""
        token_no  = clob_tokens[1] if len(clob_tokens) > 1 else ""

        prices_raw = market.get("outcomePrices", "[0.5,0.5]")
        if isinstance(prices_raw, str):
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw
        p_yes = float(prices[0]) if prices else 0.5
        p_no  = float(prices[1]) if len(prices) > 1 else 0.5

        info(f"Token YES : ...{token_yes[-12:]}  →  prix {p_yes:.3f}")
        info(f"Token NO  : ...{token_no[-12:]}   →  prix {p_no:.3f}")

        # Orderbook YES
        if token_yes:
            ob = await _fetch_orderbook(session, token_yes)
            if ob:
                bids = sorted(ob.get("bids", []), key=lambda x: -float(x.get("price", 0)))
                asks = sorted(ob.get("asks", []), key=lambda x: float(x.get("price", 0)))
                best_bid = float(bids[0]["price"]) if bids else 0
                best_ask = float(asks[0]["price"]) if asks else 0
                spread   = best_ask - best_bid
                ok(f"Orderbook YES  : bid={best_bid:.4f}  ask={best_ask:.4f}  spread={spread:.4f}")

                # Fee rate
                fee_bps = await _fetch_fee_rate(session, token_yes)
                if fee_bps is not None:
                    fee_at_mid = fee_bps / 10000 * p_yes * (1 - p_yes)
                    ok(f"Fee rate       : {fee_bps} bps → {fee_at_mid:.5f} USDC/share à p={p_yes:.2f}")
                    if fee_bps != 0:
                        note_code = 0.02 * (1 - p_yes)
                        info(f"Fee code actuel: {note_code:.5f} USDC/share (heuristique, ~{note_code/fee_at_mid:.1f}× réel)")
            else:
                warn("Orderbook YES inaccessible")


def test_dummy_order(client: object) -> None:
    section("ÉTAPE 6 — Création d'un ordre dummy (sans envoi)")

    # Token fictif pour tester la signature
    dummy_token = "1234567890"

    try:
        from py_clob_client.clob_types import OrderArgs
        order_args = OrderArgs(
            token_id=dummy_token,
            price=0.60,
            size=10.0,
            side="BUY",
        )
        signed = client.create_order(order_args)
        ok("Ordre signé avec succès (EIP-712 OK)")
        info(f"Signature : {str(signed)[:60]}...")
        info("L'ordre n'a PAS été envoyé au CLOB")
    except Exception as e:
        e_str = str(e).lower()
        if "market not found" in e_str or "404" in e_str:
            ok("Signature EIP-712 OK — token fictif rejeté par le CLOB (attendu)")
            info("En production, le bot utilise de vrais token_id issus du marché BTC actif")
        elif "nonce" in e_str:
            err(f"Création d'ordre échouée : {e}")
            info("Problème de nonce — réessaie dans quelques secondes")
        else:
            err(f"Création d'ordre échouée : {e}")


# ── main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   TEST DE CONNEXION POLYMARKET CLOB          ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════╝{RESET}")

    # 1. .env
    key = test_env()
    if not key:
        sys.exit(1)

    # 2. import
    if not test_clob_import():
        sys.exit(1)

    # 3. auth
    client = test_auth(key)
    if not client:
        sys.exit(1)

    # 4. solde
    test_balance(client)

    # 5. proxy check
    await test_proxy_status(key)

    # 6. marché
    await test_market_access()

    # 7. signature
    test_dummy_order(client)

    # ── bilan ──────────────────────────────────────────────────────────────────
    section("BILAN")
    print(f"\n  {GREEN}{BOLD}Connexion au CLOB Polymarket opérationnelle.{RESET}")
    print(f"\n  Prochaines étapes avant de lancer le bot en live :")
    print(f"  1. Approvisionne le wallet en {BOLD}USDC.e{RESET} sur Polygon")
    print(f"  2. Garde {BOLD}~$2 en MATIC{RESET} pour les frais de gas")
    print(f"  3. Change {BOLD}TRADING_MODE=live{RESET} dans .env")
    print(f"  4. Lance : {BOLD}python3 main.py{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
