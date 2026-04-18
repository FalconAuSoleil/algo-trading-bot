"""Diagnostic — trouve où se trouvent tes USDC sur Polygon.

Vérifie les 3 adresses candidates (EOA, RELAYER_API_KEY_ADDRESS,
POLYMARKET_FUNDER) et liste leurs balances USDC.e + MATIC.

Usage:
    python diagnose_wallet.py
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account

load_dotenv()

RPC = os.getenv("POLYGON_RPC", "https://polygon.drpc.org")
USDCE = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_NATIVE = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")

ERC20_ABI = [
    {"inputs": [{"name": "account", "type": "address"}],
     "name": "balanceOf",
     "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
]

# Polymarket proxy factory — utilisée pour dériver l'adresse du proxy Safe
# à partir de l'EOA. Les comptes web MetaMask ont un Safe déployé ici.
PROXY_FACTORY = Web3.to_checksum_address("0xaacFeEa03eb1561C4e67d661e40682Bd20E3541b")


def _bal(w3: Web3, token_addr: str, account: str) -> float:
    c = w3.eth.contract(address=token_addr, abi=ERC20_ABI)
    raw = c.functions.balanceOf(Web3.to_checksum_address(account)).call()
    return raw / 1_000_000


def main() -> None:
    w3 = Web3(Web3.HTTPProvider(RPC))
    print(f"\nRPC: {RPC}")
    print(f"Connecté: {w3.is_connected()}  | chainId={w3.eth.chain_id}\n")

    priv = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    wallet_cfg = os.getenv("POLYMARKET_WALLET_ADDRESS", "")
    relayer_addr = os.getenv("RELAYER_API_KEY_ADDRESS", "")
    funder_cfg = os.getenv("POLYMARKET_FUNDER", "")

    candidates = []
    if priv:
        try:
            eoa = Account.from_key(priv).address
            candidates.append(("EOA (dérivée clé privée)", eoa))
        except Exception as exc:
            print(f"[!] private_key invalide: {exc}")
    if wallet_cfg:
        candidates.append(("POLYMARKET_WALLET_ADDRESS (.env)", wallet_cfg))
    if relayer_addr:
        candidates.append(("RELAYER_API_KEY_ADDRESS (.env)", relayer_addr))
    if funder_cfg:
        candidates.append(("POLYMARKET_FUNDER (.env)", funder_cfg))

    # Dédupe en gardant l'ordre
    seen = set()
    unique = []
    for label, addr in candidates:
        key = addr.lower()
        if key not in seen:
            seen.add(key)
            unique.append((label, addr))

    if not unique:
        print("Aucune adresse à vérifier. Remplis le .env.")
        return

    print("-" * 78)
    print(f"{'LABEL':<40}  {'ADRESSE':<44}  {'USDC.e':>10}  {'USDC':>10}  {'MATIC':>8}")
    print("-" * 78)
    for label, addr in unique:
        try:
            usdce = _bal(w3, USDCE, addr)
            usdc = _bal(w3, USDC_NATIVE, addr)
            matic = w3.eth.get_balance(Web3.to_checksum_address(addr)) / 1e18
            print(f"{label:<40}  {addr:<44}  {usdce:>10.2f}  {usdc:>10.2f}  {matic:>8.4f}")
        except Exception as exc:
            print(f"{label:<40}  {addr:<44}  ERROR: {exc}")

    print("-" * 78)
    print()
    print("Interpretation :")
    print("  - Si l'EOA a >$0 USDC.e  -> signature_type=0 OK (config actuelle)")
    print("  - Si SEUL le proxy (autre adresse) a les fonds :")
    print("      -> C'est un Polymarket Proxy. Mets dans .env :")
    print("        POLYMARKET_SIGNATURE_TYPE=1  (email/Magic) ou 2 (MetaMask)")
    print("        POLYMARKET_FUNDER=<adresse qui tient les USDC>")
    print("  - Si aucune n'a de USDC -> depose sur polymarket.com d'abord.")
    print()
    print("Astuce : va sur https://polygonscan.com/address/<ton_adresse>")
    print("         onglet 'Token' pour voir ce qu'elle detient vraiment.\n")


if __name__ == "__main__":
    main()
