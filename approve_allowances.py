"""One-time script to approve Polymarket operators to spend your USDC.e + CTF.

Modes:
  - EOA direct (POLYMARKET_SIGNATURE_TYPE=0 ou non défini) :
      approuve via la clé privée. Requiert POL (MATIC) sur l'EOA pour le gas.
  - Proxy Magic.link (POLYMARKET_SIGNATURE_TYPE=1) ou MetaMask (=2) :
      le wallet on-chain est POLYMARKET_FUNDER. On ne peut PAS approuver depuis
      l'EOA — il faut passer par l'UI Polymarket (un seul clic "Enable trading"
      sur polymarket.com après connexion). Le script va juste VÉRIFIER l'état
      des allowances sur le funder et te dire quoi faire.

Reads POLYMARKET_PRIVATE_KEY, POLYMARKET_SIGNATURE_TYPE, POLYMARKET_FUNDER from .env.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0").strip() or "0")
FUNDER = os.getenv("POLYMARKET_FUNDER", "").strip()

if not PRIVATE_KEY:
    sys.exit("ERROR: POLYMARKET_PRIVATE_KEY missing from .env")

RPC = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")

USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")

OPERATORS = {
    "CTF Exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "NegRisk CTF Exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "NegRisk Adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
}

ERC20_ABI = [{
    "inputs": [{"name": "spender", "type": "address"},
               {"name": "amount", "type": "uint256"}],
    "name": "approve",
    "outputs": [{"name": "", "type": "bool"}],
    "stateMutability": "nonpayable", "type": "function",
}, {
    "inputs": [{"name": "owner", "type": "address"},
               {"name": "spender", "type": "address"}],
    "name": "allowance",
    "outputs": [{"name": "", "type": "uint256"}],
    "stateMutability": "view", "type": "function",
}, {
    "inputs": [{"name": "account", "type": "address"}],
    "name": "balanceOf",
    "outputs": [{"name": "", "type": "uint256"}],
    "stateMutability": "view", "type": "function",
}]

ERC1155_ABI = [{
    "inputs": [{"name": "operator", "type": "address"},
               {"name": "approved", "type": "bool"}],
    "name": "setApprovalForAll",
    "outputs": [],
    "stateMutability": "nonpayable", "type": "function",
}, {
    "inputs": [{"name": "owner", "type": "address"},
               {"name": "operator", "type": "address"}],
    "name": "isApprovedForAll",
    "outputs": [{"name": "", "type": "bool"}],
    "stateMutability": "view", "type": "function",
}]

MAX_UINT = 2**256 - 1


def check_only(w3: Web3, owner: str) -> None:
    """Read-only audit for proxy mode : aucune tx envoyée, pas besoin de gas."""
    usdc = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)
    ctf = w3.eth.contract(address=CTF, abi=ERC1155_ABI)

    pol_bal = w3.from_wei(w3.eth.get_balance(owner), "ether")
    try:
        usdc_bal = usdc.functions.balanceOf(owner).call() / 1e6
    except Exception:
        usdc_bal = 0.0
    print(f"Owner (funder/proxy): {owner}")
    print(f"  POL:  {pol_bal:.4f}  (non requis en mode proxy)")
    print(f"  USDC.e: {usdc_bal:.4f}")
    print()

    all_ok = True
    for name, spender_str in OPERATORS.items():
        spender = Web3.to_checksum_address(spender_str)
        print(f"--- {name} ({spender[:10]}...) ---")
        usdc_allow = usdc.functions.allowance(owner, spender).call()
        ctf_allow = ctf.functions.isApprovedForAll(owner, spender).call()
        usdc_ok = usdc_allow > 10**20
        print(f"  USDC.e allowance: {'OK (' + str(usdc_allow // 10**6) + '+ USDC)' if usdc_ok else 'MISSING'}")
        print(f"  CTF approval:     {'OK' if ctf_allow else 'MISSING'}")
        if not (usdc_ok and ctf_allow):
            all_ok = False
        print()

    if all_ok:
        print("[OK] Toutes les allowances sont OK. Tu peux trader.")
    else:
        print("[MISSING] Allowances manquantes sur le proxy. Fix :")
        print("   1. Va sur polymarket.com, connecte-toi avec ton compte Magic.link")
        print("   2. Dépose >= $1, lance un petit pari manuel (Buy sur n'importe")
        print("      quel marché). Polymarket signe automatiquement les approvals")
        print("      via meta-tx (gasless côté toi).")
        print("   3. Relance ce script pour vérifier.")


def approve_eoa(w3: Web3, acct) -> None:
    """Mode EOA direct : signe les tx depuis la clé privée. Requiert POL."""
    addr = acct.address
    pol_bal = w3.from_wei(w3.eth.get_balance(addr), "ether")
    print(f"EOA Wallet: {addr}")
    print(f"  POL balance: {pol_bal:.4f}")
    if pol_bal < 0.05:
        print()
        print("[FAIL] POL insuffisant pour le gas (~0.05 POL requis pour 6 tx).")
        print("   Envoie au moins 0.1 POL sur l'EOA avant de relancer.")
        sys.exit(1)
    print()

    usdc = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)
    ctf = w3.eth.contract(address=CTF, abi=ERC1155_ABI)

    for name, spender_str in OPERATORS.items():
        spender = Web3.to_checksum_address(spender_str)
        print(f"--- {name} ({spender[:10]}...) ---")

        current = usdc.functions.allowance(addr, spender).call()
        if current > 10**20:
            print(f"  USDC.e already approved ({current / 1e6:.0f} USDC)")
        else:
            tx = usdc.functions.approve(spender, MAX_UINT).build_transaction({
                "from": addr, "nonce": w3.eth.get_transaction_count(addr),
                "gas": 100_000, "maxFeePerGas": w3.to_wei(250, "gwei"),
                "maxPriorityFeePerGas": w3.to_wei(60, "gwei"),
            })
            signed = acct.sign_transaction(tx)
            h = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  USDC.e approve tx: {h.hex()}")
            w3.eth.wait_for_transaction_receipt(h, timeout=120)
            print("  USDC.e approved")

        already = ctf.functions.isApprovedForAll(addr, spender).call()
        if already:
            print("  CTF already approved")
        else:
            tx = ctf.functions.setApprovalForAll(spender, True).build_transaction({
                "from": addr, "nonce": w3.eth.get_transaction_count(addr),
                "gas": 100_000, "maxFeePerGas": w3.to_wei(250, "gwei"),
                "maxPriorityFeePerGas": w3.to_wei(60, "gwei"),
            })
            signed = acct.sign_transaction(tx)
            h = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  CTF approve tx: {h.hex()}")
            w3.eth.wait_for_transaction_receipt(h, timeout=120)
            print("  CTF approved")
        print()

    print("Done. Restart the bot.")


def main() -> None:
    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = w3.eth.account.from_key(PRIVATE_KEY)

    print(f"Signature type: {SIGNATURE_TYPE} "
          f"({'EOA direct' if SIGNATURE_TYPE == 0 else 'Magic.link proxy' if SIGNATURE_TYPE == 1 else 'MetaMask proxy'})")
    print(f"EOA (signer): {acct.address}")
    if FUNDER:
        print(f"Funder (proxy): {Web3.to_checksum_address(FUNDER)}")
    print()

    if SIGNATURE_TYPE == 0:
        # Mode EOA direct : on approuve depuis la clé privée
        approve_eoa(w3, acct)
    else:
        # Mode proxy : on ne peut pas approuver depuis l'EOA (elle ne détient
        # rien). On audit les allowances du funder et on explique quoi faire.
        if not FUNDER:
            sys.exit("ERROR: SIGNATURE_TYPE!=0 mais POLYMARKET_FUNDER manquant dans .env")
        owner = Web3.to_checksum_address(FUNDER)
        print("Mode proxy détecté — vérification read-only des allowances du funder.")
        print("(Les approvals d'un proxy se font via l'UI Polymarket, pas par ce script.)")
        print()
        check_only(w3, owner)


if __name__ == "__main__":
    main()
