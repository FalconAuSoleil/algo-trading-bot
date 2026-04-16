"""Redeem winning CTF positions on Polymarket (EOA mode).

In EOA mode (signature_type=0), Polymarket does NOT auto-redeem winning
shares when a market resolves. This script scans the trades DB for all
resolved markets, checks on-chain ERC-1155 balances, and calls
CTF.redeemPositions to convert winning shares into USDC.e.

Only handles standard CTF markets (neg_risk=False). For neg_risk markets
you'd need NegRiskAdapter.redeemPositions — add when needed.

Usage:
    python3 redeem_positions.py          # dry-run: show what would be redeemed
    python3 redeem_positions.py --go     # actually send transactions
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

load_dotenv()

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
if not PRIVATE_KEY:
    sys.exit("ERROR: POLYMARKET_PRIVATE_KEY missing from .env")

RPC = os.getenv("POLYGON_RPC", "https://polygon.drpc.org")
DB_PATH = os.getenv("DB_PATH", "data/trades.db")

CTF = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
PARENT_COLLECTION = b"\x00" * 32  # root collection for binary markets

CTF_ABI = [
    {"inputs": [{"name": "account", "type": "address"},
                {"name": "id", "type": "uint256"}],
     "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"}],
     "name": "redeemPositions", "outputs": [],
     "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "conditionId", "type": "bytes32"}],
     "name": "payoutDenominator", "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
]

USDC_ABI = [
    {"inputs": [{"name": "account", "type": "address"}],
     "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
]


def main() -> None:
    dry_run = "--go" not in sys.argv

    w3 = Web3(Web3.HTTPProvider(RPC))
    if not w3.is_connected():
        sys.exit(f"ERROR: cannot connect to RPC {RPC}")

    acct = Account.from_key(PRIVATE_KEY)
    addr = acct.address
    print(f"Wallet       : {addr}")
    print(f"RPC          : {RPC}")
    print(f"Mode         : {'DRY-RUN' if dry_run else 'LIVE (will broadcast)'}")
    print()

    ctf = w3.eth.contract(address=CTF, abi=CTF_ABI)
    usdc = w3.eth.contract(address=USDC_E, abi=USDC_ABI)

    # Collect ALL live trades with a token_id, grouped by conditionId.
    # We don't filter on DB outcome because the bot sometimes marks a trade
    # as 'pending' or 'error' while on-chain shares still exist — we want
    # to redeem based on what's really on-chain, not the DB view.
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT market_id, token_id, side, outcome FROM trades "
        "WHERE mode = 'live' AND token_id != ''"
    ).fetchall()
    con.close()

    if not rows:
        print("No live trades with a token_id in DB — nothing to redeem.")
        return

    # market_id -> list of (token_id, side, outcome)
    by_condition: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for market_id, token_id, side, outcome in rows:
        by_condition[market_id].append((token_id, side, outcome))

    print(f"Found {len(rows)} trades across {len(by_condition)} markets.\n")

    bal_before = usdc.functions.balanceOf(addr).call() / 1e6
    print(f"USDC.e before: ${bal_before:.4f}\n")

    to_redeem: list[tuple[str, float]] = []  # (conditionId, total_shares)

    for condition_id, trades in by_condition.items():
        # Check balance of each outcome token
        total_shares = 0.0
        details = []
        for token_id, side, outcome in trades:
            try:
                tok_int = int(token_id)
            except ValueError:
                continue
            bal = ctf.functions.balanceOf(addr, tok_int).call() / 1e6
            details.append((side, outcome, bal))
            total_shares += bal

        if total_shares <= 0:
            continue

        # Check if market is resolved on-chain (payoutDenominator > 0).
        # CTF.redeemPositions reverts on unresolved markets.
        try:
            cid_hex = condition_id[2:] if condition_id.startswith("0x") else condition_id
            cid_bytes = bytes.fromhex(cid_hex)
            denom = ctf.functions.payoutDenominator(cid_bytes).call()
        except Exception as exc:
            print(f"Market {condition_id[:10]}… | payoutDenominator error: {exc}")
            continue

        if denom == 0:
            print(
                f"Market {condition_id[:10]}…{condition_id[-6:]}  "
                f"UNRESOLVED on-chain — skipping (shares={total_shares:.4f})"
            )
            continue

        print(f"Market {condition_id[:10]}…{condition_id[-6:]}")
        for side, outcome, bal in details:
            marker = {"won": "WIN", "lost": "LOSE"}.get(outcome, outcome.upper())
            print(f"  {side} shares (DB={marker}): {bal:.4f}")
        to_redeem.append((condition_id, total_shares))
        print()

    if not to_redeem:
        print("Nothing to redeem — no on-chain shares left for resolved markets.")
        return

    print(f"=> Will redeem {len(to_redeem)} market(s).")
    if dry_run:
        print("   (dry-run; re-run with --go to broadcast transactions)")
        return

    nonce = w3.eth.get_transaction_count(addr)
    chain_id = w3.eth.chain_id

    for condition_id, _ in to_redeem:
        cid_bytes = bytes.fromhex(condition_id[2:] if condition_id.startswith("0x") else condition_id)
        assert len(cid_bytes) == 32, f"bad conditionId length: {condition_id}"

        try:
            gas_price = w3.eth.gas_price
            max_fee = int(gas_price * 2)
            priority = int(w3.to_wei(30, "gwei"))

            tx = ctf.functions.redeemPositions(
                USDC_E, PARENT_COLLECTION, cid_bytes, [1, 2],
            ).build_transaction({
                "from": addr,
                "nonce": nonce,
                "chainId": chain_id,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": priority,
                "gas": 300_000,
            })
            signed = acct.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"[redeem] {condition_id[:10]}… tx={tx_hash.hex()}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
            status = "OK" if receipt.status == 1 else "FAILED"
            print(f"  -> {status} | gas={receipt.gasUsed} | block={receipt.blockNumber}")
            nonce += 1
            time.sleep(1)
        except Exception as exc:
            print(f"  -> ERROR: {exc}")

    bal_after = usdc.functions.balanceOf(addr).call() / 1e6
    print(f"\nUSDC.e after : ${bal_after:.4f}")
    print(f"Delta        : +${bal_after - bal_before:.4f}")


if __name__ == "__main__":
    main()
