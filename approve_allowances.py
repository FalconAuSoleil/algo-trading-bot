"""One-time script to approve Polymarket operators to spend your USDC.e + CTF.

Run once after funding the wallet. Requires POL (MATIC) for gas (~$0.05 total).
Reads POLYMARKET_PRIVATE_KEY from .env.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
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


def main() -> None:
    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = w3.eth.account.from_key(PRIVATE_KEY)
    addr = acct.address
    print(f"Wallet: {addr}")
    print(f"POL balance: {w3.from_wei(w3.eth.get_balance(addr), 'ether'):.4f}")
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


if __name__ == "__main__":
    main()
