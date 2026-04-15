"""Read-only check of on-chain USDC balances and allowances for both
USDC.e (legacy bridged) and native USDC, against all 3 Polymarket operators.
"""
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("https://polygon.drpc.org"))
addr = "0xbba3AE1ed3Bd2910C79242e8764Bf6AAC82CFF9b"

USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_NATIVE = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")

OPS = {
    "CTF Exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "NegRisk CTF Exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "NegRisk Adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
}

ABI = [
    {"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],
     "name":"allowance","outputs":[{"name":"","type":"uint256"}],
     "stateMutability":"view","type":"function"},
    {"inputs":[{"name":"account","type":"address"}],
     "name":"balanceOf","outputs":[{"name":"","type":"uint256"}],
     "stateMutability":"view","type":"function"},
]

for label, tok in [("USDC.e", USDC_E), ("USDC native", USDC_NATIVE)]:
    c = w3.eth.contract(address=tok, abi=ABI)
    bal = c.functions.balanceOf(addr).call()
    print(f"{label} ({tok}): balance = {bal / 1e6:.4f}")
    for name, op in OPS.items():
        a = c.functions.allowance(addr, Web3.to_checksum_address(op)).call()
        msg = f"{a / 1e6:.0f} USDC" if a < 10**20 else "UNLIMITED"
        print(f"  -> {name}: {msg}")
    print()
