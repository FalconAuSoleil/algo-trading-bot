"""Manually sell one or more open positions at market via FAK.

Reads trade metadata from data/trades.db, checks on-chain share balance,
fetches current best_bid from the CLOB, and places a SELL FAK order.
On success, updates the trade row to outcome='closed_early'.

Usage:
    python3 sell_position.py                 # list pending live trades
    python3 sell_position.py 3               # sell trade id=3 (prompts)
    python3 sell_position.py 3 4 6           # sell multiple
    python3 sell_position.py 3 --yes         # skip confirmation
    python3 sell_position.py 3 --slippage 3  # accept fill up to 3% below best_bid
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime

import requests
from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

load_dotenv()

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
if not PRIVATE_KEY:
    sys.exit("ERROR: POLYMARKET_PRIVATE_KEY missing from .env")

CLOB_URL = "https://clob.polymarket.com"
RPC = os.getenv("POLYGON_RPC", "https://polygon.drpc.org")
DB_PATH = os.getenv("DB_PATH", "data/trades.db")
CTF = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")

CTF_ABI = [
    {"inputs": [{"name": "account", "type": "address"},
                {"name": "id", "type": "uint256"}],
     "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
]


def get_pending() -> list[tuple]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, slug, side, entry_price, size_usd "
        "FROM trades WHERE outcome='pending' AND mode='live' ORDER BY id"
    ).fetchall()
    con.close()
    return rows


def print_pending(rows: list[tuple]) -> None:
    if not rows:
        print("No pending live trades.")
        return
    print(f"{'id':<5} {'slug':<32} {'side':<4} {'entry':<7} {'size':<8}")
    print("-" * 60)
    for tid, slug, side, entry, size in rows:
        print(f"{tid:<5} {slug:<32} {side:<4} {entry:<7.4f} ${size:<7.2f}")


def prompt_select(rows: list[tuple]) -> list[int]:
    """Ask user which trade ids to sell — accepts space/comma-separated,
    'all' to pick every row, or empty to cancel."""
    print_pending(rows)
    valid_ids = {r[0] for r in rows}
    print()
    raw = input(
        "Which trade(s) to sell? [ids space-separated, 'all', or empty=cancel] : "
    ).strip().lower()
    if not raw:
        return []
    if raw == "all":
        return [r[0] for r in rows]
    picked: list[int] = []
    for chunk in raw.replace(",", " ").split():
        try:
            tid = int(chunk)
        except ValueError:
            print(f"  ignoring invalid: {chunk!r}")
            continue
        if tid not in valid_ids:
            print(f"  ignoring unknown trade id: {tid}")
            continue
        if tid not in picked:
            picked.append(tid)
    return picked


def fetch_trade(trade_id: int) -> dict:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM trades WHERE id=?", (trade_id,)
    ).fetchone()
    con.close()
    if row is None:
        sys.exit(f"ERROR: trade id={trade_id} not found")
    if row["outcome"] != "pending":
        sys.exit(
            f"ERROR: trade {trade_id} already resolved (outcome={row['outcome']})"
        )
    if row["mode"] != "live":
        sys.exit(f"ERROR: trade {trade_id} is not in live mode")
    return dict(row)


def fetch_orderbook(token_id: str) -> dict:
    r = requests.get(f"{CLOB_URL}/book", params={"token_id": token_id}, timeout=8)
    r.raise_for_status()
    return r.json()


def best_bid(book: dict) -> float:
    bids = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
    return float(bids[0]["price"]) if bids else 0.0


def resolve_early(trade_id: int, pnl: float, exit_price: float) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "UPDATE trades SET outcome=?, pnl=?, resolved_at=?, "
        "exit_price=?, exit_reason=? WHERE id=?",
        ("closed_early", pnl, time.time(), exit_price, "manual", trade_id),
    )
    con.commit()
    con.close()


def sell_one(trade_id: int, auto_confirm: bool, slippage_pct: float) -> None:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        OrderArgs, OrderType, PartialCreateOrderOptions,
    )
    from py_clob_client.constants import POLYGON
    from py_clob_client.order_builder.constants import SELL

    t = fetch_trade(trade_id)
    token_id = t["token_id"]
    if not token_id:
        sys.exit(f"ERROR: trade {trade_id} has no token_id in DB")
    size_usd_at_entry = float(t["size_usd"])

    print(f"\n=== Trade id={trade_id} ({t['slug']}) ===")
    print(f"Side            : {t['side']}")
    print(f"Entry           : {t['entry_price']:.4f}  (size=${size_usd_at_entry:.2f})")

    # Check actual shares on-chain (authoritative)
    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = Account.from_key(PRIVATE_KEY)
    addr = acct.address
    ctf = w3.eth.contract(address=CTF, abi=CTF_ABI)
    bal_raw = ctf.functions.balanceOf(addr, int(token_id)).call()
    shares = bal_raw / 1e6
    print(f"On-chain shares : {shares:.4f}")
    if shares <= 0:
        print("  -> no shares held; marking DB as closed_early with $0")
        resolve_early(trade_id, -size_usd_at_entry, 0.0)
        return

    book = fetch_orderbook(token_id)
    bid = best_bid(book)
    if bid <= 0:
        sys.exit(f"ERROR: no bids on this side — cannot place a SELL")
    limit_price = round(bid * (1.0 - slippage_pct / 100.0), 4)
    expected_usd = shares * bid
    print(f"Best bid        : {bid:.4f}")
    print(f"Limit (w/ slip) : {limit_price:.4f}   (slippage={slippage_pct:.1f}%)")
    print(f"Expected USD    : ~${expected_usd:.2f}")
    expected_pnl = expected_usd - size_usd_at_entry
    print(f"Expected PnL    : ${expected_pnl:+.2f}")

    if not auto_confirm:
        resp_in = input("\nConfirm SELL? [y/N] ").strip().lower()
        if resp_in not in ("y", "yes"):
            print("Skipped.")
            return

    print("\n--- Placing SELL FAK ---")
    client = ClobClient(
        host=CLOB_URL, chain_id=POLYGON, key=PRIVATE_KEY, signature_type=0,
    )
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)

    tick_size = client.get_tick_size(token_id)
    neg_risk = client.get_neg_risk(token_id)
    step = float(tick_size)
    snapped = round(round(limit_price / step) * step, 4)

    order_args = OrderArgs(
        token_id=token_id,
        price=snapped,
        size=round(shares, 2),
        side=SELL,
    )
    options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
    signed = client.create_order(order_args, options)
    t0 = time.time()
    resp = client.post_order(signed, OrderType.FAK)
    elapsed = time.time() - t0

    print(f"\n--- Response ({elapsed:.2f}s) ---")
    print(json.dumps(resp, indent=2, default=str))

    if not resp.get("success"):
        print(f"\nRejected: {resp.get('errorMsg', 'unknown')}")
        return

    # SELL: makingAmount = shares sold, takingAmount = USD received
    try:
        m = float(resp.get("makingAmount", 0) or 0)
        t_val = float(resp.get("takingAmount", 0) or 0)
    except (TypeError, ValueError):
        m, t_val = 0.0, 0.0
    if max(m, t_val) > 1000:
        m /= 1_000_000
        t_val /= 1_000_000
    shares_sold, usd_received = m, t_val
    fill_price = usd_received / shares_sold if shares_sold > 0 else snapped
    pnl = usd_received - size_usd_at_entry

    print("\n--- Summary ---")
    print(f"Shares sold     : {shares_sold:.4f}")
    print(f"USD received    : ${usd_received:.4f}")
    print(f"Fill price      : {fill_price:.4f}")
    print(f"PnL             : ${pnl:+.4f}")

    resolve_early(trade_id, pnl, fill_price)
    print(f"DB updated: trade {trade_id} -> closed_early, pnl=${pnl:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trade_ids", nargs="*", type=int)
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--slippage", type=float, default=2.0,
                    help="accept fill up to N%% below best_bid (default 2.0)")
    args = ap.parse_args()

    trade_ids = list(args.trade_ids)
    if not trade_ids:
        rows = get_pending()
        if not rows:
            print("No pending live trades.")
            return
        trade_ids = prompt_select(rows)
        if not trade_ids:
            print("Cancelled.")
            return

    print(f"\n=== {datetime.now().isoformat(timespec='seconds')} ===")
    for tid in trade_ids:
        try:
            sell_one(tid, args.yes, args.slippage)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"  ERROR on trade {tid}: {exc}")


if __name__ == "__main__":
    main()
