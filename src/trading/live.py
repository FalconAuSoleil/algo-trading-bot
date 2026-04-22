"""Live trading engine — executes real orders on Polymarket CLOB.

Uses py-clob-client for proper EIP-712 / HMAC order signing.
Requires POLYMARKET_PRIVATE_KEY in .env (EOA mode — address derived from key).

Trade flow:
  1. Receive Signal with action="BUY" and filters_passed=True
  2. Determine correct outcome token_id from signal.token_id_yes / token_id_no
  3. Create a signed FAK market order via ClobClient.create_market_order()
  4. Submit to CLOB; track as pending resolution
  5. Resolve after market close via Polymarket past-results API

BUGFIX (v3): previous version used conditionId as tokenID in the order
payload, which is wrong. The CLOB needs the YES or NO *outcome* token ID.
These are now carried on the Signal object and populated by main.py.

v3.6: BTC price fallback delay increased 30s→120s to reduce incorrect
resolutions when the API is temporarily slow. Also add restore_pending()
to reload in-flight trades from DB after a crash or restart.

v3.9: fixed fetch_outcome call — was missing slug as first arg, causing
silent TypeError and trades never resolving.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Callable, Optional

from src.config import PolymarketConfig
from src.engine.signal import Signal
from src.trading.portfolio import Portfolio, Position
from src.utils.db import Database, TradeRecord
from src.utils.logger import setup_logger

log = setup_logger("trading.live")

# py-clob-client is required for live trading.
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.constants import POLYGON
    _HAS_CLOB = True
except ImportError:
    _HAS_CLOB = False
    log.warning(
        "[Live] py-clob-client not installed. "
        "Run: pip install py-clob-client>=0.6"
    )

# Side constants (avoids importing the enum in the order payload)
_BUY = "BUY"
_SELL = "SELL"

# Seconds after expiry before falling back to BTC price for resolution.
# Increased from 30s to 120s (v3.6) to reduce incorrect resolutions
# when the Polymarket API is temporarily slow to publish outcomes.
_FALLBACK_DELAY_SEC = 600  # v5.3: était 120s, mais BTC fallback après 2min
                           # produisait des fake wins (prix BTC bouge entre la
                           # fin du marché et le fallback). On attend 10 min —
                           # l'API past-results finit toujours par répondre.


class LiveTrader:
    """
    Executes real trades on Polymarket via the CLOB API.

    Uses FAK (Fill-And-Kill) orders for immediate execution — matches
    whatever rests at or better than our limit price, cancels the rest.
    Requires valid Polygon wallet private key and USDC balance.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        db: Database,
        poly_cfg: PolymarketConfig,
    ):
        self.portfolio = portfolio
        self.db = db
        self.cfg = poly_cfg
        self._client: Optional[object] = None
        self._pending_resolutions: dict[int, dict] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Trade IDs rejected/errored since last pop — consumed by main.py
        # to push rejected rows to the dashboard.
        self._rejected_trade_ids: list[int] = []
        # Per-token market params cached to avoid refetching on every order
        self._tick_size_cache: dict[str, str] = {}
        self._neg_risk_cache: dict[str, bool] = {}
        # Serialize on-chain redemption txs so nonces don't collide when
        # multiple wins resolve in the same cycle.
        self._redeem_lock: Optional[asyncio.Lock] = None
        self._redeemed_conditions: set[str] = set()

    async def start(self) -> None:
        """Initialise the CLOB client and derive/set API credentials."""
        self._loop = asyncio.get_event_loop()
        self._redeem_lock = asyncio.Lock()

        if not _HAS_CLOB:
            log.error(
                "[Live] Cannot start: py-clob-client not installed."
            )
            return

        if not self.cfg.private_key:
            log.error(
                "[Live] Cannot start: POLYMARKET_PRIVATE_KEY not set in .env"
            )
            return

        if not self.cfg.wallet_address:
            log.error(
                "[Live] Cannot start: POLYMARKET_WALLET_ADDRESS not set in .env"
            )
            return

        # Log credential summary at startup
        wallet_short = self.cfg.wallet_address[:6] + "..." + self.cfg.wallet_address[-4:]
        sig_labels = {0: "EOA direct", 1: "Email/Magic proxy", 2: "Polymarket Proxy (MetaMask)"}
        log.info(f"[Live] Wallet EOA      : {wallet_short}")
        log.info(f"[Live] Signature type  : {self.cfg.signature_type} ({sig_labels.get(self.cfg.signature_type, '?')})")
        if self.cfg.signature_type != 0:
            if not self.cfg.funder:
                log.error(
                    "[Live] signature_type=%d mais POLYMARKET_FUNDER vide — "
                    "remplis l'adresse du proxy qui détient les USDC dans .env",
                    self.cfg.signature_type,
                )
                return
            funder_short = self.cfg.funder[:6] + "..." + self.cfg.funder[-4:]
            log.info(f"[Live] Funder (proxy)  : {funder_short}  (adresse qui détient les USDC)")
        if self.cfg.relayer_api_key:
            rk_short = self.cfg.relayer_api_key[:8] + "..."
            log.info(f"[Live] Relayer API key : {rk_short}  (stocké, dérivation auto depuis clé privée)")
        if self.cfg.api_key and self.cfg.api_secret and self.cfg.api_passphrase:
            log.info("[Live] Auth mode       : L2 explicite (api_key + secret + passphrase)")
        else:
            log.info("[Live] Auth mode       : dérivation depuis clé privée (create_or_derive_api_creds)")

        try:
            # Run synchronous ClobClient init in executor to avoid blocking
            def _init_client():
                kwargs = dict(
                    host=self.cfg.clob_url,
                    chain_id=POLYGON,
                    key=self.cfg.private_key,
                    signature_type=self.cfg.signature_type,
                )
                if self.cfg.signature_type != 0 and self.cfg.funder:
                    kwargs["funder"] = self.cfg.funder
                client = ClobClient(**kwargs)
                # Priorité : creds explicites dans .env → sinon dérivation depuis private_key.
                # RELAYER_API_KEY (UUID) seul ne suffit pas : il faut aussi api_secret
                # et api_passphrase. Sans les 3, on dérive — même résultat, juste plus lent.
                if self.cfg.api_key and self.cfg.api_secret and self.cfg.api_passphrase:
                    from py_clob_client.clob_types import ApiCreds
                    creds = ApiCreds(
                        api_key=self.cfg.api_key,
                        api_secret=self.cfg.api_secret,
                        api_passphrase=self.cfg.api_passphrase,
                    )
                    log.info("[Live] Credentials chargées depuis .env")
                else:
                    creds = client.create_or_derive_api_creds()
                    log.info(
                        f"[Live] Credentials dérivées depuis clé privée "
                        f"(api_key={creds.api_key[:8]}...)"
                    )
                client.set_api_creds(creds)
                return client

            self._client = await self._loop.run_in_executor(None, _init_client)

            # Set USDC + CTF token allowances for the Exchange contract.
            # Required once per wallet — allows the CLOB to debit USDC on BUY
            # and move conditional tokens on SELL (early exit).
            # Only submits the on-chain transaction if not already approved.
            # py-clob-client v0.34.6 removed set_allowances() — allowances must
            # be approved directly via the USDC.e / CTF ERC-20 contracts on
            # Polygon. We only check and warn here; if a BUY order fails with
            # an allowance error, surface it clearly instead of crashing.
            try:
                def _check_allowances_onchain():
                    from web3 import Web3
                    from eth_account import Account
                    rpc = os.getenv("POLYGON_RPC", "https://polygon.drpc.org")
                    w3 = Web3(Web3.HTTPProvider(rpc))
                    # En mode proxy (signature_type != 0), les fonds et les
                    # allowances sont sur le funder, pas sur l'EOA.
                    if self.cfg.signature_type != 0 and self.cfg.funder:
                        owner = Web3.to_checksum_address(self.cfg.funder)
                    else:
                        owner = Account.from_key(self.cfg.private_key).address
                    usdce = Web3.to_checksum_address(
                        "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
                    )
                    ops = [
                        ("CTF Exchange", "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"),
                        ("NegRisk CTF", "0xC5d563A36AE78145C45a50134d48A1215220f80a"),
                        ("NegRisk Adapter", "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"),
                    ]
                    abi = [{
                        "inputs": [
                            {"name": "owner", "type": "address"},
                            {"name": "spender", "type": "address"},
                        ],
                        "name": "allowance",
                        "outputs": [{"name": "", "type": "uint256"}],
                        "stateMutability": "view", "type": "function",
                    }]
                    c = w3.eth.contract(address=usdce, abi=abi)
                    return [
                        (name, c.functions.allowance(
                            owner, Web3.to_checksum_address(op)
                        ).call())
                        for name, op in ops
                    ]
                results = await self._loop.run_in_executor(
                    None, _check_allowances_onchain
                )
                missing = [name for name, a in results if a < 10**20]
                if missing:
                    log.warning(
                        "[Live] USDC.e allowance not set for: %s — run "
                        "approve_allowances.py before trading",
                        ", ".join(missing),
                    )
                else:
                    log.info(
                        "[Live] USDC.e UNLIMITED allowance verified "
                        "on-chain for all 3 Polymarket operators — ready to trade"
                    )
            except Exception as exc:
                log.warning("[Live] On-chain allowance check failed: %s", exc)

            # Sync portfolio balance with real wallet USDC.e balance
            try:
                def _fetch_balance():
                    from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                    bal = self._client.get_balance_allowance(
                        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                    )
                    return float(bal.get("balance", 0)) / 1_000_000
                real_usdc = await self._loop.run_in_executor(None, _fetch_balance)
                if real_usdc > 0:
                    self.portfolio.balance = real_usdc
                    self.portfolio.initial_balance = real_usdc
                    self.portfolio._peak_balance = real_usdc
                    log.info("=" * 50)
                    log.info("  WALLET BALANCE LIVE")
                    log.info("  USDC.e dispo  : $%.2f", real_usdc)
                    log.info("  Wallet        : %s", self.cfg.wallet_address)
                    log.info("  Capital actif : $%.2f  (100%% du wallet)", real_usdc)
                    log.info("  Max par pari  : $%.2f  (%.0f%% Kelly × %.0f%% cap)",
                             real_usdc * 0.04 * 0.25,
                             0.25 * 100, 0.04 * 100)
                    log.info("=" * 50)
                elif real_usdc == 0:
                    log.warning("=" * 50)
                    log.warning("  WALLET BALANCE = $0.00")
                    log.warning("  Verifie le wallet %s", self.cfg.wallet_address)
                    log.warning("  ou lance approve_allowances.py")
                    log.warning("=" * 50)
            except Exception as exc:
                log.warning("[Live] Balance sync failed: %s", exc, exc_info=True)

            try:
                from eth_account import Account
                wallet_addr = Account.from_key(self.cfg.private_key).address
            except Exception:
                wallet_addr = "unknown"
            log.info("[Live] ClobClient ready | wallet=%s", wallet_addr)
        except Exception as exc:
            log.error("[Live] ClobClient init failed: %s", exc)
            self._client = None

    async def stop(self) -> None:
        self._client = None
        log.info("[Live] Trader stopped")

    async def restore_pending(self) -> None:
        """Reload in-flight trades from DB after a crash or restart.

        Called by main.py on startup. Without this, any trade that was
        pending when the process died stays as outcome='pending' in the
        DB forever and the capital is never returned to the portfolio.

        Also rebuilds portfolio.open_positions — otherwise close_position
        later can't find the trade (returns ('error', 0.0)) and the PnL
        is never reflected in the portfolio balance.

        v3.6 addition. v4.1: also restores portfolio.open_positions.
        """
        pending = await self.db.get_pending_trades(mode="live")
        restored = 0
        for row in pending:
            trade_id = row["id"]
            if trade_id in self._pending_resolutions:
                continue  # already tracked
            approx_start = row["timestamp"] - row.get("time_remaining_sec", 300)
            end_time = row["timestamp"] + row.get("time_remaining_sec", 300)
            self._pending_resolutions[trade_id] = {
                "market_id": row["market_id"],
                "side": row["side"],
                "end_time": end_time,
                "reference_price": 0.0,  # not stored in trades table
                "start_time": approx_start,
                "duration": 300,
                "slug": row.get("slug", ""),
                "strategy_used": "chainlink_arb",
                "token_id": row.get("token_id", ""),
            }

            # Rebuild portfolio.open_positions so close_position can later
            # resolve this trade and actually update the balance.
            if trade_id not in self.portfolio.open_positions:
                entry_price = float(row.get("entry_price") or 0.0)
                size_usd = float(row.get("size_usd") or 0.0)
                shares = size_usd / entry_price if entry_price > 0 else 0.0
                pos = Position(
                    trade_id=trade_id,
                    market_id=row["market_id"],
                    side=row["side"],
                    entry_price=entry_price,
                    size_usd=size_usd,
                    shares=shares,
                    entry_time=approx_start,
                    market_end_time=end_time,
                    delta_at_entry=float(row.get("delta") or 0.0),
                    p_true_at_entry=float(row.get("p_true") or 0.0),
                    edge_at_entry=float(row.get("edge") or 0.0),
                    slug=row.get("slug", ""),
                    duration_seconds=int(row.get("time_remaining_sec") or 300),
                )
                self.portfolio.open_positions[trade_id] = pos
            restored += 1
        if restored:
            log.info(
                "[Live] Restored %d pending trade(s) from DB after restart",
                restored,
            )

    async def execute(self, signal: Signal) -> Optional[int]:
        """
        Execute a live trade based on signal.

        Returns:
            trade_id if order was filled, None if skipped or rejected.
        """
        if signal.action != "BUY" or not signal.filters_passed:
            return None

        # Plancher minimum : MAX(MIN_BET_USD config, 5 shares au prix d'entrée).
        # - Polymarket rejette les ordres < ~$1 notional OU < 5 shares.
        # - MIN_BET_USD est configurable via .env (défaut 1.0).
        # - On clamp vers le haut SEULEMENT si ça ne dépasse pas MAX_BET_FRACTION
        #   du vrai solde on-chain (sinon Kelly dit $0.10 mais on forcerait 40%
        #   du wallet — absurde). Cette vérif est faite plus bas après fetch.
        POLY_MIN_SHARES = 5
        cfg_min = float(os.environ.get("MIN_BET_USD", "1.0"))
        poly_min_usd = max(cfg_min, POLY_MIN_SHARES * signal.entry_price)
        size_usd = signal.size_usd
        kelly_was_below_min = size_usd < poly_min_usd
        if kelly_was_below_min:
            log.info(
                "[Live] Kelly $%.2f < Polymarket min $%.2f (MIN_BET_USD=%.1f ou %d shares @ %.4f) — "
                "clamp candidat, vérification ratio vs balance après fetch",
                size_usd, poly_min_usd, cfg_min, POLY_MIN_SHARES, signal.entry_price,
            )
            size_usd = poly_min_usd
        if signal.time_remaining_sec < 45.0:
            log.warning(
                "[Live] Rejected: t_rem=%.0fs < 45s safety guard",
                signal.time_remaining_sec,
            )
            return None
        if self._client is None:
            log.error("[Live] Client not initialized — check credentials")
            return None

        # Determine the correct outcome token ID
        if signal.side == "YES":
            token_id = signal.token_id_yes
        else:
            token_id = signal.token_id_no

        if not token_id:
            log.error(
                "[Live] Missing token_id for side=%s market=%s. "
                "Ensure main.py populates sig.token_id_yes/no from MarketInfo.",
                signal.side, signal.market_id[:16],
            )
            return None

        entry_price = signal.entry_price
        if entry_price <= 0 or entry_price >= 1.0:
            log.error("[Live] Invalid entry price: %.4f", entry_price)
            return None

        # Fetch la balance USDC réelle on-chain — le portfolio.balance local
        # peut être désynchronisé si des tokens winning ne sont pas encore
        # redeem (ils comptent en PnL côté DB mais les USDC sont bloqués).
        try:
            def _fetch_real_balance():
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                bal = self._client.get_balance_allowance(
                    params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                )
                return float(bal.get("balance", 0)) / 1_000_000
            real_balance = await self._loop.run_in_executor(None, _fetch_real_balance)
        except Exception as exc:
            log.warning("[Live] Real balance fetch failed, using local: %s", exc)
            real_balance = self.portfolio.balance

        # Never bet more than the real on-chain balance (safety cap).
        # 0.98 au lieu de 0.95 : le buffer de 2% couvre les fees + arrondis.
        max_size = round(real_balance * 0.98, 2)
        if max_size < poly_min_usd:
            log.warning(
                "[Live] Balance on-chain $%.2f insuffisante pour le minimum Polymarket $%.2f -> skip",
                real_balance, poly_min_usd,
            )
            return None

        # Garde-fou anti-sur-exposition : si Kelly disait "trop petit" et que le
        # clamp forcé nous ferait miser > MAX_BET_FRACTION du wallet, on skip.
        # Adaptatif selon la bankroll :
        #   - < $20  : cap 50% (sinon la règle des 5 shares × ~0.65 = $3.25
        #              bloque TOUT trade sous un cap de 25%)
        #   - >= $20 : cap 25% (sizing sain une fois la bankroll décente)
        # Override via env MAX_BET_FRACTION_OF_BALANCE si besoin.
        default_cap = 0.50 if real_balance < 20.0 else 0.25
        MAX_BET_FRACTION = float(os.environ.get("MAX_BET_FRACTION_OF_BALANCE", str(default_cap)))
        if kelly_was_below_min and real_balance > 0 and poly_min_usd > real_balance * MAX_BET_FRACTION:
            log.warning(
                "[Live] Skip: Kelly $%.2f était trop petit, clamp à $%.2f = %.0f%% du solde $%.2f "
                "(> cap %.0f%%). Pas assez de bankroll pour ce marché — on passe.",
                signal.size_usd, poly_min_usd,
                (poly_min_usd / real_balance * 100), real_balance,
                MAX_BET_FRACTION * 100,
            )
            return None
        if size_usd > max_size:
            log.info(
                "[Live] Size capped $%.2f -> $%.2f (98%% of real balance $%.2f)",
                size_usd, max_size, real_balance,
            )
            size_usd = max(poly_min_usd, max_size)

        shares = round(size_usd / entry_price, 2)

        # Record as pending in DB before submitting (idempotent crash safety)
        record = TradeRecord(
            market_id=signal.market_id,
            slug=signal.slug,
            side=signal.side,
            entry_price=entry_price,
            size_usd=size_usd,
            delta=signal.delta_chainlink,
            sigma=signal.sigma,
            p_true=signal.p_true,
            p_market=signal.p_market,
            edge=signal.edge,
            time_remaining_sec=signal.time_remaining_sec,
            oracle_age_sec=signal.oracle_age_sec,
            mode="live",
            token_id=token_id,
        )
        trade_id = await self.db.insert_trade(record)

        # Marketable BUY via create_market_order + FAK (Fill-And-Kill):
        # - amount is in USD (dollar amount to spend)
        # - price is the worst-price limit (slippage cap)
        # - FAK accepts partial fills and cancels the rest (FOK was rejecting
        #   valid orders the moment any part of the requested size was missing
        #   at the exact limit price).
        try:
            def _place_order():
                from py_clob_client.clob_types import (
                    MarketOrderArgs, OrderType, PartialCreateOrderOptions,
                )
                tick_size = self._tick_size_cache.get(token_id)
                if tick_size is None:
                    tick_size = self._client.get_tick_size(token_id)
                    self._tick_size_cache[token_id] = tick_size
                neg_risk = self._neg_risk_cache.get(token_id)
                if neg_risk is None:
                    neg_risk = self._client.get_neg_risk(token_id)
                    self._neg_risk_cache[token_id] = neg_risk
                # Snap to tick grid + add 2 ticks of upward buffer on BUY.
                # On fast markets (high-edge arb targets), best_ask can drift
                # by multiple ticks during the ~1s signature+HTTP flight. A
                # single-tick buffer was insufficient (observed trade_id=9).
                step = float(tick_size)
                limit_price = round(round(entry_price / step) * step + 2 * step, 4)
                log.info(
                    "[Live] Submitting FAK BUY | %s %s | amount=$%.2f "
                    "entry=%.4f limit=%.4f (tick=%s) | CL_age=%.0fs",
                    signal.side, signal.market_id[:16], size_usd,
                    entry_price, limit_price, tick_size,
                    signal.oracle_age_sec,
                )
                order_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=round(size_usd, 2),
                    side=_BUY,
                    price=limit_price,
                    order_type=OrderType.FAK,
                )
                options = PartialCreateOrderOptions(
                    tick_size=tick_size, neg_risk=neg_risk,
                )
                signed = self._client.create_market_order(order_args, options)
                return self._client.post_order(signed, OrderType.FAK)

            resp = await self._loop.run_in_executor(None, _place_order)

            if resp and resp.get("success"):
                # For a BUY: makingAmount = USD spent, takingAmount = shares
                # received. py-clob-client may return either raw 6-decimal
                # fixed-point integers ("2200000") OR already-scaled decimals
                # ("2.2"). Detect by magnitude: a real BUY is never >$1000,
                # so values above 1000 must be the raw 6-decimal form.
                try:
                    m = float(resp.get("makingAmount", 0) or 0)
                    t = float(resp.get("takingAmount", 0) or 0)
                except (TypeError, ValueError):
                    m, t = 0.0, 0.0
                if max(m, t) > 1000:
                    m /= 1_000_000
                    t /= 1_000_000
                spent_usd, got_shares = m, t
                if got_shares > 0 and spent_usd > 0:
                    fill_price = spent_usd / got_shares
                    actual_shares = got_shares
                    size_usd = spent_usd  # may be < requested on partial fill
                else:
                    fill_price = entry_price
                    actual_shares = size_usd / fill_price

                log.info(
                    "[Live] FILLED | trade_id=%d | fill=%.4f | shares=%.2f "
                    "| spent=$%.2f | raw m=%s t=%s",
                    trade_id, fill_price, actual_shares, spent_usd,
                    resp.get("makingAmount"), resp.get("takingAmount"),
                )

                pos = Position(
                    trade_id=trade_id,
                    market_id=signal.market_id,
                    side=signal.side,
                    entry_price=fill_price,
                    size_usd=size_usd,
                    shares=actual_shares,
                    entry_time=time.time(),
                    market_end_time=time.time() + signal.time_remaining_sec,
                    delta_at_entry=signal.delta_chainlink,
                    p_true_at_entry=signal.p_true,
                    edge_at_entry=signal.edge,
                    slug=signal.slug,
                    duration_seconds=signal.market_duration,
                )

                if not self.portfolio.open_position(pos):
                    await self.db.resolve_trade(trade_id, "cancelled", 0.0)
                    return None

                self._pending_resolutions[trade_id] = {
                    "market_id": signal.market_id,
                    "side": signal.side,
                    "end_time": pos.market_end_time,
                    "reference_price": signal.reference_price,
                    "start_time": signal.market_start_time,
                    "duration": signal.market_duration,
                    "slug": signal.slug,
                    "strategy_used": signal.strategy_used,
                    "token_id": token_id,
                }
                return trade_id

            else:
                err_msg = resp.get("errorMsg", str(resp)) if resp else "no response"
                err_lower = err_msg.lower()
                if "minimum" in err_lower or "size" in err_lower:
                    log.warning("[Live] REJECTED (size too small) | trade_id=%d | %s", trade_id, err_msg)
                elif "insufficient" in err_lower or "balance" in err_lower or "fund" in err_lower:
                    log.warning("[Live] REJECTED (insufficient funds) | trade_id=%d | %s", trade_id, err_msg)
                elif "not found" in err_lower or "market" in err_lower:
                    log.warning("[Live] REJECTED (market closed/not found) | trade_id=%d | %s", trade_id, err_msg)
                elif "fok" in err_lower or "fak" in err_lower or "fill" in err_lower:
                    log.info("[Live] FAK not matched (no resting liquidity at limit) | trade_id=%d", trade_id)
                else:
                    log.warning("[Live] Order REJECTED | trade_id=%d | %s", trade_id, err_msg)
                await self.db.resolve_trade(trade_id, "rejected", 0.0)
                self._rejected_trade_ids.append(trade_id)
                return None

        except Exception as exc:
            # py-clob-client raises PolyApiException even for business-level
            # rejections like FAK no-match. Classify those as 'rejected'
            # rather than 'error' — they are not code failures.
            msg = str(exc).lower()
            is_liquidity_rejection = (
                "no orders found" in msg
                or "fak orders are partially filled or killed" in msg
                or "fok orders are fully filled or killed" in msg
                or "couldn't be fully filled" in msg
            )
            if is_liquidity_rejection:
                log.info(
                    "[Live] FAK not matched (liquidity moved) | trade_id=%d",
                    trade_id,
                )
                await self.db.resolve_trade(trade_id, "rejected", 0.0)
            else:
                log.error("[Live] Order execution error: %s", exc, exc_info=True)
                await self.db.resolve_trade(trade_id, "error", 0.0)
            self._rejected_trade_ids.append(trade_id)
            return None

    async def sell_position(
        self, trade_id: int, exit_price: float, reason: str
    ) -> Optional[float]:
        """Sell a position early by placing a SELL FAK order on the CLOB.

        v5: Early exit feature. Places a real SELL order to liquidate
        the position at the current best_bid price.

        Returns:
            PnL if sold, None if position not found or order rejected.
        """
        if trade_id not in self._pending_resolutions:
            log.warning("[Live] sell_position: trade %d not pending", trade_id)
            return None
        if self._client is None or self._loop is None:
            log.error("[Live] sell_position: client not initialized")
            return None

        info = self._pending_resolutions[trade_id]
        token_id = info.get("token_id")
        if not token_id:
            log.error("[Live] sell_position: no token_id for trade %d", trade_id)
            return None

        pos = self.portfolio.open_positions.get(trade_id)
        if pos is None:
            log.warning("[Live] sell_position: no open position for trade %d", trade_id)
            return None

        try:
            log.info(
                "[Live] SELL order | trade=%d | shares=%.2f @ %.4f | reason=%s",
                trade_id, pos.shares, exit_price, reason,
            )

            def _place_sell():
                from py_clob_client.clob_types import (
                    OrderArgs, OrderType, PartialCreateOrderOptions,
                )
                tick_size = self._tick_size_cache.get(token_id)
                if tick_size is None:
                    tick_size = self._client.get_tick_size(token_id)
                    self._tick_size_cache[token_id] = tick_size
                neg_risk = self._neg_risk_cache.get(token_id)
                if neg_risk is None:
                    neg_risk = self._client.get_neg_risk(token_id)
                    self._neg_risk_cache[token_id] = neg_risk
                step = float(tick_size)
                limit_price = round(round(exit_price / step) * step, 4)
                order_args = OrderArgs(
                    token_id=token_id,
                    price=limit_price,
                    size=round(pos.shares, 2),
                    side=_SELL,
                )
                options = PartialCreateOrderOptions(
                    tick_size=tick_size, neg_risk=neg_risk,
                )
                signed = self._client.create_order(order_args, options)
                return self._client.post_order(signed, OrderType.FAK)

            resp = await self._loop.run_in_executor(None, _place_sell)

            if resp and resp.get("success"):
                # SELL: makingAmount = shares sold, takingAmount = USD received
                # Scale-detect (see BUY path comment above).
                try:
                    m = float(resp.get("makingAmount", 0) or 0)
                    t = float(resp.get("takingAmount", 0) or 0)
                    if max(m, t) > 1000:
                        m /= 1_000_000
                        t /= 1_000_000
                    fill_price = t / m if m > 0 else exit_price
                except (TypeError, ValueError, ZeroDivisionError):
                    fill_price = exit_price
                outcome, pnl = self.portfolio.close_position_early(
                    trade_id, fill_price
                )
                if outcome == "error":
                    return None

                await self.db.resolve_trade_early(
                    trade_id, outcome, pnl, fill_price, reason
                )
                del self._pending_resolutions[trade_id]

                log.info(
                    "[Live] EARLY EXIT FILLED | trade=%d | fill=%.4f | PnL=$%.2f",
                    trade_id, fill_price, pnl,
                )
                return pnl
            else:
                err = resp.get("errorMsg", str(resp)) if resp else "no response"
                log.warning(
                    "[Live] SELL order REJECTED | trade=%d | %s",
                    trade_id, err,
                )
                return None

        except Exception as exc:
            log.error("[Live] Sell execution error: %s", exc, exc_info=True)
            return None

    async def _redeem_winning_position(
        self, trade_id: int, condition_id: str, token_id: str,
    ) -> None:
        """Redeem a winning CTF position on-chain (EOA mode).

        In EOA mode Polymarket does not auto-redeem winning shares when a
        market resolves. This calls CTF.redeemPositions to convert the
        outcome tokens back into USDC.e. Safe to skip if the balance is 0
        (already redeemed) or if the market uses neg-risk (handled via a
        different adapter — updown markets are all standard CTF, so in
        practice the skip path only triggers on an ecosystem change).
        """
        if condition_id in self._redeemed_conditions:
            return

        # Skip neg_risk markets — they use NegRiskAdapter.redeemPositions,
        # not supported yet. Current updown markets are all non-neg-risk.
        try:
            neg_risk = self._neg_risk_cache.get(token_id)
            if neg_risk is None and self._client is not None:
                neg_risk = await self._loop.run_in_executor(
                    None, self._client.get_neg_risk, token_id,
                )
                self._neg_risk_cache[token_id] = neg_risk
            if neg_risk:
                log.warning(
                    "[Redeem] trade_id=%d is neg-risk — auto-redeem not "
                    "implemented, use NegRiskAdapter manually",
                    trade_id,
                )
                return
        except Exception as exc:
            log.warning("[Redeem] neg_risk lookup failed: %s", exc)

        async with self._redeem_lock:
            try:
                tx_hash, paid = await self._loop.run_in_executor(
                    None, self._send_redeem_tx, condition_id, token_id,
                )
                self._redeemed_conditions.add(condition_id)
                if tx_hash is None:
                    log.info(
                        "[Redeem] trade_id=%d skipped — no on-chain shares left",
                        trade_id,
                    )
                else:
                    log.info(
                        "[Redeem] trade_id=%d OK | shares=%.4f | tx=%s",
                        trade_id, paid, tx_hash,
                    )
            except Exception as exc:
                log.error(
                    "[Redeem] trade_id=%d failed: %s — run redeem_positions.py "
                    "to retry manually", trade_id, exc,
                )

    def _send_redeem_tx(
        self, condition_id: str, token_id: str,
    ) -> tuple[Optional[str], float]:
        """Blocking helper — builds, signs and broadcasts redeemPositions.

        Returns (tx_hash_hex, shares_redeemed). tx_hash is None if the
        on-chain balance was already 0 (nothing to do).
        """
        from eth_account import Account
        from web3 import Web3

        rpc = os.getenv("POLYGON_RPC", "https://polygon.drpc.org")
        w3 = Web3(Web3.HTTPProvider(rpc))
        acct = Account.from_key(self.cfg.private_key)
        addr = acct.address

        ctf_addr = Web3.to_checksum_address(
            "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
        )
        usdce = Web3.to_checksum_address(
            "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        )
        abi = [
            {"inputs": [{"name": "account", "type": "address"},
                        {"name": "id", "type": "uint256"}],
             "name": "balanceOf",
             "outputs": [{"name": "", "type": "uint256"}],
             "stateMutability": "view", "type": "function"},
            {"inputs": [{"name": "collateralToken", "type": "address"},
                        {"name": "parentCollectionId", "type": "bytes32"},
                        {"name": "conditionId", "type": "bytes32"},
                        {"name": "indexSets", "type": "uint256[]"}],
             "name": "redeemPositions", "outputs": [],
             "stateMutability": "nonpayable", "type": "function"},
        ]
        ctf = w3.eth.contract(address=ctf_addr, abi=abi)

        bal = ctf.functions.balanceOf(addr, int(token_id)).call()
        if bal == 0:
            return None, 0.0

        cid_hex = condition_id[2:] if condition_id.startswith("0x") else condition_id
        cid_bytes = bytes.fromhex(cid_hex)
        if len(cid_bytes) != 32:
            raise ValueError(f"bad conditionId length: {condition_id}")

        nonce = w3.eth.get_transaction_count(addr)
        gas_price = w3.eth.gas_price
        tx = ctf.functions.redeemPositions(
            usdce, b"\x00" * 32, cid_bytes, [1, 2],
        ).build_transaction({
            "from": addr,
            "nonce": nonce,
            "chainId": w3.eth.chain_id,
            "maxFeePerGas": int(gas_price * 2),
            "maxPriorityFeePerGas": int(w3.to_wei(30, "gwei")),
            "gas": 300_000,
        })
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status != 1:
            raise RuntimeError(f"redeem reverted in block {receipt.blockNumber}")
        return tx_hash.hex(), bal / 1e6

    async def check_resolutions(
        self,
        btc_chainlink_price: float,
        fetch_outcome: Optional[Callable[[str, float, int], object]] = None,
    ) -> list[dict]:
        """
        Check if pending positions have resolved.

        Prefers the real Polymarket API outcome; falls back to BTC
        price comparison only after _FALLBACK_DELAY_SEC (120s) if
        the API is still pending. Delay increased from 30s in v3.6
        to reduce incorrect resolutions on slow API responses.

        fetch_outcome signature: (slug: str, start_time: float, duration: int) -> Optional[str]
        """
        now = time.time()
        resolved = []

        for trade_id, info in list(self._pending_resolutions.items()):
            if now < info["end_time"] + 10:
                continue  # too early

            ref = info["reference_price"]
            side = info["side"]
            if ref <= 0 and now < info["end_time"] + _FALLBACK_DELAY_SEC:
                # No reference price (restored from DB) — wait for API only
                pass

            # Try real outcome from Polymarket API
            real_outcome = None
            if fetch_outcome:
                try:
                    real_outcome = await fetch_outcome(
                        info.get("slug", ""),          # v3.9: was missing
                        info.get("start_time", 0),
                        info.get("duration", 300),
                    )
                except Exception as exc:
                    log.warning("[Live] fetch_outcome error for trade %d: %s", trade_id, exc)

            if real_outcome in ("up", "down"):
                won = (real_outcome == "up") if side == "YES" else (real_outcome == "down")
                log.info(
                    "[Live] Resolved trade %d via API | outcome=%s side=%s won=%s",
                    trade_id, real_outcome, side, won,
                )
            elif ref > 0 and now >= info["end_time"] + _FALLBACK_DELAY_SEC:
                # API still not ready after _FALLBACK_DELAY_SEC — use BTC price
                # as last resort. Note: this uses the CURRENT price, not the
                # price at expiry, which may differ if BTC moved post-resolution.
                btc_above = btc_chainlink_price >= ref
                won = btc_above if side == "YES" else not btc_above
                log.warning(
                    "[Live] Resolved trade %d via BTC FALLBACK (%ds wait) | "
                    "btc=$%.2f ref=$%.2f side=%s won=%s — "
                    "verify manually if outcome seems wrong",
                    trade_id, int(now - info["end_time"]),
                    btc_chainlink_price, ref, side, won,
                )
            else:
                continue  # still waiting for API

            outcome, pnl = self.portfolio.close_position(trade_id, won)
            await self.db.resolve_trade(trade_id, outcome, pnl)

            if won:
                condition_id = info.get("market_id", "")
                token_id = info.get("token_id", "")
                if condition_id and token_id:
                    asyncio.create_task(
                        self._redeem_winning_position(trade_id, condition_id, token_id)
                    )

            del self._pending_resolutions[trade_id]

            resolved.append({
                "trade_id": trade_id,
                "outcome": outcome,
                "pnl": pnl,
                "side": side,
                "btc_price": btc_chainlink_price,
                "ref_price": ref,
                "slug": info.get("slug", ""),
                "duration": info.get("duration", 300),
                "strategy_used": info.get("strategy_used", "chainlink_arb"),
            })

        return resolved

    def pop_rejected_trades(self) -> list[int]:
        """Return and clear the list of recently rejected/errored trade IDs.

        Called by main.py after each execute() to push rejections to the
        dashboard. Safe to call even if no rejections occurred.
        """
        result = self._rejected_trade_ids[:]
        self._rejected_trade_ids.clear()
        return result

    @property
    def pending_count(self) -> int:
        return len(self._pending_resolutions)
