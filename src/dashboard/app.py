"""Dashboard API — FastAPI backend with WebSocket for real-time updates."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.utils.logger import setup_logger

log = setup_logger("dashboard")

TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="BTC Sniper Dashboard", version="1.0.0")


class DashboardState:
    """Shared state between the trading engine and dashboard."""

    def __init__(self):
        self.signal: dict = {}
        self.portfolio_stats: dict = {}
        self.recent_trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.daily_pnl: list[dict] = []
        self.market_state: dict = {}
        self.feeds_status: dict = {
            "binance": False,
            "chainlink": False,
            "polymarket": False,
        }
        self._ws_clients: list[WebSocket] = []
        self._db = None

    def set_db(self, db) -> None:
        self._db = db

    async def broadcast(self, data: dict) -> None:
        """Send update to all connected WebSocket clients."""
        if not self._ws_clients:
            return
        msg = json.dumps(data, default=str)
        disconnected = []
        for ws in self._ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._ws_clients.remove(ws)

    async def update_signal(self, signal_data: dict) -> None:
        self.signal = signal_data
        await self.broadcast({
            "type": "signal",
            "data": signal_data,
        })

    async def update_portfolio(self, stats: dict) -> None:
        self.portfolio_stats = stats
        await self.broadcast({
            "type": "portfolio",
            "data": stats,
        })

    async def update_trade(self, trade: dict) -> None:
        # Replace existing trade by id, or insert at front
        trade_id = trade.get("id") or trade.get("trade_id")
        if trade_id:
            self.recent_trades = [
                t for t in self.recent_trades
                if (t.get("id") or t.get("trade_id")) != trade_id
            ]
        self.recent_trades.insert(0, trade)
        self.recent_trades = self.recent_trades[:100]
        await self.broadcast({
            "type": "trade",
            "data": trade,
        })

    async def update_market(self, market_data: dict) -> None:
        self.market_state = market_data
        await self.broadcast({
            "type": "market",
            "data": market_data,
        })

    async def update_feeds(self, feeds: dict) -> None:
        self.feeds_status = feeds
        await self.broadcast({
            "type": "feeds",
            "data": feeds,
        })

    async def refresh_from_db(self) -> None:
        """Load historical data from database."""
        if not self._db:
            return
        try:
            self.recent_trades = await self._db.get_recent_trades(100)
            self.equity_curve = await self._db.get_equity_curve()
            self.daily_pnl = await self._db.get_daily_pnl()
        except Exception as e:
            log.error("DB refresh error: %s", e)

    def get_full_state(self) -> dict:
        return {
            "signal": self.signal,
            "portfolio": self.portfolio_stats,
            "recent_trades": self.recent_trades[:100],
            "equity_curve": self.equity_curve[-200:],
            "daily_pnl": self.daily_pnl[-30:],
            "market": self.market_state,
            "feeds": self.feeds_status,
            "server_time": time.time(),
        }


# Global state instance
dashboard_state = DashboardState()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/state")
async def get_state():
    return dashboard_state.get_full_state()


@app.get("/api/trades")
async def get_trades():
    return {"trades": dashboard_state.recent_trades[:50]}


@app.get("/api/equity")
async def get_equity():
    return {"curve": dashboard_state.equity_curve}


@app.get("/api/stats")
async def get_stats():
    if dashboard_state._db:
        return await dashboard_state._db.get_trade_stats()
    return dashboard_state.portfolio_stats


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    dashboard_state._ws_clients.append(websocket)
    log.info("[Dashboard] WebSocket client connected")

    # Send initial state
    try:
        await websocket.send_text(
            json.dumps({
                "type": "init",
                "data": dashboard_state.get_full_state(),
            }, default=str)
        )
    except Exception:
        pass

    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Client can request refresh
            if data == "refresh":
                await dashboard_state.refresh_from_db()
                await websocket.send_text(
                    json.dumps({
                        "type": "init",
                        "data": dashboard_state.get_full_state(),
                    }, default=str)
                )
    except WebSocketDisconnect:
        dashboard_state._ws_clients.remove(websocket)
        log.info("[Dashboard] WebSocket client disconnected")
    except Exception:
        if websocket in dashboard_state._ws_clients:
            dashboard_state._ws_clients.remove(websocket)
