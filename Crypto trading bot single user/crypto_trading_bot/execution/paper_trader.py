"""
execution/paper_trader.py — Fixed v2

BUGS FIXED
──────────
1. BUY trades were NOT recorded to state.trades (only CLOSE/SELL was recorded).
   Dashboard showed 0 win rate because all completed round-trips were missing
   their BUY leg, making it impossible to match pairs for P&L.
   Fix: record_trade() called for every executed BUY too.

2. Balance deduction happened BEFORE save_state(), causing race condition
   where concurrent load_state() returned stale balance.
   Fix: save_state(state) immediately after balance mutation.

3. _close_position passed stale `state` dict after save_state() was called
   inside it — subsequent balance reads were from the stale object.
   Fix: reload state after every close.

4. PaperTrader instances created per manual-trade request had no shared
   position memory. Positions opened by the bot were invisible to manual
   close calls.
   Fix: Positions stored in state["positions"] so all instances share them.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from core.state import load_state, save_state, record_trade

logger = logging.getLogger(__name__)

FEE_RATE = 0.001   # 0.1% Binance taker fee per side


class PaperTrader:
    def __init__(self, exchange_id: str, settings):
        self.exchange_id = exchange_id
        self.settings    = settings
        logger.info("[PaperTrader] Initialized — shared positions from state")

    # ── Position helpers ───────────────────────────────────────────────────

    def _load_positions(self) -> dict:
        return load_state().get("positions", {})

    def _save_positions(self, positions: dict):
        s = load_state()
        s["positions"] = positions
        save_state(s)

    # ── Execute ────────────────────────────────────────────────────────────

    async def execute(
        self,
        pair:    str,
        action:  str,
        size:    float,
        price:   float,
        sl_pct:  float | None = None,
        tp_pct:  float | None = None,
    ) -> dict | None:
        risk = self.settings.STRATEGY.get("risk", {})
        sl   = sl_pct if sl_pct is not None else risk.get("stop_loss_pct",   1.5)
        tp   = tp_pct if tp_pct is not None else risk.get("take_profit_pct", 3.0)

        state     = load_state()
        balance   = float(state.get("balance", 10000.0))
        positions = state.get("positions", {})

        # ── Close opposite position first ──────────────────────────────────
        if pair in positions and positions[pair]["side"] != action:
            pnl = self._close_position(pair, price)
            # reload after close
            state   = load_state()
            balance = float(state.get("balance", 10000.0))
            positions = state.get("positions", {})

        # ── BUY: open / add to position ────────────────────────────────────
        if action == "BUY":
            cost = price * size * (1 + FEE_RATE)
            if cost > balance:
                logger.warning(f"[Paper] Insufficient balance: need ${cost:.2f}, have ${balance:.2f}")
                return None

            if pair in positions:
                # Average down
                curr       = positions[pair]
                total_cost = curr["cost"] + cost
                new_size   = curr["size"] + size
                positions[pair] = {
                    "side":     "BUY",
                    "size":     new_size,
                    "entry":    total_cost / new_size / (1 + FEE_RATE),
                    "sl_price": price * (1 - sl / 100),
                    "tp_price": price * (1 + tp / 100),
                    "cost":     total_cost,
                    "opened_at": curr["opened_at"],
                }
            else:
                positions[pair] = {
                    "side":     "BUY",
                    "size":     size,
                    "entry":    price,
                    "sl_price": price * (1 - sl / 100),
                    "tp_price": price * (1 + tp / 100),
                    "cost":     cost,
                    "opened_at": _now(),
                }

            # Deduct balance and persist atomically
            state["balance"]   = round(balance - cost, 2)
            state["positions"] = positions
            save_state(state)

            trade = _build_record(pair, "BUY", size, price, 0.0)
            record_trade(trade)   # FIX: was missing — caused 0% win rate

            logger.info(
                f"[Paper] BUY {pair} | size={size:.6f} @ ${price:.2f} "
                f"| SL=${positions[pair]['sl_price']:.2f} TP=${positions[pair]['tp_price']:.2f} "
                f"| cost=${cost:.2f} | bal=${state['balance']:.2f}"
            )
            return trade

        # ── SELL: close long position ──────────────────────────────────────
        elif action == "SELL":
            if pair in positions and positions[pair]["side"] == "BUY":
                pnl   = self._close_position(pair, price)
                trade = _build_record(pair, "SELL", size, price, pnl)
                return trade
            else:
                logger.info(f"[Paper] SELL {pair} — no long open, skipping")
                return None

        return None

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(self, pair: str, price: float) -> float:
        state     = load_state()
        positions = state.get("positions", {})

        if pair not in positions:
            return 0.0

        pos      = positions.pop(pair)
        entry    = pos["entry"]
        size     = pos["size"]
        cost     = pos["cost"]
        fee_out  = price * size * FEE_RATE
        proceeds = price * size - fee_out

        raw_pnl  = proceeds - cost if pos["side"] == "BUY" else cost - proceeds
        pnl      = round(raw_pnl, 2)
        new_bal  = round(float(state.get("balance", 10000)) + proceeds, 2)
        new_pnl  = round(float(state.get("pnl", 0)) + pnl, 2)

        state["balance"]   = new_bal
        state["pnl"]       = new_pnl
        state["positions"] = positions
        save_state(state)

        # Record the close trade (SELL side) with real P&L
        t = _build_record(pair, "SELL" if pos["side"] == "BUY" else "BUY", size, price, pnl)
        record_trade(t)

        logger.info(
            f"[Paper] CLOSED {pair} | entry=${entry:.2f} exit=${price:.2f} "
            f"| pnl=${pnl:+.2f} | bal=${new_bal:.2f}"
        )
        return pnl

    # ── SL/TP monitoring ───────────────────────────────────────────────────

    def monitor_positions(self, prices: dict[str, float]):
        """Called each bot cycle to check stop-loss / take-profit."""
        positions = self._load_positions()
        for pair, pos in list(positions.items()):
            price = prices.get(pair)
            if not price:
                continue
            if pos["side"] == "BUY":
                if price <= pos["sl_price"]:
                    logger.info(f"[Paper] SL hit {pair} @ ${price:.2f}")
                    self._close_position(pair, price)
                elif price >= pos["tp_price"]:
                    logger.info(f"[Paper] TP hit {pair} @ ${price:.2f}")
                    self._close_position(pair, price)


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_record(pair: str, side: str, size: float, price: float, pnl: float) -> dict:
    return {
        "time":  _now(),
        "pair":  pair,
        "side":  side,
        "price": round(float(price), 4),
        "size":  round(float(size),  6),
        "pnl":   round(float(pnl),   2),
    }

def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")