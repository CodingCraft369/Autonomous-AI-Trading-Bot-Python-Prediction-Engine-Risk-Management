"""
execution/trade_executor.py

Live trade execution via CCXT.
Uses settings.EXCHANGE_API_KEY / EXCHANGE_API_SECRET loaded from .env

SAFETY FEATURES
───────────────
• Dry-run check: verifies API keys exist before attempting any order
• Minimum order size validation
• Catches and logs all CCXT exceptions without crashing the bot
• Records every executed trade to state.py

SUPPORTED EXCHANGES (via CCXT)
──────────────────────────────
• Binance  (default)
• Bybit
• Kraken
• Coinbase Advanced Trade (set EXCHANGE_ID=coinbase)

HOW IT WORKS
────────────
1. Receives action (BUY/SELL), pair, size, price from bot_engine
2. Places a MARKET order for immediate fill
3. Optionally sets stop-loss and take-profit as separate limit orders
4. Records the filled trade to state.py
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any

import ccxt.async_support as ccxt_async

from core.state import record_trade

logger = logging.getLogger(__name__)

# Minimum order values in USDT to avoid "notional too small" rejections
MIN_NOTIONAL = {
    "binance":  10.0,
    "bybit":    5.0,
    "kraken":   5.0,
    "coinbase": 1.0,
}


class TradeExecutor:
    def __init__(self, exchange_id: str, settings):
        self.exchange_id = exchange_id
        self.settings    = settings
        self._exchange   = None
        self._init_exchange()

    def _init_exchange(self):
        """Create the CCXT exchange instance using saved API credentials."""
        api_key    = getattr(self.settings, "EXCHANGE_API_KEY",    "")
        api_secret = getattr(self.settings, "EXCHANGE_API_SECRET", "")
        sandbox    = getattr(self.settings, "SANDBOX",             False)

        if not api_key or not api_secret:
            logger.error(
                "[TradeExecutor] No API credentials found. "
                "Add your exchange API key in Settings → Exchange API Keys."
            )
            return

        exchange_class = getattr(ccxt_async, self.exchange_id, None)
        if exchange_class is None:
            logger.error(f"[TradeExecutor] Unknown exchange: {self.exchange_id}")
            return

        try:
            self._exchange = exchange_class({
                "apiKey":        api_key,
                "secret":        api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                },
            })

            if sandbox:
                try:
                    self._exchange.set_sandbox_mode(True)
                    logger.info(f"[TradeExecutor] {self.exchange_id} — SANDBOX mode")
                except Exception:
                    logger.warning(f"[TradeExecutor] {self.exchange_id} does not support sandbox — using live")
            else:
                logger.info(f"[TradeExecutor] {self.exchange_id} — LIVE mode ⚡")

        except Exception as e:
            logger.error(f"[TradeExecutor] Failed to initialize {self.exchange_id}: {e}")
            self._exchange = None

    async def execute(
        self,
        pair:    str,
        action:  str,
        size:    float,
        price:   float,
        sl_pct:  float | None = None,
        tp_pct:  float | None = None,
    ) -> dict | None:
        """
        Place a live market order.
        Returns trade record dict if successful, None if failed/rejected.
        """
        if self._exchange is None:
            logger.error("[TradeExecutor] Exchange not initialized — check API keys")
            return None

        risk    = self.settings.STRATEGY.get("risk", {})
        sl      = sl_pct if sl_pct is not None else risk.get("stop_loss_pct",   1.5)
        tp      = tp_pct if tp_pct is not None else risk.get("take_profit_pct", 3.0)
        min_val = MIN_NOTIONAL.get(self.exchange_id, 10.0)

        # Validate minimum order size
        notional = price * size
        if notional < min_val:
            logger.warning(
                f"[TradeExecutor] Order too small: ${notional:.2f} < ${min_val} minimum for {self.exchange_id}"
            )
            return None

        side = "buy" if action == "BUY" else "sell"

        try:
            logger.info(
                f"[Live] Placing {action} | {pair} | size={size:.6f} @ ~${price:.2f} "
                f"| SL={sl:.2f}% TP={tp:.2f}%"
            )

            # Market order for immediate fill
            order = await self._exchange.create_market_order(
                symbol=pair,
                side=side,
                amount=size,
            )

            filled_price = float(order.get("average") or order.get("price") or price)
            filled_size  = float(order.get("filled")  or size)
            order_id     = str(order.get("id", ""))

            logger.info(
                f"[Live] ✓ Order filled | id={order_id} | "
                f"filled={filled_size:.6f} @ ${filled_price:.4f}"
            )

            # Place SL/TP orders (best-effort — not all exchanges support OCO on spot)
            await self._place_sl_tp(pair, side, filled_size, filled_price, sl, tp)

            trade = {
                "time":     _now(),
                "pair":     pair,
                "side":     action,
                "price":    round(filled_price, 4),
                "size":     round(filled_size,  6),
                "order_id": order_id,
                "pnl":      0.0,   # P&L calculated when closed
            }
            record_trade(trade)
            return trade

        except ccxt_async.InsufficientFunds as e:
            logger.error(f"[TradeExecutor] Insufficient funds: {e}")
            return None

        except ccxt_async.InvalidOrder as e:
            logger.error(f"[TradeExecutor] Invalid order: {e}")
            return None

        except ccxt_async.ExchangeError as e:
            logger.error(f"[TradeExecutor] Exchange error: {e}")
            return None

        except ccxt_async.NetworkError as e:
            logger.error(f"[TradeExecutor] Network error: {e}")
            return None

        except Exception as e:
            logger.error(f"[TradeExecutor] Unexpected error: {e}")
            return None

    async def _place_sl_tp(
        self,
        pair:          str,
        entry_side:    str,
        size:          float,
        entry_price:   float,
        sl_pct:        float,
        tp_pct:        float,
    ):
        """
        Place stop-loss and take-profit orders after a fill.
        Uses limit orders since not all spot exchanges support OCO.
        """
        close_side = "sell" if entry_side == "buy" else "buy"

        sl_price = entry_price * (1 - sl_pct / 100) if entry_side == "buy" \
                   else entry_price * (1 + sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100) if entry_side == "buy" \
                   else entry_price * (1 - tp_pct / 100)

        try:
            # Take-profit limit order
            await self._exchange.create_limit_order(
                symbol=pair,
                side=close_side,
                amount=size,
                price=round(tp_price, 4),
            )
            logger.info(f"[Live] TP order placed @ ${tp_price:.4f}")
        except Exception as e:
            logger.warning(f"[TradeExecutor] TP order failed (may not be supported): {e}")

        try:
            # Stop-loss — use stop-market if available, fallback to limit
            if hasattr(self._exchange, "create_order"):
                await self._exchange.create_order(
                    symbol=pair,
                    type="stop_market" if "stop_market" in self._exchange.options.get("types", {}) else "stop_loss",
                    side=close_side,
                    amount=size,
                    price=round(sl_price, 4),
                    params={"stopPrice": round(sl_price, 4)},
                )
                logger.info(f"[Live] SL order placed @ ${sl_price:.4f}")
        except Exception as e:
            logger.warning(f"[TradeExecutor] SL order failed (manual monitoring needed): {e}")

    async def get_balance(self) -> dict[str, float]:
        """Fetch live account balance from exchange."""
        if self._exchange is None:
            return {}
        try:
            bal = await self._exchange.fetch_balance()
            # bal["free"] is a simple dict of {currency: amount}
            return {k: float(v) for k, v in bal.get("free", {}).items() if v and float(v) > 0}
        except Exception as e:
            logger.error(f"[TradeExecutor] Balance fetch failed: {e}")
            return {}

    async def close(self):
        if self._exchange:
            await self._exchange.close()


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")