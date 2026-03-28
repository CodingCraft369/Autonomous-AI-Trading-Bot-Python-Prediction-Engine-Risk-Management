"""
risk_manager.py
Handles stop-loss, take-profit, position sizing, and daily loss limits.
"""
from datetime import date
from logging_monitor.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    def __init__(self, cfg):
        self.cfg         = cfg
        self.open_trades = {}
        self.daily_pnl   = 0.0
        self.last_reset  = date.today()
        self.account_bal = 10000.0

    def _reset_daily(self):
        today = date.today()
        if today > self.last_reset:
            self.daily_pnl  = 0.0
            self.last_reset = today

    def approve(self, signal, price, pair):
        self._reset_daily()

        # No trade needed for HOLD
        if signal["action"] == "HOLD":
            return False

        # Check daily loss limit
        limit = self.cfg["daily_loss_limit_pct"] / 100 * self.account_bal
        if self.daily_pnl <= -limit:
            logger.warning("Daily loss limit reached.")
            return False

        # Count open trades safely
        open_count = 0
        for k in self.open_trades:
            open_count += 1

        max_trades = int(self.cfg["max_open_trades"])

        if open_count >= max_trades:
            logger.warning("Max open trades reached: " + str(open_count))
            return False

        # No duplicate BUY on same pair
        if pair in self.open_trades and signal["action"] == "BUY":
            logger.warning("Already have position for " + pair)
            return False

        return True

    def position_size(self, price, side):
        if price <= 0:
            return 0.0
        alloc = self.cfg["max_position_pct"] / 100 * self.account_bal
        size  = alloc / price
        return round(size, 6)

    def get_sl_tp(self, entry, side):
        sl_pct = self.cfg["stop_loss_pct"]   / 100
        tp_pct = self.cfg["take_profit_pct"] / 100
        if side == "BUY":
            sl = entry * (1 - sl_pct)
            tp = entry * (1 + tp_pct)
        else:
            sl = entry * (1 + sl_pct)
            tp = entry * (1 - tp_pct)
        return round(sl, 4), round(tp, 4)

    def record_trade(self, pair, side, size, entry):
        sl, tp = self.get_sl_tp(entry, side)
        self.open_trades[pair] = {
            "side":  side,
            "size":  size,
            "entry": entry,
            "sl":    sl,
            "tp":    tp,
        }

    def check_exits(self, pair, current_price):
        if pair not in self.open_trades:
            return None
        trade = self.open_trades[pair]
        if trade["side"] == "BUY":
            if current_price <= trade["sl"]:
                return "STOP_LOSS"
            if current_price >= trade["tp"]:
                return "TAKE_PROFIT"
        else:
            if current_price >= trade["sl"]:
                return "STOP_LOSS"
            if current_price <= trade["tp"]:
                return "TAKE_PROFIT"
        return None

    def close_trade(self, pair, exit_price):
        if pair not in self.open_trades:
            return 0.0
        trade = self.open_trades.pop(pair)
        pnl = (exit_price - trade["entry"]) * trade["size"]
        if trade["side"] == "SELL":
            pnl = -pnl
        self.daily_pnl   += pnl
        self.account_bal += pnl
        logger.info(
            "Closed " + pair +
            " | pnl=" + str(round(pnl, 2)) +
            " | balance=" + str(round(self.account_bal, 2))
        )
        return pnl