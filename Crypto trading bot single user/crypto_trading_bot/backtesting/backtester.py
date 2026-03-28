"""
backtesting/backtester.py

Enhanced backtester that uses AI predictions for realistic simulation.
Identical logic to live/paper trading for accurate performance testing.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from logging_monitor.logger import get_logger

logger = get_logger(__name__)


class Backtester:
    def __init__(self, pairs, ind_eng, sig_gen, risk_mgr, settings):
        self.pairs = pairs
        self.ind_eng = ind_eng
        self.sig_gen = sig_gen
        self.risk = risk_mgr
        self.settings = settings

        # Performance tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.current_balance = 10000.0
        self.starting_balance = 10000.0
        self.open_position: Dict[str, Any] | None = None

    async def run(self, start_date, end_date):
        """Standard backtest without AI (for comparison)"""
        from data.historical_data import fetch_historical

        results = {}
        for pair in self.pairs:
            logger.info(f"Backtesting {pair}...")
            df = fetch_historical(pair, self.settings.TIMEFRAME, start_date, end_date)
            if df is not None and not df.empty:
                results[pair] = self._simulate(df, pair, use_ai=False)
            else:
                logger.warning(f"No data for {pair}")

        return results

    async def run_with_ai(self, start_date, end_date):
        """AI-enhanced backtest with realistic simulation"""
        from data.historical_data import fetch_historical

        results = {}
        for pair in self.pairs:
            logger.info(f"AI Backtesting {pair}...")
            df = fetch_historical(pair, self.settings.TIMEFRAME, start_date, end_date)
            if df is not None and not df.empty:
                results[pair] = await self._simulate_with_ai(df, pair)
            else:
                logger.warning(f"No data for {pair}")

        return results

    async def _simulate_with_ai(self, df: pd.DataFrame, pair: str) -> Dict:
        """Simulate trading with AI predictions"""
        self.trades = []
        self.equity_curve = [self.starting_balance]
        self.current_balance = self.starting_balance
        self.open_position = None

        # Warmup period for indicators
        warmup = 50

        for i in range(warmup, len(df)):
            window = df.iloc[:i]
            current = df.iloc[i]

            # Calculate indicators
            ind = self.ind_eng.calculate(window)

            # Build ticker
            ticker = {
                "last": float(current["close"]),
                "bid": float(current["close"]) * 0.9995,
                "ask": float(current["close"]) * 1.0005,
                "high": float(current["high"]),
                "low": float(current["low"]),
                "volume": float(current.get("volume", 0)),
                "percentage": 0.0,
            }

            # Build candles for AI
            candles = []
            for j in range(max(0, i-30), i):
                row = df.iloc[j]
                candles.append({
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })

            # Get AI signal
            signal = await self.sig_gen.generate_with_ai(
                indicators=ind,
                ticker=ticker,
                pair=pair,
                candles=candles,
                recent_trades=self.trades[-10:],
            )

            price = float(current["close"])

            # Execute logic
            await self._execute_backtest_logic(pair, price, signal, ind, current)

            # Record equity
            equity = self.current_balance
            if self.open_position:
                if self.open_position["action"] == "BUY":
                    equity += (price - self.open_position["entry"]) * self.open_position["size"]
                else:
                    equity += (self.open_position["entry"] - price) * self.open_position["size"]
            self.equity_curve.append(equity)

        # Close any open position at the end
        if self.open_position:
            final_price = float(df.iloc[-1]["close"])
            self._close_position(final_price, "End of backtest")

        return self._calculate_metrics()

    async def _execute_backtest_logic(self, pair: str, price: float, signal: dict, ind: dict, candle: pd.Series):
        """Execute trading logic in backtest"""
        action = signal["action"]
        ai_conf = signal.get("ai_confidence", 0)
        ai_action = signal.get("ai_action", "HOLD")

        # Check for position exit
        if self.open_position:
            pos = self.open_position
            entry = pos["entry"]
            action_type = pos["action"]

            # Calculate P&L
            if action_type == "BUY":
                pnl_pct = (price - entry) / entry * 100
            else:
                pnl_pct = (entry - price) / entry * 100

            # Check SL/TP
            sl_pct = pos.get("sl_pct", 1.5)
            tp_pct = pos.get("tp_pct", 3.0)

            exit_reason = None
            if pnl_pct <= -sl_pct:
                exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
            elif pnl_pct >= tp_pct:
                exit_reason = f"Take Profit ({pnl_pct:.2f}%)"
            elif ai_conf >= 0.60 and ai_action != action_type and ai_action != "HOLD":
                exit_reason = f"AI Reversal ({ai_action}, conf: {ai_conf:.2f})"

            if exit_reason:
                self._close_position(price, exit_reason)
                return

        # Check for entry
        elif not self.open_position and action in ("BUY", "SELL"):
            # Confirm signal
            if self._confirm_entry_signal(signal, ind, price):
                await self._open_position(pair, price, signal, ind)

    def _confirm_entry_signal(self, signal: dict, ind: dict, price: float) -> bool:
        """Confirm entry signal for backtest"""
        ai_action = signal.get("ai_action", "HOLD")
        base_action = signal["action"]
        ai_conf = signal.get("ai_confidence", 0)

        # Must have AI agreement
        if ai_action != base_action:
            return False

        # Confidence threshold
        if ai_conf < 0.45:
            return False

        # Technical confirmation
        rsi = ind.get("rsi", 50)
        macd_hist = ind.get("macd_hist", 0)

        if base_action == "BUY" and rsi > 75:
            return False
        if base_action == "SELL" and rsi < 25:
            return False

        return True

    async def _open_position(self, pair: str, price: float, signal: dict, ind: dict):
        """Open a position in backtest"""
        action = signal["action"]

        # Calculate size (1% risk per trade)
        risk_pct = self.settings.STRATEGY["risk"].get("max_position_pct", 5.0) / 100
        position_value = self.current_balance * risk_pct
        size = position_value / price

        # Calculate SL/TP
        pred_close = signal.get("ai_predicted_close", 0)
        support = signal.get("ai_support", 0)
        resistance = signal.get("ai_resistance", 0)
        pred_conf = signal.get("ai_prediction_confidence", 0)

        default_sl = self.settings.STRATEGY["risk"].get("stop_loss_pct", 1.5)
        default_tp = self.settings.STRATEGY["risk"].get("take_profit_pct", 3.0)

        if pred_conf >= 0.50 and pred_close > 0:
            if action == "BUY" and pred_close > price:
                tp_pct = max(1.0, (pred_close - price) / price * 100)
                sl_pct = max(0.5, (price - support) / price * 100) if support > 0 else default_sl
            elif action == "SELL" and pred_close < price:
                tp_pct = max(1.0, (price - pred_close) / price * 100)
                sl_pct = max(0.5, (resistance - price) / price * 100) if resistance > 0 else default_sl
            else:
                sl_pct, tp_pct = default_sl, default_tp
        else:
            sl_pct, tp_pct = default_sl, default_tp

        self.open_position = {
            "pair": pair,
            "action": action,
            "entry": price,
            "size": size,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "ai_target": pred_close,
            "entry_time": datetime.now(),
        }

        logger.info(f"[BACKTEST] ENTRY {action} {pair} @ ${price:.4f} | Size: {size:.4f} | SL: {sl_pct:.2f}% | TP: {tp_pct:.2f}%")

    def _close_position(self, price: float, reason: str):
        """Close a position in backtest"""
        if not self.open_position:
            return

        pos = self.open_position
        entry = pos["entry"]
        size = pos["size"]
        action = pos["action"]

        # Calculate P&L
        if action == "BUY":
            pnl = (price - entry) * size
            pnl_pct = (price - entry) / entry * 100
        else:
            pnl = (entry - price) * size
            pnl_pct = (entry - price) / entry * 100

        self.current_balance += pnl

        trade_record = {
            "pair": pos["pair"],
            "action": action,
            "entry": entry,
            "exit": price,
            "size": size,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
        }
        self.trades.append(trade_record)

        emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"[BACKTEST] EXIT {pos['pair']} | {reason} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Balance: ${self.current_balance:.2f}")

        self.open_position = None

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "final_balance": self.current_balance,
            }

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        # Calculate additional metrics
        total_pnl = sum(t["pnl"] for t in self.trades)
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        peak = self.starting_balance
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / self.starting_balance * 100, 2),
            "final_balance": round(self.current_balance, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "trades": self.trades,
        }

    def print_report(self, results: Dict[str, Dict]):
        """Print formatted backtest report"""
        print("\n" + "="*70)
        print("📊 BACKTEST REPORT - AI-Enhanced Strategy")
        print("="*70)

        total_pnl = 0
        total_trades = 0

        for pair, r in results.items():
            print(f"\n🪙 {pair}:")
            print(f"   Total Trades:  {r.get('total_trades', 0)}")
            print(f"   Win Rate:      {r.get('win_rate', 0):.1f}%")
            print(f"   Total P&L:     ${r.get('total_pnl', 0):.2f} ({r.get('total_pnl_pct', 0):.2f}%)")
            print(f"   Final Balance: ${r.get('final_balance', 0):.2f}")
            print(f"   Profit Factor: {r.get('profit_factor', 0):.2f}")
            print(f"   Max Drawdown:  {r.get('max_drawdown_pct', 0):.2f}%")

            total_pnl += r.get("total_pnl", 0)
            total_trades += r.get("total_trades", 0)

        print("\n" + "="*70)
        print(f"📈 COMBINED RESULTS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Total P&L:    ${total_pnl:.2f}")
        print("="*70 + "\n")