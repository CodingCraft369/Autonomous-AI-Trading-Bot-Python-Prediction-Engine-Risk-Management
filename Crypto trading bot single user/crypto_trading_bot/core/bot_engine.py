"""
core/bot_engine.py
Passes raw candles to generate_with_ai() so DeepSeek can do
candle prediction.  Also uses predicted support/resistance to
set dynamic stop-loss and take-profit when AI confidence is high.

FIXED: Removed Unicode arrow characters (→) that cause encoding errors on Windows
FIXED: Added trade recording to state history and open position tracking for dashboard
"""
import asyncio
import time
import traceback
import datetime
import ccxt.async_support as ccxt_async

from data.market_data import MarketDataCollector
from indicators.engine import IndicatorEngine
from ai_engine.signal_generator import SignalGenerator
from risk.risk_manager import RiskManager
from execution.paper_trader import PaperTrader
from execution.trade_executor import TradeExecutor
from backtesting.backtester import Backtester
from logging_monitor.logger import get_logger
from core.state import update_signal, set_running, load_state, record_trade

logger = get_logger(__name__)


class BotEngine:
    def __init__(self, mode, pairs, exchange_id, settings):
        self.mode        = mode
        self.pairs       = pairs
        self.exchange_id = exchange_id
        self.settings    = settings
        self.indicator   = IndicatorEngine(settings.STRATEGY)
        self.sig_gen     = SignalGenerator(settings.STRATEGY)
        self.risk_mgr    = RiskManager(settings.STRATEGY["risk"])
        
        # Track open positions and prices for dashboard and profit taking
        self.open_positions = {} 
        self.last_prices = {}

        if mode == "live":
            self.collector = MarketDataCollector(exchange_id, settings)
            self.executor  = TradeExecutor(exchange_id, settings)
        elif mode == "paper":
            self.collector = MarketDataCollector(exchange_id, settings)
            self.executor  = PaperTrader(exchange_id, settings)
        else:
            self.collector = None
            self.executor  = None

    async def run(self, start_date=None, end_date=None):
        if self.mode == "backtest":
            await self._run_backtest(start_date, end_date)
            return
        set_running(True)
        logger.info(f"Bot running in {self.mode} mode for {self.pairs}")

        # Sync balance once at start
        await self._sync_account_data()

        while True:
            try:
                state = load_state()
                if not state.get("auto_trading", True):
                    logger.info("Auto trading OFF -- waiting...")
                    await asyncio.sleep(30)
                    continue
                for pair in self.pairs:
                    await self._process_pair(pair)
                
                # Check for TP/SL on open positions
                await self._check_profit_taking()
                
                # Periodically sync balance (every cycle)
                await self._sync_account_data()

                wait = self._tf_seconds()
                logger.info(f"Cycle complete. Next in {wait}s...")
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                logger.info("Bot stopped.")
                set_running(False)
                break
            except Exception as e:
                logger.error(f"Loop error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(15)
                await self._sync_account_data() # Try to sync on error too

    async def _process_pair(self, pair: str):
        try:
            logger.info(f"Fetching {pair}...")
            candles_df = await self.collector.get_candles(pair)
            ticker     = await self.collector.get_ticker(pair)

            if candles_df is None or candles_df.empty:
                logger.warning(f"No candle data for {pair}")
                return
            if ticker is None or ticker.get("last") is None:
                logger.warning(f"No ticker for {pair}")
                return

            # -- Update last price for profit taking check --
            price  = float(ticker["last"])
            self.last_prices[pair] = price

            # -- Convert DataFrame to list of dicts for AI ----
            candle_list = []
            for _, row in candles_df.tail(30).iterrows():
                candle_list.append({
                    "open":   float(row.get("open",   0)),
                    "high":   float(row.get("high",   0)),
                    "low":    float(row.get("low",    0)),
                    "close":  float(row.get("close",  0)),
                    "volume": float(row.get("volume", 0)),
                })
            # -- Indicators ------------------------------------
            ind = self.indicator.calculate(candles_df)
            rsi = float(ind.get("rsi", 50))
            macd = float(ind.get("macd_hist", 0.0))

            # -- Fast Async AI Execution (Non-blocking) --------
            async def _background_ai():
                try:
                    state = load_state()
                    old = state.get("signals", {}).get(pair, {})
                    # -- OPTIMIZATION: Don't set "processing" if we have valid old data -- 
                    # This keeps the last prediction on dashboard until the NEW one is ready.
                    if not old.get("ai_used") or old.get("status") != "active":
                        update_signal(
                            pair, old.get("action", "HOLD"), price, rsi,
                            status="processing",
                            ai_used=old.get("ai_used", False),
                            ai_action=old.get("ai_action", ""),
                            pred_ts=old.get("_pred_ts")
                        )

                    recent_trades = state.get("trades", [])[-10:]
                    logger.info(f"[AI Worker] Starting DeepSeek for {pair}...")
                    
                    sig = await self.sig_gen.generate_with_ai(
                        indicators    = ind,
                        ticker        = ticker,
                        pair          = pair,
                        candles       = candle_list,
                        recent_trades = recent_trades,
                    )
                    
                    # Update state silently when AI finishes
                    # IMPORTANT: Always set status="active" to unlock UI even on failure
                    _ai_candles = sig.get("ai_candles", [])
                    update_signal(
                        pair, sig.get("action", "HOLD"), price, rsi,
                        ai_used              = sig.get("ai_used", False),
                        ai_action            = sig.get("action", ""),
                        ai_reasoning         = sig.get("ai_reasoning", ""),
                        ai_confidence        = sig.get("ai_confidence", 0.0),
                        ai_predicted_close   = sig.get("ai_predicted_close", 0.0),
                        ai_predicted_high    = sig.get("ai_predicted_high", 0.0),
                        ai_predicted_low     = sig.get("ai_predicted_low", 0.0),
                        ai_predicted_direction = sig.get("ai_predicted_direction", ""),
                        ai_support           = sig.get("ai_support", 0.0),
                        ai_resistance        = sig.get("ai_resistance", 0.0),
                        ai_prediction_confidence = sig.get("ai_prediction_confidence", 0.0),
                        ai_provider          = sig.get("ai_provider", ""),
                        ai_candles           = _ai_candles,
                        num_candles          = len(_ai_candles) if _ai_candles else 6,
                        status               = "active",
                        expiry_ts            = (int(time.time()) + 60) if not sig.get("ai_used") else None
                    )
                    
                    if sig.get("ai_used"):
                        logger.info(f"[AI Worker] {pair} finished: {sig.get('ai_predicted_direction', '')}")
                    else:
                        logger.warning(f"[AI Worker] {pair} skipped AI (fallback used)")

                except Exception as e:
                    logger.error(f"[AI Worker Error] {pair}: {e}")
                    # Unlock UI and retry in 60s on crash
                    update_signal(pair, "HOLD", price, rsi, status="active", expiry_ts=int(time.time()) + 60)

            # Fire and forget the AI request with per-pair lock to avoid stacking
            if self.settings.STRATEGY.get("ai", {}).get("enabled", False):
                if not hasattr(self, "ai_tasks"): self.ai_tasks = {}
                existing = self.ai_tasks.get(pair)

                # Only trigger AI if the previous prediction has EXPIRED
                # This stops "AI THINKING..." from showing up before it's needed
                import time
                now_ts = int(time.time())
                state = load_state()
                cached_sig = state.get("signals", {}).get(pair, {})
                expiry_ts = cached_sig.get("_expiry_ts", 0)

                if now_ts >= expiry_ts:
                    if existing is None or existing.done():
                        task = asyncio.create_task(_background_ai())
                        self.ai_tasks[pair] = task # Reference protects from GC
                
                # Ensure the UI stays "active" if we have any cached AI data
                # This prevents the loop from accidentally resetting status to "active" 
                # if the background task is already in "processing" mode.
                pass 


            # For the main synchronous thread, we use raw indicators 
            state  = load_state()
            cached_sig = state.get("signals", {}).get(pair, {})
            signal = self.sig_gen.generate(ind, ticker) # basic tech fallback
            
            # Blend cached AI manually if it's fresh enough (e.g. within 10 mins)
            import time
            now_ts = int(time.time())
            ai_ts = cached_sig.get("_pred_ts", 0)
            
            if (now_ts - ai_ts) < 600 and cached_sig.get("ai_used"):
                signal["action"] = cached_sig.get("ai_action") or signal["action"]
                signal["ai_used"] = True
                signal["ai_prediction_confidence"] = cached_sig.get("ai_prediction_confidence", 0.0)
                signal["ai_support"] = cached_sig.get("ai_support", 0.0)
                signal["ai_resistance"] = cached_sig.get("ai_resistance", 0.0)
                signal["ai_predicted_close"] = cached_sig.get("ai_predicted_close", 0.0)

            action = signal["action"]
            score  = float(signal.get("score", 0.0))
            
            logger.info(
                f"{pair} | price=${price:.2f} | RSI={rsi:.1f} "
                f"| MACD={macd:.4f} | signal={action} "
                f"| score={score:.4f} (Background AI: {'ON' if self.settings.STRATEGY.get('ai', {}).get('enabled') else 'OFF'})"
            )

            update_signal(
                pair, action, price, rsi,
                ai_used              = cached_sig.get("ai_used", False),
                ai_action            = cached_sig.get("ai_action", ""),
                ai_reasoning         = cached_sig.get("ai_reasoning", ""),
                ai_confidence        = cached_sig.get("ai_confidence", 0.0),
                ai_predicted_close   = cached_sig.get("ai_predicted_close", 0.0),
                ai_predicted_high    = cached_sig.get("ai_predicted_high", 0.0),
                ai_predicted_low     = cached_sig.get("ai_predicted_low", 0.0),
                ai_predicted_direction = cached_sig.get("ai_predicted_direction", ""),
                ai_support           = cached_sig.get("ai_support", 0.0),
                ai_resistance        = cached_sig.get("ai_resistance", 0.0),
                ai_prediction_confidence = cached_sig.get("ai_prediction_confidence", 0.0),
                pred_ts              = cached_sig.get("_pred_ts"), 
                expiry_ts            = cached_sig.get("_expiry_ts"),
                status               = cached_sig.get("status", "active")
            )

            # -- Risk approval ---------------------------------
            approved = self.risk_mgr.approve(signal, price, pair)
            if not approved:
                return

            # -- Execute ---------------------------------------
            if action in ("BUY", "SELL"):
                size = self.risk_mgr.position_size(price, action)
                if size <= 0:
                    logger.warning(f"Size=0 for {pair}, skipping")
                    return

                # -- Dynamic SL/TP from AI prediction ---------
                extra_kwargs = {}
                pred_conf = signal.get("ai_prediction_confidence", 0.0)
                sl_pct = self.settings.STRATEGY["risk"]["stop_loss_pct"] # Default
                tp_pct = self.settings.STRATEGY["risk"]["take_profit_pct"] # Default

                if signal.get("ai_used") and pred_conf >= 0.60:
                    support    = signal.get("ai_support",    0.0)
                    resistance = signal.get("ai_resistance", 0.0)
                    pred_close = signal.get("ai_predicted_close", 0.0)

                    if action == "BUY" and support > 0 and pred_close > price:
                        sl_pct = max(0.5, (price - support) / price * 100)
                        tp_pct = max(1.0, (pred_close - price) / price * 100)
                        logger.info(
                            f"[AI-SL/TP] BUY {pair}: support={support:.4f} "
                            f"pred_close={pred_close:.4f} -> SL={sl_pct:.2f}% TP={tp_pct:.2f}%"
                        )

                    elif action == "SELL" and resistance > 0 and pred_close < price:
                        sl_pct = max(0.5, (resistance - price) / price * 100)
                        tp_pct = max(1.0, (price - pred_close) / price * 100)
                        logger.info(
                            f"[AI-SL/TP] SELL {pair}: resistance={resistance:.4f} "
                            f"pred_close={pred_close:.4f} -> SL={sl_pct:.2f}% TP={tp_pct:.2f}%"
                        )
                
                extra_kwargs = {"sl_pct": round(sl_pct, 2), "tp_pct": round(tp_pct, 2)}

                logger.info(
                    f"Executing {action} | {pair} | size={size} | ${price:.2f}"
                )
                result = await self.executor.execute(pair, action, size, price, **extra_kwargs)
                
                if result is not None:
                    # 1. Update internal position tracker
                    self._update_local_positions(pair, action, size, price, result, extra_kwargs)
                    
                    # 2. Notify risk manager (internal stats)
                    self.risk_mgr.record_trade(pair, action, size, price)

        except ccxt_async.NetworkError as e:
            logger.warning(f"Network error {pair}: {e}")
        except ccxt_async.ExchangeError as e:
            logger.warning(f"Exchange error {pair}: {e}")
        except KeyError as e:
            logger.error(f"Missing key {pair}: {e}\n{traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Error {pair}: {e}\n{traceback.format_exc()}")

    def _update_local_positions(self, pair, action, size, price, result, extra_kwargs={}):
        """Track open positions for dashboard display and SL/TP checks"""
        timestamp = datetime.datetime.now().isoformat()
        
        if action == "BUY":
            # Add or update position
            if pair not in self.open_positions:
                self.open_positions[pair] = {
                    "entry_price": price,
                    "size": size,
                    "timestamp": timestamp,
                    "sl_pct": extra_kwargs.get("sl_pct", 1.5),
                    "tp_pct": extra_kwargs.get("tp_pct", 3.0),
                    "symbol": pair
                }
            else:
                # Average down / Increase position
                curr = self.open_positions[pair]
                total_cost = (curr["entry_price"] * curr["size"]) + (price * size)
                new_size = curr["size"] + size
                curr["entry_price"] = total_cost / new_size
                curr["size"] = new_size
                
                # Update SL/TP if new ones are provided (e.g. from AI)
                if "sl_pct" in extra_kwargs: curr["sl_pct"] = extra_kwargs["sl_pct"]
                if "tp_pct" in extra_kwargs: curr["tp_pct"] = extra_kwargs["tp_pct"]
        
        elif action == "SELL":
            # Reduce or close position
            if pair in self.open_positions:
                curr = self.open_positions[pair]
                if size >= curr["size"] * 0.99: # Close if size is close to full
                    del self.open_positions[pair]
                else:
                    curr["size"] -= size

    async def _check_profit_taking(self):
        """Check open positions against current price for SL/TP"""
        if not self.open_positions:
            return

        for pair, pos in list(self.open_positions.items()):
            # Use last known price if available, otherwise might be stale
            price = self.last_prices.get(pair)
            if not price: 
                continue
            
            entry = pos["entry_price"]
            sl_pct = pos.get("sl_pct", 1.5)
            tp_pct = pos.get("tp_pct", 3.0)
            
            # Simple Long logic
            pct_change = (price - entry) / entry * 100
            
            action = None
            reason = ""
            
            if pct_change <= -sl_pct:
                action = "SELL"
                reason = f"Stop Loss hit ({pct_change:.2f}%)"
            elif pct_change >= tp_pct:
                action = "SELL"
                reason = f"Take Profit hit ({pct_change:.2f}%)"
                
            if action:
                logger.info(f"[{reason}] Executing {action} for {pair}")
                size = pos["size"]
                result = await self.executor.execute(pair, action, size, price)
                if result:
                    self._update_local_positions(pair, action, size, price, result)
                    # Note: record_trade is done inside executor
                    self.risk_mgr.record_trade(pair, action, size, price)

    async def _run_backtest(self, start_date, end_date):
        if not start_date or not end_date:
            logger.error("Backtest needs --start-date and --end-date")
            return
        bt = Backtester(self.pairs, self.indicator, self.sig_gen, self.risk_mgr, self.settings)
        results = await bt.run(start_date, end_date)
        bt.print_report(results)
        return results

    async def _sync_account_data(self):
        """Synchronize real balance from exchange with bot_state.json and RiskManager"""
        try:
            state = load_state()
            current_bal = float(state.get("balance", 10000.0))

            if self.mode == "live" and self.executor:
                logger.info("Syncing live balance from exchange...")
                balances = await self.executor.get_balance()
                # Use USDT or first available stable/pair base if possible
                total_val = balances.get("USDT", current_bal)
                # If USDT is not there, check for others or sum up (simplified for now)
                
                state["balance"] = round(float(total_val), 2)
                # If this is the first sync, update start_balance
                if state.get("start_balance", 0) <= 0 or state.get("last_update") == "—":
                    state["start_balance"] = state["balance"]
                
                from core.state import save_state
                save_state(state)
                current_bal = state["balance"]

            # Update RiskManager's local copy
            self.risk_mgr.account_bal = current_bal
            logger.info(f"[Sync] Balance synchronized: ${current_bal:.2f}")

        except Exception as e:
            logger.error(f"[Sync] Failed to sync account data: {e}")

    def _tf_seconds(self) -> int:
        return {"1m":60,"5m":300,"15m":900,"1h":3600,"4h":14400,"1d":86400}.get(
            self.settings.TIMEFRAME, 300
        )

    async def force_ai_refresh(self, pair: str = None):
        """Force the bot to re-analyze AI signals immediately by clearing expiry timestamps"""
        from core.state import load_state, save_state, update_signal
        state = load_state()
        signals = state.get("signals", {})
        
        target_pairs = [pair] if pair else self.pairs
        for p in target_pairs:
            if p in signals:
                signals[p]["_expiry_ts"] = 0 # Mark as expired
                # Force status to processing immediately so UI shows "AI THINKING"
                old = signals[p]
                update_signal(
                    p, old.get("action", "HOLD"), old.get("price", 0), old.get("rsi", 50),
                    status="processing",
                    ai_used=False,
                    expiry_ts=0
                )
        
        # Clear existing task trackers and cancel any running ones so they can be re-created
        if hasattr(self, "ai_tasks"):
            if pair:
                task = self.ai_tasks.get(pair)
                if task: task.cancel()
                if pair in self.ai_tasks: del self.ai_tasks[pair]
            else:
                for task in self.ai_tasks.values():
                    task.cancel()
                self.ai_tasks = {}
        
        logger.info(f"[BotEngine] AI refresh forced (Immediate UI Feedback) for {pair or 'all pairs'}")