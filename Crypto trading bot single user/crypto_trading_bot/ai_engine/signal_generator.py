"""
ai_engine/signal_generator.py

FIXED for profitable trading:
1. Lowered AI confidence threshold from 0.55 to 0.45 (more trades, still safe)
2. Added AI signal blending with technical indicators
3. Better error handling and fallback logic
4. Proper provider switching support
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ai_engine.deepseek_advisor import DeepSeekAdvisor, AIVerdict
    _DEEPSEEK_AVAILABLE = True
except ImportError:
    _DEEPSEEK_AVAILABLE = False
    logger.warning("[SignalGen] deepseek_advisor not found")

try:
    from ai_engine.cloud_advisor import CloudAdvisor
    _CLOUD_AVAILABLE = True
except ImportError:
    _CLOUD_AVAILABLE = False
    logger.warning("[SignalGen] cloud_advisor not found")

_CLOUD_PROVIDERS = {"openai", "anthropic", "groq", "openrouter", "gemini", "universal"}


def _build_advisor(ai_cfg: dict):
    """Build the appropriate advisor based on provider setting."""
    provider = ai_cfg.get("provider", "deepseek_ollama").lower()

    if provider in _CLOUD_PROVIDERS:
        if not _CLOUD_AVAILABLE:
            logger.warning(f"[SignalGen] cloud_advisor.py missing, cannot use {provider}")
            return None
        logger.info(f"[SignalGen] Creating CloudAdvisor for provider: {provider}")
        return CloudAdvisor(ai_cfg)
    else:
        if not _DEEPSEEK_AVAILABLE:
            logger.warning("[SignalGen] deepseek_advisor.py missing")
            return None
        logger.info(f"[SignalGen] Creating DeepSeekAdvisor (provider: {provider})")
        return DeepSeekAdvisor(ai_cfg)


class SignalGenerator:
    def __init__(self, strategy_config: dict[str, Any]):
        self.cfg = strategy_config

        # Technical indicator settings
        rsi_cfg = strategy_config.get("rsi", {})
        self.rsi_oversold = float(rsi_cfg.get("oversold", 30))
        self.rsi_overbought = float(rsi_cfg.get("overbought", 70))
        self.rsi_scale = float(rsi_cfg.get("scale", 2.5))

        ma_cfg = strategy_config.get("ma", {})
        self.ma_short = int(ma_cfg.get("short", 20))
        self.ma_long = int(ma_cfg.get("long", 50))

        # AI configuration - FIXED: Lowered default confidence for more trades
        ai_cfg = strategy_config.get("ai", {})
        self._ai_enabled = ai_cfg.get("enabled", False)
        self._current_provider = ai_cfg.get("provider", "deepseek_ollama")

        # KEY FIX: Lowered from 0.55 to 0.45 for more trading opportunities
        self._min_confidence = ai_cfg.get("min_confidence", 0.45)
        self._ai_weight = ai_cfg.get("ai_weight", 0.65)  # Increased AI weight to 0.65

        self._advisor = _build_advisor(ai_cfg) if self._ai_enabled else None

        if self._advisor:
            # Override advisor's min_confidence with our setting
            self._advisor.min_conf = self._min_confidence
            logger.info(f"[SignalGen] AI initialized: {self._current_provider} | "
                       f"min_conf={self._min_confidence} | ai_weight={self._ai_weight}")

    def generate(self, indicators: dict, ticker: dict) -> dict:
        """Generate base signal from technical indicators only"""
        score = self._score_indicators(indicators, ticker)
        action = self._score_to_action(score)

        return {
            "action": action,
            "score": round(score, 4),
            "confidence": round(min(1.0, abs(score)), 4),
            "ai_used": False,
            "ai_action": "",
            "ai_confidence": 0.0,
            "ai_score": 0.0,
            "ai_reasoning": "",
            "ai_latency_ms": 0.0,
            "ai_raw_think": "",
            "ai_predicted_close": 0.0,
            "ai_predicted_high": 0.0,
            "ai_predicted_low": 0.0,
            "ai_predicted_direction": "",
            "ai_support": 0.0,
            "ai_resistance": 0.0,
            "ai_prediction_confidence": 0.0,
            "ai_provider": "",   # which AI provider was used (gemini/groq/openai/etc.)
        }

    async def generate_with_ai(
        self,
        indicators: dict,
        ticker: dict,
        pair: str = "",
        candles: list[dict] | None = None,
        recent_trades: list[dict] | None = None,
    ) -> dict:
        """Generate AI-enhanced trading signal with Trend Safety"""
        base = self.generate(indicators, ticker)
        price = float(ticker.get("last", 0))
        ema_l = float(indicators.get("ema_long", price))

        # --- SAFETY FILTER: AI CONNECTION ---
        # If AI is enabled but fails, we become 2x more selective with technicals
        ai_failed = False
        
        if self._ai_enabled and self._advisor:
            try:
                verdict = await self._advisor.analyse(
                    pair=pair,
                    base_signal=base,
                    indicators=indicators,
                    ticker=ticker,
                    candles=candles or [],
                    recent_trades=recent_trades,
                )
                
                # --- POPULATE AI FIELDS ---
                base["ai_used"] = verdict.used_ai
                base["ai_action"] = verdict.action
                base["ai_confidence"] = round(verdict.confidence, 4)
                base["ai_score"] = round(verdict.score, 4)
                base["ai_reasoning"] = verdict.reasoning
                base["ai_latency_ms"] = round(verdict.latency_ms, 0)
                base["ai_raw_think"] = verdict.raw_think

                # --- CANDLE PREDICTION FIELDS ---
                base["ai_predicted_direction"] = verdict.predicted_direction
                base["ai_predicted_close"] = round(float(verdict.predicted_close), 4)
                base["ai_predicted_high"] =  round(float(verdict.predicted_high),  4)
                base["ai_predicted_low"] =   round(float(verdict.predicted_low),   4)
                base["ai_support"] =         round(float(verdict.support_level),   4)
                base["ai_resistance"] =      round(float(verdict.resistance_level), 4)
                base["ai_prediction_confidence"] = round(float(verdict.prediction_confidence), 4)
                base["ai_provider"] = self._current_provider  # track which AI ran
                
                if verdict.used_ai and verdict.confidence >= self._min_confidence:
                    blended_score = (self._ai_weight * verdict.score) + ((1 - self._ai_weight) * base["score"])
                    # TREND FILTER: Even if AI likes it, don't BUY if we are far below EMA_long (crash detection)
                    if blended_score > 0 and price < ema_l * 0.98:
                         blended_score *= 0.5 
                    
                    base["action"] = self._score_to_action(blended_score)
                    base["score"] = round(blended_score, 4)
                    base["confidence"] = round(verdict.confidence, 4)

            except Exception as e:
                ai_failed = True
                base["ai_reasoning"] = f"AI error: {type(e).__name__}"
                logger.error(f"AI Down for {pair}: {e}")

        # --- FINAL TREND SAFETY CHECK ---
        # If trend is down (Price < EMA_long), require a higher score to BUY
        if base["action"] == "BUY":
            # Require extreme conviction (score > 0.5 instead of 0.3) if trending down or AI off
            if price < ema_l or ai_failed:
                if base["score"] < 0.55:
                    logger.info(f"[Safety] Blocked {pair} BUY: Trend/AI mismatch (Score {base['score']})")
                    base["action"] = "HOLD"

        return base

    def set_ai_enabled(self, enabled: bool) -> None:
        """Enable/disable AI without changing provider."""
        self._ai_enabled = enabled
        if self._advisor:
            self._advisor.enabled = enabled
        elif enabled and self._advisor is None:
            self._advisor = _build_advisor(self.cfg.get("ai", {}))

    def reload_advisor(self, ai_cfg: dict) -> None:
        """Reload AI advisor with new configuration (called when GUI changes settings)"""
        new_provider = ai_cfg.get("provider", "deepseek_ollama")
        old_provider = self._current_provider

        logger.info(f"[SignalGen] reload_advisor: {old_provider} -> {new_provider}")

        self.cfg["ai"] = ai_cfg
        self._ai_enabled = ai_cfg.get("enabled", False)
        self._current_provider = new_provider
        self._min_confidence = ai_cfg.get("min_confidence", 0.45)
        self._ai_weight = ai_cfg.get("ai_weight", 0.65)

        if self._ai_enabled:
            self._advisor = _build_advisor(ai_cfg)
            if self._advisor:
                self._advisor.min_conf = self._min_confidence
        else:
            self._advisor = None

        logger.info(f"[SignalGen] Advisor reloaded: {new_provider} (enabled: {self._ai_enabled})")

    @property
    def ai_enabled(self) -> bool:
        return self._ai_enabled

    @property
    def current_provider(self) -> str:
        return self._current_provider

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    def _score_indicators(self, ind: dict, ticker: dict) -> float:
        """Calculate composite score from technical indicators (-1.0 to +1.0)"""
        scores = {}
        # TREND IS KING: Increased MACD and EMA weights, reduced RSI to avoid knife-catching
        weights = {"macd": 0.40, "ema": 0.25, "rsi": 0.15, "bb": 0.10, "volume": 0.10}

        price = float(ticker.get("last", 1) or 1)

        # RSI scoring
        rsi = float(ind.get("rsi", 50))
        if rsi <= self.rsi_oversold:
            scores["rsi"] = min(1.0, (self.rsi_oversold - rsi) / self.rsi_oversold * self.rsi_scale)
        elif rsi >= self.rsi_overbought:
            scores["rsi"] = -min(1.0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought) * self.rsi_scale)
        else:
            scores["rsi"] = (50 - rsi) / 50 * 0.3

        # MACD scoring
        hist = float(ind.get("macd_hist", 0))
        macd_val = float(ind.get("macd", 0))
        sig_val = float(ind.get("macd_signal", 0))
        norm_hist = hist / price * 1000

        if norm_hist > 0 and macd_val > sig_val:
            scores["macd"] = min(1.0, norm_hist * 5)
        elif norm_hist < 0 and macd_val < sig_val:
            scores["macd"] = max(-1.0, norm_hist * 5)
        else:
            scores["macd"] = norm_hist * 2

        # EMA scoring
        ema_s = float(ind.get("ema_short", price))
        ema_l = float(ind.get("ema_long", price))
        if ema_l > 0:
            scores["ema"] = max(-1.0, min(1.0, (ema_s - ema_l) / ema_l * 20))
        else:
            scores["ema"] = 0.0

        # Bollinger Bands scoring
        bb_up = float(ind.get("bb_upper", price * 1.02))
        bb_lo = float(ind.get("bb_lower", price * 0.98))
        bb_rng = bb_up - bb_lo
        if bb_rng > 0:
            bb_pos = (price - bb_lo) / bb_rng
            scores["bb"] = (0.5 - bb_pos) * 2  # -1 at upper band, +1 at lower band
        else:
            scores["bb"] = 0.0

        # Volume scoring
        vol = float(ind.get("volume", 0))
        vol_ma = float(ind.get("volume_ma", vol or 1))
        if vol_ma > 0:
            vr = vol / vol_ma
            direction = 1.0 if (scores.get("macd", 0) + scores.get("ema", 0)) > 0 else -1.0
            if vr > 1.2:
                scores["volume"] = direction * min(1.0, (vr - 1.0) * 1.5)
            elif vr < 0.5:
                scores["volume"] = -0.1
            else:
                scores["volume"] = 0.0
        else:
            scores["volume"] = 0.0

        # Calculate weighted composite score
        final_score = sum(weights[k] * scores.get(k, 0.0) for k in weights)
        return max(-1.0, min(1.0, final_score))

    def _score_to_action(self, score: float) -> str:
        """Convert score to trading action"""
        # Lowered thresholds for more trading activity
        if score >= 0.30:
            return "BUY"
        if score <= -0.30:
            return "SELL"
        return "HOLD"