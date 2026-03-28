"""
indicators/engine.py
Orchestrates all indicators, returns unified result dict.

FIX: Added ema_short/ema_long aliases (signal_generator reads these keys).
     Also added macd/macd_signal aliases and volume_ma for volume scoring.
     Previously ma_short/ma_long were returned but never read → EMA score=0
     → Market Strength gauge always showed 50%.
"""
from indicators.rsi import compute_rsi
from indicators.macd import compute_macd
from indicators.moving_averages import compute_ma
from indicators.bollinger_bands import compute_bb


class IndicatorEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def calculate(self, df):
        close  = df["close"]
        volume = df["volume"] if "volume" in df.columns else None

        rsi  = compute_rsi(close, self.cfg["rsi"]["period"])
        macd = compute_macd(
            close,
            self.cfg["macd"]["fast"],
            self.cfg["macd"]["slow"],
            self.cfg["macd"]["signal"],
        )
        ma   = compute_ma(close, self.cfg["ma"]["short"], self.cfg["ma"]["long"])
        bb   = compute_bb(close, self.cfg["bb"]["period"], self.cfg["bb"]["std_dev"])

        # Volume moving average (20-period simple) for volume scoring
        vol_ma = None
        if volume is not None:
            vol_ma = volume.rolling(window=20).mean()

        result = {
            # RSI
            "rsi":          float(rsi.iloc[-1]),

            # MACD — both naming conventions so nothing breaks
            "macd":         float(macd["macd"].iloc[-1]),
            "macd_line":    float(macd["macd"].iloc[-1]),
            "macd_signal":  float(macd["signal"].iloc[-1]),
            "macd_hist":    float(macd["histogram"].iloc[-1]),

            # Moving averages — BOTH aliases (signal_generator uses ema_short/ema_long)
            "ma_short":     float(ma["short"].iloc[-1]),
            "ma_long":      float(ma["long"].iloc[-1]),
            "ema_short":    float(ma["short"].iloc[-1]),   # ← FIX: was missing
            "ema_long":     float(ma["long"].iloc[-1]),    # ← FIX: was missing

            # Bollinger Bands
            "bb_upper":     float(bb["upper"].iloc[-1]),
            "bb_lower":     float(bb["lower"].iloc[-1]),
            "bb_mid":       float(bb["mid"].iloc[-1]),

            # Price
            "close":        float(close.iloc[-1]),
        }

        # Volume fields (safe — may not always be present)
        if volume is not None:
            result["volume"]    = float(volume.iloc[-1])
            result["volume_ma"] = float(vol_ma.iloc[-1]) if vol_ma is not None and not vol_ma.isnull().iloc[-1] else float(volume.iloc[-1])
        else:
            result["volume"]    = 0.0
            result["volume_ma"] = 0.0

        return result