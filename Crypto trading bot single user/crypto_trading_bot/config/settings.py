"""
config/settings.py
Bot configuration — supports both stateless (RAM-only) and legacy (disk-based) modes.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    def __init__(self, config_dict: dict = None):
        # ── Strategy / indicator defaults ─────────────────────
        self.STRATEGY = {
            "rsi":  {"oversold": 30, "overbought": 70, "period": 14, "scale": 2.5},
            "ma":   {"short": 20, "long": 50},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bb":   {"period": 20, "std_dev": 2.0},
            "risk": {
                "stop_loss_pct": 1.5, "take_profit_pct": 3.0,
                "max_position_pct": 5.0, "max_open_trades": 3,
                "daily_loss_limit_pct": 3.0,
            },
            "ai": {
                "enabled":        False,
                "provider":       "deepseek_ollama",
                "model":          "deepseek-r1:7b",
                "ollama_url":     "http://localhost:11434",
                "timeout":        300,
                "min_confidence": 0.55,
                "ai_weight":      0.60,
            },
        }
        self.TIMEFRAME = "5m"
        self.EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
        self.SANDBOX     = os.getenv("SANDBOX", "false").lower() in ("true", "1", "yes")

        # If a config dict is provided (stateless/multi-user), use it
        if config_dict:
            self._apply_dict(config_dict)
        else:
            # Otherwise fallback to disk (legacy/single-user)
            self.load_from_file()

        self._reload_credentials()

    def _apply_dict(self, config: dict):
        """Apply a raw config dict (nested) into self.STRATEGY and attrs."""
        for section in ("rsi", "ma", "macd", "bb", "risk", "ai"):
            if section in config and isinstance(config[section], dict):
                self.STRATEGY[section].update(config[section])
        if "timeframe"   in config: self.TIMEFRAME   = config["timeframe"]
        if "exchange_id" in config: self.EXCHANGE_ID = config["exchange_id"]
        if "sandbox"     in config: self.SANDBOX     = bool(config["sandbox"])

    def _reload_credentials(self):
        """Read API key/secret for current EXCHANGE_ID from env."""
        _prefix = str(self.EXCHANGE_ID).upper()
        self.EXCHANGE_API_KEY    = os.getenv(f"{_prefix}_API_KEY",    "")
        self.EXCHANGE_API_SECRET = os.getenv(f"{_prefix}_API_SECRET", "")
        # Aliases
        self.API_KEY    = self.EXCHANGE_API_KEY
        self.API_SECRET = self.EXCHANGE_API_SECRET

    def load_from_file(self):
        """LEGACY: Merge saved strategy config from disk."""
        path = Path("config/strategy_config.json")
        if path.exists():
            try:
                self._apply_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception: pass

    def has_api_keys(self) -> bool:
        return bool(self.EXCHANGE_API_KEY and self.EXCHANGE_API_SECRET)

    def to_dict(self) -> dict:
        return {
            **self.STRATEGY,
            "timeframe":    self.TIMEFRAME,
            "exchange_id":  self.EXCHANGE_ID,
            "sandbox":      self.SANDBOX,
            "has_api_keys": self.has_api_keys(),
        }

def load_settings(config_dict: dict = None) -> Settings:
    return Settings(config_dict=config_dict)