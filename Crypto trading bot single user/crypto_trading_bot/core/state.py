"""
core/state.py — Stateless in-memory architecture
==================================================
DESIGN: No JSON files. No disk writes. Pure in-memory state.
- Supports 1000+ concurrent users (stateless server)
- All user-specific data lives in browser localStorage/IndexedDB
- Server holds only the current trading session's live state
- State resets cleanly on server restart (bot re-syncs from exchange)
"""
import threading
import datetime
import time

_lock  = threading.Lock()

_DEFAULTS = {
    "auto_trading": False,
    "mode":         "paper",
    "running":      False,
    "balance":      10000.0,
    "start_balance":10000.0,
    "pnl":          0.0,
    "trades":       [],
    "signals":      {},
    "pairs":        ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT"],
    "last_update":  "—",
}

# Single in-memory state dict — shared across the whole process
_state: dict = {}


def _init():
    global _state
    if not _state:
        _state = dict(_DEFAULTS)
        _state["trades"]  = []
        _state["signals"] = {}


# ── Public API ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    with _lock:
        _init()
        return dict(_state)   # shallow copy — safe for readers


def save_state(new_state: dict) -> None:
    with _lock:
        _init()
        _state.update(new_state)
        _state["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_auto_trading(v: bool):
    with _lock:
        _init()
        _state["auto_trading"] = v


def set_running(v: bool):
    with _lock:
        _init()
        _state["running"] = v


def set_mode(mode: str):
    with _lock:
        _init()
        _state["mode"] = mode


def set_pairs(pairs: list):
    with _lock:
        _init()
        _state["pairs"] = pairs


def reset_balance():
    with _lock:
        _init()
        _state["balance"]       = 10000.0
        _state["start_balance"] = 10000.0
        _state["pnl"]           = 0.0
        _state["trades"]        = []
        _state["signals"]       = {}


def update_balance(new_balance: float, pnl: float):
    with _lock:
        _init()
        _state["balance"] = round(float(new_balance), 2)
        _state["pnl"]     = round(float(pnl), 2)


def record_trade(trade: dict):
    with _lock:
        _init()
        trades = _state.setdefault("trades", [])
        trades.append(trade)
        _state["trades"] = trades[-500:]
        _state["pnl"]    = round(sum(float(t.get("pnl", 0)) for t in _state["trades"]), 2)


def update_signal(
    pair:                     str,
    action:                   str,
    price:                    float,
    rsi:                      float,
    ai_used:                  bool  = False,
    ai_reasoning:             str   = "",
    ai_confidence:            float = 0.0,
    ai_predicted_close:       float = 0.0,
    ai_predicted_high:        float = 0.0,
    ai_predicted_low:         float = 0.0,
    ai_predicted_direction:   str   = "",
    ai_support:               float = 0.0,
    ai_resistance:            float = 0.0,
    ai_prediction_confidence: float = 0.0,
    pred_ts:                  int   = None,
    status:                   str   = "active",
    ai_action:                str   = "",
    expiry_ts:                int   = None,
    score:                    float = 0.0,
    ai_provider:              str   = "",
    ai_candles:               list  = None,
    num_candles:              int   = 6,
):
    """
    Update signal state for a pair.

    Expiry = pred_ts + (num_candles × timeframe_seconds)
    For 5m TF and 6 candles → 30 minutes of valid prediction window.
    """
    from config.settings import load_settings
    settings          = load_settings()
    base_tf           = settings.TIMEFRAME
    lookup            = {"m": 60, "h": 3600, "d": 86400}
    timeframe_seconds = 300
    if base_tf and base_tf[-1] in lookup and base_tf[:-1].isdigit():
        timeframe_seconds = int(base_tf[:-1]) * lookup[base_tf[-1]]

    now_ts = int(time.time())

    with _lock:
        _init()
        existing = _state.get("signals", {}).get(pair, {})

        final_pred_ts = pred_ts if pred_ts is not None else existing.get("_pred_ts", now_ts)

        n_candles     = max(1, num_candles)
        pred_duration = timeframe_seconds * n_candles   # e.g. 5min × 6 = 30min

        if expiry_ts is not None:
            final_exp_ts = expiry_ts
        elif existing.get("_expiry_ts") and existing["_expiry_ts"] > now_ts:
            # Keep existing expiry if prediction is still live — don't reset mid-cycle
            final_exp_ts = existing["_expiry_ts"]
        else:
            final_exp_ts = final_pred_ts + pred_duration

        # Preserve timestamp on minor updates so UI doesn't flicker
        if status in ("active", "tech", "processing") and existing.get("time"):
            final_time = existing["time"]
        else:
            final_time = datetime.datetime.now().strftime("%H:%M:%S")

        _state.setdefault("signals", {})[pair] = {
            "status":                   status,
            "action":                   action,
            "ai_action":                ai_action,
            "price":                    round(float(price), 4),
            "rsi":                      round(float(rsi), 2),
            "score":                    round(float(score), 4),
            "time":                     final_time,
            "ai_used":                  ai_used,
            "ai_reasoning":             ai_reasoning,
            "ai_confidence":            round(float(ai_confidence), 3),
            "ai_predicted_close":       round(float(ai_predicted_close), 4),
            "ai_predicted_high":        round(float(ai_predicted_high),  4),
            "ai_predicted_low":         round(float(ai_predicted_low),   4),
            "ai_predicted_direction":   ai_predicted_direction,
            "ai_support":               round(float(ai_support),    4),
            "ai_resistance":            round(float(ai_resistance), 4),
            "ai_prediction_confidence": round(float(ai_prediction_confidence), 3),
            "ai_provider":              ai_provider,
            "ai_candles":               ai_candles or [],
            "ai_num_candles":           n_candles,
            "_pred_ts":                 final_pred_ts,
            "_expiry_ts":               final_exp_ts,
            "_candle_id":               f"C-{final_pred_ts // max(timeframe_seconds, 1)}",
        }
        _state["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")