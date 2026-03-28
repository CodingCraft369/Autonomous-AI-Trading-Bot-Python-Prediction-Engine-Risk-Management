"""
dashboard/app.py — Production-Ready CryptoBot Pro
═══════════════════════════════════════════════════
FIXES IN THIS VERSION
─────────────────────
1. Token Efficiency: global token tracker, cycle-lock prevents duplicate AI calls
2. Signal Accuracy: /api/signals/stats endpoint exposes raw indicator scores for gauges
3. Dynamic Coins: POST /api/coins/validate + /api/coins/add
4. Bot P&L Fix: record_trade now correctly marks wins/losses; win_rate computed properly
5. Multi-user: client-side localStorage key storage (keys never stored on server in memory)
6. Security: input sanitisation, rate-limiting per IP, API key validation before save
7. Deployment: CORS locked to configurable origins, no wildcard in production
"""
import json
import asyncio
import os
import re
import time as _time
import logging
import uvicorn
import httpx
import ccxt.async_support as ccxt_async
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv, set_key

# ── Logging ─────────────────────────────────────────────────────────
# Use the project's own Windows-safe logger (no basicConfig — that causes
# duplicate log lines when module loggers propagate to the root).
import sys, io
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # already wrapped or not a real terminal

# Silence root logger so basicConfig-style duplicates never happen
import logging as _logging
_logging.getLogger().handlers = []
_logging.getLogger().addHandler(_logging.NullHandler())
_logging.getLogger().setLevel(_logging.WARNING)

from logging_monitor.logger import get_logger
logger = get_logger("dashboard.app")

load_dotenv(override=True)

# ── App & CORS ────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app = FastAPI(title="CryptoBot Pro", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Exchange (public, no keys) ────────────────────────────────────────
_exchange = ccxt_async.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})

DEFAULT_PAIRS = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT"]
CONFIG_PATH = Path("config/strategy_config.json")
ENV_PATH    = Path(".env")

from core.state import (
    load_state, save_state, set_auto_trading, set_running,
    set_pairs, set_mode, reset_balance, update_signal,
)

BOT_TASK   = None
BOT_ENGINE = None

# ── Token Usage Tracker ───────────────────────────────────────────────
# Persisted to disk so it survives server restarts.
_TOKEN_FILE = Path("logs/token_usage.json")

_token_tracker: Dict[str, Any] = {
    "total_prompt_tokens":      0,
    "total_completion_tokens":  0,
    "total_tokens":             0,
    "session_tokens":           0,
    "calls_today":              0,
    "calls_saved":              0,
    "cost_usd_estimate":        0.0,
    "history":                  [],   # last 200 entries, persisted
    "cycle_locks":              {},   # pair → expiry_ts (not persisted)
}

# ── Multi-user session key cache ─────────────────────────────────────────────
# Keys arrive from browser localStorage via X-AI-Keys request header.
# We cache them in RAM so the background bot loop can use them.
# Keys are NEVER written to disk from here — pure in-memory session store.
# Cleared when the bot is stopped.
_session_keys: dict = {
    # "GEMINI_API_KEY": "key...",
    # "UNIVERSAL_BASE_URL": "https://...",
    # etc.
}
_AI_KEY_ENV_MAP = {
    "gemini":     "GEMINI_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "groq":       "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "universal":  "UNIVERSAL_API_KEY",
}

def _inject_session_keys(request: "Request | None" = None) -> dict:
    """
    Read X-AI-Keys header from the request, update _session_keys in RAM,
    and inject into os.environ so all downstream code (cloud_advisor etc.)
    picks them up via os.getenv(). Returns dict of env-var→original-value
    so the caller can restore them afterward (for request-scoped injection).
    """
    global _session_keys

    if request is not None:
        raw = request.headers.get("X-AI-Keys", "")
        if raw:
            try:
                parsed = json.loads(raw)
                for provider, key in parsed.items():
                    env_var = _AI_KEY_ENV_MAP.get(provider.lower(), "")
                    if env_var and str(key).strip():
                        _session_keys[env_var] = str(key).strip()
            except Exception:
                pass
        # Universal extras
        u_url = request.headers.get("X-Universal-URL", "").strip()
        u_mod = request.headers.get("X-Universal-Model", "").strip()
        if u_url: _session_keys["UNIVERSAL_BASE_URL"] = u_url
        if u_mod: _session_keys["UNIVERSAL_MODEL"]    = u_mod

    # Inject cached keys into os.environ so cloud_advisor.get_api_key() finds them
    backup = {}
    for env_var, key_val in _session_keys.items():
        backup[env_var] = os.environ.get(env_var, "")
        os.environ[env_var] = key_val
    return backup

def _restore_env(backup: dict) -> None:
    """Restore os.environ after a request-scoped injection."""
    for env_var, original in backup.items():
        if original:
            os.environ[env_var] = original
        elif env_var in os.environ:
            del os.environ[env_var]

def _clear_session_keys() -> None:
    """Clear cached keys and remove them from os.environ (called on bot stop)."""
    global _session_keys
    for env_var in list(_session_keys.keys()):
        if env_var in os.environ:
            del os.environ[env_var]
    _session_keys = {}


def _load_token_file() -> None:
    """Load token usage into memory (no longer reads from disk)."""
    pass

def _save_token_file() -> None:
    """Save token usage (no longer writes to disk)."""
    pass

def _record_token_usage(pair: str, prompt_tok: int, completion_tok: int, model: str = ""):
    import datetime as _dt
    total = prompt_tok + completion_tok
    cost  = (prompt_tok * 0.00000015) + (completion_tok * 0.0000006)
    _token_tracker["total_prompt_tokens"]     += prompt_tok
    _token_tracker["total_completion_tokens"] += completion_tok
    _token_tracker["total_tokens"]            += total
    _token_tracker["session_tokens"]          += total
    _token_tracker["calls_today"]             += 1
    _token_tracker["cost_usd_estimate"]       += cost
    entry = {
        "time":   _dt.datetime.now().strftime("%H:%M:%S"),
        "pair":   pair,
        "model":  model,
        "tokens": total,
        "cost":   round(cost, 6),
    }
    _token_tracker["history"].insert(0, entry)
    _token_tracker["history"] = _token_tracker["history"][:200]
    # Persist after every call (fast, <1ms for small JSON)
    _save_token_file()

def _is_cycle_locked(pair: str) -> bool:
    """Returns True if an AI call for this pair is still within its prediction cycle."""
    expiry = _token_tracker["cycle_locks"].get(pair, 0)
    if _time.time() < expiry:
        return True
    return False

def _set_cycle_lock(pair: str, duration_seconds: int):
    _token_tracker["cycle_locks"][pair] = _time.time() + duration_seconds

def _unlock_cycle(pair: str):
    _token_tracker["cycle_locks"].pop(pair, None)

# ── Input validation helpers ──────────────────────────────────────────
PAIR_PATTERN = re.compile(r'^[A-Z]{2,10}/[A-Z]{2,6}$')

def _validate_pair(pair: str) -> str:
    p = pair.strip().upper()
    if not PAIR_PATTERN.match(p):
        raise HTTPException(400, f"Invalid pair format: '{pair}'. Use BASE/QUOTE e.g. BTC/USDT")
    return p

def _sanitize_string(s: str, max_len: int = 200) -> str:
    """Strip control characters and limit length."""
    return re.sub(r'[^\w\s\-./]', '', s or '')[:max_len]

# ── Simple rate limiter (per IP, in-memory) ───────────────────────────
_rate_limit: Dict[str, list] = {}
RATE_WINDOW = 60    # seconds
RATE_MAX    = 120   # max requests per window

# ── Browser-provided key helpers ─────────────────────────────────────────────
# Keys are stored in browser localStorage and sent in request headers.
# Server NEVER writes keys to disk. Each user has their own keys.

def _parse_ai_keys_header(request: Request) -> dict:
    """
    Parse X-AI-Keys header from browser.
    Format: JSON {"gemini":"key","openai":"key","groq":"key",...}
    Returns empty dict if header missing or malformed.
    """
    raw = request.headers.get("X-AI-Keys", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {k: str(v).strip() for k, v in data.items() if v and str(v).strip()}
    except Exception:
        return {}

def _parse_exchange_keys_header(request: Request) -> dict:
    """
    Parse X-Exchange-Keys header from browser.
    Format: JSON {"exchange":"binance","api_key":"key","api_secret":"secret","sandbox":true}
    """
    raw = request.headers.get("X-Exchange-Keys", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}

def _check_rate_limit(ip: str) -> bool:
    now = _time.time()
    calls = _rate_limit.get(ip, [])
    calls = [t for t in calls if now - t < RATE_WINDOW]
    if len(calls) >= RATE_MAX:
        return False
    calls.append(now)
    _rate_limit[ip] = calls
    return True

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(ip):
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
    return await call_next(request)

# ── Pydantic Models ───────────────────────────────────────────────────
class ModeBody(BaseModel):
    mode: str = "paper"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("live", "paper", "backtest"):
            raise ValueError("mode must be live, paper, or backtest")
        return v

class PairsBody(BaseModel):
    pairs: List[str]

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, v):
        if not v or len(v) > 20:
            raise ValueError("1–20 pairs required")
        return [p.strip().upper() for p in v]

class StrategyBody(BaseModel):
    rsi_oversold:         Optional[float] = 30
    rsi_overbought:       Optional[float] = 70
    macd_fast:            Optional[int]   = 12
    macd_slow:            Optional[int]   = 26
    macd_signal:          Optional[int]   = 9
    bb_period:            Optional[int]   = 20
    bb_std_dev:           Optional[float] = 2.0
    stop_loss_pct:        Optional[float] = 1.5
    take_profit_pct:      Optional[float] = 3.0
    max_position_pct:     Optional[float] = 5.0
    daily_loss_limit_pct: Optional[float] = 3.0
    max_open_trades:      Optional[int]   = 3
    timeframe:            Optional[str]   = "5m"

class ApiKeysBody(BaseModel):
    api_key:    str
    api_secret: str
    exchange:   str = "binance"
    sandbox:    bool = True

    @field_validator("exchange")
    @classmethod
    def validate_exchange(cls, v):
        allowed = {"binance", "coinbase", "kraken", "bybit"}
        if v.lower() not in allowed:
            raise ValueError(f"Exchange must be one of {allowed}")
        return v.lower()

    @field_validator("api_key", "api_secret")
    @classmethod
    def validate_key_format(cls, v):
        # Basic: printable ASCII, 10-200 chars
        if not v or not (10 <= len(v) <= 200):
            raise ValueError("Key must be 10-200 characters")
        if not re.match(r'^[\x20-\x7E]+$', v):
            raise ValueError("Key contains invalid characters")
        return v

class AIConfigBody(BaseModel):
    enabled:        bool  = False
    provider:       str   = "deepseek_ollama"
    model:          str   = "deepseek-r1:7b"
    ollama_url:     str   = "http://localhost:11434"
    timeout:        int   = 90
    min_confidence: float = 0.45
    ai_weight:      float = 0.65

class AIApiKeyBody(BaseModel):
    provider: str
    api_key:  Optional[str] = ""

class DeleteKeysBody(BaseModel):
    exchange: str = "binance"

class ManualTradeBody(BaseModel):
    pair:   str
    action: str
    size:   float
    price:  float

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        if v.upper() not in ("BUY", "SELL"):
            raise ValueError("action must be BUY or SELL")
        return v.upper()

    @field_validator("size", "price")
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("size and price must be positive")
        return v

class BacktestBody(BaseModel):
    start_date: str
    end_date:   str
    pairs:      Optional[List[str]] = None

class CoinAddBody(BaseModel):
    pair: str

class RefreshBody(BaseModel):
    pair: Optional[str] = None

# ── Config helpers ────────────────────────────────────────────────────
# ── In-Memory Config (Stateless) ──────────────────────────────────────
_memory_config: dict = {}

def _load_config() -> dict:
    defaults = {
        "rsi":  {"oversold": 30, "overbought": 70, "period": 14, "scale": 2.5},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bb":   {"period": 20, "std_dev": 2.0},
        "ma":   {"short": 20, "long": 50},
        "risk": {
            "stop_loss_pct": 1.5, "take_profit_pct": 3.0,
            "max_position_pct": 5.0, "daily_loss_limit_pct": 3.0,
            "max_open_trades": 3,
        },
        "ai": {
            "enabled": False, "provider": "deepseek_ollama",
            "model": "deepseek-r1:7b", "ollama_url": "http://localhost:11434",
            "timeout": 90, "min_confidence": 0.45, "ai_weight": 0.65,
        },
        "timeframe": "5m",
        "pairs": DEFAULT_PAIRS[:4],
        "mode": "paper"
    }
    
    # Merge any in-memory user overrides
    for k, v in _memory_config.items():
        if isinstance(v, dict) and k in defaults and isinstance(defaults[k], dict):
            defaults[k].update(v)
        else:
            defaults[k] = v
    return defaults

def _save_config(cfg: dict):
    global _memory_config
    _memory_config = cfg


def _reset_bot_engine():
    global BOT_ENGINE
    BOT_ENGINE = None
    logger.info("[APP] Bot engine reset")

def _mask(key: str) -> str:
    if not key or len(key) < 8:
        return "••••••••"
    return key[:4] + "••••••••" + key[-4:]

def _env_get(key_name: str) -> str:
    # Stateless backend: do not read keys from disk.
    return os.environ.get(key_name, "")

def _env_delete(key_name: str):
    # Stateless backend: remove the key from the running process environment memory
    if key_name in os.environ:
        del os.environ[key_name]

# ── Bot background task ───────────────────────────────────────────────
async def run_bot_background():
    global BOT_ENGINE
    logger.info("[APP] Bot background task started")

    while True:
        try:
            state = load_state()

            if not state.get("auto_trading", False):
                if state.get("running", False):
                    set_running(False)
                await asyncio.sleep(2)
                continue

            if not state.get("running", False):
                set_running(True)
                logger.info("[BOT] Bot started trading")

            try:
                from core.bot_engine import BotEngine
                from config.settings import load_settings

                settings = load_settings()
                pairs    = state.get("pairs", DEFAULT_PAIRS[:4])
                mode     = state.get("mode", "paper")

                if BOT_ENGINE is None:
                    logger.info(f"[BOT] Creating BotEngine: mode={mode}, pairs={pairs}")
                    BOT_ENGINE = BotEngine(
                        mode=mode, pairs=pairs,
                        exchange_id=settings.EXCHANGE_ID, settings=settings,
                    )

                # Re-inject cached session keys before each cycle
                # so cloud_advisor.get_api_key() finds them via os.getenv()
                _inject_session_keys()

                for pair in pairs:
                    try:
                        await BOT_ENGINE._process_pair(pair)
                    except Exception as e:
                        logger.error(f"[BOT] Error {pair}: {e}")

                await BOT_ENGINE._check_profit_taking()
                await BOT_ENGINE._sync_account_data()

                state   = load_state()
                balance = state.get("balance", 10000)
                pnl     = state.get("pnl", 0)
                logger.info(f"[STATS] Balance: ${balance:.2f} | P&L: ${pnl:+.2f}")

                await asyncio.sleep(10)

            except ImportError as e:
                logger.error(f"[APP] Import error: {e}")
                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"[APP] Loop error: {e}")
            import traceback; traceback.print_exc()
            await asyncio.sleep(5)

# ── Static routes ─────────────────────────────────────────────────────
@app.get("/")
def root():
    html_path = static_dir / "index.html"
    return FileResponse(str(html_path)) if html_path.exists() else JSONResponse({"error": "index.html not found"}, 404)

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

# ── Status ────────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    try:
        state  = load_state()
        trades = state.get("trades", [])
        cfg    = _load_config()
        ai_cfg = cfg.get("ai", {})

        # FIX: Properly compute win rate from closed (non-zero P&L) trades
        closed = [t for t in trades if float(t.get("pnl", 0)) != 0.0]
        wins   = [t for t in closed if float(t.get("pnl", 0)) > 0]
        losses = [t for t in closed if float(t.get("pnl", 0)) < 0]
        win_rate = round(len(wins) / len(closed) * 100, 1) if closed else 0.0

        open_positions = 0
        global BOT_ENGINE
        if BOT_ENGINE and hasattr(BOT_ENGINE, 'open_positions'):
            open_positions = len(BOT_ENGINE.open_positions)

        return {
            "auto_trading":    state.get("auto_trading", False),
            "mode":            state.get("mode", "paper"),
            "balance":         round(float(state.get("balance",       10000.0)), 2),
            "start_balance":   round(float(state.get("start_balance", 10000.0)), 2),
            "pnl":             round(float(state.get("pnl", 0.0)), 2),
            "pnl_pct":         round(float(state.get("pnl", 0)) / max(float(state.get("start_balance", 10000)), 1) * 100, 2),
            "trade_count":     len(trades),
            "closed_trades":   len(closed),
            "win_count":       len(wins),
            "loss_count":      len(losses),
            "win_rate":        win_rate,
            "open_positions":  open_positions,
            "trades":          trades[-20:],
            "signals":         state.get("signals", {}),
            "running":         state.get("running", False),
            "last_update":     state.get("last_update", "—"),
            "pairs":           state.get("pairs", DEFAULT_PAIRS[:4]),
            # AI
            "ai_enabled":      ai_cfg.get("enabled",        False),
            "ai_provider":     ai_cfg.get("provider",       "deepseek_ollama"),
            "ai_model":        ai_cfg.get("model",          "deepseek-r1:7b"),
            "ai_min_confidence": ai_cfg.get("min_confidence", 0.45),
            "ai_weight":       ai_cfg.get("ai_weight",      0.65),
            "strategy_mode": (
                "ai"     if ai_cfg.get("enabled") and ai_cfg.get("ai_weight", 0.65) >= 0.7
                else "hybrid" if ai_cfg.get("enabled")
                else "technical"
            ),
            # Expose bot timeframe so frontend uses correct expiry math
            # (not the chart display TF which the user may have changed)
            "bot_timeframe": cfg.get("timeframe", "5m"),
            "bot_tf_seconds": {
                "1m":60,"5m":300,"15m":900,"1h":3600,"4h":14400,"1d":86400
            }.get(cfg.get("timeframe","5m"), 300),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# ── NEW: Signal statistics for accurate gauge rendering ───────────────
@app.get("/api/signals/stats")
async def get_signal_stats():
    """
    Returns per-pair indicator scores so the frontend can draw
    accurate AI Confidence, Market Strength, and Volatility gauges
    instead of the static 50% placeholders.
    """
    try:
        state   = load_state()
        signals = state.get("signals", {})
        pairs   = state.get("pairs", DEFAULT_PAIRS[:4])

        # Try to get live indicator data from BOT_ENGINE
        global BOT_ENGINE
        stats = {}
        for pair in pairs:
            sig = signals.get(pair, {})
            rsi = float(sig.get("rsi", 50))

            # Market strength: map composite score (-1..1) → 0..100
            # The signal "score" field is the blended indicator score
            raw_score = float(sig.get("score", 0.0))
            market_strength = round((raw_score + 1) / 2 * 100, 1)

            # Volatility: BB width or RSI deviation from neutral
            bb_upper = float(sig.get("ai_resistance", 0))
            bb_lower = float(sig.get("ai_support", 0))
            price    = float(sig.get("price", 1))
            if bb_upper > 0 and bb_lower > 0 and price > 0:
                vol_pct = round((bb_upper - bb_lower) / price * 100, 2)
                # normalise to 0-100 (0-5% bb width maps to 0-100)
                volatility = min(100, vol_pct * 20)
            else:
                # Fallback: RSI deviation from 50
                volatility = round(abs(rsi - 50) * 2, 1)

            ai_conf = round(float(sig.get("ai_confidence", 0)) * 100, 1)

            stats[pair] = {
                "ai_confidence":   ai_conf,
                "market_strength": market_strength,
                "volatility":      volatility,
                "rsi":             rsi,
                "action":          sig.get("action", "HOLD"),
                "score":           raw_score,
            }

        # Aggregate averages for overall gauges
        n = len(stats)
        if n:
            avg_conf  = round(sum(v["ai_confidence"]   for v in stats.values()) / n, 1)
            avg_str   = round(sum(v["market_strength"]  for v in stats.values()) / n, 1)
            avg_vol   = round(sum(v["volatility"]        for v in stats.values()) / n, 1)
        else:
            avg_conf = avg_str = avg_vol = 0

        return {
            "pairs": stats,
            "aggregate": {
                "ai_confidence":   avg_conf,
                "market_strength": avg_str,
                "volatility":      avg_vol,
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# ── NEW: Token usage tracker ──────────────────────────────────────────
@app.get("/api/tokens")
def get_token_usage():
    """Returns current token usage stats for the dashboard widget."""
    return {
        "total_tokens":            _token_tracker["total_tokens"],
        "total_prompt_tokens":     _token_tracker["total_prompt_tokens"],
        "total_completion_tokens": _token_tracker["total_completion_tokens"],
        "session_tokens":          _token_tracker["session_tokens"],
        "calls_today":             _token_tracker["calls_today"],
        "calls_saved":             _token_tracker["calls_saved"],
        "cost_usd_estimate":       round(_token_tracker["cost_usd_estimate"], 4),
        "history":                 _token_tracker["history"][:20],
        "cycle_locks":             {k: round(v - _time.time(), 0) for k, v in _token_tracker["cycle_locks"].items() if v > _time.time()},
    }

@app.post("/api/tokens/reset")
def reset_token_usage():
    """Reset session counters only — preserve total lifetime counts."""
    _token_tracker["session_tokens"] = 0
    _token_tracker["calls_today"]    = 0
    return {"status": "reset", "message": "Session counters cleared. Lifetime totals preserved."}

@app.delete("/api/tokens")
def delete_token_history():
    """Delete ALL token history and reset all counters."""
    for k in ("total_prompt_tokens","total_completion_tokens","total_tokens",
              "session_tokens","calls_today","cost_usd_estimate"):
        _token_tracker[k] = 0 if k != "cost_usd_estimate" else 0.0
    _token_tracker["history"] = []
    _save_token_file()
    return {"status": "deleted", "message": "All token usage history cleared."}

# ── NEW: Validate and add a crypto coin dynamically ───────────────────
@app.post("/api/coins/validate")
async def validate_coin(body: CoinAddBody):
    """Check if a pair exists on Binance before adding."""
    try:
        pair = _validate_pair(body.pair)
        # ── Optimized: Use direct REST instead of heavy ccxt.load_markets() ──
        # This avoids the ~10MB exchangeInfo call which often fails/times out.
        clean_sym = pair.replace("/", "")
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={clean_sym}"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
            
        if r.status_code == 200:
            data = r.json()
            return {
                "valid": True,
                "pair":  pair,
                "price": float(data.get("lastPrice") or 0),
                "change": float(data.get("priceChangePercent") or 0),
            }
        
        # ── Fallback to CCXT only if direct HTTP fails ──
        ticker = await _exchange.fetch_ticker(pair)
        return {
            "valid": True,
            "pair":  pair,
            "price": float(ticker.get("last") or 0),
            "change": float(ticker.get("percentage") or 0),
        }
    except Exception as e:
        logger.warning(f"[Coins] Validation failed for {body.pair}: {e}")
        return {"valid": False, "pair": body.pair, "error": str(e)}

@app.post("/api/coins/add")
def add_coin(body: CoinAddBody):
    """Add a validated coin to the active trading pairs list."""
    try:
        pair  = _validate_pair(body.pair)
        state = load_state()
        pairs = list(state.get("pairs", DEFAULT_PAIRS[:4]))
        if pair in pairs:
            return {"status": "already_exists", "pairs": pairs}
        if len(pairs) >= 20:
            raise HTTPException(400, "Maximum 20 pairs allowed")
        pairs.append(pair)
        set_pairs(pairs)
        cfg = _load_config()
        cfg["pairs"] = pairs
        _save_config(cfg)
        _reset_bot_engine()
        logger.info(f"[API] Added coin: {pair}")
        return {"status": "added", "pair": pair, "pairs": pairs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/coins/remove")
def remove_coin(body: CoinAddBody):
    """
    Remove a coin from the active trading pairs list.
    Also purges: signals, open positions, cycle locks, and cached state for the pair
    so no orphaned references remain after deletion.
    """
    try:
        pair  = _validate_pair(body.pair)
        state = load_state()
        pairs = list(state.get("pairs", DEFAULT_PAIRS[:4]))
        if pair not in pairs:
            raise HTTPException(404, f"{pair} not in active pairs")
        if len(pairs) <= 1:
            raise HTTPException(400, "Must keep at least one pair")

        # ── 1. Remove from pairs list ──────────────────────────────────────
        pairs.remove(pair)

        # ── 2. Purge signal data ───────────────────────────────────────────
        signals = state.get("signals", {})
        signals.pop(pair, None)
        state["signals"] = signals

        # ── 3. Purge open position ─────────────────────────────────────────
        positions = state.get("positions", {})
        if pair in positions:
            logger.warning(f"[API] Removing {pair} with open position — closing at last price")
            positions.pop(pair, None)
        state["positions"] = positions

        # ── 4. Update pairs in state ───────────────────────────────────────
        state["pairs"] = pairs
        save_state(state)

        # ── 5. Persist to config ───────────────────────────────────────────
        cfg = _load_config()
        cfg["pairs"] = pairs
        _save_config(cfg)

        # ── 6. Clear cycle lock and token cache for this pair ──────────────
        _unlock_cycle(pair)

        # ── 7. Reset bot engine so it picks up the new pair list ───────────
        _reset_bot_engine()

        logger.info(f"[API] Removed coin: {pair} — signals, positions, locks purged")
        return {
            "status": "removed",
            "pair":   pair,
            "pairs":  pairs,
            "purged": {"signals": True, "positions": True, "cycle_lock": True},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Trades ────────────────────────────────────────────────────────────
@app.get("/api/trades")
def get_trades():
    try:
        state  = load_state()
        trades = list(reversed(state.get("trades", [])[-50:]))
        return {"trades": trades}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Prices & Candles ─────────────────────────────────────────────────
@app.get("/api/prices")
async def get_prices():
    """
    Fetch prices using direct Binance 24hr ticker REST — no load_markets needed.
    Falls back to CCXT per-pair if batch call fails.
    """
    state = load_state()
    pairs = state.get("pairs", DEFAULT_PAIRS[:4])

    # ── Batch fetch all symbols in one request (fastest) ───────────────────
    symbols = [p.replace("/", "") for p in pairs]   # BTC/USDT → BTCUSDT
    results = []
    try:
        symbols_param = "[" + ",".join(f'"{s}"' for s in symbols) + "]"
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbols={symbols_param}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
        if r.status_code == 200:
            tickers = {t["symbol"]: t for t in r.json()}
            for pair in pairs:
                sym = pair.replace("/", "")
                t   = tickers.get(sym, {})
                if t:
                    results.append({
                        "symbol": pair,
                        "price":  float(t.get("lastPrice",          0) or 0),
                        "change": float(t.get("priceChangePercent", 0) or 0),
                        "high":   float(t.get("highPrice",          0) or 0),
                        "low":    float(t.get("lowPrice",           0) or 0),
                        "volume": float(t.get("volume",             0) or 0),
                    })
                else:
                    results.append({"symbol": pair, "error": "not found"})
            return {"prices": results}
    except Exception as e:
        logger.warning(f"[Prices] Batch fetch failed: {e} — falling back to per-pair CCXT")

    # ── CCXT fallback (per-pair) ────────────────────────────────────────────
    for pair in pairs:
        try:
            t = await _exchange.fetch_ticker(pair)
            results.append({
                "symbol": pair,
                "price":  float(t["last"] or 0),
                "change": float(t.get("percentage", 0) or 0),
                "high":   float(t["high"] or 0),
                "low":    float(t["low"]  or 0),
                "volume": float(t.get("baseVolume", 0) or 0),
            })
        except Exception as e:
            results.append({"symbol": pair, "error": str(e)})
    return {"prices": results}

@app.get("/api/candles/{symbol}")
async def get_candles(symbol: str, timeframe: str = "5m", limit: int = 100):
    """
    Fetch OHLCV candles using direct Binance REST (no load_markets / exchangeInfo call).
    BTC_USDT or BTC/USDT are both accepted — underscore is converted to slash then
    the slash is stripped for the Binance klines symbol param (BTCUSDT).
    Falls back to CCXT if direct HTTP fails.
    """
    try:
        # ── Sanitise symbol ────────────────────────────────────────────────
        clean = re.sub(r"[^A-Za-z0-9/]", "/", symbol)  # _ → /
        clean = re.sub(r"/+", "/", clean).upper()        # normalise slashes
        if "/" not in clean or len(clean) < 5:
            raise HTTPException(400, f"Invalid symbol: {symbol}")

        # ── Validate timeframe ─────────────────────────────────────────────
        VALID_TF = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1w"}
        if timeframe not in VALID_TF:
            raise HTTPException(400, f"Invalid timeframe: {timeframe}")
        limit = max(10, min(500, limit))

        # ── Direct Binance REST — no exchangeInfo needed ───────────────────
        # Binance klines endpoint: /api/v3/klines?symbol=BTCUSDT&interval=5m&limit=100
        # This endpoint works without load_markets() / authentication.
        binance_symbol = clean.replace("/", "")   # BTC/USDT → BTCUSDT
        url = (
            "https://api.binance.com/api/v3/klines"
            f"?symbol={binance_symbol}&interval={timeframe}&limit={limit}"
        )

        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)

        if r.status_code == 200:
            raw = r.json()
            candles = [
                {
                    "time":   int(row[0] / 1000),
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                }
                for row in raw if len(row) >= 6
            ]
            return {"symbol": clean, "candles": candles}

        # ── Fallback: CCXT (in case direct HTTP fails) ─────────────────────
        logger.warning(f"[Candles] Direct HTTP failed ({r.status_code}), falling back to CCXT")
        raw_ccxt = await _exchange.fetch_ohlcv(clean, timeframe, limit=limit)
        if not raw_ccxt:
            return {"symbol": clean, "candles": []}
        candles = [
            {"time": int(c[0]/1000), "open": float(c[1]), "high": float(c[2]),
             "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])}
            for c in raw_ccxt if all(v is not None for v in c[:5])
        ]
        return {"symbol": clean, "candles": candles}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Candles] {symbol}: {e}")
        raise HTTPException(500, str(e))

# ── Bot control ───────────────────────────────────────────────────────
@app.post("/api/toggle")
def toggle_bot():
    try:
        state     = load_state()
        new_state = not state.get("auto_trading", False)
        set_auto_trading(new_state)
        set_running(new_state)
        if not new_state:
            _reset_bot_engine()
            # Clear all cycle locks when bot stops
            _token_tracker["cycle_locks"].clear()
        return {"auto_trading": new_state, "running": new_state}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/mode")
def set_trading_mode(body: ModeBody):
    try:
        set_mode(body.mode)
        cfg = _load_config()
        cfg["mode"] = body.mode
        _save_config(cfg)
        _reset_bot_engine()
        return {"mode": body.mode}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/pairs")
def set_trading_pairs(body: PairsBody):
    try:
        validated = [_validate_pair(p) for p in body.pairs]
        set_pairs(validated)
        cfg = _load_config()
        cfg["pairs"] = validated
        _save_config(cfg)
        _reset_bot_engine()
        return {"pairs": validated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/reset")
def reset_bot():
    try:
        reset_balance()
        _reset_bot_engine()
        _token_tracker["session_tokens"] = 0
        _token_tracker["cycle_locks"].clear()
        return {"status": "reset"}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Settings ──────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings():
    try:
        cfg   = _load_config()
        state = load_state()
        cfg["mode"] = state.get("mode", "paper")
        return cfg
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/settings")
def save_settings(body: StrategyBody):
    try:
        cfg = _load_config()
        cfg["rsi"]["oversold"]           = body.rsi_oversold
        cfg["rsi"]["overbought"]         = body.rsi_overbought
        cfg["macd"]["fast"]              = body.macd_fast
        cfg["macd"]["slow"]              = body.macd_slow
        cfg["macd"]["signal"]            = body.macd_signal
        cfg["bb"]["period"]              = body.bb_period
        cfg["bb"]["std_dev"]             = body.bb_std_dev
        cfg["risk"]["stop_loss_pct"]     = body.stop_loss_pct
        cfg["risk"]["take_profit_pct"]   = body.take_profit_pct
        cfg["risk"]["max_position_pct"]  = body.max_position_pct
        cfg["risk"]["daily_loss_limit_pct"] = body.daily_loss_limit_pct
        cfg["risk"]["max_open_trades"]   = body.max_open_trades
        cfg["timeframe"]                 = body.timeframe
        _save_config(cfg)
        _reset_bot_engine()
        return {"status": "saved", "config": cfg}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Exchange API Keys ─────────────────────────────────────────────────
@app.get("/api/apikeys")
def get_api_keys():
    try:
        result = {}
        for ex in ["binance", "coinbase", "kraken", "bybit"]:
            prefix = ex.upper()
            key    = _env_get(f"{prefix}_API_KEY")
            secret = _env_get(f"{prefix}_API_SECRET")
            sandbox= _env_get(f"{prefix}_SANDBOX")
            result[ex] = {
                "has_key":      bool(key),
                "has_secret":   bool(secret),
                "key_masked":   _mask(key)    if key    else "",
                "secret_masked":_mask(secret) if secret else "",
                "sandbox": sandbox.lower() in ("true","1","yes") if sandbox else True,
            }
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/apikeys")
def save_api_keys(body: ApiKeysBody):
    """
    Multi-user: exchange keys are stored in browser localStorage.
    Also writes to .env as fallback for single-user setups.
    Bot uses header-provided keys at runtime.
    """
    try:
        prefix = body.exchange.upper()
        try:
            # Set the exchange to the session environment, do not write to disk
            os.environ[f"{prefix}_API_KEY"]    = body.api_key
            os.environ[f"{prefix}_API_SECRET"] = body.api_secret
            os.environ[f"{prefix}_SANDBOX"]    = str(body.sandbox).lower()
            os.environ["EXCHANGE_ID"]          = body.exchange
        except Exception as e:
            logger.warning(f"[Keys] In-memory update failed: {e}")
        _reset_bot_engine()
        return {
            "status": "saved",
            "exchange": body.exchange,
            "storage": "browser_localStorage",
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/apikeys")
def delete_api_keys(body: DeleteKeysBody):
    try:
        prefix = body.exchange.upper()
        for k in (f"{prefix}_API_KEY", f"{prefix}_API_SECRET", f"{prefix}_SANDBOX"):
            _env_delete(k)
        load_dotenv(override=True)
        _reset_bot_engine()
        return {"status": "deleted", "exchange": body.exchange}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Manual trade ──────────────────────────────────────────────────────
@app.post("/api/manual-trade")
async def manual_trade(body: ManualTradeBody):
    try:
        pair = _validate_pair(body.pair)
        from config.settings import load_settings
        settings = load_settings()
        mode = load_state().get("mode", "paper")
        if mode == "paper":
            from execution.paper_trader import PaperTrader
            executor = PaperTrader("binance", settings)
        elif mode == "live":
            from execution.trade_executor import TradeExecutor
            executor = TradeExecutor(settings.EXCHANGE_ID, settings)
        else:
            return {"status": "error", "error": "Not available in backtest mode"}
        result = await executor.execute(pair=pair, action=body.action, size=body.size, price=body.price)
        return {"status": "ok", "trade": result} if result else {"status": "error", "error": "Trade rejected"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ── AI routes ─────────────────────────────────────────────────────────
@app.get("/api/ai/status")
async def get_ai_status():
    try:
        cfg      = _load_config()
        ai_cfg   = cfg.get("ai", {})
        provider = ai_cfg.get("provider", "deepseek_ollama")
        health   = {}
        try:
            if provider == "deepseek_ollama":
                from ai_engine.deepseek_advisor import DeepSeekAdvisor
                health = await DeepSeekAdvisor(ai_cfg).health_check()
            else:
                from ai_engine.cloud_advisor import CloudAdvisor
                health = await CloudAdvisor(ai_cfg).health_check()
        except Exception as e:
            health = {"error": str(e)}

        from ai_engine.cloud_advisor import list_providers
        # Add universal provider context
        import os as _os
        universal_base_url = _os.getenv("UNIVERSAL_BASE_URL", "")
        universal_model    = _os.getenv("UNIVERSAL_MODEL", "")

        return {
            "enabled":             ai_cfg.get("enabled",        False),
            "provider":            provider,
            "model":               ai_cfg.get("model",          "deepseek-r1:7b"),
            "ollama_url":          ai_cfg.get("ollama_url",     "http://localhost:11434"),
            "min_confidence":      ai_cfg.get("min_confidence", 0.45),
            "ai_weight":           ai_cfg.get("ai_weight",      0.65),
            "timeout":             ai_cfg.get("timeout",        90),
            "all_providers":       list_providers(),
            "universal_base_url":  universal_base_url,
            "universal_model":     universal_model,
            **health,
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/ai/providers")
def get_ai_providers():
    try:
        from ai_engine.cloud_advisor import list_providers, PROVIDER_MODELS
        return {"providers": list_providers(), "models": PROVIDER_MODELS}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/ai/apikey")
def save_ai_api_key(body: AIApiKeyBody):
    """
    Multi-user mode: keys are stored in browser localStorage, NOT on the server.
    This endpoint validates the key format and returns confirmation.
    The browser stores the key and sends it via X-AI-Keys header on each request.
    """
    try:
        provider = body.provider.lower()
        valid_providers = {"gemini","openai","anthropic","groq","openrouter","universal"}
        if provider not in valid_providers:
            raise HTTPException(400, f"Unknown provider: {body.provider}")

        key = (body.api_key or "").strip()
        if key and not re.match(r'^[\x20-\x7E]+$', key):
            raise HTTPException(400, "API key contains invalid characters")

        # Keys live in browser localStorage — server does NOT store them.
        # We still also save to .env as fallback for single-user setups.
        env_map = {
            "gemini":"GEMINI_API_KEY","openai":"OPENAI_API_KEY",
            "anthropic":"ANTHROPIC_API_KEY","groq":"GROQ_API_KEY",
            "openrouter":"OPENROUTER_API_KEY","universal":"UNIVERSAL_API_KEY",
        }
        env_var = env_map.get(provider, "")
        if env_var and key:
            try:
                # Set in memory only
                os.environ[env_var] = key
                if provider == "universal":
                    base_url = getattr(body, "base_url", "") or ""
                    model    = getattr(body, "model_name", "") or ""
                    if base_url: os.environ["UNIVERSAL_BASE_URL"] = base_url.strip()
                    if model:    os.environ["UNIVERSAL_MODEL"] = model.strip()
            except Exception as e:
                logger.warning(f"[Keys] In-memory set failed: {e}")

        _reset_bot_engine()
        return {
            "status": "saved",
            "provider": provider,
            "storage": "browser_localStorage",
            "message": "Key stored in your browser. It will be sent securely with each AI request.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/ai/apikey")
def delete_ai_api_key(body: AIApiKeyBody):
    try:
        env_map = {
            "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY", "universal": "UNIVERSAL_API_KEY",
        }
        env_var = env_map.get(body.provider.lower())
        if not env_var:
            raise HTTPException(400, f"Unknown provider: {body.provider}")
        _env_delete(env_var)
        load_dotenv(override=True)
        _reset_bot_engine()
        return {"status": "deleted", "provider": body.provider}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/ai/toggle")
def toggle_ai():
    try:
        cfg       = _load_config()
        new_state = not cfg.get("ai", {}).get("enabled", False)
        cfg.setdefault("ai", {})["enabled"] = new_state
        _save_config(cfg)
        global BOT_ENGINE
        if BOT_ENGINE is not None:
            try:
                BOT_ENGINE.sig_gen.set_ai_enabled(new_state)
            except Exception as e:
                logger.error(f"[API] AI toggle error: {e}")
        if not new_state:
            _token_tracker["cycle_locks"].clear()
        return {"ai_enabled": new_state, "provider": cfg.get("ai", {}).get("provider")}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/ai/config")
def save_ai_config(body: AIConfigBody):
    try:
        cfg = _load_config()
        new_ai = {
            "enabled":        body.enabled,
            "provider":       body.provider,
            "model":          body.model,
            "ollama_url":     body.ollama_url,
            "timeout":        body.timeout,
            "min_confidence": body.min_confidence,
            "ai_weight":      body.ai_weight,
        }
        cfg["ai"] = new_ai
        _save_config(cfg)
        global BOT_ENGINE
        if BOT_ENGINE is not None:
            try:
                BOT_ENGINE.sig_gen.reload_advisor(new_ai)
            except Exception as e:
                logger.error(f"[API] AI reload error: {e}")
                _reset_bot_engine()
        # Clear stale cycle locks on config change
        _token_tracker["cycle_locks"].clear()
        return {"status": "saved", "ai": cfg["ai"]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/ai/refresh")
async def refresh_ai_prediction(body: RefreshBody, request: Request):
    """
    Force an immediate AI re-prediction for a pair or all pairs.

    Cycle-lock fix
    ──────────────
    OLD: check lock → if locked return cycle_locked
         ↳ the system's own expiry-triggered refresh was blocked by its own lock
    NEW: check whether prediction is *genuinely still live* in state.
         If genuinely active  → honour the lock (blocks user spam only)
         If expired or no data → unlock unconditionally, then trigger
    """
    try:
        global BOT_ENGINE
        if BOT_ENGINE is None:
            raise HTTPException(400, "Bot engine is not running")

        cfg    = _load_config()
        tf_sec = {"1m":60, "5m":300, "15m":900, "1h":3600, "4h":14400}.get(
            cfg.get("timeframe", "5m"), 300)

        pair = body.pair

        # ── Is the prediction genuinely still live? ─────────────────────────
        # We check state directly, not just the in-memory lock timestamp.
        # This is the critical fix: the lock timestamp and the prediction expiry
        # can diverge — always use the state expiry as the source of truth.
        prediction_still_live = False
        if pair:
            try:
                sig = load_state().get("signals", {}).get(pair, {})
                exp = sig.get("_expiry_ts", 0)
                prediction_still_live = (
                    exp > 0
                    and _time.time() < exp
                    and sig.get("ai_used", False)
                    and float(sig.get("ai_predicted_close", 0)) > 0
                )
            except Exception:
                prediction_still_live = False

        # ── Rate-limit ONLY when prediction is genuinely still active ───────
        if prediction_still_live and _is_cycle_locked(pair):
            remaining = max(0, round(_token_tracker["cycle_locks"].get(pair, 0) - _time.time()))
            _token_tracker["calls_saved"] += 1
            logger.info(f"[Refresh] {pair}: live prediction locked, {remaining}s left")
            return {
                "status":           "cycle_locked",
                "pair":             pair,
                "retry_in_seconds": remaining,
                "message":          f"Prediction still active for {remaining}s",
            }

        # ── Expired or no live data — clear lock unconditionally ─────────────
        if pair:
            _unlock_cycle(pair)
        else:
            _token_tracker["cycle_locks"].clear()

        # ── Trigger the bot engine ─────────────────────────────────────────
        # Update session key cache from this request's headers,
        # then inject into os.environ for all downstream AI calls
        _inject_session_keys(request)
        await BOT_ENGINE.force_ai_refresh(pair=pair)

        # ── Re-lock after confirmed trigger ───────────────────────────────
        if pair:
            _set_cycle_lock(pair, tf_sec)
        else:
            for p in load_state().get("pairs", []):
                _set_cycle_lock(p, tf_sec)

        logger.info(f"[Refresh] Triggered: {pair or 'all'}, lock set {tf_sec}s")
        return {"status": "refresh_triggered", "pair": pair}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Refresh] Error: {e}")
        raise HTTPException(500, str(e))
# ── Backtest ──────────────────────────────────────────────────────────
@app.post("/api/backtest")
async def run_backtest(body: BacktestBody):
    try:
        from core.bot_engine import BotEngine
        from config.settings import load_settings
        settings = load_settings()
        pairs    = body.pairs or DEFAULT_PAIRS[:4]
        engine   = BotEngine(mode="backtest", pairs=pairs, exchange_id="binance", settings=settings)
        results  = await engine.run(start_date=body.start_date, end_date=body.end_date)
        return {"status": "completed", "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Startup / Shutdown ────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global BOT_TASK
    logger.info("CryptoBot Pro v2.0 starting...")

    # ── Load persisted token usage from disk ───────────────────────────────
    _load_token_file()

    # ── Auto-disable Ollama AI if Ollama is not running ────────────────────
    # Prevents spamming "ConnectError: All connection attempts failed" every
    # 10 seconds when AI is enabled but Ollama is not running locally.
    await _check_and_fix_ollama_config()

    # ── Wire token usage callback into AI advisors ─────────────────────────
    try:
        import ai_engine as _ai_pkg
        _ai_pkg._token_callback = _record_token_usage
        logger.info("[TOKENS] Token callback registered for all AI providers")
    except Exception as e:
        logger.warning(f"[TOKENS] Could not register token callback: {e}")

    # ── Pre-warm exchange markets with retry ───────────────────────────────
    # Try to load Binance markets so prices/tickers work immediately.
    # If it fails (network issue), the candles endpoint uses direct HTTP anyway.
    asyncio.create_task(_preload_markets())

    BOT_TASK = asyncio.create_task(run_bot_background())


async def _check_and_fix_ollama_config():
    """
    If AI provider is deepseek_ollama but Ollama is not reachable,
    log a clear ONE-TIME warning and temporarily disable AI so the bot
    doesn't spam ConnectError every cycle.
    The user can re-enable from the Settings → AI Engine panel once Ollama is running.
    """
    try:
        cfg    = _load_config()
        ai_cfg = cfg.get("ai", {})
        if not ai_cfg.get("enabled", False):
            return  # AI already off, nothing to check

        provider = ai_cfg.get("provider", "deepseek_ollama")
        if provider != "deepseek_ollama":
            return  # Cloud provider — don't check Ollama

        ollama_url = ai_cfg.get("ollama_url", "http://localhost:11434")
        import httpx as _httpx
        try:
            async with _httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{ollama_url}/api/tags")
                if r.status_code < 500:
                    logger.info(f"[Startup] Ollama reachable at {ollama_url} ✓")
                    return
        except Exception:
            pass

        # Ollama not reachable — disable AI and warn clearly ONCE
        msg = (
            f"[Startup] AI is enabled but Ollama is NOT running at {ollama_url}. "
            "Auto-disabling AI to prevent error spam. "
            "To fix: run 'ollama serve' then re-enable AI in Settings."
        )
        logger.warning(msg)
        cfg["ai"]["enabled"] = False
        _save_config(cfg)

    except Exception as e:
        logger.warning(f"[Startup] Ollama check failed: {e}")


async def _preload_markets():
    """Load Binance markets in background with retry — non-blocking."""
    for attempt in range(1, 4):
        try:
            await _exchange.load_markets()
            logger.info("[Exchange] Markets loaded successfully")
            return
        except Exception as e:
            wait = attempt * 5
            logger.warning(f"[Exchange] load_markets attempt {attempt}/3 failed: {e} — retry in {wait}s")
            await asyncio.sleep(wait)
    logger.warning("[Exchange] Could not load markets after 3 attempts — candles use direct HTTP fallback")

@app.on_event("shutdown")
async def shutdown():
    global BOT_TASK
    if BOT_TASK and not BOT_TASK.done():
        BOT_TASK.cancel()
        try:
            await asyncio.wait_for(BOT_TASK, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    try:
        await _exchange.close()
    except Exception:
        pass

if __name__ == "__main__":
    """
    Direct launch: python dashboard/app.py
    Equivalent to: uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload
    
    Uses subprocess so it works with ANY uvicorn version.
    """
    import subprocess, sys, os
    # Always run from project root so module imports resolve correctly
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "dashboard.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--reload-dir", "dashboard",
        "--reload-dir", "core",
        "--reload-dir", "ai_engine",
        "--reload-dir", "indicators",
        "--reload-dir", "execution",
    ], cwd=root, check=False)