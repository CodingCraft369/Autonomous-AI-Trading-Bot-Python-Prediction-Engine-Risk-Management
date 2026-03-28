"""
data/market_data.py
Fetches live market data from Binance.

DESIGN: All data methods use direct Binance REST endpoints (no load_markets /
exchangeInfo required). This avoids the NetworkError that occurs when Binance
rate-limits or blocks the exchangeInfo endpoint on first connection.

Direct endpoints used:
  Candles  → GET /api/v3/klines
  Ticker   → GET /api/v3/ticker/24hr
  OrderBook→ GET /api/v3/depth
"""
import asyncio
import httpx
import pandas as pd
import ccxt.async_support as ccxt
from logging_monitor.logger import get_logger

logger = get_logger(__name__)

BINANCE_REST = "https://api.binance.com/api/v3"


class MarketDataCollector:
    def __init__(self, exchange_id, settings):
        # CCXT instance kept for fallback and for order execution
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.timeframe      = settings.TIMEFRAME
        self.markets_loaded = False
        self._load_lock     = None
        # Shared httpx client (created on first use)
        self._http: httpx.AsyncClient | None = None
        logger.info("MarketDataCollector ready — direct REST + CCXT fallback")

    # ── HTTP client ───────────────────────────────────────────────────
    def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15, follow_redirects=True)
        return self._http

    # ── Timeframe → Binance interval string ──────────────────────────
    @staticmethod
    def _tf(timeframe: str) -> str:
        # CCXT format (5m, 1h) == Binance interval format — pass through
        return timeframe

    # ── Direct Binance klines (no exchangeInfo needed) ────────────────
    async def get_candles(self, symbol: str, limit: int = 200, timeframe: str = None) -> pd.DataFrame:
        """Fetch OHLCV candles using direct Binance klines REST endpoint."""
        # BTC/USDT → BTCUSDT
        binance_sym = symbol.replace("/", "")
        interval    = self._tf(timeframe or self.timeframe)
        url = f"{BINANCE_REST}/klines?symbol={binance_sym}&interval={interval}&limit={limit}"

        try:
            r = await self._client().get(url)
            r.raise_for_status()
            raw = r.json()
            df = pd.DataFrame(raw, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            df = df.astype({
                "open": float, "high": float, "low": float,
                "close": float, "volume": float,
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            logger.warning(f"[MarketData] Direct klines failed for {symbol}: {e} — trying CCXT")
            # CCXT fallback
            await self._ensure_markets()
            raw_ccxt = await self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(raw_ccxt, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df

    # ── Direct Binance 24hr ticker ────────────────────────────────────
    async def get_ticker(self, symbol: str) -> dict:
        """Fetch ticker using direct Binance 24hr ticker REST endpoint."""
        binance_sym = symbol.replace("/", "")
        url = f"{BINANCE_REST}/ticker/24hr?symbol={binance_sym}"

        try:
            r = await self._client().get(url)
            r.raise_for_status()
            t = r.json()
            # Normalise to CCXT ticker format so existing code works unchanged
            return {
                "symbol":     symbol,
                "last":       float(t.get("lastPrice", 0)),
                "bid":        float(t.get("bidPrice",  0)),
                "ask":        float(t.get("askPrice",  0)),
                "high":       float(t.get("highPrice", 0)),
                "low":        float(t.get("lowPrice",  0)),
                "baseVolume": float(t.get("volume",    0)),
                "percentage": float(t.get("priceChangePercent", 0)),
                "change":     float(t.get("priceChange", 0)),
                "open":       float(t.get("openPrice",  0)),
                "close":      float(t.get("lastPrice",  0)),
                "info":       t,
            }
        except Exception as e:
            logger.warning(f"[MarketData] Direct ticker failed for {symbol}: {e} — trying CCXT")
            await self._ensure_markets()
            return await self.exchange.fetch_ticker(symbol)

    # ── Order book (CCXT, rarely called) ─────────────────────────────
    async def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        await self._ensure_markets()
        return await self.exchange.fetch_order_book(symbol, depth)

    # ── Background market load (non-blocking, best-effort) ───────────
    def _get_lock(self):
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        return self._load_lock

    async def _ensure_markets(self):
        """Try to load markets once. Non-blocking best-effort — never raises."""
        if self.markets_loaded:
            return
        async with self._get_lock():
            if self.markets_loaded:
                return
            for attempt in range(1, 4):
                try:
                    await self.exchange.load_markets()
                    self.markets_loaded = True
                    logger.info("[MarketData] Markets loaded via CCXT")
                    return
                except Exception as e:
                    wait = attempt * 3
                    logger.warning(
                        f"[MarketData] load_markets {attempt}/3 failed: {e}"
                        + (f" — retry in {wait}s" if attempt < 3 else "")
                    )
                    if attempt < 3:
                        await asyncio.sleep(wait)
            # Mark done so we stop retrying
            self.markets_loaded = True

    async def close(self):
        try:
            await self.exchange.close()
        except Exception:
            pass
        if self._http and not self._http.is_closed:
            await self._http.aclose()