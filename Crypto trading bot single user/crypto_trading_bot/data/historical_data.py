"""
historical_data.py
Fetches historical OHLCV candles from Binance for backtesting.
Uses the synchronous CCXT client (no async needed for backtesting).
"""

import ccxt
import pandas as pd
from datetime import datetime
from logging_monitor.logger import get_logger

logger = get_logger(__name__)


def fetch_historical(symbol: str, timeframe: str,
                     start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV candles between start_date and end_date.

    Parameters:
        symbol     : e.g. 'BTC/USDT'
        timeframe  : e.g. '5m', '1h', '1d'
        start_date : e.g. '2024-01-01'
        end_date   : e.g. '2024-12-31'

    Returns:
        pandas DataFrame with columns:
        timestamp, open, high, low, close, volume
    """

    logger.info(f"Fetching historical data for {symbol} "
                f"from {start_date} to {end_date}")

    # Use Binance public API — no API key needed for historical data
    exchange = ccxt.binance({
        "enableRateLimit": True,
    })

    # Convert start and end dates to millisecond timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").timestamp() * 1000)

    all_candles = []
    current_ts  = start_ts

    # Binance returns max 1000 candles per request
    # so we loop until we have all data in the date range
    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_ts,
                limit=1000,
            )

            if not candles:
                break

            all_candles.extend(candles)

            # Move timestamp forward to last fetched candle + 1ms
            current_ts = candles[-1][0] + 1

            logger.info(f"Fetched {len(all_candles)} candles so far...")

            # Stop if we have passed end date
            if candles[-1][0] >= end_ts:
                break

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            break

    if not all_candles:
        logger.warning("No historical data returned. Check symbol and dates.")
        return pd.DataFrame(
            columns=["timestamp","open","high","low","close","volume"]
        )

    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=["timestamp","open","high","low","close","volume"]
    )

    # Convert timestamp from milliseconds to readable datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Filter to exact date range
    df = df[df.index <= pd.to_datetime(end_date)]

    logger.info(f"Done. Total candles fetched: {len(df)}")

    return df