#!/usr/bin/env python3
# """Entry point. Supports live, paper, and backtest modes."""
import asyncio, argparse
from config.settings import Settings
from core.bot_engine import BotEngine
from logging_monitor.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="AI Crypto Trading Bot")
    p.add_argument("--mode", choices=["live","paper","backtest"], default="paper")
    p.add_argument("--pairs", nargs="+", default=["BTC/USDT"])
    p.add_argument("--exchange", default="binance")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date",   default=None)
    return p.parse_args()

async def main():
    args = parse_args()
    settings = Settings()
    logger.info("Starting bot mode=" + args.mode)
    engine = BotEngine(
        mode=args.mode, pairs=args.pairs,
        exchange_id=args.exchange, settings=settings,
    )
    await engine.run(start_date=args.start_date, end_date=args.end_date)

if __name__ == "__main__":
    asyncio.run(main())