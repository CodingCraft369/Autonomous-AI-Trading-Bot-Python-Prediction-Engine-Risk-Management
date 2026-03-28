import asyncio
import logging
import traceback
from ai_engine.deepseek_advisor import DeepSeekAdvisor

logging.basicConfig(level=logging.DEBUG)

async def test():
    print("Initializing DeepSeekAdvisor...")
    adv = DeepSeekAdvisor({'enabled': True, 'model': 'deepseek-r1:1.5b', 'timeout': 300})
    
    candles = [
        {'time': 1710000000, 'open': 70000, 'high': 70500, 'low': 69500, 'close': 70200, 'volume': 1000},
        {'time': 1710000300, 'open': 70200, 'high': 70300, 'low': 69800, 'close': 70100, 'volume': 1200},
    ]
    
    try:
        print(f"Health check: {await adv.health_check()}")
        print("Running _query()...")
        v = await adv._query('BTC/USDT', {'action':'HOLD','score':0}, {'rsi': 50}, {'last': 70100}, candles)
        print("\n\nResult Action:", v.action)
        print("Result Pred Close:", v.predicted_close)
        print("Used AI:", v.used_ai)
    except Exception as e:
        print("\n\nException occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
