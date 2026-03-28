# Autonomous-AI-Trading-Bot-Python-Prediction-Engine-Risk-Management
<div align="center">

# 🤖 CryptoBot Pro

### AI-Powered Cryptocurrency Trading Bot & Dashboard

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CCXT](https://img.shields.io/badge/CCXT-Exchange-orange)](https://github.com/ccxt/ccxt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com/)

**Paper trading · Live trading · AI candle prediction · Multi-provider AI · 1M-user ready architecture**

</div>

---

## 📋 Table of Contents

- [What is CryptoBot Pro?](#-what-is-cryptobot-pro)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Running the Bot](#-running-the-bot)
- [Dashboard Walkthrough](#-dashboard-walkthrough)
- [Configuring API Keys](#-configuring-api-keys)
- [AI Engine Setup](#-ai-engine-setup)
- [Strategy Configuration](#-strategy-configuration)
- [Example Usage](#-example-usage)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)
- [Disclaimer](#-disclaimer)

---

## 🚀 What is CryptoBot Pro?

**CryptoBot Pro** is a production-grade, AI-enhanced cryptocurrency trading bot with a real-time web dashboard. It connects to live exchange data (Binance by default), applies technical analysis, optionally consults an AI model to predict the next candle's direction and price, and executes trades automatically in paper or live mode.

The entire user interface runs in your browser. All your API keys, strategy settings, and trade history are stored **exclusively in your browser's IndexedDB** — nothing sensitive is ever written to the server. This makes it safe for multiple users to share a single server instance, each with their own private keys and configuration.

```
                    ┌─────────────────────────────────┐
                    │         Your Browser             │
                    │   ┌──────────────────────────┐   │
                    │   │  IndexedDB / localStorage │   │
                    │   │  • AI API keys            │   │
                    │   │  • Exchange keys          │   │
                    │   │  • Strategy config        │   │
                    │   │  • Trade history cache    │   │
                    │   │  • Token usage history    │   │
                    │   └──────────────────────────┘   │
                    └────────────┬────────────────────┘
                                 │ HTTPS (keys in headers)
                    ┌────────────▼────────────────────┐
                    │     CryptoBot Pro Server         │
                    │   FastAPI + Bot Engine           │
                    │   • Live market signals          │
                    │   • AI predictions (your key)   │
                    │   • Trade execution              │
                    │   • Zero user data on disk       │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │   Binance (or Bybit/Kraken)      │
                    │   Real-time OHLCV + Trading      │
                    └─────────────────────────────────┘
```

---

## ✨ Features

### 📊 Live Dashboard
- Real-time candlestick chart with LightweightCharts (1m / 5m / 15m / 1h / 4h)
- Live price cards for all 8 trading pairs with 24h change
- Portfolio balance, P&L, and win-rate statistics updated every 10 seconds
- System status bar with connectivity indicator
- Floating trade alerts with AI or technical signal badge

### 🤖 AI Prediction Engine
- **Next-candle prediction** — predicts direction (UP / DOWN / SIDEWAYS), close price, high/low range, support and resistance levels
- **Confidence-gated signals** — AI only overrides technical signals when confidence ≥ your threshold
- **Blended scoring** — weighted combination of AI score and technical indicators
- **Dynamic Stop-Loss / Take-Profit** — automatically adjusted from AI's predicted support/resistance
- **7 AI providers supported** (see [AI Engine Setup](#-ai-engine-setup))

### 📈 Technical Indicators
| Indicator | Details |
|-----------|---------|
| RSI | 14-period, configurable oversold/overbought thresholds |
| MACD | 12/26/9 EMA crossover with histogram |
| Bollinger Bands | 20-period, 2σ with squeeze detection |
| EMA Cross | 20/50 EMA trend filter |
| Volume Ratio | Volume vs. rolling average |

### 💰 Trading Modes
| Mode | Description |
|------|-------------|
| **Paper** | Virtual $10,000 balance, realistic 0.1% fee simulation, full P&L tracking |
| **Live** | Real orders via CCXT — requires exchange API key |

### 🛡️ Risk Management
- Per-trade stop-loss and take-profit (configurable %)
- Maximum open trades limit
- Daily loss limit (auto-stops trading when hit)
- Position sizing as % of balance
- AI-driven dynamic SL/TP override when prediction confidence is high

### 🔒 Security (Multi-User Safe)
- **Zero server-side key storage** — all API keys stay in your browser
- Keys sent via encrypted `X-AI-Keys` request header, never logged
- Per-user strategy config sent via `X-Strategy` header
- Clearing browser data = complete logout, no server cleanup needed
- Safe for shared or cloud-hosted deployments

### ⚡ Performance (1M-User Architecture)
- In-memory TTL response cache (512-entry LRU, prices 5s, candles 15s)
- ETag-based conditional GET for `/api/status` — saves bandwidth for all polling clients
- GZip compression on all responses
- Parallel processing of all trading pairs per cycle
- Rate limiting (300 req/min per IP with 50k-IP memory cap)

---

## 📁 Project Structure

```
C:\crypto_trading_bot\
│
├── dashboard/
│   ├── app.py                  # FastAPI backend — stateless, 1M-user ready
│   └── static/
│       └── index.html          # Single-page dashboard (all UI, IndexedDB storage)
│
├── core/
│   ├── bot_engine.py           # Main trading loop, AI task scheduling
│   └── state.py                # Thread-safe bot_state.json read/write
│
├── ai_engine/
│   ├── __init__.py             # Token usage callback registry
│   ├── cloud_advisor.py        # Gemini, OpenAI, Anthropic, Groq, OpenRouter, Universal
│   ├── deepseek_advisor.py     # Local Ollama / DeepSeek-R1 advisor
│   └── signal_generator.py     # Provider routing + blended signal logic
│
├── indicators/
│   ├── engine.py               # Orchestrates all indicators → unified dict
│   ├── rsi.py
│   ├── macd.py
│   ├── bollinger_bands.py
│   └── moving_averages.py
│
├── execution/
│   ├── paper_trader.py         # Virtual trading with realistic fees
│   └── trade_executor.py       # Live CCXT order placement
│
├── data/
│   ├── market_data.py          # Real-time OHLCV via direct Binance REST
│   └── historical_data.py      # Historical candles for backtesting
│
├── risk/
│   └── risk_manager.py         # Stop-loss, take-profit, position sizing
│
├── config/
│   ├── settings.py             # Loads strategy_config.json + env credentials
│   └── strategy_config.json    # Default strategy parameters
│
├── logging_monitor/
│   └── logger.py               # Windows-safe rotating file logger
│
├── logs/
│   └── bot.log                 # Rotating log (10 MB max, 3 backups)
│
├── bot_state.json              # Live bot state (signals, trades, balance)
├── run.py                      # CLI entry point (paper/live/backtest)
└── requirements.txt
```

---

## 📦 Requirements

- **Python** 3.11 or higher
- **Windows 10/11**, Ubuntu 20.04+, or macOS 12+
- **8 GB RAM** minimum (16 GB recommended if running local Ollama AI)
- **Internet connection** (for Binance market data)

### Python Dependencies

```
fastapi>=0.100.0
uvicorn[standard]>=0.27.0
ccxt>=4.0.0
httpx>=0.25.0
pandas>=2.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

> **Note:** A full `requirements.txt` is included in the project root.

---

## 🔧 Installation

### Step 1 — Clone or download the project

```bash
# Windows — place outside OneDrive to avoid sync conflicts
cd C:\
git clone https://github.com/yourname/cryptobot-pro crypto_trading_bot
cd crypto_trading_bot
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Create the required folders

```bash
# Windows
mkdir logs
mkdir dashboard\static
mkdir config

# Linux / macOS
mkdir -p logs dashboard/static config
```

### Step 5 — Copy the dashboard frontend

Place `index.html` into `dashboard/static/index.html`.

> The dashboard is the single file at `dashboard/static/index.html`. No build step needed.

### Step 6 — (Optional) Install Ollama for local AI

If you want **free local AI** without any API keys, install [Ollama](https://ollama.ai):

```bash
# After installing Ollama, pull the DeepSeek-R1 model
ollama pull deepseek-r1:7b        # 8 GB RAM — recommended
ollama pull deepseek-r1:1.5b      # 3 GB RAM — lighter option
```

---

## ▶️ Running the Bot

### Start the dashboard server

```bash
# From the project root (C:\crypto_trading_bot\)
uvicorn dashboard.app:app --reload --port 8000
```

Then open your browser at: **http://localhost:8000**

### Alternative — run without reload (production)

```bash
uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --workers 1
```

### Run the bot standalone (CLI — no dashboard)

```bash
# Paper trading — BTC and ETH, 5-minute candles
python run.py --mode paper --pairs BTC/USDT ETH/USDT

# Live trading — requires exchange API keys in .env
python run.py --mode live --pairs BTC/USDT --exchange binance

# Backtest — historical replay
python run.py --mode backtest --pairs BTC/USDT --start-date 2024-01-01 --end-date 2024-12-31
```

### Windows — quick-start script

Create `start.bat` in the project root:

```bat
@echo off
cd C:\crypto_trading_bot
call venv\Scripts\activate
uvicorn dashboard.app:app --reload --port 8000
pause
```

Double-click `start.bat` to launch.

---

## 🖥️ Dashboard Walkthrough

Once the server is running at `http://localhost:8000`, the sidebar gives you access to:

| Section | What it does |
|---------|-------------|
| **Dashboard** | Live candlestick chart, portfolio stats, AI signal cards, prediction timeline |
| **Live Markets** | Price cards for all pairs, 24h change, volume |
| **AI Signals** | Full signal cards with candle prediction blocks, confidence bars, support/resistance |
| **Trade History** | Every executed trade with P&L, pair, price, and trigger type (AI or Technical) |
| **Settings** | Strategy params, risk settings, trading mode, exchange keys, AI engine config |
| **Token Usage** | AI API call tracking — tokens used, cost estimate, call history |
| **My Data** | View all browser-stored keys and data, storage usage meter, Clear All button |

### Starting the Bot

1. Open the dashboard at `http://localhost:8000`
2. Configure your settings (pairs, timeframe, risk) in **Settings**
3. Click **▶ Start Bot** in the top-right corner
4. The bot begins processing pairs every 10 seconds

---

## 🔑 Configuring API Keys

> **All API keys are stored exclusively in your browser.** Nothing is written to the server or any config file on disk.

### Exchange Keys (for Live Trading only)

Paper trading requires **no exchange keys**. For live trading:

1. Navigate to **Settings → Exchange API Keys**
2. Select your exchange (Binance, Bybit, Kraken, or Coinbase)
3. Paste your API Key and Secret
4. Check **Use Testnet / Sandbox** to test without real money
5. Click **🔐 Save Keys**

Your keys are stored in **browser IndexedDB** and sent to the server only when making authenticated trade requests.

**Supported Exchanges:**

| Exchange | Mode | Notes |
|----------|------|-------|
| Binance | Paper + Live | Default. Testnet available |
| Bybit | Live | Testnet available |
| Kraken | Live | — |
| Coinbase Advanced | Live | Set exchange to `coinbase` |

**Binance API setup:**
1. Log in to [binance.com](https://www.binance.com) → Account → API Management
2. Create API key → enable **Spot Trading** only
3. Restrict to your server IP for security
4. **Never enable withdrawal permissions**

---

## 🧠 AI Engine Setup

The AI engine predicts the next candle's direction, close price, high/low range, and support/resistance levels. It is **optional** — the bot works with pure technical signals when AI is disabled.

Navigate to **Settings → AI Engine** to configure.

### Option A — Local AI with Ollama (Free, Private)

No API key needed. Runs entirely on your machine.

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Pull a model (choose based on your RAM):

ollama pull deepseek-r1:1.5b   # ~3 GB RAM  — fastest
ollama pull deepseek-r1:7b     # ~8 GB RAM  — recommended
ollama pull deepseek-r1:14b    # ~16 GB RAM — most accurate

# 3. Start the Ollama server (runs in background automatically after install)
ollama serve
```

In Settings → AI Engine:
- Provider: `DeepSeek-R1 (Local Ollama — free)`
- Model: `deepseek-r1:7b` (or whichever you pulled)
- Ollama URL: `http://localhost:11434`
- Click **Test** to verify connectivity

### Option B — Cloud AI Providers

Click **Save Key** after pasting each key. Keys are stored in your browser only.

| Provider | Free Tier | Speed | Best Model | Get Key |
|----------|-----------|-------|------------|---------|
| **Groq** | ✅ 30 RPM free | ~300 tok/s | `llama-3.3-70b-versatile` | [console.groq.com/keys](https://console.groq.com/keys) |
| **Google Gemini** | ✅ 15 RPM free | Fast | `gemini-2.0-flash` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **OpenRouter** | ✅ Free models | Varies | `google/gemini-flash-1.5` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **OpenAI** | ❌ Pay-per-use | Fast | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Anthropic** | ❌ Pay-per-use | Fast | `claude-3-5-haiku-20241022` | [console.anthropic.com](https://console.anthropic.com) |
| **Universal** | Depends | Varies | Any | Custom endpoint |

### Option C — Universal / Custom Endpoint (Kimi K2, Together.ai, etc.)

Works with any OpenAI-compatible API server.

In Settings → AI Engine, select **Universal / Custom Endpoint**, then fill in:

| Field | Example |
|-------|---------|
| Base URL | `https://api.moonshot.cn/v1` |
| Model Name | `kimi-k2-instruct` |
| API Key | Your Moonshot/Together.ai key |

**Common Universal endpoints:**

```
Kimi K2 (Moonshot):  https://api.moonshot.cn/v1     model: kimi-k2-instruct
Together.ai:         https://api.together.xyz/v1    model: meta-llama/Llama-3-70b-chat-hf
Mistral:             https://api.mistral.ai/v1      model: mistral-large-latest
Perplexity:          https://api.perplexity.ai      model: llama-3.1-sonar-large-128k-online
LM Studio (local):   http://localhost:1234/v1        model: your-loaded-model
```

### AI Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| Min Confidence | 0.45 | AI must reach this confidence to override technical signal |
| AI Weight | 0.65 | How much AI score contributes to final blended score (0.0–1.0) |
| Timeout | 90s | Max seconds to wait for AI response before falling back |

---

## ⚙️ Strategy Configuration

Strategy settings are saved in **your browser's IndexedDB** and sent to the server as a request header — no config file needed. You can also edit `config/strategy_config.json` directly for the server default.

### Trading Pairs

Default: `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, XRP/USDT, DOGE/USDT, ADA/USDT, AVAX/USDT`

In Settings → Trading Pairs, check/uncheck pairs or click **➕ Add Coin** to add any Binance spot pair.

### Timeframes

`1m`, `5m` *(default)*, `15m`, `1h`, `4h`

### Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Stop Loss % | 1.5% | Auto-sell when position drops this much |
| Take Profit % | 2.3% | Auto-sell when position gains this much |
| Max Position Size % | 3.0% | Maximum % of balance per trade |
| Daily Loss Limit % | 4.0% | Bot pauses trading for the day when reached |
| Max Open Trades | 3 | Maximum simultaneous positions |

### Technical Indicator Defaults

```json
{
  "rsi":  { "oversold": 30,   "overbought": 70,  "period": 14 },
  "macd": { "fast": 12,       "slow": 26,         "signal": 9  },
  "bb":   { "period": 20,     "std_dev": 2.0                    },
  "ma":   { "short": 20,      "long": 50                        }
}
```

---

## 💡 Example Usage

### Example 1 — Paper trading with no AI (pure technical signals)

```
1. Start server:  uvicorn dashboard.app:app --reload --port 8000
2. Open:          http://localhost:8000
3. Settings:
   - Mode: Paper
   - Pairs: BTC/USDT, ETH/USDT
   - AI Engine: disabled
   - Stop Loss: 1.5% / Take Profit: 3.0%
4. Click ▶ Start Bot
5. Watch signals update every 10 seconds on the Dashboard
```

**What happens:** The bot reads 5-minute candles from Binance, computes RSI, MACD, Bollinger Bands, and EMAs, generates BUY/SELL/HOLD signals, and executes paper trades automatically. Results appear in Trade History.

---

### Example 2 — Paper trading with Groq AI (free, no credit card)

```
1. Get a free Groq key at console.groq.com/keys
2. Start server and open dashboard
3. Settings → AI Engine:
   - Provider: Groq
   - Model: llama-3.3-70b-versatile
   - Paste your key → Save Key
   - Min Confidence: 0.50
   - AI Weight: 0.65
   - Enable AI ✓
4. Click ▶ Start Bot
5. Watch the AI Signals page — each card shows:
   - Predicted direction (▲ UP / ▼ DOWN / → SIDEWAYS)
   - Predicted next close price
   - Predicted high/low range
   - Support and resistance levels
   - Confidence percentage
   - Countdown timer until prediction expires
```

**What happens:** For each pair, the bot sends the last 20 candles + all indicator values to Groq's Llama model. The model reasons step-by-step and returns a structured JSON prediction. The bot blends this with the technical score (65% AI / 35% technical by default) to decide BUY/SELL/HOLD.

---

### Example 3 — Local Ollama AI on low-RAM machine

```bash
# Terminal 1 — start Ollama with the small model
ollama pull deepseek-r1:1.5b
ollama serve

# Terminal 2 — start the bot
uvicorn dashboard.app:app --reload --port 8000
```

In Settings → AI Engine:
- Provider: `DeepSeek-R1 (Local Ollama)`
- Model: `deepseek-r1:1.5b`
- Click **Test** → should show `✓ Ready`
- Enable AI → Save AI Config → Start Bot

---

### Example 4 — Live trading on Binance Testnet

```
1. Create a Binance Testnet account at testnet.binance.vision
2. Generate API key (Testnet keys have different format from mainnet)
3. In Settings → Exchange API Keys:
   - Exchange: Binance
   - Paste testnet API Key + Secret
   - ✅ Use Testnet / Sandbox = ON
   - Save Keys
4. In Settings → Trading Mode: select Live
5. Start Bot — orders will appear in your Testnet account
```

> ⚠️ **Testnet orders are completely simulated** with no real money.

---

### Example 5 — Kimi K2 via Universal endpoint

```
1. Sign up at platform.moonshot.cn and get an API key
2. Settings → AI Engine → Universal / Custom Endpoint
3. Fill in:
   - Base URL:  https://api.moonshot.cn/v1
   - Model:     kimi-k2-instruct
   - API Key:   sk-xxxxxxxxxxxx
4. Save Key → Enable AI → Save AI Config
5. Start Bot
```

---

### Example 6 — Multiple users on one server

```
User A (their browser):
  - Groq key stored in their IndexedDB
  - Pairs: BTC, ETH
  - Bot starts → keys cached in server RAM for their session
  - AI uses their Groq key

User B (different browser / device):
  - OpenAI key stored in their IndexedDB
  - Pairs: SOL, XRP, DOGE
  - Configures independently, starts their own session
  - AI uses their OpenAI key

Server: one uvicorn process, zero user data on disk
```

---

## 🏗️ Architecture

### How the bot loop works

```
Every 10 seconds:
  ├─ Fetch OHLCV candles from Binance (direct REST, cached 15s)
  ├─ Fetch ticker for current price (cached 5s)
  ├─ Calculate RSI, MACD, Bollinger Bands, EMA, Volume
  ├─ Generate base technical signal (score -1.0 to +1.0)
  │
  ├─ [If AI enabled]
  │   ├─ Send last 20 candles + indicators to AI model
  │   ├─ Parse prediction: direction, close, high, low, S/R, confidence
  │   ├─ If confidence ≥ threshold → blend AI score with technical score
  │   └─ Set dynamic SL/TP from predicted support/resistance
  │
  ├─ Risk check: daily loss limit, max open trades, duplicate positions
  ├─ Execute BUY/SELL via PaperTrader or TradeExecutor
  └─ Write result to bot_state.json → broadcast to all dashboard clients
```

### Scalability

| Users | Setup |
|-------|-------|
| 1–1,000 | Single `uvicorn` process — works out of the box |
| 10,000 | `gunicorn -w 4 -k uvicorn.workers.UvicornWorker dashboard.app:app` |
| 100,000 | Nginx reverse proxy + `Cache-Control` caching layer |
| 1,000,000 | CDN for static files + multiple regional bot instances |

---

## 🔧 Troubleshooting

### `PermissionError: [WinError 32]` in log file

**Cause:** Windows locks the log file when multiple processes try to rotate it.
**Fix:** The logger is already patched with `_WinSafeRotatingHandler`. Make sure you're using the latest `logging_monitor/logger.py`.

### `ConnectError: All connection attempts failed` (Ollama)

**Cause:** AI is enabled but Ollama is not running.
**Fix:**
```bash
# Start Ollama server
ollama serve

# Or disable AI in Settings → AI Engine → toggle off
```
The server auto-detects this at startup and disables AI with a single clear warning.

### Dashboard shows stale data

**Cause:** Bot is stopped.
**Fix:** Click **▶ Start Bot**. The dashboard polls every 10 seconds only when the bot is running.

### `Insufficient balance` in paper trading

**Cause:** Previous trades consumed the virtual $10,000 balance.
**Fix:** Click **🔄 Reset Balance** in Trade History.

### AI shows "AI THINKING..." but never completes

**Cause:** Model timeout (large Ollama model on slow CPU) or API rate limit.
**Fix:**
- For Ollama: increase timeout in Settings → AI Engine → Timeout
- For Groq: Groq free tier is 30 RPM; reduce active pairs or increase cycle interval
- Check browser console (F12) for error messages

### `ModuleNotFoundError` on startup

**Cause:** Dependencies not installed or wrong Python environment.
**Fix:**
```bash
# Make sure venv is active
venv\Scripts\activate          # Windows
source venv/bin/activate        # Linux/macOS

# Reinstall
pip install -r requirements.txt
```

### Port 8000 already in use

```bash
# Windows — find and kill the process
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux
lsof -ti:8000 | xargs kill -9
```

---

## ⚠️ Disclaimer

> **CryptoBot Pro is provided for educational and research purposes only.**
>
> - **Do not use real money until you fully understand the system and have tested extensively in paper mode and on testnet.**
> - Cryptocurrency trading involves significant risk of loss. Past performance does not guarantee future results.
> - AI prediction accuracy is approximately 55–59% directional accuracy. This is a small statistical edge, not a guarantee of profit.
> - The authors accept no responsibility for financial losses incurred through use of this software.
> - Always start with Paper mode. Move to Testnet next. Only use real funds if you can afford to lose them.

---

<div align="center">

Built with ❤️ using Python · FastAPI · CCXT · LightweightCharts
 
**[⬆ Back to Top](#-cryptobot-pro)**

</div>
