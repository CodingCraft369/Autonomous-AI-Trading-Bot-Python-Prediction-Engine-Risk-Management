@echo off
:: CryptoBot Pro — Start dashboard server
:: Equivalent to: uvicorn dashboard.app:app --reload --port 8000
:: Place in: C:\crypto_trading_bot\start.bat

title CryptoBot Pro

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo  Starting CryptoBot Pro at http://localhost:8000
echo  Press Ctrl+C to stop
echo.

uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload
pause