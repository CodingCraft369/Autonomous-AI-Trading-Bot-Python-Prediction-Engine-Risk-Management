@echo off
:: CryptoBot Pro — First-time setup / repair
:: Run this ONCE to fix uvicorn and install all dependencies
:: Place in: C:\crypto_trading_bot\setup.bat

title CryptoBot Pro - Setup

echo.
echo  ========================================
echo   CryptoBot Pro - Setup / Repair
echo  ========================================
echo.

cd /d "%~dp0"

:: Activate virtualenv
if exist "venv\Scripts\activate.bat" (
    echo  [INFO] Activating virtualenv...
    call venv\Scripts\activate.bat
) else (
    echo  [WARN] No venv found - using system Python
    echo  [HINT] Create one with: python -m venv venv
)

echo.
echo  [STEP 1] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo  [STEP 2] Installing / upgrading all dependencies...
echo  (This fixes the uvicorn --reload bug on Windows Python 3.11)
pip install -r requirements.txt --upgrade

echo.
echo  ========================================
echo   Setup complete!
echo.
echo   Start the bot with:
echo     uvicorn dashboard.app:app --reload --port 8000
echo.
echo   OR double-click: start.bat
echo  ========================================
echo.
pause