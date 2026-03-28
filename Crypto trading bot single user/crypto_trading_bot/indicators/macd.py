"""MACD — Moving Average Convergence Divergence."""
def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": macd_line - signal_line,
    }