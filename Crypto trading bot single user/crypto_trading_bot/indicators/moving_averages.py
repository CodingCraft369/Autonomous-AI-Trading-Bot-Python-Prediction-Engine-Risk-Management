"""Simple Moving Averages."""
def compute_ma(close, short=20, long=50):
    return {
        "short": close.rolling(window=short).mean(),
        "long":  close.rolling(window=long).mean(),
    }