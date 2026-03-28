"""Bollinger Bands."""
def compute_bb(close, period=20, std_dev=2):
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return {
        "upper": mid + std_dev * std,
        "mid":   mid,
        "lower": mid - std_dev * std,
    }