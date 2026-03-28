"""
Alpaca Connector — handles all communication with Alpaca Markets API
Pulls bar data, calculates indicators, places and manages trades
Supports multi-symbol scanning
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATA_BASE_URL     = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type":        "application/json"
}

# Tradeable symbols — stocks, ETFs, crypto
TRADEABLE_PAIRS = [
    # Gold & commodities (ETFs)
    "GLD",   # Gold ETF
    "SLV",   # Silver ETF
    "GDX",   # Gold miners ETF
    # Major indices ETFs
    "SPY",   # S&P 500
    "QQQ",   # NASDAQ 100
    "DIA",   # Dow Jones
    "IWM",   # Russell 2000
    # High-liquidity stocks
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "TSLA",  # Tesla
    "AMZN",  # Amazon
    "META",  # Meta
    "GOOGL", # Alphabet
]

TIMEFRAME_MAP = {
    "15M": "15Min",
    "1H":  "1Hour",
    "4H":  "4Hour",
}


# ─────────────────────────────────────────────
# ACCOUNT
# ─────────────────────────────────────────────

def get_account_info():
    url = f"{ALPACA_BASE_URL}/v2/account"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    acc = r.json()
    return {
        "balance":     float(acc["equity"]),
        "equity":      float(acc["equity"]),
        "unrealized":  float(acc.get("unrealized_pl", 0)),
        "margin_used": float(acc.get("initial_margin", 0)),
    }


# ─────────────────────────────────────────────
# MARKET DATA + INDICATORS
# ─────────────────────────────────────────────

def get_candles(symbol, timeframe="1H", count=200):
    tf = TIMEFRAME_MAP.get(timeframe, "1Hour")

    # Calculate start time based on count and timeframe
    now = datetime.now(timezone.utc)
    if timeframe == "15M":
        start = now - timedelta(minutes=15 * count)
    elif timeframe == "1H":
        start = now - timedelta(hours=count)
    elif timeframe == "4H":
        start = now - timedelta(hours=4 * count)
    else:
        start = now - timedelta(hours=count)

    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")

    url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": tf,
        "start":     start_str,
        "limit":     count,
        "feed":      "iex",
        "sort":      "asc"
    }

    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    bars = r.json().get("bars", [])

    if not bars:
        return pd.DataFrame()

    rows = []
    for b in bars:
        rows.append({
            "time":   b["t"],
            "open":   float(b["o"]),
            "high":   float(b["h"]),
            "low":    float(b["l"]),
            "close":  float(b["c"]),
            "volume": int(b["v"])
        })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df


def calculate_indicators(df):
    """Add EMA20, EMA50, EMA200, RSI14, ATR14, MACD, BB, StochRSI to dataframe"""
    if len(df) < 50:
        return df

    # EMAs
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI 14
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR 14
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"]  = true_range.ewm(com=13, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12        = df["close"].ewm(span=12, adjust=False).mean()
    ema26        = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]   = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"]   = df["macd"] - df["signal"]

    # Bollinger Bands (20, 2)
    rolling_mean   = df["close"].rolling(20).mean()
    rolling_std    = df["close"].rolling(20).std()
    df["bb_upper"] = rolling_mean + (2 * rolling_std)
    df["bb_lower"] = rolling_mean - (2 * rolling_std)
    df["bb_mid"]   = rolling_mean

    # Stochastic RSI
    rsi_min        = df["rsi"].rolling(14).min()
    rsi_max        = df["rsi"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100

    # Swing highs / lows
    df["swing_high"] = df["high"].rolling(5, center=True).max() == df["high"]
    df["swing_low"]  = df["low"].rolling(5,  center=True).min() == df["low"]

    # Trend
    df["trend"] = df["ema20"] > df["ema50"]

    return df


def get_market_summary(symbol):
    """Pull multi-timeframe data + indicators for Gemini to analyze"""
    summary = {}
    for tf in ["4H", "1H", "15M"]:
        try:
            df = get_candles(symbol, tf, 200)
            if df.empty or len(df) < 52:
                return None
            df = calculate_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]

            recent      = df.tail(30)
            swing_highs = recent[recent["swing_high"]]["high"].tolist()[-3:]
            swing_lows  = recent[recent["swing_low"]]["low"].tolist()[-3:]

            summary[tf] = {
                "time":                str(last["time"]),
                "open":                round(last["open"],  5),
                "high":                round(last["high"],  5),
                "low":                 round(last["low"],   5),
                "close":               round(last["close"], 5),
                "prev_close":          round(prev["close"], 5),
                "ema20":               round(last["ema20"],  5),
                "ema50":               round(last["ema50"],  5),
                "ema200":              round(last["ema200"], 5),
                "rsi":                 round(float(last["rsi"]),       2),
                "atr":                 round(float(last["atr"]),       5),
                "macd":                round(float(last["macd"]),      6),
                "macd_signal":         round(float(last["signal"]),    6),
                "macd_hist":           round(float(last["hist"]),      6),
                "bb_upper":            round(float(last["bb_upper"]),  5),
                "bb_lower":            round(float(last["bb_lower"]),  5),
                "bb_mid":              round(float(last["bb_mid"]),    5),
                "stoch_rsi":           round(float(last["stoch_rsi"]), 2),
                "trend":               "BULLISH" if last["trend"] else "BEARISH",
                "candles_above_ema20": int((df["close"].tail(10) > df["ema20"].tail(10)).sum()),
                "swing_highs":         [round(x, 5) for x in swing_highs],
                "swing_lows":          [round(x, 5) for x in swing_lows],
            }
        except Exception:
            return None

    return summary


# ─────────────────────────────────────────────
# OPEN POSITIONS
# ─────────────────────────────────────────────

def get_open_trades(symbol=None):
    url = f"{ALPACA_BASE_URL}/v2/positions"
    r   = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    positions = r.json()

    # Normalize to same shape main.py expects
    trades = []
    for p in positions:
        trades.append({
            "instrument":    p["symbol"],
            "currentUnits":  p["qty"],
            "unrealizedPL":  float(p.get("unrealized_pl", 0)),
            "id":            p["asset_id"],
        })

    if symbol:
        return [t for t in trades if t["instrument"] == symbol]
    return trades


def get_closed_trades_today():
    """Get trades closed today for P&L tracking"""
    from datetime import date
    today_str = date.today().isoformat() + "T00:00:00Z"

    url = f"{ALPACA_BASE_URL}/v2/account/activities/FILL"
    params = {"after": today_str, "direction": "desc", "page_size": 50}
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    activities = r.json()

    # Normalize to match risk_manager expectations
    trades = []
    for a in activities:
        if a.get("type") == "fill":
            trades.append({
                "id":         a.get("id"),
                "realizedPL": float(a.get("price", 0)) * float(a.get("qty", 0)) * (-1 if a.get("side") == "sell" else 1),
            })
    return trades


# ─────────────────────────────────────────────
# PLACE ORDER
# ─────────────────────────────────────────────

def place_order(symbol, direction, units, sl_price, tp_price):
    """
    direction: 'buy' or 'sell'
    units: number of shares/units
    Uses bracket order for automatic SL and TP
    """
    side = "buy" if direction == "buy" else "sell"
    qty  = abs(int(units))

    if qty < 1:
        qty = 1

    order_body = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          side,
        "type":          "market",
        "time_in_force": "day",
        "order_class":   "bracket",
        "stop_loss": {
            "stop_price": str(round(sl_price, 2))
        },
        "take_profit": {
            "limit_price": str(round(tp_price, 2))
        }
    }

    url = f"{ALPACA_BASE_URL}/v2/orders"
    r   = requests.post(url, headers=HEADERS, json=order_body, timeout=15)

    if r.status_code not in [200, 201]:
        return {"success": False, "error": r.text}

    data = r.json()
    return {
        "success":   True,
        "trade_id":  data.get("id"),
        "price":     data.get("filled_avg_price") or "market",
        "units":     qty,
        "direction": direction,
        "sl":        sl_price,
        "tp":        tp_price,
    }
