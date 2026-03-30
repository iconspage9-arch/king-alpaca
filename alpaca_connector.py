"""
Alpaca Connector — handles all communication with Alpaca Markets API
Pulls bar data, calculates indicators, places and manages trades
"""

import os
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATA_BASE_URL     = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type":        "application/json"
}

TRADEABLE_PAIRS = [
    "BTC/USD",
]

TIMEFRAME_MAP = {
    "15M": "15Min",
    "1H":  "1Hour",
    "4H":  "4Hour",
}


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


def get_candles(symbol, timeframe="1H", count=200):
    tf = TIMEFRAME_MAP.get(timeframe, "1Hour")

    now = datetime.now(timezone.utc)
    if timeframe == "15M":
        start = now - timedelta(days=30)
    elif timeframe == "1H":
        start = now - timedelta(days=30)
    elif timeframe == "4H":
        start = now - timedelta(days=60)
    else:
        start = now - timedelta(days=30)

    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")

    is_crypto = "/" in symbol
    if is_crypto:
        url = f"{DATA_BASE_URL}/v1beta3/crypto/us/bars"
        params = {"symbols": symbol, "timeframe": tf, "start": start_str, "sort": "asc"}
    else:
        url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
        params = {"timeframe": tf, "start": start_str, "limit": count, "feed": "iex", "sort": "asc"}

    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()
    bars = raw["bars"][symbol] if is_crypto else raw.get("bars", [])

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
    if len(df) < 20:
        return df

    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"]  = true_range.ewm(com=13, adjust=False).mean()

    ema12        = df["close"].ewm(span=12, adjust=False).mean()
    ema26        = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]   = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"]   = df["macd"] - df["signal"]

    rolling_mean   = df["close"].rolling(20).mean()
    rolling_std    = df["close"].rolling(20).std()
    df["bb_upper"] = rolling_mean + (2 * rolling_std)
    df["bb_lower"] = rolling_mean - (2 * rolling_std)
    df["bb_mid"]   = rolling_mean

    rsi_min         = df["rsi"].rolling(14).min()
    rsi_max         = df["rsi"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100

    df["swing_high"] = df["high"].rolling(3, center=True).max() == df["high"]
    df["swing_low"]  = df["low"].rolling(3,  center=True).min() == df["low"]
    df["trend"]      = df["ema20"] > df["ema50"]

    return df


def get_market_summary(symbol):
    # Fetch live price first — this is what the LLM must use as entry price
    try:
        live_price = get_live_price(symbol)
    except Exception as e:
        print(f"{symbol}: could not fetch live price: {e}", flush=True)
        return None

    summary = {"live_price": round(live_price, 5)}

    for tf in ["4H", "1H", "15M"]:
        try:
            df = get_candles(symbol, tf, 200)
            if df.empty or len(df) < 10:
                print(f"{symbol} {tf}: got {len(df)} candles", flush=True)
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
        except Exception as e:
            print(f"{symbol} {tf} error: {e}", flush=True)
            return None

    return summary


def get_open_trades(symbol=None):
    url = f"{ALPACA_BASE_URL}/v2/positions"
    r   = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    positions = r.json()

    trades = []
    for p in positions:
        trades.append({
            "instrument":   p["symbol"],
            "currentUnits": p["qty"],
            "unrealizedPL": float(p.get("unrealized_pl", 0)),
            "id":           p["asset_id"],
        })

    if symbol:
        return [t for t in trades if t["instrument"] == symbol]
    return trades


def get_closed_trades_today():
    from datetime import date
    today_str = date.today().isoformat() + "T00:00:00Z"

    url = f"{ALPACA_BASE_URL}/v2/account/activities/FILL"
    params = {"after": today_str, "direction": "desc", "page_size": 50}
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    activities = r.json()

    trades = []
    for a in activities:
        if a.get("type") == "fill":
            trades.append({
                "id":         a.get("id"),
                "realizedPL": float(a.get("price", 0)) * float(a.get("qty", 0)) * (-1 if a.get("side") == "sell" else 1),
            })
    return trades


def get_live_price(symbol):
    """Fetch the latest trade price for a symbol."""
    is_crypto = "/" in symbol
    if is_crypto:
        url = f"{DATA_BASE_URL}/v1beta3/crypto/us/latest/trades"
        params = {"symbols": symbol}
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return float(data["trades"][symbol]["p"])
    else:
        url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/trades/latest"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return float(r.json()["trade"]["p"])


def place_order(symbol, direction, units, sl_price, tp_price):
    side = "buy" if direction == "buy" else "sell"
    qty  = abs(int(units))

    if qty < 1:
        qty = 1

    # --- Fetch live price and validate TP before placing ---
    try:
        live_price = get_live_price(symbol)
    except Exception as e:
        return {"success": False, "error": f"Could not fetch live price: {e}"}

    if side == "sell":
        # For SELL: TP must be below live price (we're targeting a drop)
        if tp_price >= live_price:
            return {
                "success": False,
                "error": (
                    f"TP {tp_price} is at or above live price {live_price} for a SELL — "
                    f"market has already moved past target. Skipping stale signal."
                )
            }
        # Also validate SL is above live price
        if sl_price <= live_price:
            return {
                "success": False,
                "error": (
                    f"SL {sl_price} is at or below live price {live_price} for a SELL — "
                    f"invalid structure. Skipping."
                )
            }
    else:  # buy
        # For BUY: TP must be above live price
        if tp_price <= live_price:
            return {
                "success": False,
                "error": (
                    f"TP {tp_price} is at or below live price {live_price} for a BUY — "
                    f"market has already moved past target. Skipping stale signal."
                )
            }
        # Also validate SL is below live price
        if sl_price >= live_price:
            return {
                "success": False,
                "error": (
                    f"SL {sl_price} is at or above live price {live_price} for a BUY — "
                    f"invalid structure. Skipping."
                )
            }

    # Re-check RR ratio using live price (not stale entry price from LLM)
    if side == "sell":
        live_sl_dist = abs(sl_price - live_price)
        live_tp_dist = abs(live_price - tp_price)
    else:
        live_sl_dist = abs(live_price - sl_price)
        live_tp_dist = abs(tp_price - live_price)

    if live_sl_dist == 0:
        return {"success": False, "error": "SL distance is zero — cannot place order."}

    live_rr = live_tp_dist / live_sl_dist
    if live_rr < 2.0:
        return {
            "success": False,
            "error": (
                f"Live RR is {live_rr:.2f} (live price: {live_price}) — "
                f"below 2.0 minimum after price movement. Skipping."
            )
        }
    # -------------------------------------------------------

    # Alpaca does NOT support bracket orders for crypto — place legs separately.
    url = f"{ALPACA_BASE_URL}/v2/orders"

    # 1) Main market order
    main_order = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          side,
        "type":          "market",
        "time_in_force": "gtc",
    }
    r = requests.post(url, headers=HEADERS, json=main_order, timeout=15)
    if r.status_code not in [200, 201]:
        return {"success": False, "error": r.text}

    data     = r.json()
    trade_id = data.get("id")

    # 2) Take-profit limit order (opposite side, reduce-only)
    exit_side = "sell" if side == "buy" else "buy"
    tp_order = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          exit_side,
        "type":          "limit",
        "time_in_force": "gtc",
        "limit_price":   str(round(tp_price, 2)),
        "reduce_only":   True,
    }
    r_tp = requests.post(url, headers=HEADERS, json=tp_order, timeout=15)
    if r_tp.status_code not in [200, 201]:
        print(f"WARNING: TP order failed: {r_tp.text}", flush=True)

    # 3) Stop-loss stop order (opposite side, reduce-only)
    sl_order = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          exit_side,
        "type":          "stop",
        "time_in_force": "gtc",
        "stop_price":    str(round(sl_price, 2)),
        "reduce_only":   True,
    }
    r_sl = requests.post(url, headers=HEADERS, json=sl_order, timeout=15)
    if r_sl.status_code not in [200, 201]:
        print(f"WARNING: SL order failed: {r_sl.text}", flush=True)

    return {
        "success":   True,
        "trade_id":  trade_id,
        "price":     data.get("filled_avg_price") or "market",
        "units":     qty,
        "direction": direction,
        "sl":        sl_price,
        "tp":        tp_price,
        "live_rr":   round(live_rr, 2),
    }
