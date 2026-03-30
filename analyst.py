"""
Groq Analyst — sends market data to Groq (Llama 3.3 70B) and gets trade decisions.
Free tier: 30 requests/min, 14,400/day — more than enough for this bot.
"""

import os
import json
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.3-70b-versatile"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type":  "application/json"
}

SYSTEM_PROMPT = """You are an elite institutional trader managing a $5,000 funded trading challenge account.

CHALLENGE RULES (non-negotiable):
- Starting balance: $5,000
- Profit target: $500 (stop trading at $5,500)
- Max total drawdown: $500 (never go below $4,500)
- Max daily loss: $250
- Risk per trade: 1% of current balance
- Minimum Risk:Reward ratio: 1:2

YOUR STRATEGY — use ALL of the following to confirm a setup:

TREND (4H bias):
- EMA20 > EMA50 > EMA200 = strong bull trend
- EMA20 < EMA50 < EMA200 = strong bear trend
- Only trade in the direction of the 4H trend

STRUCTURE (1H):
- Identify swing highs and swing lows
- Only enter after a confirmed Break of Structure (BOS) with a clean retest
- SL must sit behind the most recent swing high/low

ENTRY TRIGGER (15M):
- MACD histogram crossover confirming direction
- RSI not overbought/oversold at entry (avoid RSI > 75 for buys, < 25 for sells)
- Stochastic RSI confirming momentum direction
- Bollinger Band squeeze breakout adds extra confidence
- Price must be near value area (near EMA20 on 1H), not extended

CONFLUENCE SCORING (must be 7/10 or higher to trade):
- 4H trend aligned: +2
- 1H structure break confirmed: +2
- 15M entry trigger (MACD + RSI): +2
- Stoch RSI momentum confirmation: +1
- ATR SL makes sense (1-2x ATR): +1
- Clean price action (no choppy wicks): +1
- No major news or weekend risk: +1

RISK RULES:
- SL must be structural (behind swing high/low), NOT arbitrary
- TP must give at least 1:2 RR
- Validate SL distance is 1-2x ATR on entry timeframe
- If market is ranging or unclear: NO_TRADE — capital protection is priority
- If daily or total loss limits are near: be more conservative

You will receive multi-timeframe data (4H → 1H → 15M) with full indicator suite.
You must respond ONLY in valid JSON. No text outside the JSON object.

JSON format:
{
  "decision": "BUY" | "SELL" | "NO_TRADE",
  "confidence": 1-10,
  "reasoning": "brief explanation covering trend, structure, trigger",
  "entry_price": float or null,
  "sl_price": float or null,
  "tp_price": float or null,
  "rr_ratio": float or null,
  "timeframe_used": "15M" | "1H" | "4H",
  "pattern": "describe the pattern/setup (e.g. BOS retest, BB squeeze, MACD crossover)"
}"""


def analyze_market(symbol: str, market_summary: dict, account_info: dict, open_trades: list) -> dict:
    """Send market data to Groq and get trade decision"""

    user_message = f"""
Analyze this pair: {symbol}

IMPORTANT: Use the "live_price" field as your entry price — this is the real-time market price.
Do NOT use candle close prices as entry. SL and TP must be calculated from live_price.

Multi-timeframe market data:
{json.dumps(market_summary, indent=2)}

Account status:
- Balance: ${account_info['balance']:.2f}
- Equity: ${account_info['equity']:.2f}
- Unrealized P&L: ${account_info['unrealized']:.2f}
- Open trades: {len(open_trades)}

Risk status:
- Daily loss used: ${account_info.get('daily_used', 0):.2f}
- Daily loss remaining: ${account_info.get('daily_remaining', 250):.2f}
- Total drawdown used: ${account_info.get('total_loss', 0):.2f}
- Total drawdown remaining: ${account_info.get('total_remaining', 500):.2f}

Analyze top-down: 4H bias → 1H structure → 15M entry.
Score the setup out of 10 using the confluence scoring system.
Only recommend BUY or SELL if confidence is 7 or higher AND all three timeframes agree.
Respond in JSON only. No markdown, no extra text.
"""

    payload = {
        "model":       MODEL,
        "temperature": 0.1,
        "max_tokens":  1000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ]
    }

    try:
        r = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=30)
        r.raise_for_status()

        content = r.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        decision = json.loads(content)
        decision["symbol"] = symbol
        return decision

    except Exception as e:
        return {
            "symbol":         symbol,
            "decision":       "NO_TRADE",
            "confidence":     0,
            "reasoning":      f"Analysis failed: {e}",
            "entry_price":    None,
            "sl_price":       None,
            "tp_price":       None,
            "rr_ratio":       None,
            "timeframe_used": None,
            "pattern":        None,
        }


def pick_best_setup(decisions: list) -> dict:
    """Pick the best tradeable setup from all decisions"""
    tradeable = [
        d for d in decisions
        if d.get("decision") in ("BUY", "SELL")
        and d.get("confidence", 0) >= 7
        and d.get("sl_price") is not None
        and d.get("tp_price") is not None
        and d.get("rr_ratio") is not None
        and d.get("rr_ratio", 0) >= 2.0
    ]

    if not tradeable:
        return {
            "symbol":         None,
            "decision":       "NO_TRADE",
            "confidence":     0,
            "reasoning":      "No qualifying setup found across all pairs this cycle.",
            "entry_price":    None,
            "sl_price":       None,
            "tp_price":       None,
            "rr_ratio":       None,
            "timeframe_used": None,
            "pattern":        None,
        }

    return max(tradeable, key=lambda d: (d["confidence"], d.get("rr_ratio", 0)))
