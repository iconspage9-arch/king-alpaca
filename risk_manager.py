"""
Risk Manager
- Max total drawdown: $500  (account never goes below $4,500)
- Max daily loss:     $250
- Profit target:      $500  (stop trading at $5,500 — challenge passed!)
- Risk per trade:     1% of balance
- Min RR:             1:2
"""

import os
import json
import logging
from datetime import date

log = logging.getLogger(__name__)

RISK_FILE = "/tmp/risk_state.json"

STARTING_BALANCE = float(os.environ.get("STARTING_BALANCE",   5000))
MAX_TOTAL_LOSS   = float(os.environ.get("MAX_TOTAL_DRAWDOWN",  500))
MAX_DAILY_LOSS   = float(os.environ.get("MAX_DAILY_LOSS",      250))
PROFIT_TARGET    = float(os.environ.get("PROFIT_TARGET",       500))
RISK_PER_TRADE   = float(os.environ.get("RISK_PER_TRADE_PCT",  1.0))


def _load():
    try:
        with open(RISK_FILE) as f:
            return json.load(f)
    except Exception:
        return {"daily_loss": 0.0, "last_date": str(date.today()), "seen_trade_ids": []}


def _save(state):
    try:
        with open(RISK_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        log.error(f"Failed to save risk state: {e}")


def _reset_if_new_day(state):
    today = str(date.today())
    if state.get("last_date") != today:
        log.info(f"New day ({today}) — resetting daily loss counter")
        state["daily_loss"] = 0.0
        state["last_date"]  = today
    return state


def sync_daily_pnl_from_alpaca(closed_trades: list):
    """
    Sync today's closed trade P&L from Alpaca fill activities.
    Only counts losses, accumulates across the day.
    """
    state   = _reset_if_new_day(_load())
    seen    = set(state.get("seen_trade_ids", []))
    changed = False

    for trade in closed_trades:
        trade_id = trade.get("id")
        if trade_id in seen:
            continue

        realized_pl = float(trade.get("realizedPL", 0))
        seen.add(trade_id)

        if realized_pl < 0:
            state["daily_loss"] = round(state.get("daily_loss", 0) + abs(realized_pl), 2)
            log.info(f"Recorded loss ${abs(realized_pl):.2f} from trade {trade_id} — daily total: ${state['daily_loss']:.2f}")

        changed = True

    state["seen_trade_ids"] = list(seen)
    if changed:
        _save(state)

    return state


def can_trade(current_balance: float, closed_trades: list = None):
    if closed_trades is not None:
        state = sync_daily_pnl_from_alpaca(closed_trades)
    else:
        state = _reset_if_new_day(_load())

    total_loss = STARTING_BALANCE - current_balance
    daily_loss = state.get("daily_loss", 0.0)
    profit     = current_balance - STARTING_BALANCE

    reasons = []

    if profit >= PROFIT_TARGET:
        reasons.append(
            f"🏆 PROFIT TARGET HIT! Account at ${current_balance:.2f} "
            f"(+${profit:.2f}) — challenge passed! Bot halted."
        )
    if total_loss >= MAX_TOTAL_LOSS:
        reasons.append(
            f"⛔ Total drawdown limit hit: lost ${total_loss:.2f} of ${MAX_TOTAL_LOSS:.2f} max"
        )
    if daily_loss >= MAX_DAILY_LOSS:
        reasons.append(
            f"⛔ Daily loss limit hit: ${daily_loss:.2f} of ${MAX_DAILY_LOSS:.2f} max"
        )

    if reasons:
        return False, " | ".join(reasons)
    return True, "✅ Risk OK"


def get_risk_dollar(balance: float) -> float:
    return round(balance * RISK_PER_TRADE / 100, 2)


def calculate_units(balance: float, sl_distance_price: float, price: float, instrument: str = "SPY") -> int:
    """
    Calculate shares to trade based on 1% risk.
    units = risk_$ / sl_distance_per_share
    """
    risk_dollar = get_risk_dollar(balance)

    if sl_distance_price <= 0:
        return 1

    units = int(risk_dollar / sl_distance_price)

    # Caps by instrument type
    if instrument in ("GLD", "SLV", "GDX"):
        units = max(1, min(units, 500))    # ETFs
    elif instrument in ("SPY", "QQQ", "DIA", "IWM"):
        units = max(1, min(units, 100))    # Index ETFs
    else:
        units = max(1, min(units, 200))    # Stocks

    return units


def get_status(current_balance: float, closed_trades: list = None) -> dict:
    if closed_trades is not None:
        state = sync_daily_pnl_from_alpaca(closed_trades)
    else:
        state = _reset_if_new_day(_load())

    daily_used = state.get("daily_loss", 0.0)
    total_loss = STARTING_BALANCE - current_balance
    profit     = current_balance - STARTING_BALANCE

    return {
        "balance":          round(current_balance, 2),
        "profit":           round(profit, 2),
        "profit_target":    PROFIT_TARGET,
        "profit_remaining": round(PROFIT_TARGET - profit, 2),
        "daily_used":       round(daily_used, 2),
        "daily_remaining":  round(MAX_DAILY_LOSS - daily_used, 2),
        "total_loss":       round(total_loss, 2),
        "total_remaining":  round(MAX_TOTAL_LOSS - total_loss, 2),
    }
