"""
Main Bot Loop — runs 24/7 on Render
Every 15 minutes:
  1. Checks risk limits (daily loss, max drawdown, profit target)
  2. Syncs today's closed trades for accurate P&L tracking
  3. Scans ALL tradeable symbols for setups
  4. Picks the highest-confidence setup
  5. Places trade if it passes all filters
"""

import time
import logging
import os
from datetime import datetime

import alpaca_connector as alpaca
import analyst
import risk_manager as risk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/bot.log")
    ]
)
log = logging.getLogger(__name__)

CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL_SECONDS", 300))


def log_decision(decision: dict):
    log.info("─" * 60)
    log.info(f"SYMBOL       : {decision.get('symbol', 'N/A')}")
    log.info(f"DECISION     : {decision['decision']}")
    log.info(f"CONFIDENCE   : {decision['confidence']}/10")
    log.info(f"PATTERN      : {decision.get('pattern', 'N/A')}")
    log.info(f"REASONING    : {decision['reasoning']}")
    if decision["decision"] != "NO_TRADE":
        log.info(f"ENTRY        : {decision['entry_price']}")
        log.info(f"SL           : {decision['sl_price']}")
        log.info(f"TP           : {decision['tp_price']}")
        log.info(f"RR           : {decision['rr_ratio']}")
    log.info("─" * 60)


def log_status(status: dict):
    log.info(
        f"💰 Balance: ${status['balance']:.2f} | "
        f"Profit: +${status['profit']:.2f} / ${status['profit_target']:.2f} target | "
        f"Daily loss: ${status['daily_used']:.2f} / $250 | "
        f"Max drawdown: ${status['total_loss']:.2f} / $500"
    )


def run_cycle():
    log.info(f"⏱  Cycle started at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # 1. Get account info
    try:
        account = alpaca.get_account_info()
    except Exception as e:
        log.error(f"Failed to get account info: {e}")
        return

    # 2. Sync today's closed trades for accurate daily P&L
    try:
        closed_today = alpaca.get_closed_trades_today()
    except Exception as e:
        log.warning(f"Could not fetch closed trades: {e}")
        closed_today = None

    # 3. Check all risk limits
    tradeable, reason = risk.can_trade(account["balance"], closed_trades=closed_today)
    status = risk.get_status(account["balance"], closed_trades=closed_today)
    log_status(status)

    if not tradeable:
        log.warning(f"🛑 TRADING HALTED: {reason}")
        return

    # 4. Check if already in a position (only 1 open at a time)
    all_open = alpaca.get_open_trades()
    if all_open:
        log.info(f"📊 Already in {len(all_open)} open position(s) — skipping entry")
        for t in all_open:
            log.info(f"   → {t['instrument']} | qty: {t['currentUnits']} | P&L: ${float(t.get('unrealizedPL', 0)):.2f}")
        return

    # 5. Scan all symbols for setups
    log.info(f"🔍 Scanning {len(alpaca.TRADEABLE_PAIRS)} symbols for setups...")
    decisions = []

    for symbol in alpaca.TRADEABLE_PAIRS:
        log.info(f"   Checking {symbol}...")
        try:
            market_summary = alpaca.get_market_summary(symbol)
            if market_summary is None:
                log.info(f"   {symbol}: Insufficient data, skipping")
                continue
        except Exception as e:
            log.warning(f"   {symbol}: Data fetch failed ({e}), skipping")
            continue

        decision = analyst.analyze_market(symbol, market_summary, {**account, **status}, all_open)
        decisions.append(decision)

        if decision["decision"] != "NO_TRADE":
            log.info(
                f"   {symbol}: {decision['decision']} | "
                f"Confidence: {decision['confidence']}/10 | "
                f"RR: {decision.get('rr_ratio', 'N/A')} | "
                f"Pattern: {decision.get('pattern', 'N/A')}"
            )
        else:
            log.info(f"   {symbol}: NO TRADE — {decision['reasoning'][:60]}")

        time.sleep(1)

    # 6. Pick the best setup
    best = analyst.pick_best_setup(decisions)
    log_decision(best)

    if best["decision"] == "NO_TRADE":
        log.info("✋ No qualifying setup found this cycle — staying flat")
        return

    symbol = best["symbol"]

    # 7. Final validation
    if best["confidence"] < 6:
        log.info(f"⚠️  Confidence too low ({best['confidence']}/10) — skipping")
        return

    if not best["sl_price"] or not best["tp_price"]:
        log.info("⚠️  Missing SL or TP — skipping")
        return

    if best.get("rr_ratio") and best["rr_ratio"] < 2.0:
        log.info(f"⚠️  RR too low ({best['rr_ratio']}) — skipping")
        return

    # 8. Calculate position size
    sl_distance = abs(best["entry_price"] - best["sl_price"])
    units = risk.calculate_units(account["balance"], sl_distance, best["entry_price"], symbol)

    log.info(f"📦 Placing order: {best['decision']} {units} shares of {symbol}")

    # 9. Place the trade
    try:
        result = alpaca.place_order(
            symbol    = symbol,
            direction = best["decision"].lower(),
            units     = units,
            sl_price  = best["sl_price"],
            tp_price  = best["tp_price"],
        )
    except Exception as e:
        log.error(f"Order placement failed: {e}")
        return

    if result["success"]:
        live_rr = result.get("live_rr", best["rr_ratio"])
        log.info(
            f"✅ Trade placed! | {symbol} {best['decision']} | "
            f"ID: {result['trade_id']} | Price: {result['price']} | "
            f"Units: {units} | SL: {best['sl_price']} | TP: {best['tp_price']} | "
            f"Signal RR: {best['rr_ratio']} | Live RR: {live_rr}"
        )
    else:
        log.warning(f"⚠️  Order skipped: {result['error']}")


if __name__ == "__main__":
    log.info("🚀 Alpaca Multi-Symbol Trading Bot started")
    log.info(f"   Symbols   : {len(alpaca.TRADEABLE_PAIRS)} instruments")
    log.info(f"   Interval  : every {CHECK_INTERVAL}s ({CHECK_INTERVAL // 60} min)")
    log.info(f"   Balance   : ${os.environ.get('STARTING_BALANCE', 5000)}")
    log.info(f"   Target    : +${os.environ.get('PROFIT_TARGET', 500)}")
    log.info(f"   Max loss  : -${os.environ.get('MAX_TOTAL_DRAWDOWN', 500)}")
    log.info(f"   Daily cap : -${os.environ.get('MAX_DAILY_LOSS', 250)}")
    log.info(f"   Alpaca    : {alpaca.ALPACA_BASE_URL}")
    log.info("─" * 60)

    while True:
        try:
            run_cycle()
        except Exception as e:
            log.error(f"Unexpected error in cycle: {e}", exc_info=True)
        log.info(f"💤 Sleeping {CHECK_INTERVAL}s until next cycle...")
        time.sleep(CHECK_INTERVAL)
