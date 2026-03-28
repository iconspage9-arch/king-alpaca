"""
Minimal web server so the bot can run as a Render Web Service.
Exposes a /health endpoint and a /logs endpoint to view recent activity.
"""

import threading
import logging
import os
from flask import Flask, jsonify

app = Flask(__name__)
log = logging.getLogger(__name__)

@app.route("/")
def index():
    return jsonify({
        "status": "running",
        "bot": "Claude Multi-Pair Trading Bot",
        "message": "Bot is active. Check /health or /logs."
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/logs")
def logs():
    try:
        with open("/tmp/bot.log", "r") as f:
            lines = f.readlines()
        # Return last 100 lines
        return "<pre>" + "".join(lines[-100:]) + "</pre>"
    except Exception as e:
        return f"No logs yet: {e}", 200


def start_bot():
    """Run the trading bot loop in a background thread"""
    import main
    # main.py's while loop runs here
    while True:
        try:
            main.run_cycle()
        except Exception as e:
            log.error(f"Bot cycle error: {e}", exc_info=True)
        import time
        time.sleep(int(os.environ.get("CHECK_INTERVAL_SECONDS", 900)))


if __name__ == "__main__":
    # Start bot in background thread
    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()

    # Start Flask on the port Render expects
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
