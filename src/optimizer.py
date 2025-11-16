import random

def suggest_next_params(db, user_params):
    """
    Agentic logic:
    1) Use user inputs if provided
    2) If missing, infer from DB
    3) If DB empty, explore randomly
    """

    # ----------------------------
    # 1. TICKERS
    # ----------------------------
    if "tickers" in user_params:
        tickers = user_params["tickers"]
    else:
        past = db.get_recent_runs(limit=20)
        if past:
            tickers = [past[0]["ticker"]]
        else:
            tickers = ["AAPL"]

    # ----------------------------
    # 2. START DATE
    # ----------------------------
    if "start" in user_params:
        start = user_params["start"]
    else:
        past = db.get_all_starts()
        if past:
            start = min(past)
        else:
            start = "2010-01-01"

    # ----------------------------
    # 3. RSI window
    # ----------------------------
    if "rsi_window" in user_params:
        rsi = user_params["rsi_window"]
    else:
        best = db.get_best_params(metric="sharpe", limit=20)
        if best and "rsi_window" in best[0]:
            base = best[0]["rsi_window"]
            rsi = max(3, min(40, int(base * (1 + random.uniform(-0.2, 0.2)))))
        else:
            rsi = random.randint(5, 30)

    return {
        "tickers": tickers,
        "start": start,
        "rsi_window": rsi
    }