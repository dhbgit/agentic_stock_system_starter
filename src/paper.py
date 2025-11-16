import pandas as pd

def long_flat_strategy(preds: pd.DataFrame, entry_prob=0.55, exit_prob=0.50):
    """Naive paper trading on prediction stream (long/flat)."""
    preds = preds.sort_index().copy()
    position = 0
    equity = 1.0
    returns = []

    for dt, row in preds.iterrows():
        p = row['y_prob']
        ret_next = 1 if row['y_true']==1 else -1  # proxy; in real use, map to actual forward return
        if position == 0 and p >= entry_prob:
            position = 1
        elif position == 1 and p < exit_prob:
            position = 0
        equity *= (1 + (0.005 * position * ret_next))  # toy PnL
        returns.append((dt, equity, position))

    return pd.DataFrame(returns, columns=["date","equity","position"]).set_index("date")
