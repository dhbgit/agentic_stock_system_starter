import argparse, sqlite3, os
import pandas as pd
from datetime import timedelta
from .data import load_prices, merge_macro, load_macro_csv
from .features import add_tech_features
from .labeler import make_labels
from .model import train_rf, metrics_classifier

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "agent.db")

def walkforward_symbol(ticker, start, end, label_horizon=5, model_params=None, refit_every_days=21, macros=None, rsi_window=14):
    px = load_prices(ticker, start, end)
    if macros:
        px = merge_macro(px, macros)
    feat = add_tech_features(px, rsi_window=rsi_window)
    y = make_labels(feat, horizon_days=label_horizon)
    X = feat.loc[y.index].copy()

    dates = X.index.unique()
    if len(dates) < 300:
        raise ValueError("Not enough data after feature engineering.")

    split0 = dates[200]  # first 200 points train warmup
    model = None
    preds = []
    last_refit = None

    for dt in dates[200:-label_horizon]:
        idx_train = X.index <= dt
        idx_test  = X.index == dt

        # refit periodically
        if (last_refit is None) or ((dt - (last_refit or dt)) >= pd.Timedelta(days=refit_every_days)):
            model = train_rf(X.loc[idx_train], y.loc[idx_train], model_params or {"n_estimators":300,"max_depth":6,"min_samples_leaf":5})
            last_refit = dt

        p = model.predict_proba(X.loc[idx_test])[:,1][0]
        preds.append({"date": dt, "ticker": ticker, "y_true": int(y.loc[idx_test].values[0]), "y_prob": float(p)})

    pred_df = pd.DataFrame(preds).set_index("date")
    pred_df["y_pred"] = (pred_df["y_prob"] >= 0.55).astype(int)

    m = metrics_classifier(pred_df["y_true"], pred_df["y_prob"], pred_df["y_pred"])
    return pred_df, m

def run_backtest(tickers, start, end=None, rsi_window=14, **kwargs):
    all_metrics = {}
    
    for t in tickers:
        pred_df, metrics = walkforward_symbol(
            t, start, end,
            label_horizon=kwargs.get("horizon", 5),
            refit_every_days=kwargs.get("refit_days", 21),
            macros={},
            rsi_window=rsi_window
        )

        # Merge ticker into metrics
        metrics["ticker"] = t

        # Log backtest result if you want
        log_to_sqlite("metrics", metrics)
        log_to_sqlite("preds", pred_df.reset_index())

        all_metrics[t] = metrics

    return all_metrics

def log_to_sqlite(table, df_or_dict):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if isinstance(df_or_dict, dict):
        pd.DataFrame([df_or_dict]).to_sql(table, conn, if_exists="append", index=False)
    else:
        df_or_dict.to_sql(table, conn, if_exists="append")
    conn.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--refit_days", type=int, default=21)
    ap.add_argument("--rsi_window", type=int, default=14)
    args = ap.parse_args()

    all_preds = []
    for t in args.tickers:
        pred_df, metrics = walkforward_symbol(
            t, args.start, args.end,
            label_horizon=args.horizon,
            refit_every_days=args.refit_days,
            macros={},  # add macro dfs here if available
            rsi_window=args.rsi_window,
        )
        metrics.update({"ticker": t})
        log_to_sqlite("metrics", metrics)
        log_to_sqlite("preds", pred_df.reset_index())
        all_preds.append(pred_df.assign(ticker=t))

    print("Done. Metrics logged to artifacts/agent.db")

if __name__ == "__main__":
    main()
