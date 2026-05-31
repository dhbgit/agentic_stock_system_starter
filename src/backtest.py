"""
Walk-forward backtesting engine + daily prediction helpers.

Primary entry points:
  - walkforward_symbol()   : Full historical walk-forward (used by tune.py for evaluation)
  - train_and_save_model() : Train RF on all data and cache to disk (used by pipeline)
  - predict_latest()       : Load cached model + predict the most recent day (used by pipeline)
"""

import os
import json
import sqlite3
import joblib
import pandas as pd
import yaml
from datetime import datetime, timedelta

from .data import load_prices, merge_macro, load_macro_csv
from .features import add_tech_features
from .labeler import make_labels
from .model import train_ensemble, metrics_classifier, portfolio_metrics

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "agent.db")

DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
}
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
}


# ---------------------------------------------------------------------------
# Walk-forward evaluation (backtesting / tuning)
# ---------------------------------------------------------------------------

def walkforward_symbol(
    ticker, start, end,
    label_horizon=1,
    model_params=None,
    refit_every_days=21,
    macros=None,
    rsi_window=14,
    light_features=False,
    transaction_costs=0.0,
):
    """
    Walk-forward evaluation for a single ticker using the Ensemble.
    """
    px = load_prices(ticker, start, end)
    if macros:
        px = merge_macro(px, macros)
    feat = add_tech_features(px, rsi_window=rsi_window, light=light_features)
    y, fwd_ret = make_labels(feat, horizon_days=label_horizon)
    X = feat.loc[y.index].copy()

    dates = X.index.unique()
    if len(dates) < 300:
        raise ValueError(f"Not enough data for {ticker} after feature engineering ({len(dates)} rows).")

    model = None
    preds = []
    last_refit = None

    for dt in dates[200:-label_horizon]:
        idx_train = X.index <= dt
        idx_test = X.index == dt

        # Refit periodically
        needs_refit = (last_refit is None) or (
            (dt - last_refit) >= pd.Timedelta(days=refit_every_days)
        )
        if needs_refit:
            rfp = model_params.get("rf", DEFAULT_RF_PARAMS) if model_params else DEFAULT_RF_PARAMS
            lp  = model_params.get("lgbm", DEFAULT_LGBM_PARAMS) if model_params else DEFAULT_LGBM_PARAMS
            model = train_ensemble(X.loc[idx_train], y.loc[idx_train], rfp, lp)
            last_refit = dt

        p = model.predict_proba(X.loc[idx_test])[:, 1][0]
        preds.append({"date": dt, "ticker": ticker,
                       "y_true": int(y.loc[idx_test].values[0]),
                       "y_prob": float(p)})

    pred_df = pd.DataFrame(preds).set_index("date")
    pred_df["y_pred"] = (pred_df["y_prob"] >= 0.55).astype(int)

    m = metrics_classifier(pred_df["y_true"], pred_df["y_prob"], pred_df["y_pred"])
    strat_returns = pred_df["y_prob"] * fwd_ret.loc[pred_df.index] - transaction_costs
    m.update(portfolio_metrics(strat_returns))

    return pred_df, m


# ---------------------------------------------------------------------------
# Daily pipeline helpers: train, cache, and predict
# ---------------------------------------------------------------------------

def _cache_paths(ticker, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    # Ensemble cache name
    model_path = os.path.join(cache_dir, f"{ticker}_ensemble.pkl")
    meta_path = os.path.join(cache_dir, f"{ticker}_ensemble_meta.json")
    return model_path, meta_path


def _model_is_fresh(ticker, cache_dir, refit_every_days=21):
    """Return True if a cached model exists and is younger than refit_every_days."""
    model_path, meta_path = _cache_paths(ticker, cache_dir)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    try:
        trained_on = datetime.fromisoformat(meta["trained_on"])
        return (datetime.now() - trained_on).days < refit_every_days
    except Exception:
        return False


def train_and_save_model(ticker, start="2020-01-01", end=None,
                          model_params=None, label_horizon=1,
                          cache_dir="artifacts/models"):
    """
    Train an Ensemble model on all available data and save it to disk.
    """
    px = load_prices(ticker, start, end)
    feat = add_tech_features(px)
    y, _ = make_labels(feat, horizon_days=label_horizon)
    X = feat.loc[y.index].copy()

    # Train on everything except the last `label_horizon` rows (no lookahead)
    X_train = X.iloc[:-label_horizon]
    y_train = y.iloc[:-label_horizon]

    rfp = model_params.get("rf", DEFAULT_RF_PARAMS) if model_params else DEFAULT_RF_PARAMS
    lp  = model_params.get("lgbm", DEFAULT_LGBM_PARAMS) if model_params else DEFAULT_LGBM_PARAMS
    
    model = train_ensemble(X_train, y_train, rfp, lp)

    # Compute recent metrics (last 60 trading days held out)
    window = min(60, len(X_train) // 2)
    X_val = X_train.iloc[-window:]
    y_val = y_train.iloc[-window:]
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= 0.55).astype(int)
    m = metrics_classifier(y_val, y_prob_val, y_pred_val)

    # Cache model + metadata
    model_path, meta_path = _cache_paths(ticker, cache_dir)
    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "trained_on": datetime.now().isoformat(),
            "ticker": ticker,
            "start": start,
            "n_train": len(X_train),
            "auc": m.get("auc"),
            "acc": m.get("acc"),
            "f1": m.get("f1"),
        }, f, indent=2)

    return model, X, y, m


def predict_latest(ticker, start="2020-01-01", end=None,
                   model_params=None, label_horizon=1,
                   cache_dir="artifacts/models", refit_every_days=21,
                   force_retrain=False):
    """
    Main daily prediction function using Ensemble + Calibration.
    """
    retrained = False
    metrics = {"auc": None, "acc": None, "f1": None}

    if force_retrain or not _model_is_fresh(ticker, cache_dir, refit_every_days):
        # Full retrain
        model, X, y, metrics = train_and_save_model(
            ticker, start=start, end=end,
            model_params=model_params, label_horizon=label_horizon,
            cache_dir=cache_dir,
        )
        retrained = True
    else:
        # Load cached model
        model_path, _ = _cache_paths(ticker, cache_dir)
        model = joblib.load(model_path)
        
        # Download recent data for the last feature row
        recent_start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        px = load_prices(ticker, recent_start, end)
        feat = add_tech_features(px)
        y, _ = make_labels(feat, horizon_days=label_horizon)
        # BUG FIX: don't restrict X to y.index here, because y.dropna() mathematically deletes 
        # the absolute latest day (since tomorrow hasn't happened yet, so y is NaN).
        X = feat.copy()

    # Predict the absolute latest available row from Yahoo Finance
    latest_X = X.iloc[[-1]]
    latest_date = X.index[-1]
    y_prob = float(model.predict_proba(latest_X)[:, 1][0])
    y_pred = int(y_prob >= 0.55)
    
    # If the latest date has a label true, fetch it. Otherwise, it's unknown (-1)
    if latest_date in y.index:
        y_true = int(y.loc[latest_date])
    else:
        y_true = -1

    pred = {
        "date": latest_date,
        "ticker": ticker,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
    return pred, metrics, retrained


# ---------------------------------------------------------------------------
# SQLite logging
# ---------------------------------------------------------------------------

def log_to_sqlite(table, df_or_dict):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if isinstance(df_or_dict, dict):
        pd.DataFrame([df_or_dict]).to_sql(table, conn, if_exists="append", index=False)
    else:
        df_or_dict.to_sql(table, conn, if_exists="append")
    conn.close()


# ---------------------------------------------------------------------------
# Batch backtest (used by evaluation scripts)
# ---------------------------------------------------------------------------

def run_backtest(tickers, start, end=None, rsi_window=14, **kwargs):
    all_metrics = {}
    for t in tickers:
        pred_df, metrics = walkforward_symbol(
            t, start, end,
            label_horizon=kwargs.get("horizon", 1),
            refit_every_days=kwargs.get("refit_days", 21),
            macros={},
            rsi_window=rsi_window,
        )
        metrics["ticker"] = t
        log_to_sqlite("metrics", metrics)
        log_to_sqlite("preds", pred_df.reset_index())
        all_metrics[t] = metrics
    return all_metrics
