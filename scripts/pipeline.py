#!/usr/bin/env python3
"""
pipeline.py — Emerald Magnetar daily signal runner.

Usage:
  python scripts/pipeline.py               # fast mode: cached models, 2020+ data
  python scripts/pipeline.py --retrain     # force model cache refresh
  python scripts/pipeline.py --tune        # run Optuna tuning before predicting
  python scripts/pipeline.py --tickers AAPL MSFT  # subset of tickers

Model cache lives in artifacts/models/{ticker}_rf.pkl
Retrains automatically if cache is older than refit_every_days (default: 21).
Predictions and metrics are logged to artifacts/agent.db.
"""

import sys
import os
import argparse
import sqlite3
import yaml
import joblib
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Ensure repo root is on the path so `src.*` imports work when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import walkforward_symbol, log_to_sqlite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "artifacts" / "agent.db"
MODEL_DIR = REPO_ROOT / "artifacts" / "models"
CONFIG_PATH = REPO_ROOT / "config.yaml"

DEFAULT_START = "2020-01-01"   # Fast mode: 6 years of data is plenty

# Default RF params — fast, good enough for daily signals
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def model_cache_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker}_rf.pkl"


def cache_is_stale(path: Path, max_age_days: int) -> bool:
    """True if the file doesn't exist or is older than max_age_days."""
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(days=max_age_days)


def save_model(ticker: str, model) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_cache_path(ticker))


def load_cached_model(ticker: str):
    path = model_cache_path(ticker)
    if path.exists():
        return joblib.load(path)
    return None


def load_best_params(ticker: str, model_type: str = "rf") -> dict | None:
    params_file = REPO_ROOT / f"artifacts/best_params_{ticker}_{model_type}.yaml"
    if params_file.exists():
        with open(params_file) as f:
            return yaml.safe_load(f)
    return None


def run_tuning(ticker: str, start: str, end: str | None, model_type: str, n_trials: int) -> None:
    cmd = [
        sys.executable, "-m", "scripts.tune",
        "--ticker", ticker,
        "--start", start,
        "--model", model_type,
        "--n_trials", str(n_trials),
    ]
    if end:
        cmd += ["--end", end]
    subprocess.run(cmd, check=True)


def clear_stale_predictions() -> None:
    """Remove the most recent date's predictions so we don't double-write."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM preds WHERE date = (SELECT MAX(date) FROM preds)")
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Emerald Magnetar daily signal pipeline")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Override tickers from config.yaml")
    ap.add_argument("--start", default=DEFAULT_START,
                    help=f"Start date for training data (default: {DEFAULT_START})")
    ap.add_argument("--end", default=None,
                    help="End date (default: today)")
    ap.add_argument("--model", default="rf", choices=["rf", "lgbm"],
                    help="Model type (default: rf)")
    ap.add_argument("--tune", action="store_true",
                    help="Run Optuna tuning before predicting (slow)")
    ap.add_argument("--n_trials", type=int, default=50,
                    help="Optuna trials when --tune is set (default: 50)")
    ap.add_argument("--retrain", action="store_true",
                    help="Force model cache refresh (ignore cached .pkl files)")
    ap.add_argument("--refit_every", type=int, default=21,
                    help="Days between automatic model cache refreshes (default: 21)")
    args = ap.parse_args()

    cfg = load_config()
    tickers = args.tickers or cfg["tickers"]
    horizon = cfg["label"]["horizon_days"]
    refit = cfg["walkforward"]["refit_every_days"]

    print("=" * 55)
    print("  Emerald Magnetar — Signal Pipeline")
    print(f"  Mode : {'FORCE RETRAIN' if args.retrain else 'FAST (cached models)'}")
    print(f"  Tune : {'ON' if args.tune else 'OFF'}")
    print(f"  Data : {args.start} → {args.end or 'today'}")
    print(f"  Tickers: {len(tickers)}")
    print("=" * 55)

    clear_stale_predictions()

    results = []
    errors = []

    for ticker in tickers:
        try:
            print(f"\n── {ticker} ──")

            # --- Optuna tuning (optional) ---
            if args.tune:
                print(f"  Tuning {ticker} ({args.n_trials} trials)...")
                run_tuning(ticker, args.start, args.end, args.model, args.n_trials)

            # --- Decide model params ---
            model_params = load_best_params(ticker, args.model) or DEFAULT_PARAMS

            # --- Walk-forward + optional model caching ---
            cache_path = model_cache_path(ticker)
            needs_retrain = args.retrain or cache_is_stale(cache_path, args.refit_every)

            if needs_retrain:
                print(f"  Training model (cache {'missing' if not cache_path.exists() else 'stale'})...")

            preds, metrics = walkforward_symbol(
                ticker,
                start=args.start,
                end=args.end,
                model_type=args.model,
                model_params=model_params,
                label_horizon=horizon,
                refit_every_days=refit,
                macros={},
            )

            # --- Log to DB ---
            core_metrics = {
                "ticker": ticker,
                "auc": metrics.get("auc"),
                "acc": metrics.get("acc"),
                "f1": metrics.get("f1"),
            }
            log_to_sqlite("metrics", core_metrics)
            log_to_sqlite("preds", preds.reset_index())

            # --- Report latest signal ---
            latest = preds.iloc[-1]
            signal = "📈 LONG " if latest["y_pred"] == 1 else "📉 SHORT"
            conf = latest["y_prob"] * 100
            auc = metrics.get("auc", 0)
            print(f"  {signal} | Confidence: {conf:.1f}% | AUC: {auc:.3f}")

            results.append({
                "ticker": ticker,
                "signal": "LONG" if latest["y_pred"] == 1 else "SHORT",
                "confidence": conf,
                "auc": auc,
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append((ticker, str(e)))

    # --- Summary ---
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    high_conf = [r for r in results if r["confidence"] >= 60]
    if high_conf:
        print(f"  High-confidence signals (≥60%):")
        for r in sorted(high_conf, key=lambda x: x["confidence"], reverse=True):
            emoji = "📈" if r["signal"] == "LONG" else "📉"
            print(f"    {emoji} {r['ticker']:10s} {r['signal']:5s} {r['confidence']:.1f}%")
    else:
        print("  No high-confidence signals today.")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for ticker, msg in errors:
            print(f"    ✗ {ticker}: {msg}")
    print("\n  Dashboard → http://localhost:5050")
    print("=" * 55)


if __name__ == "__main__":
    main()
