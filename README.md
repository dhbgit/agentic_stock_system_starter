# Agentic Stock System (Starter)

This is a **starter project** that:
- downloads daily OHLCV with `yfinance`
- computes technical indicators with `ta`
- (optionally) adds macro regime features (CSV placeholders in `data/macro/`)
- trains a simple model (RandomForest) with **walk‑forward** evaluation
- logs predictions & metrics to **SQLite** for a self‑learning loop
- runs a **paper-trading** long/flat strategy based on predicted probability

> Goal: give you a working scaffold you can iterate on in a month‑long feedback loop.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) run a quick experiment on a few tickers
python src/backtest.py --tickers AAPL MSFT NVDA --start 2018-01-01 --end 2025-10-31

# 2) daily pipeline (fetch latest, train/update, predict, log)
python src/pipeline.py --tickers AAPL MSFT NVDA
```

## What it does

- **Features**: RSI(14), MACD(12,26,9), SMA(10/20/50/200), Bollinger(20,2), ATR(14), daily returns, rolling volatility.
- **Labels**: 5‑day forward return > 0 → 1 else 0 (classification). You can switch to regression target in `labeler.py`.
- **Walk‑forward**: expanding window with monthly re‑fit (no look‑ahead).
- **Macro**: place CSVs into `data/macro/` (e.g., `vix.csv`, `yc_10y_2y.csv`, `cpi.csv`) with columns [date,value].
- **SQLite**: `artifacts/agent.db` stores metrics, predictions, trades for later analysis.

## Disclaimers

This is **educational**. Markets are risky. Past performance ≠ future results.
No financial advice. Use paper trading only.
