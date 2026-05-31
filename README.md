# Emerald Magnetar — Agentic Stock Prediction System

A professional, daily signal pipeline that uses **machine learning** to predict next-day price direction for a curated watchlist of US, ETF, and ASX stocks. Designed with a clean, light-mode "Financial Editorial" aesthetic—your poor man's Bloomberg terminal.

---

## Quickstart

```bash
python -m venv .venv-3.11 && source .venv-3.11/bin/activate
pip install -r requirements.txt

# 1. Run the daily signal pipeline (after market close)
python scripts/pipeline.py

# 2. Start the autonomous background Agent
python src/agent/loop.py

# 3. Start the tactical dashboard (if not running)
python web/app.py
# → http://localhost:5050
```

---

## How to Read the Tactical Dashboard

The dashboard is structured as a **High-Confidence Scanner**, focusing on data density and structural conviction rather than flashy charts.

### 1. Summary Line
Located immediately below the header. It shows the total breadth of the market's high-confidence signals (≥60%) for the next trading day.
- **X Long · Y Short** — simple net direction for the watchlist.

---

### 2. The Tactical Signal Table

Each row represents one ticker where the model has reached the **60% confidence threshold**.

| Column | What it means |
|---|---|
| **Ticker & Company** | Stock code (e.g. MSFT), company name, and the **Agent's Technical Rationale** beneath it. |
| **Signal** | The model's call for **tomorrow**. ↑ Long = expects rise. ↓ Short = expects fall. |
| **Conf** | **Platt-Calibrated Probability** (Ensemble of RF + LightGBM). Represents true historical win rate, not raw tree scores. |
| **Size (Half-Kelly)** | Mathematical percentage of optimal portfolio allocation based on the calibrated `Conf` (risk management). |
| **Streak** | 🔥 8 days = the model has held the **same directional call for 8 consecutive sessions**. |
| **Trend** | Today's confidence relative to the **10-day institutional baseline**. ↑ Rising = building conviction. |
| **5d / 10d Avg** | Rolling baselines. Use these to see if today's confidence is an outlier or a structural shift. |

---

### 3. Agent Simulator (P&L Replay)
The dashboard continuously runs a **Theoretical P&L Backtest** over the Agent's recent out-of-sample predictions. It uses the exact Half-Kelly sizing logic recommended on the dashboard to calculate exactly what an initial $10k portfolio would have grown to if it followed the Agent blindly.

---

### 4. Signal Intelligence Logic
Three intelligence metrics are computed on-the-fly from historical `preds`—no extra ML overhead.

| Logic | Professional Use Case |
|---|---|
| **Streak** | Filter out "one-hit wonders." A 1-day signal is a guess; a 10-day streak is a regime. |
| **10-Day Baseline** | The "Institutional Anchor." A 10-day average filters out daily volatility to show the real trend. |
| **Trend Delta** | If confidence is 62% but the 10-day average is 65%, the signal is **fading** (↓) even if still "long." |

#### Sample Institutional Reading

```text
MSFT | ↑ Long | 60.7% | 🔥 8d | ↓ -2.9% | ∅ 63.5%
```
**Interpretation:** The model is still bullish (8-day streak), but its conviction is **fading** compared to its two-week average. A seasoned trader might wait for the trend to turn green (↑) before sizing up.

---

## Pipeline & Architecture

```
scripts/pipeline.py
  ├─ src/data.py          ← yfinance download + VIX features
  ├─ src/features.py      ← 25 technical indicators
  ├─ src/labeler.py       ← next-day return > 0 → label 1
  ├─ src/model.py         ← Calibrated Ensemble (RandomForest + LightGBM)
  ├─ src/backtest.py      ← walk-forward engine + SQLite logger
  └─ artifacts/agent.db   ← SQLite: preds, metrics tables

src/agent/ (The Agentic Layer)
  ├─ loop.py              ← The autonomous heartbeat (Self-healing data)
  ├─ reasoner.py          ← The Analyst (Generates tactical rationales)
  ├─ monitor.py           ← The Evaluator (Auto-triggers hyperparameter tuning)
  └─ portfolio.py         ← The Risk Desk (Computes Kelly Criterion sizes)

web/app.py
  ├─ /api/predictions         ← latest preds & rationales from agent.db
  ├─ /api/simulation          ← true out-of-sample historical P&L replay 
  └─ /api/signal-intelligence ← 10-day streak + trend tracking
```

### Key config (`config.yaml`)

| Setting | Default | Effect |
|---|---|---|
| `label.horizon_days` | 1 | Predict 1-day forward return |
| `walkforward.refit_every_days` | 21 | How often cached models re-train |
| `tickers` | 23 stocks | Add/remove tickers here |

---

## Model Evaluation Metrics

| Metric | Target | What it means |
|---|---|---|
| **AUC** | > 0.55 | How well the model separates Up vs Down days. |
| **Accuracy** | > 52% | % of predictions that were correct. |
| **F1** | > 0.50 | Balance between precision and recall. |

---

## Disclaimers

Educational project. Not financial advice. Markets are risky. Past model performance does not predict future returns. Use paper trading only.
