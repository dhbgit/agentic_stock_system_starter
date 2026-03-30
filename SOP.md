# Standard Operating Procedure (SOP)
## AI Stock Prediction System (Emerald Magnetar)

### 1. What does this system do?
This system uses **Machine Learning (Random Forest)** to scan 20 years of historical price data (from 2004) for 50 global assets. It looks for specific patterns in **5 Key Indicators**:
*   **RSI**: Is it overbought/oversold?
*   **MACD**: Is the momentum shifting?
*   **Bollinger Bands**: Did the price snap out of its normal range?
*   **ATR**: Is the volatility high enough to trade?
*   **Volume**: Is the move supported by real money?

It outputs a **Probability Score** (Confidence %).
*   **> 55%**: Weak Signal (Be careful)
*   **> 60%**: Strong Signal (Historical patterns suggest a high win rate)

#### 2. The Daily Workflow (The Cycle)
The model has already learned from the past. You only need to feed it *new* daily data to get today's predictions.

**Step 1: Update Data & Predict (Run this Once a Day)**
Run this command after the markets close (e.g., 5 PM NY time or 8 AM Sydney time).
```bash
python scripts/pipeline.py
```
*   **What it does:** Downloads today's prices, calculates indicators, runs the AI model, and saves the new predictions to the database.
*   **Fast by default:** Uses cached models and 2020+ data. Use `--retrain` to force a full refresh.
*   **Optional tuning:** Add `--tune` for Optuna hyperparameter search (slower, ~1h).

**Step 2: View the Dashboard**
Run this command to open your command center:
```bash
python web/app.py
```
Open **http://localhost:5050** in your browser.

### 3. How to Extract Value ("One Kick")
Don't trade everything. Focus on the **Gold**.

1.  **Filter by Color**:
    *   **Green Column (Bullish)**: Only look at these if you want to BUY.
    *   **Red Column (Bearish)**: Only look at these if you want to SHORT (sell).

2.  **Filter by Confidence (The "Lean and Mean" Approach)**:
    *   **< 55%**: Weak Signal. The system hides details for these to save your attention.
    *   **55% - 60%**: Moderate. Worth watching, but not a slam dunk.
    *   **> 60%**: **Strong Signal**. The system automatically highlights these with full details (Name, Exchange, Target Price). This is where you focus.

3.  **Use the Targets**:
    *   **Target Sell Price**: If you buy, place a Limit Sell order at this price immediately. This automates your exit strategy based on standard volatility.

### 4. Frequently Asked Questions

**Q: Do I need to keep the server running?**
A: **No.** The server (`web/app.py`) is just for *viewing* the data. You can shut it down (Ctrl+C) when you are done looking. The data is safely stored in the database (`agent.db`).

**Q: When does it learn (Re-Train)?**
A: The model is designed to "Walk Forward" and learn from new data automatically. You do NOT need a special training command.
**Simply run the pipeline:**
```bash
python scripts/pipeline.py
```
This script acts as the "Gym". It downloads the latest data, *re-trains the models* to include the newest patterns, and then outputs predictions. Cache refreshes automatically every 21 days, or force it with `--retrain`.

### 5. Maintenance Commands (The "Pit Stop")

**To Update Stock Metadata & Timestamp:**
If you want to update the "Last Price" or see "Unknown" for a new stock:
```bash
python scripts/refresh_metadata.py
```

### 6. Glossary: What do the numbers mean?

*   **Confidence % (next to ticker)**: The model's conviction in its prediction.
    *   *50-55%*: Low confidence. Card shown with reduced opacity.
    *   *55-60%*: Moderate confidence. Worth watching.
    *   *60%+*: High confidence. Full card details shown (name, price, target).
*   **Model Accuracy**: The percentage of times the AI correctly predicted the stock's direction (Up or Down).
    *   *50%* = Coin Flip / Random.
    *   *55-60%* = Decent Edge (Casino House Edge is only 52%).
    *   *60%+* = Excellent.
*   **AUC Score (Area Under Curve)**: The "Trust Score" of the model.
    *   *0.50* = The model is guessing.
    *   *1.00* = The model is perfect (impossible).
    *   *Aim for > 0.55*.
*   **Active Predictions**: The number of stocks the system is currently tracking and scanning.

### 7. Strategy: The "Centaur" Approach (AI + Human)
**Q: Can the model see the CEO quitting?**
**A: NO.** The model is a **Technical Sniper**. It looks at Price, Volume, and Momentum ("The 1000 Kicks"). It does *not* read the news.

**Your Job (The Human Edge):**
1.  **The Model says:** "ANZ is oversold, BUY." (Based on math).
2.  **You know:** "ANZ just lost its CEO and is in regulatory trouble."
3.  **Decision:** **SKIP IT.**

**Rule of Thumb:**
Use the AI to find the *opportunity* (screen 50 stocks down to 3), but use your brain to check for *disasters*. If a stock is in the news for bad reasons, ignore the Green Light.
