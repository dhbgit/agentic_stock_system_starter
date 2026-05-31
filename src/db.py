"""
Minimal SQLite wrapper for querying the agent database.
Logging writes go through backtest.log_to_sqlite() directly.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "agent.db")


class Database:
    def __init__(self, path=DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _conn(self):
        return sqlite3.connect(self.path)

    def get_latest_predictions(self, limit=100):
        """Return the most recent predictions (from the latest date in preds table)."""
        conn = self._conn()
        try:
            df = pd.read_sql_query(
                """
                SELECT * FROM preds
                WHERE date = (SELECT MAX(date) FROM preds)
                ORDER BY y_prob DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )
            return df.to_dict("records")
        except Exception:
            return []
        finally:
            conn.close()

    def get_metrics_summary(self, limit=50):
        """Return latest per-ticker metrics."""
        conn = self._conn()
        try:
            df = pd.read_sql_query(
                """
                SELECT ticker, auc, acc, f1
                FROM metrics
                ORDER BY rowid DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )
            return df.to_dict("records")
        except Exception:
            return []
        finally:
            conn.close()

    def get_signal_intelligence(self):
        """
        Computes signal intelligence for all tickers from the preds table:
          - streak       : consecutive days with the same y_pred (+ = bullish, - = bearish)
          - conf_trend   : confidence delta vs 10-day average ('rising', 'falling', 'flat')
          - conf_5d_avg  : average y_prob over the last 5 sessions (weekly proxy)
          - conf_10d_avg : average y_prob over the last 10 sessions (institutional baseline)
        Returns a dict keyed by ticker.
        """
        conn = self._conn()
        try:
            # Pull enough history for 10rd average + streak
            df = pd.read_sql_query(
                """
                SELECT ticker, date, y_pred, y_prob
                FROM preds
                ORDER BY ticker, date DESC
                """,
                conn,
                parse_dates=["date"],
            )
        except Exception:
            return {}
        finally:
            conn.close()

        if df.empty:
            return {}

        result = {}
        for ticker, grp in df.groupby("ticker"):
            grp = grp.sort_values("date", ascending=False).reset_index(drop=True)

            # --- Streak ---
            latest_pred = grp["y_pred"].iloc[0]
            streak = 1
            for i in range(1, len(grp)):
                if grp["y_pred"].iloc[i] == latest_pred:
                    streak += 1
                else:
                    break
            streak_signed = streak if latest_pred == 1 else -streak

            # --- Confidence logic (today vs 5-day and 10-day avgs) ---
            today_conf = float(grp["y_prob"].iloc[0])
            
            recent_5 = grp["y_prob"].iloc[1:6]  # prior 5 sessions
            conf_5d_avg = float(recent_5.mean()) if len(recent_5) >= 2 else today_conf
            
            recent_10 = grp["y_prob"].iloc[1:11] # prior 10 sessions
            conf_10d_avg = float(recent_10.mean()) if len(recent_10) >= 2 else today_conf

            # Trend is now anchored to the more stable 10-day baseline
            delta_10 = today_conf - conf_10d_avg
            if abs(delta_10) < 0.015:
                conf_trend = "flat"
            elif delta_10 > 0:
                conf_trend = "rising"
            else:
                conf_trend = "falling"

            result[ticker] = {
                "streak": streak_signed,
                "conf_trend": conf_trend,
                "conf_5d_avg": round(conf_5d_avg * 100, 1),
                "conf_10d_avg": round(conf_10d_avg * 100, 1),
                "today_conf": round(today_conf * 100, 1),
                "conf_delta": round(delta_10 * 100, 1),  # This is vs 10d
            }
        return result

    def get_vix_regime(self):
        """
        Derives the current VIX market regime from the most recent VIX close
        stored in the preds table (the pipeline logs it via features).
        Falls back to fetching from yfinance if not in DB.
        Returns: dict with keys 'level', 'regime', 'label'
        """
        try:
            import yfinance as yf
            vix = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
            if vix.empty:
                return {"level": None, "regime": "unknown", "label": "?"}
            level = float(vix["Close"].dropna().iloc[-1])
            if level < 15:
                regime, label, emoji = "calm", "Calm", "🟢"
            elif level < 25:
                regime, label, emoji = "uncertain", "Uncertain", "🟡"
            else:
                regime, label, emoji = "fearful", "Fearful", "🔴"
            return {"level": round(level, 2), "regime": regime, "label": label, "emoji": emoji}
        except Exception:
            return {"level": None, "regime": "unknown", "label": "?", "emoji": "⚪"}