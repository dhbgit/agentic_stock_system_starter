"""
Minimal SQLite wrapper for querying the agent database.
Logging writes go through backtest.log_to_sqlite() directly.
"""

import sqlite3
import os
import pandas as pd

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