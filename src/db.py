import sqlite3
import os
import pandas as pd
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "agent.db")

class Database:
    def __init__(self, path=DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _conn(self):
        return sqlite3.connect(self.path)

    # Return last N runs ordered by performance
    def get_recent_runs(self, limit=20):
        conn = self._conn()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM metrics ORDER BY rowid DESC LIMIT {limit}",
                conn
            )
            return df.to_dict("records")
        except:
            return []
        finally:
            conn.close()

    # Get all start dates
    def get_all_starts(self):
        conn = self._conn()
        try:
            df = pd.read_sql_query("SELECT start FROM metrics", conn)
            return df["start"].dropna().tolist()
        except:
            return []
        finally:
            conn.close()

    # Get best params by metric
    def get_best_params(self, metric="sharpe", limit=50):
        conn = self._conn()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM metrics ORDER BY {metric} DESC LIMIT {limit}",
                conn
            )
            return df.to_dict("records")
        except:
            return []
        finally:
            conn.close()

    # Insert results manually if needed

    def insert_result(self, params, result):
        conn = self._conn()

        # Convert lists/dicts into JSON strings
        safe_params = {}
        for k, v in params.items():
            if isinstance(v, (list, dict)):
                safe_params[k] = json.dumps(v)
            else:
                safe_params[k] = v

        safe_result = {}
        for k, v in result.items():
            if isinstance(v, (list, dict)):
                safe_result[k] = json.dumps(v)
            else:
                safe_result[k] = v

        row = { **safe_params, **safe_result }

        df = pd.DataFrame([row])
        df.to_sql("learning_runs", conn, if_exists="append", index=False)
        conn.close()