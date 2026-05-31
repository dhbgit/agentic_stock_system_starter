import sqlite3
import subprocess
import sys
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DB_PATH = REPO_ROOT / "artifacts" / "agent.db"

class AgentMonitor:
    def __init__(self, auc_threshold=0.52):
        self.auc_threshold = auc_threshold

    def log_event(self, msg):
        print(f"[MONITOR] {msg}")

    def get_latest_metrics(self):
        """Fetches the most recent metrics for all tickers."""
        if not DB_PATH.exists(): return pd.DataFrame()
        conn = sqlite3.connect(DB_PATH)
        try:
            # Latest auc per ticker
            query = """
                SELECT m1.ticker, m1.auc, m1.acc, m1.f1
                FROM metrics m1
                INNER JOIN (
                    SELECT ticker, MAX(rowid) as max_id
                    FROM metrics
                    GROUP BY ticker
                ) m2 ON m1.rowid = m2.max_id
            """
            return pd.read_sql_query(query, conn)
        except Exception:
            return pd.DataFrame()
        finally:
            conn.close()

    def trigger_retune(self, ticker):
        """Triggers scripts/tune.py for a specific ticker."""
        self.log_event(f"🚨 ALERT: {ticker} AUC dropped below {self.auc_threshold}. Triggering Auto-Tune...")
        cmd = [sys.executable, "-m", "scripts.tune", "--ticker", ticker, "--n_trials", "30"]
        try:
            subprocess.run(cmd, check=True)
            self.log_event(f"✅ Auto-tune for {ticker} completed.")
            return True
        except Exception as e:
            self.log_event(f"❌ Auto-tune failed: {e}")
            return False

    def health_check(self):
        """Main monitoring loop."""
        metrics = self.get_latest_metrics()
        if metrics.empty:
            self.log_event("No metrics found in DB. Skipping health check.")
            return

        for _, row in metrics.iterrows():
            ticker = row['ticker']
            auc = row['auc'] if row['auc'] is not None else 1.0
            
            if auc < self.auc_threshold:
                self.trigger_retune(ticker)
            else:
                pass # Ticker is healthy

if __name__ == "__main__":
    monitor = AgentMonitor()
    monitor.health_check()
