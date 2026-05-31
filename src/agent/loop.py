import os
import sys
import time
import subprocess
import yaml
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "config.yaml"
DB_PATH = REPO_ROOT / "artifacts" / "agent.db"

class AgentLoop:
    def __init__(self, check_interval_sec=3600):
        self.check_interval = check_interval_sec
        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)

    def log_event(self, msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [AGENT] {msg}")

    def is_market_data_stale(self):
        """Self-healing check: Is the latest prediction date older than the latest market day?
        Prevents weekend runs and uses actual market calendar updates from Yahoo Finance.
        """
        # Skip checking/running on weekends
        today = datetime.now()
        if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        if not DB_PATH.exists():
            return True

        conn = sqlite3.connect(DB_PATH)
        try:
            res = conn.execute("SELECT MAX(date) FROM preds").fetchone()[0]
            if not res: 
                return True
            last_pred_date = datetime.fromisoformat(res.split()[0]).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Query Yahoo Finance for the latest session date of the benchmark (SPY)
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1d")
            if hist.empty:
                return False
            
            latest_market_date = hist.index[0].to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
            
            # Stale if the latest market date is newer than our last prediction date
            return last_pred_date < latest_market_date
        except Exception as e:
            self.log_event(f"Error checking staleness: {e}")
            return True
        finally:
            conn.close()

    def run_pipeline(self, force_retrain=False):
        self.log_event("Triggering Signal Pipeline...")
        cmd = [sys.executable, str(REPO_ROOT / "scripts/pipeline.py")]
        if force_retrain:
            cmd.append("--retrain")
        
        try:
            subprocess.run(cmd, check=True)
            self.log_event("Pipeline completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            self.log_event(f"❌ Pipeline failed: {e}")
            return False

    def trigger_reasoning(self):
        """Pass the baton to the Reasoner for daily rationale and LLM evaluation."""
        from src.agent.reasoner import run_analysis
        self.log_event("Generating Daily Digest & LLM Rationales...")
        digest = run_analysis()
        self.log_event("Daily Digest Ready.")
        print("-" * 30)
        print(digest)
        print("-" * 30)

    def trigger_monitoring(self):
        """Self-healing: check if any ticker needs a re-tune."""
        from src.agent.monitor import AgentMonitor
        self.log_event("Starting Performance Health Check...")
        monitor = AgentMonitor()
        monitor.health_check()
        self.log_event("Health Check Complete.")

    def start(self):
        self.log_event("Agentic Heartbeat Started.")
        while True:
            if self.is_market_data_stale():
                self.log_event("Data is STALE. Starting self-healing refresh...")
                success = self.run_pipeline()
                if success:
                    self.trigger_monitoring()  # Retune if needed
                    self.trigger_reasoning()   # Generate digest
            else:
                self.log_event("Data is FRESH. Monitoring for next session...")

            time.sleep(self.check_interval)

if __name__ == "__main__":
    agent = AgentLoop()
    agent.start()
