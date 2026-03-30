try:
    import yaml
except ImportError:
    yaml = None

import json
import yfinance as yf
from pathlib import Path
import sqlite3
import os
import sys
from datetime import datetime

# Paths - Relative to repo root
REPO_DIR = Path(".")
CONFIG_PATH = REPO_DIR / "config.yaml"
# Try standard DB paths
DB_PATHS = [
    REPO_DIR / "artifacts/agent.db",
    Path(os.environ.get('STOCK_DB_PATH', 'artifacts/agent.db'))
]
OUTPUT_PATH = REPO_DIR / "web" / "metadata.json"

EXCHANGE_MAP = {
    'NMS': 'NASDAQ Global Select',
    'NGM': 'NASDAQ Global Market',
    'NCM': 'NASDAQ Capital Market',
    'NYQ': 'New York Stock Exchange (NYSE)',
    'PCX': 'NYSE Arca (ETF Exchange)',
    'PNK': 'OTC Markets (Pink Sheets)',
    'L': 'London Stock Exchange',
    'AX': 'ASX',
    'ASX': 'Australian Securities Exchange (ASX)',
    'TOR': 'Toronto Stock Exchange',
    'HK': 'Hong Kong Stock Exchange',
    'T': 'Tokyo Stock Exchange'
}

def get_db_tickers():
    """Fetch all unique tickers that have predictions in the DB"""
    for db_path in DB_PATHS:
        if db_path.exists():
            try:
                print(f"Reading tickers from DB: {db_path}")
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT ticker FROM preds")
                tickers = {row[0] for row in cursor.fetchall()}
                conn.close()
                return tickers
            except Exception as e:
                print(f"Warning: Could not read DB at {db_path}: {e}")
    return set()

def get_config_tickers():
    """Fetch tickers from config.yaml"""
    if yaml and CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)
            return set(config.get('stocks', []))
        except Exception as e:
            print(f"Warning: Could not read config at {CONFIG_PATH}: {e}")
    else:
         print("Warning: PyYAML not installed or config missing. Skipping config.yaml sources.")
    return set()

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        name = info.get('longName') or info.get('shortName') or ticker
        
        # Exchange Logic
        raw_exchange = info.get('exchange', 'Unknown')
        exchange = EXCHANGE_MAP.get(raw_exchange, raw_exchange)
        
        # Manual overrides
        if ticker.endswith('.AX'): exchange = 'Australian Securities Exchange (ASX)'
        
        # Heuristic Target Price (Current +/- 5%)
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0.0
        
        print(f"  [OK] {ticker}: {name} ({exchange}) ${current_price}")
        
        return {
            'name': name,
            'exchange': exchange,
            'currency': info.get('currency', '$'),
            'price': float(current_price) if current_price else 0.0,
            'target_bull': round(float(current_price) * 1.05, 2) if current_price else 0.0,
            'target_bear': round(float(current_price) * 0.95, 2) if current_price else 0.0,
            'fetched_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"  [FAIL] {ticker}: {e}")
        return {
            'name': ticker, 
            'exchange': 'Unknown', 
            'currency': '$', 
            'price': 0.0, 
            'target_bull': 0.0, 
            'target_bear': 0.0,
            'fetched_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def main():
    print("--- Starting Metadata Refresh ---")
    
    # 1. Gather Tickers from ALL sources
    db_tickers = get_db_tickers()
    config_tickers = get_config_tickers()
    
    all_tickers = sorted(list(db_tickers | config_tickers))
    
    if not all_tickers:
        print("CRITICAL: No tickers found in DB or Config!")
        sys.exit(1)
        
    print(f"Found {len(all_tickers)} unique stocks to process.")
    
    # 2. Fetch Metadata
    metadata = {}
    for t in all_tickers:
        metadata[t] = get_stock_info(t)
        
    # 3. Save
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"--- SUCCESS: Saves {len(metadata)} stocks to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()
