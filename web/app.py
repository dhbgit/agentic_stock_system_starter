"""
Stock Prediction Dashboard - Flask Server
Reads from existing agent.db created by agentic_stock_system_starter
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

@app.route('/api/metadata')
def get_metadata():
    """Serve the pre-calculated metadata with robust path finding."""
    try:
        # Candidate paths
        candidates = [
            Path(__file__).parent / 'metadata.json',  # Standard: Relative to app.py
            Path.cwd() / 'web' / 'metadata.json',     # Fallback: Relative to CWD
            Path.cwd() / 'metadata.json',             # Fallback: In CWD
            Path('/Users/dhb/repos/agentic_stock_system_starter/web/metadata.json') # Hardcoded Absolute
        ]
        
        target_path = None
        for p in candidates:
            if p.exists():
                target_path = p
                break
        
        if target_path:
            with open(target_path) as f:
                return jsonify(json.load(f))
        
        # DEBUG: Return error if not found
        searched = [str(p.absolute()) for p in candidates]
        print(f"METADATA NOT FOUND. Searched: {searched}")
        return jsonify({"error": "Metadata file not found", "searched": searched}), 404
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return jsonify({"error": str(e)}), 500

# DB is always one directory up from web/ in the artifacts folder
DB_PATH = Path(__file__).parent.parent / "artifacts" / "agent.db"


def get_db():
    """Get database connection"""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


def dict_factory(cursor, row):
    """Convert SQLite rows to dictionaries"""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predictions')
def get_predictions():
    """Get latest predictions for all stocks"""
    try:
        conn = get_db()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        # Get latest predictions from 'preds' table
        # Columns: index, date, ticker, y_true, y_prob, y_pred
        # Show predictions from the most recent date available
        cursor.execute('''
            SELECT ticker, y_pred as prediction, y_prob as confidence, date as created_at
            FROM preds
            WHERE date = (SELECT MAX(date) FROM preds)
            ORDER BY confidence DESC
        ''')
        predictions = cursor.fetchall()
        conn.close()
        
        return jsonify({'predictions': predictions, 'status': 'ok'})
    except Exception as e:
        return jsonify({'predictions': [], 'status': 'error', 'message': str(e)})


@app.route('/api/metrics')
def get_metrics():
    """Get model performance metrics"""
    try:
        conn = get_db()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        # Get metrics from your metrics table (Note: DB uses 'acc', not 'accuracy')
        cursor.execute('''
            SELECT ticker, auc, acc as accuracy, f1
            FROM metrics
            ORDER BY rowid DESC
            LIMIT 100
        ''')
        metrics = cursor.fetchall()
        conn.close()
        
        # Calculate averages
        if metrics:
            avg_accuracy = sum(m['accuracy'] for m in metrics if m['accuracy']) / len(metrics)
            avg_auc = sum(m['auc'] for m in metrics if m['auc']) / len(metrics)
        else:
            avg_accuracy = 0
            avg_auc = 0
        
        return jsonify({
            'metrics': metrics,
            'summary': {
                'avg_accuracy': round(avg_accuracy, 4),
                'avg_auc': round(avg_auc, 4),
                'total_records': len(metrics)
            },
            'status': 'ok'
        })
    except Exception as e:
        return jsonify({'metrics': [], 'summary': {}, 'status': 'error', 'message': str(e)})


@app.route('/api/trades')
def get_trades():
    """Get paper trading history"""
    try:
        conn = get_db()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        # Check if table exists first
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if not cursor.fetchone():
            conn.close()
            return jsonify({'trades': [], 'total_pnl': 0, 'status': 'ok', 'note': 'No paper trading data yet'})

        cursor.execute('''
            SELECT ticker, action, price, shares, pnl, date as created_at
            FROM trades
            ORDER BY date DESC
            LIMIT 50
        ''')
        trades = cursor.fetchall()
        conn.close()
        
        # Calculate total P&L
        total_pnl = sum(t['pnl'] for t in trades if t.get('pnl'))
        
        return jsonify({
            'trades': trades,
            'total_pnl': round(total_pnl, 2),
            'status': 'ok'
        })
    except Exception as e:
        return jsonify({'trades': [], 'total_pnl': 0, 'status': 'error', 'message': str(e)})


@app.route('/api/stocks')
def get_stocks():
    """Get list of tracked stocks with latest metrics"""
    try:
        conn = get_db()
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Use rowid (always present) to get the most recent row per ticker
        cursor.execute('''
            SELECT DISTINCT ticker,
                   (SELECT acc FROM metrics m2
                    WHERE m2.ticker = metrics.ticker
                    ORDER BY rowid DESC LIMIT 1) as latest_accuracy
            FROM metrics
            ORDER BY ticker
        ''')
        stocks = cursor.fetchall()
        conn.close()

        return jsonify({'stocks': stocks, 'status': 'ok'})
    except Exception as e:
        return jsonify({'stocks': [], 'status': 'error', 'message': str(e)})


@app.route('/api/signal/<string:ticker>')
def get_signal_details(ticker):
    """Get all data for a specific ticker."""
    try:
        conn = get_db()
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Predictions
        cursor.execute('''
            SELECT ticker, y_pred as prediction, y_prob as confidence, date as created_at
            FROM preds
            WHERE ticker = ?
            ORDER BY date DESC
        ''', (ticker,))
        predictions = cursor.fetchall()

        # Metrics
        cursor.execute('''
            SELECT ticker, auc, acc as accuracy, f1
            FROM metrics
            WHERE ticker = ?
            ORDER BY rowid DESC
            LIMIT 1
        ''', (ticker,))
        metrics = cursor.fetchone()

        # Trades
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        trades = []
        if cursor.fetchone():
            cursor.execute('''
                SELECT ticker, action, price, shares, pnl, date as created_at
                FROM trades
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 50
            ''', (ticker,))
            trades = cursor.fetchall()
        
        conn.close()

        return jsonify({
            'ticker': ticker,
            'predictions': predictions,
            'metrics': metrics,
            'trades': trades,
            'status': 'ok'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/health')
def health():
    """Health check endpoint"""
    try:
        conn = get_db()
        conn.execute('SELECT 1')
        conn.close()
        return jsonify({'status': 'healthy', 'database': 'connected'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print("=" * 50)
    print("Stock Prediction Dashboard")
    print("=" * 50)
    print(f"Database path: {DB_PATH}")
    print(f"Starting server at http://localhost:{port}")
    print("=" * 50)
    app.run(debug=True, port=port)
