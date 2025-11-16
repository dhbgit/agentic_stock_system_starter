import argparse, yaml, os, sqlite3, pandas as pd
from .backtest import walkforward_symbol, log_to_sqlite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    args = ap.parse_args()

    if args.config and os.path.exists(args.config):
        cfg = yaml.safe_load(open(args.config))
    else:
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))

    tickers = args.tickers or cfg["tickers"]
    start = cfg.get("start","2018-01-01")
    end = cfg.get("end",None)
    horizon = cfg["label"]["horizon_days"]
    refit = cfg["walkforward"]["refit_every_days"]

    for t in tickers:
        preds, metrics = walkforward_symbol(t, start, end, label_horizon=horizon, refit_every_days=refit, macros={})
        metrics.update({"ticker": t})
        log_to_sqlite("metrics", metrics)
        log_to_sqlite("preds", preds.reset_index())
        print(t, metrics)

if __name__ == "__main__":
    main()
