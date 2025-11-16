import argparse
from src.backtest import run_backtest
from src.db import Database
from src.optimizer import suggest_next_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--rsi_window", type=int)
    args = parser.parse_args()

    # All args (user-provided or empty)
    user_params = {k: v for k, v in vars(args).items() if v is not None}

    db = Database()

    for i in range(50):   # 50 iterations of agentic learning
        print(f"\n=== Iteration {i+1} ===")

        params = suggest_next_params(db, user_params)
        print("Chosen params:", params)

        result = run_backtest(**params)
        print("Result:", result)

        db.insert_result(params, result)

if __name__ == "__main__":
    main()