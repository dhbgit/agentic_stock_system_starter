import optuna
import yaml
from src.backtest import walkforward_symbol

def objective_rf(trial, ticker, start, end):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }

    _, metrics = walkforward_symbol(ticker, start, end, model_type='rf', model_params=params)
    return metrics["auc"]

def objective_lstm(trial, ticker, start, end):
    params = {
        "time_steps": trial.suggest_int("time_steps", 10, 100),
        "epochs": trial.suggest_int("epochs", 1, 5),
        "batch_size": trial.suggest_int("batch_size", 1, 32),
    }

    _, metrics = walkforward_symbol(ticker, start, end, model_type='lstm', model_params=params)
    return metrics["auc"]

def run_tune(ticker='AAPL', start='2018-01-01', end=None, model_type='rf', n_trials=100):
    if model_type == 'rf':
        objective = objective_rf
    elif model_type == 'lgbm':
        objective = objective_lgbm
    elif model_type == 'lstm':
        objective = objective_lstm
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, ticker, start, end), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params to a file
    with open(f"artifacts/best_params_{ticker}_{model_type}.yaml", "w") as f:
        yaml.dump(trial.params, f)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--model", default="rf", choices=["rf", "lgbm", "lstm"])
    ap.add_argument("--n_trials", type=int, default=100)
    args = ap.parse_args()

    run_tune(args.ticker, args.start, args.end, args.model, args.n_trials)
