import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import lightgbm as lgb

def train_rf(X: pd.DataFrame, y: pd.Series, params: dict) -> RandomForestClassifier:
    clf = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return clf

def train_lgbm(X: pd.DataFrame, y: pd.Series, params: dict) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(**params, random_state=42)
    clf.fit(X, y)
    return clf

def metrics_classifier(y_true, y_prob, y_pred):
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true))>1 else None,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }

def portfolio_metrics(returns: pd.Series, risk_free_rate=0.0):
    """
    Calculates portfolio metrics.
    """
    # Sharpe Ratio
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() - risk_free_rate) / downside_returns.std()

    # Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Calmar Ratio
    annual_return = returns.mean() * 252
    calmar_ratio = annual_return / abs(max_drawdown)

    return {
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
    }
