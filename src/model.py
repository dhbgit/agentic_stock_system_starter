import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class EnsembleClassifier:
    """
    Combines RF and LGBM for better calibration and lower variance.
    Supports Platt Scaling via a Logistic Regression meta-model.
    """
    def __init__(self, rf_params, lgbm_params):
        self.rf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=42)
        self.lgbm = lgb.LGBMClassifier(**lgbm_params, random_state=42, verbosity=-1)
        self.calibrator = LogisticRegression(random_state=42)
        self.is_fitted = False

    def fit(self, X, y):
        # 1. Train base models
        self.rf.fit(X, y)
        self.lgbm.fit(X, y)
        
        # 2. Calibrate (Platt Scaling) using out-of-bag or holdout predictions
        # We'll use the training set itself for simplicity here, 
        # or better: a small internal validation split if data allows.
        rf_probs = self.rf.predict_proba(X)[:, 1]
        lgbm_probs = self.lgbm.predict_proba(X)[:, 1]
        meta_X = np.stack([rf_probs, lgbm_probs], axis=1)
        
        self.calibrator.fit(meta_X, y)
        self.is_fitted = True

    def predict_proba(self, X):
        rf_probs = self.rf.predict_proba(X)[:, 1]
        lgbm_probs = self.lgbm.predict_proba(X)[:, 1]
        
        # Ensemble average or Meta-model prediction? 
        # Meta-model (calibrator) provides the 'Truth' probability.
        meta_X = np.stack([rf_probs, lgbm_probs], axis=1)
        # We return a (N, 2) array to match sklearn API: [P(0), P(1)]
        p1 = self.calibrator.predict_proba(meta_X)[:, 1]
        return np.stack([1-p1, p1], axis=1)

    def predict(self, X, threshold=0.55):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

def train_ensemble(X, y, rf_params, lgbm_params) -> EnsembleClassifier:
    clf = EnsembleClassifier(rf_params, lgbm_params)
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
