import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def train_rf(X: pd.DataFrame, y: pd.Series, params: dict) -> RandomForestClassifier:
    clf = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return clf

def metrics_classifier(y_true, y_prob, y_pred):
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true))>1 else None,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
