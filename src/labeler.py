import pandas as pd

def make_labels(df: pd.DataFrame, horizon_days: int=5, threshold: float=0.0) -> tuple[pd.Series, pd.Series]:
    fwd = df['adj_close'].shift(-horizon_days) / df['adj_close'] - 1.0
    y = (fwd > threshold).astype(int)
    return y.dropna(), fwd.dropna()
