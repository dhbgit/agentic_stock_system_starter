import pandas as pd
import yfinance as yf
from datetime import date

def load_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    Download historical price data for a ticker and return
    a normalized DataFrame with 1-D Series columns.
    """
    if end in (None, "null", "", "None"):
        end = date.today().isoformat()

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    # Normalize column names to lowercase
    df.columns = [c[0].lower().strip() if isinstance(c, tuple) else c.lower().strip() for c in df.columns]
    df.index = pd.to_datetime(df.index)

    # Handle naming differences safely
    open_col = df.get("open", None)
    high_col = df.get("high", None)
    low_col = df.get("low", None)
    close_col = df.get("close", None)
    adj_close_col = df.get("adj close", None)
    volume_col = df.get("volume", None)

    if any(x is None for x in [open_col, high_col, low_col, close_col, adj_close_col, volume_col]):
        raise KeyError(f"One or more expected columns missing for {ticker}. Columns found: {df.columns.tolist()}")

    # ✅ build new DataFrame with aligned index
    df_clean = pd.DataFrame({
        "open": open_col.values,
        "high": high_col.values,
        "low": low_col.values,
        "close": close_col.values,
        "adj_close": adj_close_col.values,
        "volume": volume_col.values,
    }, index=df.index)

    df_clean.dropna(inplace=True)
    return df_clean


def load_macro_csv(path: str, col: str = "value") -> pd.DataFrame:
    """Load a CSV with columns: date, value (or custom col)."""
    m = pd.read_csv(path)
    m["date"] = pd.to_datetime(m["date"])
    m = m.rename(columns={col: "macro_value"})
    return m[["date", "macro_value"]]


def merge_macro(price_df: pd.DataFrame, macro_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge price data with one or more macroeconomic DataFrames."""
    out = price_df.copy().reset_index().rename(columns={"index": "date"})
    for name, m in macro_dict.items():
        out = out.merge(m.rename(columns={"macro_value": name}), on="date", how="left")
    out = out.set_index("date").fillna(method="ffill")
    return out

def load_vix(start: str, end: str | None = None) -> pd.DataFrame:
    """
    Load VIX data from yfinance.
    """
    vix = load_prices("^VIX", start, end)
    vix = vix.rename(columns={"adj_close": "vix_close"})
    return vix[["vix_close"]]

