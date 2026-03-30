import pandas as pd
from fredapi import Fred

def get_fred_series(api_key, series_id, start, end):
    """
    Fetches a single series from FRED.
    """
    fred = Fred(api_key=api_key)
    series = fred.get_series(series_id, observation_start=start, observation_end=end)
    series = series.to_frame(name=series_id)
    return series

def get_fred_data(api_key, series_ids, start, end):
    """
    Fetches multiple series from FRED and merges them into a single DataFrame.
    """
    df = pd.DataFrame()
    for series_id in series_ids:
        series = get_fred_series(api_key, series_id, start, end)
        if df.empty:
            df = series
        else:
            df = df.merge(series, left_index=True, right_index=True, how='outer')
    
    df = df.ffill()
    return df
