"""
Feature engineering module for technical indicators.
Replaces pandas_ta with the modern `ta` library.
Generates a wide range of momentum, trend, volatility, and volume indicators.
"""

import pandas as pd
import ta
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


def add_tech_features(df, rsi_window=14):
    df = df.copy()

    # --- Ensure columns are 1D Series
    for col in ["open", "high", "low", "adj_close", "volume"]:
        if col not in df.columns:
            continue
        x = df[col]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        elif isinstance(x, np.ndarray):
            x = pd.Series(x.ravel(), index=df.index[: len(x)])
        elif not isinstance(x, pd.Series):
            x = pd.Series(x, index=df.index[: len(x)])
        df[col] = x

    # --- Momentum Indicators
    df["rsi"] = RSIIndicator(close=df["adj_close"], window=rsi_window).rsi()
    stoch = StochasticOscillator(
        high=df["high"], low=df["low"], close=df["adj_close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # --- Trend Indicators
    macd = MACD(close=df["adj_close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["sma_20"] = SMAIndicator(close=df["adj_close"], window=20).sma_indicator()
    df["ema_20"] = EMAIndicator(close=df["adj_close"], window=20).ema_indicator()

    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["adj_close"], window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # --- Volatility Indicators
    bb = BollingerBands(close=df["adj_close"], window=20, window_dev=2)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    df["bb_bbw"] = bb.bollinger_wband()

    df["atr_14"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["adj_close"], window=14
    ).average_true_range()

    # --- Volume Indicators
    df["obv"] = OnBalanceVolumeIndicator(
        close=df["adj_close"], volume=df["volume"]
    ).on_balance_volume()

    vwap = VolumeWeightedAveragePrice(
        high=df["high"], low=df["low"], close=df["adj_close"], volume=df["volume"], window=14
    )
    df["vwap"] = vwap.volume_weighted_average_price()

    # --- Clean up
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.bfill().ffill()

    return df