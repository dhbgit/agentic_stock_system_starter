"""
Feature engineering: technical indicators on OHLCV price data.

Full feature set (default):
  Momentum  — RSI, Stochastic %K/%D
  Trend     — MACD, SMA-20, EMA-20, ADX (+DI/-DI)
  Volatility — Bollinger Bands (w/width), ATR-14, ATR volatility-of-volatility
  Volume    — OBV, VWAP-14
  Macro     — VIX close, VIX 10-day MA/std

Light feature set (light=True):
  Returns (1/3/5-day), rolling mean (7/21-day), rolling std (14-day), RSI
"""

import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


def _to_series(x, index):
    """Coerce a column to a clean 1-D Series."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    if isinstance(x, np.ndarray):
        return pd.Series(x.ravel(), index=index[: len(x)])
    if not isinstance(x, pd.Series):
        return pd.Series(x, index=index[: len(x)])
    return x


def add_tech_features(df, rsi_window=14, light=False):
    df = df.copy()

    # Ensure all OHLCV columns are clean 1-D Series
    for col in ["open", "high", "low", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = _to_series(df[col], df.index)

    # Merge VIX data
    from .data import load_vix
    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    vix = load_vix(start, end)
    df = df.merge(vix, left_index=True, right_index=True, how="left")
    df["vix_close"] = df["vix_close"].ffill()

    if light:
        # Minimal, fast feature set
        df["ret_1"] = df["adj_close"].pct_change(1)
        df["ret_3"] = df["adj_close"].pct_change(3)
        df["ret_5"] = df["adj_close"].pct_change(5)
        df["rmean_7"] = df["adj_close"].pct_change().rolling(7).mean()
        df["rmean_21"] = df["adj_close"].pct_change().rolling(21).mean()
        df["rstd_14"] = df["adj_close"].pct_change().rolling(14).std()
        df["rsi"] = RSIIndicator(close=df["adj_close"], window=rsi_window).rsi()
    else:
        # --- Momentum
        df["rsi"] = RSIIndicator(close=df["adj_close"], window=rsi_window).rsi()
        stoch = StochasticOscillator(
            high=df["high"], low=df["low"], close=df["adj_close"], window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # --- Trend
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

        # --- Volatility
        bb = BollingerBands(close=df["adj_close"], window=20, window_dev=2)
        df["bb_bbm"] = bb.bollinger_mavg()
        df["bb_bbh"] = bb.bollinger_hband()
        df["bb_bbl"] = bb.bollinger_lband()
        df["bb_bbw"] = bb.bollinger_wband()
        df["atr_14"] = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["adj_close"], window=14
        ).average_true_range()
        df["atr_14_std_14"] = df["atr_14"].rolling(14).std()

        # --- Volume
        df["obv"] = OnBalanceVolumeIndicator(
            close=df["adj_close"], volume=df["volume"]
        ).on_balance_volume()
        df["vwap"] = VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["adj_close"], volume=df["volume"], window=14
        ).volume_weighted_average_price()

        # --- VIX features
        df["vix_close_ma_10"] = df["vix_close"].rolling(10).mean()
        df["vix_close_std_10"] = df["vix_close"].rolling(10).std()

    # Clean up
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.bfill().ffill()
    return df
