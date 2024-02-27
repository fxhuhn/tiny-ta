""" Collection of methods for chart analsys """

from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

ATRTypes = Literal["sma", "ema", "rma"]

# default='warn' disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def williams(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """
    Williams %R is showing the current closing price in relation to the high and low of the
    past N days (for a given N). Its purpose is to tell whether a stock or commodity market
    is trading near the high or the low, or somewhere in between, of its recent range.

    The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest),
    obverse of the more common 0 to 100 scale found in many technical analysis oscillators.
    A value of −100 means the close today was the lowest low of the past N days,
    and 0 means today's close was the highest high of the past N days.

    (Although sometimes the %R is adjusted by adding 100.)

    Args:
        df (pd.DataFrame): _description_
        period (int, optional): _description_. Defaults to 7.

    Returns:
        pd.Series: _description_
    """
    max_high = df.High.rolling(period).max()
    min_low = df.Low.rolling(period).min()

    williams_r = (max_high - df.Close) / (max_high - min_low) * -100

    return round(williams_r, 0)


def rsl(close: pd.Series, period: int = 23) -> pd.Series:
    """
    The relative strength according to Levy is based on the assumption that
    securities which exhibited a large relative strength in the past will
    also develop relatively strongly in future, and vice-versa.

    The formula is fairly simple:
    RSL = c / ma(c)
    where c = close price and ma = Moving Average.

    Args:
        close (pd.Series): _description_
        period (int, optional): _description_. Defaults to 23.

    Returns:
        pd.Series: _description_
    """

    return close / sma(close, period)


def rsi(close: pd.Series, period: int = 7) -> pd.Series:
    """
    The relative strength index (RSI) is intended to chart the current and
    historical strength or weakness of a stock or market based on the
    closing prices of a recent trading period.

    Args:
        close (pd.Series): _description_
        period (int, optional): _description_. Defaults to 7.

    Returns:
        pd.Series: _description_
    """

    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = close.diff(1)

    # Make the positive gains (up) and negative gains (down) Series
    up = delta.where(delta > 0, 0.0)
    down = -1 * delta.where(delta < 0, 0.0)

    # Calculate the EWMA
    roll_up = up.ewm(min_periods=period, adjust=False, alpha=(1 / period)).mean()
    roll_down = down.ewm(min_periods=period, adjust=False, alpha=1 / period).mean()

    # Calculate the RSI based on EWMA
    rs = roll_up / roll_down
    rs_index = 100.0 - (100.0 / (1.0 + rs))

    return round(rs_index, 0)


def resample_week(df: pd.DataFrame) -> pd.DataFrame:
    # expected columns Date, Open, High, Low, Close

    # TODO work with indices
    df.reset_index(inplace=True)

    try:
        df.Date = df["Date"].astype("datetime64[ns]")
    except Exception as e:
        print(df, e)

    # note calendar week and year
    df["week"] = df["Date"].dt.strftime("%y-%W")

    df = df.groupby("week").agg(
        Date=("Date", "last"),
        Low=("Low", "min"),
        High=("High", "max"),
        Open=("Open", "first"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    df.reset_index(inplace=True)
    df.drop(columns=["week"], inplace=True)
    df.Date = pd.to_datetime(df.Date)

    return df.set_index("Date")


def ema(close: pd.Series, period: int = 200) -> pd.Series:
    return round(
        close.ewm(
            span=period, min_periods=period, adjust=False, ignore_na=False
        ).mean(),
        2,
    )


def sma(close: pd.Series, period: int = 200) -> pd.Series:
    return round(close.rolling(period).mean(), 2)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return round((close - close.shift(period)) / close.shift(period) * 100)


def atr(
    df: pd.DataFrame, intervall: int = 14, smoothing: ATRTypes = "sma"
) -> pd.Series:
    # Ref: https://stackoverflow.com/a/74282809/

    def rma(s: pd.Series, intervall: int) -> pd.Series:
        return s.ewm(
            alpha=1 / intervall, min_periods=intervall, adjust=False, ignore_na=False
        ).mean()

    def sma(s: pd.Series, intervall: int) -> pd.Series:
        return s.rolling(intervall).mean()

    def ema(s: pd.Series, intervall: int) -> pd.Series:
        return s.ewm(
            span=intervall, min_periods=10, adjust=False, ignore_na=False
        ).mean()

    high, low, prev_close = df["High"], df["Low"], df["Close"].shift()
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)

    if smoothing == "rma":
        return rma(tr, intervall)
    elif smoothing == "ema":
        return ema(tr, intervall)
    elif smoothing == "ema":
        return sma(tr, intervall)
    else:
        raise ValueError(f"unknown smothing type {smoothing}")


def swings(df):
    # local extrema
    df["swing_low"] = df.iloc[
        argrelextrema(data=df["Low"].values, comparator=np.less_equal, order=5)[0]
    ]["Low"]
    df["swing_high"] = df.iloc[
        argrelextrema(data=df["High"].values, comparator=np.greater_equal, order=5)[0]
    ]["High"]

    swings = list(df["swing_low"].dropna().values) + list(
        df["swing_high"].dropna().values
    )
    swings.sort()
    df["sr"] = 0
    df["rs"] = 0
    df["sr_low"] = np.nan
    df["sr_close"] = np.nan

    diff = 0.00025
    for index, row in df.iterrows():
        if [x for x in swings if abs(x - row["High"]) < diff * x]:
            df.loc[index, "rs"] = 1

        if [x for x in swings if abs(x - row["Low"]) < diff * x]:
            df.loc[index, "sr"] = 1
            continue
        if [x for x in swings if abs(x - row["Close"]) < diff * x]:
            df.loc[index, "sr"] = 1
            df.loc[index, "rs"] = 1
    return df


if __name__ == "__main__":
    pass
