import pandas as pd
import numpy as np


def doji(df) -> pd.Series:

    # https://thepatternsite.com/LongLegDoji.html
    
    close = df.Close
    open  = df.Open
    high  = df.High
    low   = df.Low

    body = abs(df.Open-df.Close)
    shadow_up = df.High - df[['Open', 'Close']].max(axis=1)
    shadow_low = df[['Open', 'Close']].min(axis=1) - df.Low

    cond_1 = body / (high - low) < .1
    cond_2 = shadow_up > 3 * body
    cond_3 = shadow_low > 3 * body
    
    doji = cond_1 & cond_2 & cond_3
    
    df['doji'] = 0
    df.loc[doji, 'doji'] = 1
    
    return df['doji']


def inside_candle(df:pd.DataFrame) -> pd.Series:
    high  = df.High
    low   = df.Low

    prev_high  = df.High.shift(1)
    prev_low   = df.Low.shift(1)
    
    inside = (prev_high > high) & (prev_low < low)
    
    df['inside'] = 0
    df.loc[inside, 'inside'] = 1

    return df['inside']


if __name__ == "__main__":
    pass
