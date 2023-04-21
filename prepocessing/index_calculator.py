# -*- coding = utf-8 -*-

import numpy as np
import pandas as pd


# 获取股票最近30个交易日的5日、10日、20日、30日线的历史数据
def calculate_average_line(df):
    df = (df.set_index('date')
          .sort_index()
          .assign(day_5=lambda x: x['close'].rolling(window=5).mean(),
                  day_10=lambda x: x['close'].rolling(window=10).mean(),
                  day_20=lambda x: x['close'].rolling(window=20).mean(),
                  day_30=lambda x: x['close'].rolling(window=30).mean())
          .reset_index())
    return df


# 计算MACD指标
def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    df = (df.assign(fast_ema=lambda x: x['close'].ewm(span=fast_period, min_periods=fast_period).mean())
          .assign(slow_ema=lambda x: x['close'].ewm(span=slow_period, min_periods=slow_period).mean())
          .assign(MACD=lambda x: x['fast_ema'] - x['slow_ema'])
          .assign(SIGNAL=lambda x: x['MACD'].ewm(span=signal_period, min_periods=signal_period).mean()))
    return df


# 计算KDJ指标
def calculate_kdj(df, n=9, m1=3, m2=3):
    C = df['close']
    H = df['high']
    L = df['low']

    # 通过rolling函数计算RSV
    RSV = (C - L.rolling(window=n).min()) / (H.rolling(window=n).max() - L.rolling(window=n).min()) * 100
    RSV = RSV.fillna(50)  # 填充缺失值

    # 通过EMA计算K值和D值
    K = pd.Series(np.zeros(len(df)), index=df.index)
    D = pd.Series(np.zeros(len(df)), index=df.index)

    for i in range(1, len(df)):
        K[i] = 2 / m1 * K[i - 1] + 1 / m1 * RSV[i]
        D[i] = 2 / m2 * D[i - 1] + 1 / m2 * K[i]

    # 计算J值
    J = 3 * K - 2 * D

    # 将KDJ指标添加到原始dataframe中
    df = df.assign(K=K.values, D=D.values, J=J.values)

    return df


# 计算DMI指标
def calculate_dmi(df, n1=14, n2=6):
    # 初始化上升动向值（PDM）、下降动向值（MDM）和真实波幅（TR）
    PDM = []
    MDM = []
    TR = []
    for i in range(len(df)):
        if i == 0:
            PDM.append(0)
            MDM.append(0)
            TR.append(0)
        else:
            high_diff = df.iloc[i]['high'] - df.iloc[i - 1]['high']
            low_diff = df.iloc[i - 1]['low'] - df.iloc[i]['low']
            PDM_value = 0 if high_diff <= 0 or high_diff <= low_diff else high_diff
            MDM_value = 0 if low_diff <= 0 or low_diff <= high_diff else low_diff
            PDM.append(PDM_value)
            MDM.append(MDM_value)
            TR_value = max(df.iloc[i]['high'] - df.iloc[i]['low'], abs(df.iloc[i]['high'] - df.iloc[i - 1]['close']),
                           abs(df.iloc[i]['low'] - df.iloc[i - 1]['close']))
            TR.append(TR_value)
    df['PDM'] = PDM
    df['MDM'] = MDM
    df['TR'] = TR

    # 计算14日平均TR、+DI和-DI
    df['TR14'] = df['TR'].rolling(n1).sum()
    df['PDM14'] = df['PDM'].rolling(n1).sum()
    df['MDM14'] = df['MDM'].rolling(n1).sum()
    df['PDI14'] = df['PDM14'] / df['TR14'] * 100
    df['MDI14'] = df['MDM14'] / df['TR14'] * 100

    # 计算DX、ADX和ADXR
    df['DX'] = abs(df['PDI14'] - df['MDI14']) / (df['PDI14'] + df['MDI14']) * 100
    df['ADX'] = df['DX'].rolling(n2).mean()
    df['ADXR'] = (df['ADX'].shift(n2) + df['ADX']) / 2

    # 删除无用的列
    df = df.drop(columns=['TR', 'PDM', 'MDM', 'TR14', 'PDM14', 'MDM14', 'DX'])

    return df


# 计算布林带指标
def calculate_bollinger_bands(df, window_size=20, num_of_std=2):
    return df.assign(rolling_mean=df['close'].rolling(window=window_size).mean(),
                     rolling_std=df['close'].rolling(window=window_size).std(),
                     upper_band=lambda x: x['rolling_mean'] + (x['rolling_std'] * num_of_std),
                     lower_band=lambda x: x['rolling_mean'] - (x['rolling_std'] * num_of_std))


# 计算ATR指标
def calculate_average_true_range(df, window_size=14):
    high_to_low = df['high'] - df['low']
    high_to_prev_close = np.abs(df['high'] - df['close'].shift())
    low_to_prev_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_to_low, high_to_prev_close, low_to_prev_close], axis=1).max(axis=1)
    return df.assign(ATR=true_range.rolling(window=window_size).mean())
