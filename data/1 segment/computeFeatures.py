import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler


# Parameters for long-term features
long_term_window = 20         # 20-period window for long-term moving average and volatility
momentum_periods = [5, 10, 20]  # Longer momentum: differences over 5, 10, and 20 periods
ema_span = 20                 # EMA span for a smoother, long-term trend



# Read CSV file and parse the datetime column
df = pd.read_csv('/TradingModel_V4/1 segment/all_stocks_intraday_1segment.csv',
                 parse_dates=['datetime'])

# Sort the data by symbol and datetime to ensure correct time-series order
df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

# Identify numeric columns (excludes 'symbol' and 'datetime')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


# Scale the Numeric Features Using Min-Max Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(df[numeric_cols])
df_scaled = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)



# Feature Engineering: Basic Long-term Features & Momentum
new_feature_dfs = []

for col in numeric_cols:
    # Long-term Rolling Mean per symbol
    roll_mean = df.groupby('symbol')[col].rolling(window=long_term_window, min_periods=1).mean()\
                 .reset_index(level=0, drop=True)
    roll_mean.name = f'{col}_roll_mean'
    
    # Long-term Rolling Standard Deviation per symbol
    roll_std = df.groupby('symbol')[col].rolling(window=long_term_window, min_periods=1).std()\
                .reset_index(level=0, drop=True)
    roll_std.name = f'{col}_roll_std'
    
    # Exponential Moving Average (EMA)
    ema = df.groupby('symbol')[col].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean())
    ema.name = f'{col}_EMA'
    
    new_feature_dfs.extend([roll_mean, roll_std, ema])
    
    # Momentum features: differences over longer periods
    for period in momentum_periods:
        mom_series = df.groupby('symbol')[col].diff(period)
        mom_series.name = f'{col}_mom{period}'
        new_feature_dfs.append(mom_series)


# ATR requires high, low, and close. Compute the true range (TR):
df['prev_close'] = df.groupby('symbol')['close'].shift(1)
df['TR'] = df[['high', 'low']].apply(lambda row: row['high'] - row['low'], axis=1)
df['TR2'] = abs(df['high'] - df['prev_close'])
df['TR3'] = abs(df['low'] - df['prev_close'])
df['TR'] = df[['TR', 'TR2', 'TR3']].max(axis=1)

# ATR over a 20-period window (long-term ATR)
ATR = df.groupby('symbol')['TR'].rolling(window=long_term_window, min_periods=1).mean()\
      .reset_index(level=0, drop=True)
df['ATR'] = ATR

# ATR14: ATR computed over a 14-period window
ATR14 = df.groupby('symbol')['TR'].rolling(window=14, min_periods=1).mean()\
         .reset_index(level=0, drop=True)
df['ATR14'] = ATR14


# Fast EMA (12 periods) and Slow EMA (26 periods) using close price
df['EMA12'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
df['EMA26'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['MACD'] = df['EMA12'] - df['EMA26']
df['MACD_signal'] = df.groupby('symbol')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
df['MACD_hist'] = df['MACD'] - df['MACD_signal']



# 14-period Stochastic Oscillator
lowest_low_14 = df.groupby('symbol')['low'].transform(lambda x: x.rolling(window=14, min_periods=1).min())
highest_high_14 = df.groupby('symbol')['high'].transform(lambda x: x.rolling(window=14, min_periods=1).max())
df['Stoch14'] = (df['close'] - lowest_low_14) / (highest_high_14 - lowest_low_14) * 100

# Single period Stochastic Oscillator (intra-day)
df['Stoch1'] = (df['close'] - df['low']) / (df['high'] - df['low']) * 100



# Add New Indicator Features to the List
indicator_features = ['ATR', 'ATR14', 'MACD', 'MACD_signal', 'MACD_hist', 'Stoch14', 'Stoch1']
new_feature_dfs.extend([df[col] for col in indicator_features])


# Selective Composite Features (using Scaled Values)
# ---------------------
# For instance: ATR x MACD, MACD x Stoch14, ATR14 x MACD
df['ATR_x_MACD'] = df['ATR'] * df['MACD']
df['MACD_x_Stoch14'] = df['MACD'] * df['Stoch14']
df['ATR14_x_MACD'] = df['ATR14'] * df['MACD']

df['ATR_x_Stoch14'] = df['ATR'] * df['Stoch14']
df['ATR14_x_MACD_signal'] = df['ATR14'] * df['MACD_signal']
df['ATR_x_MACD_hist'] = df['ATR'] * df['MACD_hist']
df['ATR14_x_Stoch1'] = df['ATR14'] * df['Stoch1']

df['ATR_x_MACD_diff'] = df['ATR'] * (df['MACD'] - df['MACD_signal'])
df['ATR14_x_MACD_hist'] = df['ATR14'] * df['MACD_hist']
df['ATR_x_Stoch1'] = df['ATR'] * df['Stoch1']

df['MACD_x_Stoch1'] = df['MACD'] * df['Stoch1']
df['MACD_hist_x_Stoch14'] = df['MACD_hist'] * df['Stoch14']
df['MACD_hist_x_MACD_signal'] = df['MACD_hist'] * df['MACD_signal']

df['MACD_div_ATR'] = df['MACD'] / df['ATR'].replace(0, np.nan)
df['Stoch14_div_ATR14'] = df['Stoch14'] / df['ATR14'].replace(0, np.nan)


selective_composites = ['ATR_x_MACD', 'MACD_x_Stoch14', 'ATR14_x_MACD']
composite_features = [df[comp] for comp in selective_composites]



# Concatenate New Features with Original Data
df_new_features = pd.concat(new_feature_dfs + composite_features, axis=1)
df_final = pd.concat([df, df_new_features], axis=1)

# Drop temporary columns
df_final = df_final.drop(columns=['prev_close', 'TR', 'TR2', 'TR3'])
df_final['datetime'] = pd.to_datetime(df_final['datetime'])
df_final = df_final[df_final['datetime'] >= '2024-01-31 04:00:00']


# Save Transformed Data
output_filename = 'stockData_1segment_Features.csv'
df_final.to_csv(output_filename, index=False)
print(f"Transformed dataset saved to '{output_filename}'")
