import pandas as pd
import numpy as np

# Read the CSV file and parse the datetime column
df = pd.read_csv("/TradingModel_V4/all_stocks_intraday_hourly.csv", parse_dates=["datetime"])

# Sort data by symbol and datetime to ensure correct order
df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

# Create a date column for daily grouping
df['date'] = df['datetime'].dt.date

# For each symbol and date, create a grouping variable that divides the day into pairs of rows (2 rows per segment).
# Assuming exactly 16 segments per day, grouping by pairs yields 8 segments per day.
df['group_id'] = df.groupby(['symbol', 'date']).cumcount() // 2


agg_funcs = {
    'datetime': 'first',
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'symbol': 'first'
}

# Aggregate the data for each symbol, date, and group_id
df_agg = df.groupby(['symbol', 'date', 'group_id'], as_index=False).agg(agg_funcs)

# Drop the helper columns used for grouping, here we can drop 'date' and 'group_id' if not needed
df_agg = df_agg.drop(columns=['group_id', 'date'])

# Ensure the column order matches the original file
df_agg = df_agg[['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

# Save the aggregated data to a new CSV file
output_filename = 'all_stocks_intraday_8segments.csv'
df_agg.to_csv(output_filename, index=False)
print(f"Aggregated dataset saved to '{output_filename}'")
