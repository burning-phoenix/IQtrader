import pandas as pd

# Define the file names for each resolution and their associated weights
files_and_weights = {
    '1': {'file': '/TradingModel_V4/aggregated data/stockData_1segment_Features_feature_selection_summary.csv', 'weight': 0.1},
    '2': {'file': '/TradingModel_V4/aggregated data/stockData_2segments_Features_feature_selection_summary.csv', 'weight': 0.15},
    '4': {'file': '/TradingModel_V4/aggregated data/stockData_4segments_Features_feature_selection_summary.csv', 'weight': 0.5},
    '8': {'file': '/TradingModel_V4/aggregated data/stockData_8segments_Features_feature_selection_summary.csv', 'weight': 0.25}
}

# List of metric columns to process
metric_cols = ['Correlation', 'Mutual Information', 'Permutation Importance', 'Tree-Based Importance']

# Read each CSV, using the first column as the index so that it contains the feature names.
dfs = {}
for res, info in files_and_weights.items():
    # Read CSV with the first column as the index (assumes the feature names are stored there)
    df = pd.read_csv(info['file'], index_col=0)
    # Reset index to create a column named 'feature' and ensure it's a string
    df = df.reset_index().rename(columns={'index': 'feature'})
    df['feature'] = df['feature'].astype(str)
    # Rename metric columns to include the resolution suffix (e.g., "Correlation_1")
    for col in metric_cols:
        df.rename(columns={col: f"{col}_{res}"}, inplace=True)
    # Keep only 'feature' and the renamed metric columns
    cols_to_keep = ['feature'] + [f"{col}_{res}" for col in metric_cols]
    dfs[res] = df[cols_to_keep]

# Merge all dataframes on the 'feature' column (outer join to capture all features)
merged_df = None
for res, df in dfs.items():
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on='feature', how='outer')

# Fill missing values with 0 (if a feature isn't present in a given resolution)
merged_df.fillna(0, inplace=True)

# Standardize the metric columns by subtracting the mean and dividing by the standard deviation
for col in metric_cols:
    for res in files_and_weights.keys():
        col_name = f"{col}_{res}"
        mean = merged_df[col_name].mean()
        std = merged_df[col_name].std()
        merged_df[f"standardized_{col_name}"] = (merged_df[col_name] - mean) / std

# Calculate the overall score as the sum of the standardized metric columns
merged_df["overall_score"] = (
    merged_df["standardized_Correlation_1"] * files_and_weights['1']['weight'] +
    merged_df["standardized_Correlation_2"] * files_and_weights['2']['weight'] +
    merged_df["standardized_Correlation_4"] * files_and_weights['4']['weight'] +
    merged_df["standardized_Correlation_8"] * files_and_weights['8']['weight'] +
    merged_df["standardized_Mutual Information_1"] * files_and_weights['1']['weight'] +
    merged_df["standardized_Mutual Information_2"] * files_and_weights['2']['weight'] +
    merged_df["standardized_Mutual Information_4"] * files_and_weights['4']['weight'] +
    merged_df["standardized_Mutual Information_8"] * files_and_weights['8']['weight'] +
    merged_df["standardized_Permutation Importance_1"] * files_and_weights['1']['weight'] +
    merged_df["standardized_Permutation Importance_2"] * files_and_weights['2']['weight'] +
    merged_df["standardized_Permutation Importance_4"] * files_and_weights['4']['weight'] +
    merged_df["standardized_Permutation Importance_8"] * files_and_weights['8']['weight'] +
    merged_df["standardized_Tree-Based Importance_1"] * files_and_weights['1']['weight'] +
    merged_df["standardized_Tree-Based Importance_2"] * files_and_weights['2']['weight'] +
    merged_df["standardized_Tree-Based Importance_4"] * files_and_weights['4']['weight'] +
    merged_df["standardized_Tree-Based Importance_8"] * files_and_weights['8']['weight']
)

# Drop the standardized columns
standardized_cols = [f"standardized_{col}_{res}" for col in metric_cols for res in files_and_weights.keys()]
merged_df.drop(columns=standardized_cols, inplace=True)

# Keep only the necessary columns: feature and overall_score
merged_df = merged_df[["feature", "overall_score"]]

# Save the final weighted feature names and overall scores to a CSV file
output_filename = 'combinedScores.csv'
merged_df.to_csv(output_filename, index=False)
print(f"Final weighted feature names saved to '{output_filename}'")
