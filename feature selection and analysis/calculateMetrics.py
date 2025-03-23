import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

import shap


data_folder = '/TradingModel_V4/aggregated data'
csv_files = glob.glob(os.path.join(data_folder, '*.csv'))


target_column = 'target'

csv_files = ['/TradingModel_V4/aggregated data/stockData_16segments_Features.csv']


def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=True)
    
    # Compute the future price change for 1-period as the target (choose either absolute or percentage change)
    df['target'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-1)) - df['close']
    # Or for percentage change:
    # df['target'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-1)) / df['close'] - 1

    # Drop rows with missing target values (last row per symbol might have NaN due to shifting)
    df = df.dropna(subset=['target'])
    # Fill missing values per symbol to preserve time ordering
    df = df.groupby('symbol', group_keys=False).apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

    # Extract numeric features (excluding the target)
    X = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore')
    y = df['target']
    return X, y, df



# Feature Selection Methods


# 1. Correlation Matrix (Pearson)
def correlation_selection(X, y, threshold=0.1):
    df_corr = X.copy()
    df_corr[target_column] = y
    corr_matrix = df_corr.corr(method='pearson')
    target_corr = corr_matrix[target_column].drop(target_column)
    
    print("Correlation with target:")
    print(target_corr.sort_values(ascending=False))
    
    selected_features = target_corr[abs(target_corr) >= threshold].index.tolist()
    return selected_features, target_corr

# 2. Mutual Information
def mutual_information_selection(X, y):
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    print("Mutual Information Scores:")
    print(mi_series)
    return mi_series

# 3. Permutation Importance using a Random Forest surrogate model
def permutation_importance_selection(X, y):
    # Split the data to avoid overfitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importance = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    
    print("Permutation Importance Scores:")
    print(perm_importance)
    return perm_importance, model

# 4. Tree-Based Feature Importance
def tree_based_importance(model, X):
    # Assuming model is a tree-based model (RandomForest in our case)
    tree_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Tree-based Feature Importance:")
    print(tree_importance)
    return tree_importance

# 5. SHAP Values
def shap_analysis(model, X, num_features=10):
    # Use TreeExplainer for tree-based models with interventional feature perturbation.
    explainer = shap.Explainer(model, X, feature_perturbation="interventional")
    # Disable additivity check to avoid errors if slight differences occur.
    shap_values = explainer(X, check_additivity=False)
    
    # Plot summary for the top features
    shap.summary_plot(shap_values, X, max_display=num_features)
    
    return shap_values

# 6. Partial Dependence Plots (PDP) and ICE Plots
def plot_pdp_ice(model, X, features):
    # Plot PDP and ICE for the provided features.
    fig, ax = plt.subplots(figsize=(12, 6))
    display = PartialDependenceDisplay.from_estimator(
        model, X, features, kind="both", subsample=50, grid_resolution=20, ax=ax
    )
    plt.tight_layout()
    plt.show()

# ---------------------
# Process Each CSV File
# ---------------------
for file in csv_files:
    print(f"\nProcessing file: {file}")
    
    X, y, df_original = load_data(file)
    
    # Ensure that there are enough features and the target is not empty.
    if X.empty or y.empty:
        print("No numeric features or target found. Skipping file.")
        continue
    
    # 1. Correlation Matrix
    selected_corr, corr_scores = correlation_selection(X, y, threshold=0.1)
    
    # 2. Mutual Information
    mi_scores = mutual_information_selection(X, y)
    
    # 3. Permutation Importance
    perm_importance, surrogate_model = permutation_importance_selection(X, y)
    
    # 4. Tree-Based Importance (using the same surrogate model)
    tree_importance = tree_based_importance(surrogate_model, X)
    
    # 5. SHAP Analysis
    print("Generating SHAP summary plot...")
    shap_values = shap_analysis(surrogate_model, X)
    
    # 6. Partial Dependence and ICE Plots for top 3 features by tree importance
    top_features = tree_importance.head(3).index.tolist()
    print(f"Plotting PDP and ICE for top features: {top_features}")
    plot_pdp_ice(surrogate_model, X, top_features)
    
    # Optionally, save the feature selection results
    results = pd.DataFrame({
        'Correlation': corr_scores,
        'Mutual Information': mi_scores,
        'Permutation Importance': perm_importance,
        'Tree-Based Importance': tree_importance
    })
    results.sort_values(by='Tree-Based Importance', ascending=False, inplace=True)
    output_results = file.replace('.csv', '_feature_selection_summary.csv')
    results.to_csv(output_results)
    print(f"Feature selection summary saved to: {output_results}")

print("Feature selection processing complete.")
