import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Dropout, Activation,
                                     Concatenate, Add, Bidirectional, LSTM, GlobalAveragePooling1D,
                                     Dense, MultiHeadAttention, LayerNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.stats import skew
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import matplotlib.pyplot as plt


# ---------------------------
# Feature Selection for Ablation Studies
# ---------------------------
ALL_FEATURES = []       # Final selected features omitted

# ---------------------------
# Helper Functions
# ---------------------------

def get_selected_features(features_to_remove=None):
    selected = ALL_FEATURES.copy()
    if features_to_remove:
        for feat in features_to_remove:
            if feat in selected:
                selected.remove(feat)
    return selected


def add_gaussian_noise(X, noise_std):
    if noise_std > 0:
        return X + np.random.normal(0, noise_std, X.shape)
    return X


# ---------------------------
# Data Loading & Preprocessing Functions
# ---------------------------
def load_data(data_paths):
    data_dict = {}
    for res, path in data_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        data_dict[res] = df
    return data_dict

def add_target_column(data_dict, target_col='target'):
    for res, df in data_dict.items():
        if target_col not in df.columns:
            df[target_col] = df['close'].shift(-4) - df['close']
            df.dropna(inplace=True)
    return data_dict


def preprocess_data_with_pipeline(data_dict, feature_cols, target_col):
    x_pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), StandardScaler())
    y_pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), StandardScaler())

    train_df = data_dict[1]
    x_pipeline.fit(train_df[feature_cols])
    y_pipeline.fit(train_df[[target_col]])

    for res, df in data_dict.items():
        df[feature_cols] = x_pipeline.transform(df[feature_cols])
        df[target_col] = y_pipeline.transform(df[[target_col]]).flatten()

    return data_dict, x_pipeline, y_pipeline


def generate_sliding_windows(df, seq_length, feature_cols, target_col):
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df[feature_cols].iloc[i: i+seq_length].values)
        y.append(df[target_col].iloc[i+seq_length])
    return np.array(X), np.array(y)

def create_datasets(data_dict, seq_length, feature_cols, target_col):
    datasets = {}
    for res, df in data_dict.items():
        X, y = generate_sliding_windows(df, seq_length, feature_cols, target_col)
        datasets[res] = (X, y)
    return datasets

# ---------------------------
# Model Building Functions
# ---------------------------
def sliding_window_interaction_block(inputs, conv_filters, kernel_sizes, dropout_rate, regularizer):
    # Original branch: subject to dynamic soft freezing
    conv_outputs = []
    for k in kernel_sizes:
        conv = Conv1D(filters=conv_filters, kernel_size=k, padding='same',
                      kernel_regularizer=regularizer, name=f"sw_conv_{k}")(inputs)
        conv = BatchNormalization(name=f"sw_bn_{k}")(conv)
        conv = Activation('relu', name=f"sw_act_{k}")(conv)
        conv_outputs.append(conv)
    concat_conv = Concatenate(name="sw_concat")(conv_outputs)
    proj = Conv1D(filters=conv_filters * len(kernel_sizes), kernel_size=1, padding='same',
                  kernel_regularizer=regularizer, name="sw_proj")(inputs)
    frozen_branch = Add(name="sw_add")([concat_conv, proj])
    frozen_branch = Activation('relu', name="sw_act_out")(frozen_branch)
    frozen_branch = Dropout(dropout_rate, name="sw_dropout")(frozen_branch)

    # Redundant branch: not subject to soft freezing (no special names added)
    redundant = Conv1D(filters=conv_filters * len(kernel_sizes), kernel_size=1, padding='same',
                       kernel_regularizer=regularizer, name="sw_redundant_conv")(inputs)
    redundant = BatchNormalization(name="sw_redundant_bn")(redundant)
    redundant = Activation('relu', name="sw_redundant_act")(redundant)
    redundant = Dropout(dropout_rate, name="sw_redundant_dropout")(redundant)

    # Combine both branches via concatenation
    combined = Concatenate(name="sw_combined")([frozen_branch, redundant])
    return combined


def create_hybrid_model(seq_length, num_features,
                        l2_lambda=l2_lambda,
                        dropout_conv=dropout_conv,
                        dropout_lstm=dropout_lstm,
                        dropout_dense=dropout_dense,
                        conv_filters=conv_filters,
                        conv_kernel_sizes=conv_kernel_sizes,
                        lstm_units=lstm_units,
                        dense_units=dense_units,
                        num_transformer_heads=num_transformer_heads,
                        transformer_ff_dim=transformer_ff_dim):
    regularizer = tf.keras.regularizers.l2(l2_lambda)
    inputs = Input(shape=(seq_length, num_features), name="series_input")

    # Modified Sliding-Window Interaction Block with redundancy
    sw_block = sliding_window_interaction_block(inputs, conv_filters, conv_kernel_sizes, dropout_conv, regularizer)

    # The rest of the network remains unchanged.
    conv1 = Conv1D(filters=conv_filters*2, kernel_size=7, activation='relu', padding='same',
                   kernel_regularizer=regularizer, name="res_conv1")(sw_block)
    conv2 = Conv1D(filters=conv_filters*2, kernel_size=5, activation='relu', padding='same',
                   kernel_regularizer=regularizer, name="res_conv2")(conv1)
    conv2 = BatchNormalization(name="res_bn")(conv2)
    sw_block_proj = Conv1D(filters=conv_filters*2, kernel_size=1, padding='same',
                           kernel_regularizer=regularizer, name="res_proj")(sw_block)
    cnn_out = Add(name="res_add")([conv2, sw_block_proj])
    cnn_out = Dropout(dropout_conv, name="res_dropout")(cnn_out)

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizer, name="bilstm1"),
                             name="bidirectional_1")(cnn_out)
    lstm_out = BatchNormalization(name="lstm_bn_1")(lstm_out)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizer, name="bilstm2"),
                             name="bidirectional_2")(lstm_out)
    lstm_out = BatchNormalization(name="lstm_bn_2")(lstm_out)
    lstm_out = Dropout(dropout_lstm, name="lstm_dropout")(lstm_out)

    attn_key_dim = (2 * lstm_units) // num_transformer_heads
    attn_output = MultiHeadAttention(num_heads=num_transformer_heads, key_dim=attn_key_dim,
                                     name="transformer_attn")(lstm_out, lstm_out)
    attn_output = Dropout(dropout_dense, name="attn_dropout")(attn_output)
    x = Add(name="attn_add")([lstm_out, attn_output])
    x = LayerNormalization(name="attn_layernorm")(x)

    ffn = tf.keras.Sequential([
         Dense(transformer_ff_dim, activation='relu', kernel_regularizer=regularizer, name="ffn_dense1"),
         Dense(2 * lstm_units, kernel_regularizer=regularizer, name="ffn_dense2")
    ], name="transformer_ffn")
    ffn_output = ffn(x)
    ffn_output = Dropout(dropout_dense, name="ffn_dropout")(ffn_output)
    x = Add(name="ffn_add")([x, ffn_output])
    x = LayerNormalization(name="ffn_layernorm")(x)

    x = GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=regularizer, name="dense_relu")(x)
    x = BatchNormalization(name="dense_bn")(x)
    x = Dropout(dropout_dense, name="dense_dropout")(x)

    # Scaled output for more variance
    scale_factor = 1.5

    #___________________________
    # tanh is bad with variance, Exponential Linear Unit is good with variance
    # output = Dense(1, activation='tanh')(x)
    output = Dense(1, activation='elu')(x)

    output_scaled = tf.keras.layers.Lambda(lambda x: x * scale_factor)(output)

    # Use `output_scaled` as final output
    output = output_scaled


    return Model(inputs=inputs, outputs=output)

# ---------------------------
# Evaluation & Trading Simulation Function
# ---------------------------
def evaluate_model(model, X, y, threshold=0.0):
    preds = model.predict(X).flatten()
    true_var = np.var(y)
    pred_var = np.var(preds)
    true_skew = skew(y)
    pred_skew = skew(preds)
    corr = np.corrcoef(y, preds)[0, 1]
    mse = np.mean((y - preds)**2)
    sign_accuracy = np.mean(np.sign(preds) == np.sign(y))

    print("Evaluation Metrics:")
    print(f"True Variance: {true_var:.6f}")
    print(f"Predicted Variance: {pred_var:.6f}")
    print(f"True Skewness: {true_skew:.6f}")
    print(f"Predicted Skewness: {pred_skew:.6f}")
    print(f"Correlation: {corr:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"Sign Accuracy: {sign_accuracy:.4f}")

    signals = (preds > threshold).astype(int)
    strat_returns = signals * y
    cumulative_return = np.sum(strat_returns)
    strategy_var = np.var(strat_returns)
    strategy_skew = skew(strat_returns)
    print("\nTrading Simulation Metrics:")
    print(f"Total Strategy Return: {cumulative_return:.6f}")
    print(f"Strategy Return Variance: {strategy_var:.6f}")
    print(f"Strategy Return Skewness: {strategy_skew:.6f}")

    return preds, signals, strat_returns




# ---------------------------
# The details of the following helper functions and model training are omitted
# ---------------------------


# ---------------------------
# Soft Freezing via Gradient Scaling (Dynamic & Gradual Unfreezing)
# ---------------------------
class SoftFreezeModel(tf.keras.Model):
    """
    Details of the class omitted
    """

# ---------------------------
# Iterative Soft Freezing Function with Dynamic Freeze Multiplier and Gradual Unfreezing
# ---------------------------
def iterative_freeze(model, stage_resolution, stage_idx, total_stages):
    """
    Details of the function omitted
    """

# ---------------------------
# Training Function with Early Stopping on Validation Loss
# ---------------------------
def train_model(model, datasets, training_stages, batch_size=32):
    """
    Training details omitted
    """

def main():
    features_to_remove = ["EMA26", "close_mom20", "low_EMA", "close_EMA", "low_roll_std"]
    feature_cols = get_selected_features(features_to_remove)
    target_col = 'target'
    seq_length = seq_length     # seq_length is omitted
    data_paths = {
        1: '/content/stockData_1segment_Features.csv',
        2: '/content/stockData_2segments_Features.csv',
        4: '/content/stockData_4segments_Features.csv',
        8: '/content/stockData_8segments_Features.csv',
        16: '/content/stockData_16segments_Features.csv'
    }
    data_dict = load_data(data_paths)
    data_dict = add_target_column(data_dict, target_col)
    data_dict, x_pipeline, y_pipeline = preprocess_data_with_pipeline(data_dict, feature_cols, target_col)
    datasets = create_datasets(data_dict, seq_length, feature_cols, target_col)
    num_features = len(feature_cols)

    # Create the base hybrid model
    base_model = create_hybrid_model(seq_length, num_features)      # Hyperparameters are omitted
    base_model.summary()

    # Wrap the base model with SoftFreezeModel for dynamic soft freezing
    model = SoftFreezeModel(base_model, base_freeze_multiplier=0.0005)

    # Training stages (from less noisy to more noisy resolutions)
    training_stages = [
        {'resolution': 1, 'lr': 1e-4, 'epochs': 20},
        {'resolution': 2, 'lr': 5e-4, 'epochs': 15},
        {'resolution': 4, 'lr': 1e-3, 'epochs': 20},
        {'resolution': 8, 'lr': 5e-4, 'epochs': 18},
        {'resolution': 16, 'lr': 1e-4, 'epochs': 15},
        {'resolution': 8, 'lr': 5e-4, 'epochs': 10}
    ]

    trained_model = train_model(model, datasets, training_stages, batch_size=128)
    print("\nRunning Final Evaluation on Resolution 8 Data:")
    X_eval, y_eval = datasets[8]
    evaluate_model(trained_model, X_eval, y_eval, threshold=0.01)
    
    random_indices = np.random.choice(len(X_eval), size=10, replace=False)

    plt.figure(figsize=(18, 25))

    for i, idx in enumerate(random_indices, 1):
        X_sample = X_eval[idx]
        y_true_sample = y_eval[idx]

        # Predict the next value
        y_pred_sample_scaled = trained_model.predict(X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1])).flatten()[0]

        # Unscale predictions
        X_sample_unscaled = x_pipeline.inverse_transform(X_sample)
        y_true_unscaled = y_pipeline.inverse_transform([[y_true_sample]])[0,0]
        y_pred_unscaled = y_pipeline.inverse_transform([[y_pred_sample_scaled]])[0,0]

        # Historical sequence (close price or first feature)
        feature_idx = feature_cols.index('close') if 'close' in feature_cols else 0
        sequence_feature = X_sample_unscaled[:, feature_idx]

        # Convert predicted and true differences into absolute prices
        last_value = sequence_feature[-1]
        predicted_absolute_price = last_value + y_pred_unscaled
        true_absolute_price = last_value + y_true_unscaled

        plt.subplot(5, 2, i)

        # Plot historical sequence
        plt.plot(sequence_feature, marker='o', label='Historical Input Sequence')

        # Plot true and predicted next prices
        plt.scatter(len(sequence_feature), true_absolute_price, color='green', s=100, label='True Next Value')
        plt.scatter(len(sequence_feature), predicted_absolute_price, color='red', marker='x', s=100, label='Predicted Next Value')

        plt.title(f"Sample {i} (Absolute Prices)", fontsize=14)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Price (Original Scale)", fontsize=12)

        plt.xticks(
            list(range(len(sequence_feature))) + [len(sequence_feature)],
            labels=[f"t-{len(sequence_feature)-j}" for j in range(len(sequence_feature), 0, -1)] + ['Prediction'],
            rotation=45
        )
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
