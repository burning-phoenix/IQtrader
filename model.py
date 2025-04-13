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

# ---------------------------
# Feature Selection for Ablation Studies
# ---------------------------
ALL_FEATURES = [
    "close",
    "EMA26",
    "close_EMA",
    "ATR14_x_MACD_hist",
    "high_roll_std",
    "close_mom5",
    "close_mom20",
    "close_mom10",
    "low_mom5",
    "low_roll_std",
    "EMA12",
    "open_mom10",
    "low_EMA",
    "open_roll_mean",
    "ATR_x_Stoch14",
    "open_mom20"
]

def get_selected_features(features_to_remove=None):
    selected = ALL_FEATURES.copy()
    if features_to_remove:
        for feat in features_to_remove:
            if feat in selected:
                selected.remove(feat)
    return selected

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
                        l2_lambda=2.5e-6,
                        dropout_conv=0.04,
                        dropout_lstm=0.08,
                        dropout_dense=0.07,
                        conv_filters=64,
                        conv_kernel_sizes=[2,3,5],
                        lstm_units=32,
                        dense_units=16,
                        num_transformer_heads=2,
                        transformer_ff_dim=32):
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
    output = Dense(1, activation='tanh', kernel_regularizer=regularizer, name="output")(x)

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
    print("Evaluation Metrics:")
    print(f"True Variance: {true_var:.6f}")
    print(f"Predicted Variance: {pred_var:.6f}")
    print(f"True Skewness: {true_skew:.6f}")
    print(f"Predicted Skewness: {pred_skew:.6f}")
    print(f"Correlation: {corr:.6f}")
    print(f"MSE: {mse:.6f}")
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
# Soft Freezing via Gradient Scaling (Dynamic & Gradual Unfreezing)
# ---------------------------
class SoftFreezeModel(tf.keras.Model):
    def __init__(self, base_model, base_freeze_multiplier=0.002):
        super(SoftFreezeModel, self).__init__()
        self.base_model = base_model
        # Base freeze multiplier for new frozen layers.
        self.base_freeze_multiplier = base_freeze_multiplier
        # Dictionary mapping layer prefix -> current freeze multiplier.
        self.soft_freeze_dict = {}
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        new_gradients = []
        # Apply dynamic freeze multipliers to layers as needed.
        for var, grad in zip(trainable_vars, gradients):
            multiplier = 1.0
            for prefix, freeze_mult in self.soft_freeze_dict.items():
                if var.name.startswith(prefix) and grad is not None:
                    multiplier = freeze_mult
                    break
            new_gradients.append(grad * multiplier if grad is not None else None)
        self.optimizer.apply_gradients(zip(new_gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# ---------------------------
# Iterative Soft Freezing Function with Dynamic Freeze Multiplier and Gradual Unfreezing
# ---------------------------
def iterative_freeze(model, stage_resolution, stage_idx, total_stages):
    """
    Applies soft freezing on layers based on the current segment resolution.
    New layers are added with the base_freeze_multiplier.
    
    Then, for every frozen layer, update its freeze multiplier using a linear schedule 
    that increases from base_freeze_multiplier at stage 0 to 1.0 at the final stage.
    """
    new_freeze = {}
    if stage_resolution == 8:
        for layer in ["sw_conv_3", "sw_bn_3"]:
            if layer not in model.soft_freeze_dict:
                new_freeze[layer] = model.base_freeze_multiplier
        if new_freeze:
            print("After resolution 16, adding soft freeze for layers:", ", ".join(new_freeze.keys()),
                  f"with multiplier {model.base_freeze_multiplier}.")
    elif stage_resolution == 4:
        if "res_conv1" not in model.soft_freeze_dict:
            new_freeze["res_conv1"] = model.base_freeze_multiplier
            print("After resolution 8, adding soft freeze for layer: res_conv1",
                  f"with multiplier {model.base_freeze_multiplier}.")
    elif stage_resolution == 2:
        if "bidirectional_1" not in model.soft_freeze_dict:
            new_freeze["bidirectional_1"] = model.base_freeze_multiplier
            print("After resolution 4, adding soft freeze for layer: bidirectional_1",
                  f"with multiplier {model.base_freeze_multiplier}.")
    elif stage_resolution == 1:
        if "sw_conv_2" not in model.soft_freeze_dict:
            new_freeze["sw_conv_2"] = model.base_freeze_multiplier
            print("After resolution 1, adding soft freeze for layer: sw_conv_2",
                  f"with multiplier {model.base_freeze_multiplier}.")
    else:
        print(f"No iterative soft freezing applied for resolution {stage_resolution}.")

    # Update the freeze dictionary with any new layers.
    model.soft_freeze_dict.update(new_freeze)
    
    # Compute the new multiplier uniformly:
    new_multiplier = model.base_freeze_multiplier + (0.35 - model.base_freeze_multiplier) * (stage_idx / total_stages)
    
    # Update all frozen layers to the new multiplier.
    for prefix in list(model.soft_freeze_dict.keys()):
        old_val = model.soft_freeze_dict[prefix]
        model.soft_freeze_dict[prefix] = new_multiplier
        print(f"Updating {prefix}: multiplier updated from {old_val} to {new_multiplier}.")

# ---------------------------
# Training Function with Early Stopping on Validation Loss
# ---------------------------
def train_model(model, datasets, training_stages, batch_size=32):
    total_stages = len(training_stages)
    for stage_idx, stage in enumerate(training_stages, start=1):
        res = stage['resolution']
        lr = stage['lr']
        epochs = stage['epochs']

        print(f"\nStarting training stage: Resolution = {res} segments/day, LR = {lr}, Epochs = {epochs}")
        X_train, y_train = datasets[res]

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='huber')
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        # Use a validation split of 20%
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.2, callbacks=callbacks)
        
        # Apply the iterative soft freezing strategy with the new schedule.
        iterative_freeze(model, res, stage_idx, total_stages)
        
    return model

# ---------------------------
# Main Function
# ---------------------------
def main():
    features_to_remove = ["EMA26", "close_mom20", "low_EMA", "close_EMA", "low_roll_std"]
    feature_cols = get_selected_features(features_to_remove)
    target_col = 'target'
    seq_length = 40
    data_paths = {
        1: '/TradingModel_V4/aggregated data/stockData_1segment_Features.csv',
        2: '/TradingModel_V4/aggregated data/stockData_2segments_Features.csv',
        4: '/TradingModel_V4/aggregated data/stockData_4segments_Features.csv',
        8: '/TradingModel_V4/aggregated data/stockData_8segments_Features.csv',
        16: '/TradingModel_V4/aggregated data/stockData_16segments_Features.csv'
    }
    data_dict = load_data(data_paths)
    data_dict = add_target_column(data_dict, target_col)
    data_dict, x_pipeline, y_pipeline = preprocess_data_with_pipeline(data_dict, feature_cols, target_col)
    datasets = create_datasets(data_dict, seq_length, feature_cols, target_col)
    num_features = len(feature_cols)
    
    # Create the base hybrid model.
    base_model = create_hybrid_model(seq_length, num_features)
    base_model.summary()
    
    # Wrap the base model with our SoftFreezeModel for dynamic soft freezing.
    model = SoftFreezeModel(base_model, base_freeze_multiplier=0.0005)
    
    # Training stages (in descending order)
    training_stages = [
        {'resolution': 8, 'lr': 1e-4, 'epochs': 10},
        {'resolution': 4, 'lr': 1e-3, 'epochs': 15},
        {'resolution': 2, 'lr': 5e-4, 'epochs': 8},
        {'resolution': 1, 'lr': 1e-4, 'epochs': 8},
        {'resolution': 4, 'lr': 1e-4, 'epochs': 5}
    ]

    trained_model = train_model(model, datasets, training_stages, batch_size=32)
    print("\nRunning Evaluation on Resolution 4 Data:")
    X_eval, y_eval = datasets[4]
    evaluate_model(trained_model, X_eval, y_eval, threshold=0.1)

if __name__ == '__main__':
    main()
