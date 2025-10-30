import os
import argparse
import gc
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration and reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Feature definitions
# -----------------------------
FEATURES = [
    'T', 'H', 'PMS1', 'PMS2_5', 'PMS10', 'CO2', 'NO2', 'CO', 'VoC', 'C2H5OH',
    'pm25_max_2min', 'pm25_mean_2min', 'co2_max_2min', 'co2_mean_1min',
    'co2_rise_1to2min', 'T_rise_1min', 'pm10_mean_30s', 'pm10_spike',
    'H_mean_2min', 'voc_mean_1min', 'ethanol_mean_1min', 'Label', 'Confidence'
]
TARGET = 'EatSafe_Score'

NUMERIC_FEATURES = [
    'T', 'H', 'PMS1', 'PMS2_5', 'PMS10', 'CO2', 'NO2', 'CO', 'VoC', 'C2H5OH',
    'pm25_max_2min', 'pm25_mean_2min', 'co2_max_2min', 'co2_mean_1min',
    'co2_rise_1to2min', 'T_rise_1min', 'pm10_mean_30s', 'pm10_spike',
    'H_mean_2min', 'voc_mean_1min', 'ethanol_mean_1min', 'Confidence'
]
CATEGORICAL_FEATURES = ['Label']


# -----------------------------
# Utility: memory-friendly dtypes
# -----------------------------
def get_dtypes():
    # Use float32 for continuous features to reduce RAM; ints as int32.
    dtypes = {col: 'float32' for col in NUMERIC_FEATURES}
    # Int-like original sensors (okay to read as float32 for scaler consistency)
    int_like = ['PMS1', 'PMS2_5', 'PMS10', 'NO2', 'CO', 'VoC']
    for col in int_like:
        dtypes[col] = 'float32'
    dtypes['CO2'] = 'float32'
    dtypes['C2H5OH'] = 'float32'
    dtypes['Label'] = 'category'  # compact memory
    dtypes[TARGET] = 'float32'
    return dtypes


# -----------------------------
# Build preprocessor
# -----------------------------
def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder='drop',
        n_jobs=None  # leave None; parallelization can increase memory usage
    )
    return preprocessor


# -----------------------------
# Build MLP model
# -----------------------------
def build_model(input_dim: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), dtype=tf.float32),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# -----------------------------
# Plot training curves
# -----------------------------
def plot_history(history, loss_path='training_loss.png', mae_path='training_mae.png'):
    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # MAE
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['mae'], label='train_mae')
    plt.plot(history.history.get('val_mae', []), label='val_mae')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(mae_path, dpi=150)
    plt.close()


# -----------------------------
# Main training function
# -----------------------------
def train(
        csv_path: str,
        batch_size: int = 512,
        epochs: int = 50,
        test_size: float = 0.2,
        model_out: str = 'eatsafe_model.keras',
        preproc_out: str = 'preprocessor.pkl'
):
    # Read only needed columns
    usecols = FEATURES + [TARGET]
    dtypes = get_dtypes()
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, nrows=2000000)

    # Ensure category dtype for Label
    if df['Label'].dtype.name != 'category':
        df['Label'] = df['Label'].astype('category')

    # Drop rows with missing values in features or target (fast, low-memory)
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    after = len(df)
    print(f"Dropped {before - after} rows with missing values. Remaining: {after}")

    # Train/validation split
    X = df[FEATURES]
    y = df[TARGET].astype('float32').to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, shuffle=True
    )
    print(f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]}")

    # Build & fit preprocessor ONLY on training data
    preprocessor = build_preprocessor()
    print("Fitting preprocessor on training data ...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # Convert to float32 to reduce RAM
    X_train_proc = X_train_proc.astype('float32', copy=False)
    X_val_proc = X_val_proc.astype('float32', copy=False)

    # Save preprocessor
    joblib.dump(preprocessor, preproc_out)
    print(f"Saved preprocessor to {preproc_out}")

    # Free pandas frames if needed
    del X, df
    gc.collect()

    # Build and train model
    input_dim = X_train_proc.shape[1]
    model = build_model(input_dim)
    print(f"Model input dimension: {input_dim}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_out,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Starting training ...")
    history = model.fit(
        X_train_proc, y_train,
        validation_data=(X_val_proc, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save final model (best already checkpointed; this ensures a final copy too)
    model.save(model_out)
    print(f"Saved model to {model_out}")

    # Plot & save training curves
    plot_history(history, loss_path='training_loss.png', mae_path='training_mae.png')
    print("Saved training_loss.png and training_mae.png")

    # Cleanup
    del X_train_proc, X_val_proc, y_train, y_val, model
    gc.collect()
    print("Training complete.")


# -----------------------------
# Entry point
# -----------------------------
def parse_args():
    args = argparse.Namespace(
        csv='/kaggle/input/inside/eatsafe_scored_data.csv',
        batch_size=512,
        epochs=50,
        test_size=0.2,
        model_out='eatsafe_model.keras',
        preproc_out='preprocessor.pkl'
    )
    return args


if __name__ == '__main__':
    args = parse_args()
    train(
        csv_path=args.csv,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_size=args.test_size,
        model_out=args.model_out,
        preproc_out=args.preproc_out
    )