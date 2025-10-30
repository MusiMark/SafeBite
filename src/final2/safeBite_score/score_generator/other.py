#!/usr/bin/env python3
"""
train_model.py
Train a time series forecasting model on environmental sensor data.
"""

import os, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ["T","H","PMS1","PMS2_5","PMS10","CO2","NO2","CO","VoC","C2H5OH"]

# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, df, seq_len, horizon, scaler=None, fit_scaler=False):
        X = df[FEATURE_COLS].values.astype(float)

        if scaler is None:
            scaler = StandardScaler()
        if fit_scaler:
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)

        self.scaler = scaler
        self.X = X
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.seq_len - self.horizon

    def __getitem__(self, idx):
        x = self.X[idx:idx+self.seq_len]
        y = self.X[idx+self.seq_len:idx+self.seq_len+self.horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Model: LSTM + Attention
# -----------------------------
class LSTMAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, input_dim * horizon)
        self.input_dim = input_dim
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, seq_len, hidden)
        attn_weights = torch.softmax(self.attn(out), dim=1)  # (B, seq_len, 1)
        context = (out * attn_weights).sum(dim=1)  # (B, hidden)
        pred = self.fc(context)  # (B, horizon*input_dim)
        return pred.view(-1, self.horizon, self.input_dim)

# -----------------------------
# Training Loop
# -----------------------------
def train_model(train_loader, val_loader, model, optimizer, criterion, epochs=30, seq_len=None, horizon=None, scaler=None):
    best_val_r2 = -np.inf
    history = {"train_loss": [], "val_r2": [], "val_mae": [], "val_rmse": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_preds.append(pred.cpu().numpy())
                val_true.append(yb.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_true = np.concatenate(val_true, axis=0)

        r2 = r2_score(val_true.reshape(-1), val_preds.reshape(-1))
        mae = mean_absolute_error(val_true.reshape(-1), val_preds.reshape(-1))
        rmse = math.sqrt(mean_squared_error(val_true.reshape(-1), val_preds.reshape(-1)))

        history["train_loss"].append(np.mean(train_losses))
        history["val_r2"].append(r2)
        history["val_mae"].append(mae)
        history["val_rmse"].append(rmse)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss {np.mean(train_losses):.4f} | "
              f"Val R² {r2:.3f} | MAE {mae:.3f} | RMSE {rmse:.3f}")

        if r2 > best_val_r2:
            best_val_r2 = r2
            package = {
                "model": model,
                "scaler": scaler,
                "seq_len": seq_len,
                "horizon": horizon,
                "feature_cols": FEATURE_COLS
            }
            torch.save(package, "best_model.pth")
            print("✅ Saved new best model")

    # --- Visualization ---
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["val_r2"], label="Val R²")
    plt.plot(history["val_mae"], label="Val MAE")
    plt.plot(history["val_rmse"], label="Val RMSE")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    args = argparse.Namespace(
        csv="/kaggle/input/inside/12_Kitchen_window.csv",  # Path to your input CSV
        seq_len=288,  # Input sequence length (e.g., 24h @ 5min intervals)
        horizon=6,  # Prediction horizon (number of steps ahead)
        batch_size=64,  # Batch size for training
        epochs=50,  # Number of epochs
        lr=1e-3  # Learning rate
    )

    df = pd.read_csv(args.csv, parse_dates=["ts"],nrows=10000).sort_values("ts").reset_index(drop=True)

    # Train/val split
    split_idx = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:split_idx], df.iloc[split_idx:]

    train_ds = SeqDataset(df_train, args.seq_len, args.horizon, fit_scaler=True)
    val_ds = SeqDataset(df_val, args.seq_len, args.horizon, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LSTMAttn(input_dim=len(FEATURE_COLS), horizon=args.horizon).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_model(train_loader, val_loader, model, optimizer, criterion,
                epochs=args.epochs,
                seq_len=args.seq_len,
                horizon=args.horizon,
                scaler=train_ds.scaler)