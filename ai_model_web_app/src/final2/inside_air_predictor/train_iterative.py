#!/usr/bin/env python3
"""
train_iterative.py
Train a time series forecasting model with iterative forecasting in mind.
Includes 70/15/15 split with a single progress bar for the whole training.
"""

import argparse, math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ["T", "H", "PMS1", "PMS2_5", "PMS10", "CO2", "NO2", "CO", "VoC", "C2H5OH"]


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
        x = self.X[idx:idx + self.seq_len]
        y = self.X[idx + self.seq_len:idx + self.seq_len + self.horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# -----------------------------
# Model: LSTM + Attention
# -----------------------------
class LSTMAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=60):
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
def run_epoch(loader, model, criterion, optimizer=None, desc="Train"):
    """Run one epoch over a loader. If optimizer is None, it's eval mode."""
    losses = []
    preds_all, true_all = [], []

    if optimizer:
        model.train()
    else:
        model.eval()

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if optimizer:
            optimizer.zero_grad()
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(xb)
            loss = criterion(pred, yb)
            if optimizer:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        preds_all.append(pred.detach().cpu().numpy())
        true_all.append(yb.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    true_all = np.concatenate(true_all, axis=0)
    r2 = r2_score(true_all.reshape(-1), preds_all.reshape(-1))
    mae = mean_absolute_error(true_all.reshape(-1), preds_all.reshape(-1))
    rmse = math.sqrt(mean_squared_error(true_all.reshape(-1), preds_all.reshape(-1)))
    return np.mean(losses), r2, mae, rmse


def train_model(train_loader, val_loader, test_loader, model, optimizer, criterion, epochs=30, seq_len=None,
                horizon=None, scaler=None):
    best_val_r2 = -np.inf

    # Single progress bar for the entire training process
    pbar = tqdm(total=epochs, desc="Training", unit="epoch")

    for epoch in range(epochs):
        train_loss, train_r2, train_mae, train_rmse = run_epoch(train_loader, model, criterion, optimizer, desc="Train")
        val_loss, val_r2, val_mae, val_rmse = run_epoch(val_loader, model, criterion, optimizer=None, desc="Val")

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train: Loss {train_loss:.4f} R² {train_r2:.3f} MAE {train_mae:.3f} RMSE {train_rmse:.3f}")
        print(f"Val:   Loss {val_loss:.4f} R² {val_r2:.3f} MAE {val_mae:.3f} RMSE {val_rmse:.3f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            package = {
                "model": model,
                "scaler": scaler,
                "seq_len": seq_len,
                "horizon": horizon,
                "feature_cols": FEATURE_COLS
            }
            torch.save(package, "best_model.pth")
            print("✅ Saved new best model")

        # Update progress bar
        pbar.set_postfix({"Val R²": f"{val_r2:.3f}", "Val Loss": f"{val_loss:.4f}"})
        pbar.update(1)

    pbar.close()

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_r2, test_mae, test_rmse = run_epoch(test_loader, model, criterion, optimizer=None, desc="Test")
    print(f"Test:  Loss {test_loss:.4f} R² {test_r2:.3f} MAE {test_mae:.3f} RMSE {test_rmse:.3f}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = argparse.Namespace(
        csv="/kaggle/input/inside/12_Kitchen_window.csv",  # Path to your input CSV
        seq_len=100,  # Reduced from 3600 - faster training
        horizon=10,  # Reduced from 60 - faster training
        batch_size=128,  # Increased for better GPU utilization
        epochs=30,  # Reduced epochs
        lr=1e-3  # Learning rate
    )

    df = pd.read_csv(args.csv, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)

    # 70/15/15 split
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    df_train, df_val, df_test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    train_ds = SeqDataset(df_train, args.seq_len, args.horizon, fit_scaler=True)
    val_ds = SeqDataset(df_val, args.seq_len, args.horizon, scaler=train_ds.scaler)
    test_ds = SeqDataset(df_test, args.seq_len, args.horizon, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = LSTMAttn(input_dim=len(FEATURE_COLS), horizon=args.horizon).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_model(train_loader, val_loader, test_loader, model, optimizer, criterion,
                epochs=args.epochs,
                seq_len=args.seq_len,
                horizon=args.horizon,
                scaler=train_ds.scaler)