from fastapi import APIRouter, HTTPException
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from src.final2.inside_air_predictor.train_iterative import LSTMAttn


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import importlib  # add
# Provide a legacy alias for pickled models that reference 'train_iterative'
sys.modules.setdefault(
    'train_iterative',
    importlib.import_module('src.final2.inside_air_predictor.train_iterative')
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#def generate_future(input_csv, model_path, out_csv, total_seconds=1800):
def generate_future(input_csv):
    """
    input_csv: CSV containing recent data with columns:
               ts,T,H,PMS1,PMS2_5,PMS10,CO2,NO2,CO,VoC,C2H5OH
    model_path: path to best_model.pth (full package saved by training script)
    out_csv: output CSV path for future predictions
    total_seconds: total horizon in seconds (default 1800 = 30 minutes)
    """

    # Get model path
    model_path = os.path.join(
        os.path.dirname(__file__), 
        '..',
        'inside_air_predictor', 
        'best_model_v2.pth'
    )
        
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=500, 
            detail=f"Sample data file not found at {model_path}"
        )
    total_seconds = 1800

    # Load full package
    package = torch.load(model_path, map_location=DEVICE,weights_only=False)
    model = package["model"].to(DEVICE)
    scaler = package["scaler"]
    seq_len = package["seq_len"]
    horizon = package["horizon"]
    feature_cols = package["feature_cols"]

    df = input_csv

    # Convert the 'ts' column to datetime
    df['ts'] = pd.to_datetime(df['ts'])

    # Step 2: Sort by the 'ts' column
    df = df.sort_values("ts").reset_index(drop=True)

    if len(df) < seq_len:
        raise RuntimeError(f"Need at least {seq_len} rows, got {len(df)}")

    # Determine 1-second interval (or whatever the data uses)
    if len(df) >= 2:
        freq_sec = (df["ts"].iloc[1] - df["ts"].iloc[0]).total_seconds()
    else:
        raise RuntimeError("Cannot infer sampling interval from a single row.")

    if freq_sec <= 0:
        raise RuntimeError(f"Invalid sampling interval inferred: {freq_sec} seconds")

    steps_needed = int(total_seconds / freq_sec)
    if steps_needed <= 0:
        raise RuntimeError(f"Total seconds ({total_seconds}) must be >= sampling interval ({freq_sec}).")

    # Prepare initial window and timestamps
    df_input = df.iloc[-seq_len:]
    current_window = df_input[feature_cols].values.astype(float)
    last_ts = df_input["ts"].iloc[-1]

    # Iteratively forecast in chunks of `horizon`
    future_preds = []
    future_ts = []
    steps_done = 0

    model.eval()
    while steps_done < steps_needed:
        # Scale the latest seq_len window
        X_scaled = scaler.transform(current_window[-seq_len:])
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(x_tensor).cpu().numpy()[0]  # (horizon, n_features)

        preds = scaler.inverse_transform(pred_scaled)

        # Number of steps to take this iteration
        steps_to_take = min(horizon, steps_needed - steps_done)

        # Collect timestamps and predictions
        for i in range(steps_to_take):
            ts_future = last_ts + timedelta(seconds=freq_sec * (steps_done + i + 1))
            future_ts.append(ts_future)
            future_preds.append(preds[i])

        # Extend window with the new predictions for next iteration
        current_window = np.vstack([current_window, preds[:steps_to_take]])
        steps_done += steps_to_take

    # Assemble and save
    df_future = pd.DataFrame(future_preds, columns=feature_cols)
    df_future.insert(0, "ts", future_ts)
    # df_future.to_csv(out_csv, index=False)
    # print(f"✅ Saved future predictions ({len(df_future)} rows) to {out_csv}")
    print(f"✅ Generated future predictions ({len(df_future)} rows)")

    return df_future



# if __name__ == "__main__":
#     generate_future('sample.csv', 'best_model.pth', 'future_new.csv',total_seconds=1800)