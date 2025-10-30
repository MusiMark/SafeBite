from fastapi import APIRouter, HTTPException
import pickle
import torch
import numpy as np
import pandas as pd
import os
import sys
import math

from src.final2.pm25_predictor.airqo_api import airqo_api

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the necessary classes from train_model
try:
    from src.final2.pm25_predictor.train_model import OptimizedPM25Predictor, OptimizedGNNPredictor, OptimizedKGBuilder
except ImportError as e:
    print(f"Error importing from train_model: {e}")
    print("Make sure train_model.py is in the same directory.")
    sys.exit(1)


def load_model(model_path="pm25_gnn_complete_complete_model.pkl"):
    """Load the trained PM2.5 predictor safely (CPU/GPU compatible)."""
    
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Always map tensors to CPU if CUDA not available
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(model_path, 'rb') as f:
        # This ensures all tensors are loaded to the correct device
        predictor = pickle.load(f)
        predictor.device = map_location

    # Force the model itself onto the right device
    predictor.model.to(map_location)
    predictor.model.eval()

    print(f"✅ Model loaded successfully on {predictor.device}")
    return predictor



def predict_pm25(predictor, latitude, longitude, datetime_str):
    """
    Predict PM2.5 value for a single input.
    Input: (latitude, longitude, datetime_str)
    Output: PM2.5 (float)
    """
    # --- Prepare input ---
    dt = pd.to_datetime(datetime_str)
    hour = dt.hour
    day_of_week = dt.dayofweek
    day_of_year = dt.dayofyear
    month = dt.month

    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Construct feature array
    features = np.array([
        latitude, longitude,
        hour_sin, hour_cos,
        month_sin, month_cos,
        day_of_week, day_of_year
    ]).reshape(1, -1)

    # Scale features using the stored scaler
    X_scaled = predictor.scaler_features.transform(features)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(predictor.device)

    # --- Build location + time entities ---
    loc_entity = f"loc_{latitude:.4f}_{longitude:.4f}"
    time_entity = f"time_{hour:02d}"

    # If unseen, fallback to mean embeddings
    entity_to_id = predictor.kg_builder.entity_to_id
    node_features = predictor.kg_builder.node_features.to(predictor.device)
    edge_index = predictor.kg_builder.edge_index.to(predictor.device)

    with torch.no_grad():
        node_embeddings = predictor.model(node_features, edge_index)

        if loc_entity in entity_to_id and time_entity in entity_to_id:
            loc_emb = node_embeddings[entity_to_id[loc_entity]]
            time_emb = node_embeddings[entity_to_id[time_entity]]
            combined_emb = (loc_emb + time_emb) / 2
        else:
            # fallback: mean embedding (for unseen locations/times)
            combined_emb = node_embeddings.mean(dim=0)

        pred_scaled = predictor.model.predict_from_embeddings(
            combined_emb.unsqueeze(0), X_tensor
        ).cpu().numpy().flatten()[0]

    # Inverse scale prediction
    pm25 = predictor.scaler_target.inverse_transform([[pred_scaled]])[0, 0]
    return float(pm25)


# if __name__ == "__main__":
def pm25_predictor(lat, lon, dt_str):

    # Get model path
    model_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'pm25_predictor', 
        'pm25_gnn_complete_complete_model_v2.pkl'
    )
        
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=500, 
            detail=f"Sample data file not found at {model_path}"
        )
    total_seconds = 1800
    
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train the model first by running train_model.py")
        sys.exit(1)
    
    
    predictor = OptimizedPM25Predictor()
    predictor = load_model(model_path)

    pm25_value = predict_pm25(predictor, lat, lon, dt_str)

    # print(f"\n🎯 Predicted PM2.5 at ({lat}, {lon}) on {dt_str}: {pm25_value:.3f} µg/m³")

    realtime_pm25, distance = airqo_api(lat, lon)

    #Prediction Smoothening
    if distance <= 0.5:
        w = 1.0
    else:
        k = 1.9736  # decay constant derived above
        w = math.exp(-k * (distance - 0.5))
        w = max(w, 0.001)

    final = (w * realtime_pm25) + ((1 - w) * pm25_value)

    return final