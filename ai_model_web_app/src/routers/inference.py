from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import os
import sys
from typing import Optional

# Add parent directory to path to import from final2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Lazy imports - these will be loaded when first called
_anomaly_detector = None
_data_preprocessing = None
_get_latest_row = None
_get_eatscore = None
_generate_future = None
_pm25_predictor = None

def get_anomaly_detector():
    global _anomaly_detector
    if _anomaly_detector is None:
        from src.final2.anomaly_detector.infer_tgat import anomaly_detector
        _anomaly_detector = anomaly_detector
    return _anomaly_detector

def get_data_preprocessing():
    global _data_preprocessing, _get_latest_row
    if _data_preprocessing is None:
        from src.final2.data_processing import data_preprocessing, get_latest_row
        _data_preprocessing = data_preprocessing
        _get_latest_row = get_latest_row
    return _data_preprocessing, _get_latest_row

def get_eatscore_fn():
    global _get_eatscore
    if _get_eatscore is None:
        from src.final2.safeBite_score.load_and_predict import get_eatscore
        _get_eatscore = get_eatscore
    return _get_eatscore

def get_future_predictor():
    global _generate_future
    if _generate_future is None:
        from src.final2.inside_air_predictor.infer_iterative import generate_future
        _generate_future = generate_future
    return _generate_future

def get_pm25_predictor_fn():
    global _pm25_predictor
    if _pm25_predictor is None:
        from src.final2.pm25_predictor.load_and_predict import pm25_predictor
        _pm25_predictor = pm25_predictor
    return _pm25_predictor

router = APIRouter()

class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")

class InferenceResponse(BaseModel):
    eat_score_now: float
    eat_score_future: float
    current_anomaly_detected: bool
    future_anomaly_detected: bool
    external_pm25: float
    latitude: float
    longitude: float
    message: str

@router.post("/inference", response_model=InferenceResponse)
async def perform_inference(coords: Coordinates):
    """
    Perform air quality inference and anomaly detection for given coordinates.
    
    - **latitude**: Latitude coordinate (-90 to 90)
    - **longitude**: Longitude coordinate (-180 to 180)
    
    Returns current and future (30 min) EatSafe scores and anomaly detection results.
    """
    try:
        # TODO: Check database for the location's inside air quality data
        # For now, using sample data
        sample_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'final2', 
            'sample_data', 
            'sample.csv'
        )
        
        if not os.path.exists(sample_path):
            raise HTTPException(
                status_code=500, 
                detail=f"Sample data file not found at {sample_path}"
            )
        
        # Lazy load ML functions
        pm25_predictor = get_pm25_predictor_fn()
        data_preprocessing, get_latest_row = get_data_preprocessing()
        generate_future = get_future_predictor()
        anomaly_detector = get_anomaly_detector()
        get_eatscore = get_eatscore_fn()

        #Fetch location's outside pm2.5 values
        today = datetime.today()
        today = today.strftime('%Y-%m-%d %H:%M:%S')

        external_pm25 = pm25_predictor(coords.latitude, coords.longitude, today)

        print(f"the external pm2.5 is {external_pm25}")

        original_df = pd.read_csv(sample_path)

        # Get Future Air Data (30 min forecast)
        future_df = generate_future(original_df)


        # Inside air quality dataset processing
        original_df = data_preprocessing(original_df)
        future_df = data_preprocessing(future_df)

        # Reset index to make 'ts' a column again
        future_df = future_df.reset_index()
        original_df = original_df.reset_index()

        # Anomaly Detection
        original_df = anomaly_detector(original_df)
        future_df = anomaly_detector(future_df)

        future_anomaly = get_latest_row(future_df, 'ts')
        current_anomaly = get_latest_row(original_df, 'ts')

        # EatSafe Score Calculation
        eatscore_now = get_eatscore(current_anomaly)
        eatscore_future = get_eatscore(future_anomaly)

        return {
            "eat_score_now": float(eatscore_now) if not isinstance(eatscore_now, (int, float)) else eatscore_now,
            "eat_score_future": float(eatscore_future) if not isinstance(eatscore_future, (int, float)) else eatscore_future,
            "current_anomaly_detected": bool(current_anomaly.get('anomaly', False)),
            "future_anomaly_detected": bool(future_anomaly.get('anomaly', False)),
            "external_pm25": float(external_pm25) if not isinstance(external_pm25, (int, float)) else external_pm25,
            "latitude": coords.latitude,
            "longitude": coords.longitude,
            "message": "Inference completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error during inference: {str(e)}"
        )

@router.get("/status")
async def get_status():
    """Check if the inference service is available"""
    return {
        "status": "online",
        "service": "Air Quality Inference API",
        "models": {
            "anomaly_detector": "active",
            "air_quality_predictor": "active",
            "eatsafe_scorer": "active"
        }
    }