from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import os
import sys
import time
import asyncio
import hashlib
from typing import Optional, Dict, Any, List
from functools import lru_cache
import re
import glob
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Add parent directory to path to import from final2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Lazy imports - these will be loaded when first called
_anomaly_detector = None
_data_preprocessing = None
_get_latest_row = None
_get_eatscore = None
_generate_future = None
_pm25_predictor = None

# Global cache for processed locations
_locations_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = asyncio.Lock()
_cache_last_updated: Optional[datetime] = None

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

class DetailedInferenceResponse(BaseModel):
    eat_score_now: float
    eat_score_future: float
    current_anomaly: str
    future_anomaly: str
    current_confidence: float
    future_confidence: float
    external_pm25: float
    processing_time: float
    risk_level_now: str
    risk_level_future: str
    recommendation: str
    latitude: float
    longitude: float
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models: Dict[str, bool]
    dependencies: Dict[str, bool]

class StatusResponse(BaseModel):
    status: str
    service: str
    models: Dict[str, str]

def get_risk_level(score: float) -> str:
    """Convert EatSafe score to risk level
    Handles both 0-1 range and 0-100 range scores"""
    # Normalize to 0-1 range if score is in 0-100 range
    normalized_score = score / 100.0 if score > 1 else score
    
    if normalized_score >= 0.8:
        return "LOW"
    elif normalized_score >= 0.6:
        return "MODERATE"
    elif normalized_score >= 0.4:
        return "HIGH"
    else:
        return "CRITICAL"

def generate_recommendation(current_score: float, future_score: float, 
                          current_anomaly: str, future_anomaly: str) -> str:
    """Generate intelligent safety recommendation"""
    current_risk = get_risk_level(current_score)
    future_risk = get_risk_level(future_score)
    
    if current_risk == "CRITICAL" or future_risk == "CRITICAL":
        base = "🚨 DO NOT CONSUME FOOD - Immediate safety risk detected"
    elif current_risk in ["HIGH", "CRITICAL"] or future_risk in ["HIGH", "CRITICAL"]:
        base = "⚠️ AVOID FOOD CONSUMPTION - High risk conditions"
    else:
        base = "✅ Food consumption is safe under current conditions"
    
    # Add anomaly-specific information
    anomalies = []
    if current_anomaly != "normal":
        anomalies.append(f"Current: {current_anomaly.replace('_', ' ').title()}")
    if future_anomaly != "normal":
        anomalies.append(f"Expected: {future_anomaly.replace('_', ' ').title()}")
    
    if anomalies:
        base += f" ({'; '.join(anomalies)})"
    
    return base

class ModelManager:
    """Manages model loading and initialization"""
    def __init__(self):
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize_models(self):
        """Warm up models by loading them once"""
        if self._initialized:
            return
            
        async with self._lock:
            if not self._initialized:
                # Pre-load all functions to initialize models
                get_pm25_predictor_fn()
                get_data_preprocessing()
                get_future_predictor()
                get_anomaly_detector()
                get_eatscore_fn()
                
                self._initialized = True

model_manager = ModelManager()

def extract_coords_from_filename(filename: str) -> Optional[tuple]:
    """
    Extract latitude and longitude from filename.
    Expected format: lat,lon.csv (e.g., 0.31296,32.58624.csv)
    """
    try:
        # Remove .csv extension and split by comma
        basename = os.path.basename(filename)
        coords_str = basename.replace('.csv', '')
        parts = coords_str.split(',')
        
        if len(parts) == 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return (lat, lon)
    except Exception as e:
        print(f"Error extracting coords from {filename}: {e}")
    
    return None

def process_location_file(file_path: str, lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Process a single CSV file and return the inference results.
    """
    try:
        # Lazy load ML functions
        pm25_predictor = get_pm25_predictor_fn()
        data_preprocessing, get_latest_row = get_data_preprocessing()
        generate_future = get_future_predictor()
        anomaly_detector = get_anomaly_detector()
        get_eatscore = get_eatscore_fn()

        # Fetch location's outside pm2.5 values
        today = datetime.today()
        today_str = today.strftime('%Y-%m-%d %H:%M:%S')

        external_pm25 = pm25_predictor(lat, lon, today_str)
        print(f"Processing {file_path}: External PM2.5 = {external_pm25}")

        # Load and process data
        original_df = pd.read_csv(file_path)

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
        
        # Normalize scores to 0-100 range for consistency
        # If model returns 0-1 range, multiply by 100
        if eatscore_now <= 1:
            eatscore_now = eatscore_now * 100
        if eatscore_future <= 1:
            eatscore_future = eatscore_future * 100
        
        # Clamp scores to valid range
        eatscore_now = max(0, min(100, eatscore_now))
        eatscore_future = max(0, min(100, eatscore_future))

        # Extract additional information
        current_anomaly_type = str(current_anomaly.get('Label', 'normal'))
        future_anomaly_type = str(future_anomaly.get('Label', 'normal'))
        current_confidence = float(current_anomaly.get('Confidence', 0.0))
        future_confidence = float(future_anomaly.get('Confidence', 0.0))

        result = {
            "eat_score_now": float(eatscore_now),
            "eat_score_future": float(eatscore_future),
            "current_anomaly": current_anomaly_type,
            "future_anomaly": future_anomaly_type,
            "current_confidence": current_confidence,
            "future_confidence": future_confidence,
            "external_pm25": float(external_pm25),
            "risk_level_now": get_risk_level(float(eatscore_now)),
            "risk_level_future": get_risk_level(float(eatscore_future)),
            "recommendation": generate_recommendation(
                float(eatscore_now), float(eatscore_future),
                current_anomaly_type, future_anomaly_type
            ),
            "latitude": lat,
            "longitude": lon,
            "file_source": os.path.basename(file_path),
            "processed_at": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_all_locations():
    """
    Process all CSV files in the new_samples directory.
    This function is called on startup and every 5 minutes.
    """
    global _locations_cache, _cache_last_updated
    
    try:
        # Get path to new_samples directory
        new_samples_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'final2', 
            'new_samples'
        )
        
        if not os.path.exists(new_samples_path):
            print(f"Warning: new_samples directory not found at {new_samples_path}")
            return
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(new_samples_path, '*.csv'))
        print(f"\nProcessing {len(csv_files)} location files...")
        
        new_cache = {}
        processed_count = 0
        
        for csv_file in csv_files:
            # Extract coordinates from filename
            coords = extract_coords_from_filename(csv_file)
            
            if coords:
                lat, lon = coords
                location_key = f"{lat},{lon}"
                
                # Process the file
                result = process_location_file(csv_file, lat, lon)
                
                if result:
                    new_cache[location_key] = result
                    processed_count += 1
                    print(f"✓ Processed {location_key}: Score={result['eat_score_now']:.2f}")
        
        # Update global cache
        _locations_cache = new_cache
        _cache_last_updated = datetime.now()
        
        print(f"\n✅ Successfully processed {processed_count}/{len(csv_files)} locations")
        print(f"Cache updated at: {_cache_last_updated.isoformat()}\n")
        
    except Exception as e:
        print(f"Error in process_all_locations: {str(e)}")

# Background scheduler for automatic updates
scheduler = BackgroundScheduler()

def schedule_background_processing():
    """Start background processing job"""
    # DO NOT process immediately on startup - this blocks port binding
    # Instead, schedule first run after 2 minutes
    print("Scheduling background location processing...")
    
    # Schedule to run every 5 minutes, starting 2 minutes from now
    scheduler.add_job(
        process_all_locations,
        'interval',
        minutes=5,
        id='process_locations',
        replace_existing=True,
        next_run_time=datetime.now() + pd.Timedelta(minutes=2)  # First run in 2 min
    )
    scheduler.start()
    print("Background scheduler started - first run in 2 minutes, then every 5 minutes")

# Shutdown handler
def shutdown_scheduler():
    """Shutdown the scheduler gracefully"""
    if scheduler.running:
        scheduler.shutdown()
        print("Background scheduler stopped")

atexit.register(shutdown_scheduler)

model_manager = ModelManager()

@router.on_event("startup")
async def startup_event():
    """Initialize models and start background processing on startup"""
    print("🚀 Starting up inference service...")
    
    # CRITICAL: Do NOT initialize models at startup on Render
    # This causes port binding timeout. Models load on-demand instead.
    # await model_manager.initialize_models()  # DISABLED for Render deployment
    
    print("✅ Inference service ready (models will load on first request)")
    
    # Start background processing in a separate thread AFTER a delay
    # This allows the port to bind first
    import threading
    def delayed_background_start():
        import time
        time.sleep(30)  # Wait 30 seconds after startup
        schedule_background_processing()
    
    thread = threading.Thread(target=delayed_background_start, daemon=True)
    thread.start()
    print("✅ Background processing will start in 30 seconds")

@router.post("/inference", response_model=DetailedInferenceResponse)
async def perform_inference(coords: Coordinates):
    """
    Perform air quality inference and anomaly detection for given coordinates.
    
    - **latitude**: Latitude coordinate (-90 to 90)
    - **longitude**: Longitude coordinate (-180 to 180)
    
    Returns current and future (30 min) EatSafe scores and anomaly detection results.
    """
    start_time = time.time()
    
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

        # Fetch location's outside pm2.5 values
        today = datetime.today()
        today = today.strftime('%Y-%m-%d %H:%M:%S')

        external_pm25 = pm25_predictor(coords.latitude, coords.longitude, today)
        print(f"The external PM2.5 is {external_pm25}")

        # Load and process data
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
        
        # Normalize scores to 0-100 range for consistency
        # If model returns 0-1 range, multiply by 100
        if eatscore_now <= 1:
            eatscore_now = eatscore_now * 100
        if eatscore_future <= 1:
            eatscore_future = eatscore_future * 100
        
        # Clamp scores to valid range
        eatscore_now = max(0, min(100, eatscore_now))
        eatscore_future = max(0, min(100, eatscore_future))

        # Extract additional information
        current_anomaly_type = str(current_anomaly.get('Label', 'normal'))
        future_anomaly_type = str(future_anomaly.get('Label', 'normal'))
        current_confidence = float(current_anomaly.get('Confidence', 0.0))
        future_confidence = float(future_anomaly.get('Confidence', 0.0))

        processing_time = time.time() - start_time

        return {
            "eat_score_now": float(eatscore_now),
            "eat_score_future": float(eatscore_future),
            "current_anomaly": current_anomaly_type,
            "future_anomaly": future_anomaly_type,
            "current_confidence": current_confidence,
            "future_confidence": future_confidence,
            "external_pm25": float(external_pm25),
            "processing_time": round(processing_time, 3),
            "risk_level_now": get_risk_level(float(eatscore_now)),
            "risk_level_future": get_risk_level(float(eatscore_future)),
            "recommendation": generate_recommendation(
                float(eatscore_now), float(eatscore_future),
                current_anomaly_type, future_anomaly_type
            ),
            "latitude": coords.latitude,
            "longitude": coords.longitude,
            "message": "Inference completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500, 
            detail=f"Error during inference: {str(e)}"
        )

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Check if the inference service is available"""
    return {
        "status": "online",
        "service": "Air Quality Inference API",
        "models": {
            "anomaly_detector": "active",
            "air_quality_predictor": "active", 
            "eatsafe_scorer": "active",
            "pm25_predictor": "active"
        }
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check including model status"""
    try:
        # Test if models are loaded by checking if functions exist
        models_status = {
            "anomaly_detector": _anomaly_detector is not None,
            "air_quality_predictor": _generate_future is not None,
            "eatsafe_scorer": _get_eatscore is not None,
            "pm25_predictor": _pm25_predictor is not None
        }
        
        # Check basic dependencies
        dependencies_status = {
            "pandas": True,
            "numpy": True,
            "pydantic": True
        }
        
        health_result = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": models_status,
            "dependencies": dependencies_status
        }
        
        return health_result
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@router.get("/metrics")
async def get_metrics():
    """Get basic service metrics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "service": "Air Quality Inference API",
        "version": "1.0.0",
        "endpoints": [
            "/inference",
            "/status", 
            "/health",
            "/metrics"
        ],
        "models_loaded": {
            "anomaly_detector": _anomaly_detector is not None,
            "air_quality_predictor": _generate_future is not None,
            "eatsafe_scorer": _get_eatscore is not None,
            "pm25_predictor": _pm25_predictor is not None
        }
    }

# Optional: Add a test endpoint for quick validation
@router.get("/test")
async def test_endpoint():
    """Quick test endpoint to verify API is working"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "message": "Air Quality Inference API is operational"
    }

@router.get("/locations")
async def get_all_locations():
    """
    Get all cached location data.
    Returns a list of all processed locations with their air quality scores.
    """
    if not _locations_cache:
        return {
            "status": "processing",
            "message": "Locations are being processed. Please try again in a few moments.",
            "locations": [],
            "count": 0,
            "last_updated": None
        }
    
    # Convert cache dict to list
    locations_list = list(_locations_cache.values())
    
    return {
        "status": "success",
        "message": "Cached locations retrieved successfully",
        "locations": locations_list,
        "count": len(locations_list),
        "last_updated": _cache_last_updated.isoformat() if _cache_last_updated else None
    }

@router.get("/location/{lat}/{lon}")
async def get_specific_location(lat: float, lon: float):
    """
    Get cached data for a specific location.
    Returns the cached inference result if available, otherwise triggers processing.
    """
    location_key = f"{lat},{lon}"
    
    if location_key in _locations_cache:
        return {
            "status": "success",
            "message": "Location data retrieved from cache",
            "data": _locations_cache[location_key],
            "cached": True
        }
    
    # If not in cache, check if file exists and process it
    new_samples_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'final2', 
        'new_samples',
        f'{lat},{lon}.csv'
    )
    
    if os.path.exists(new_samples_path):
        # Process this specific file
        result = process_location_file(new_samples_path, lat, lon)
        if result:
            # Add to cache
            _locations_cache[location_key] = result
            return {
                "status": "success",
                "message": "Location processed and cached",
                "data": result,
                "cached": False
            }
    
    # Not found
    return {
        "status": "not_found",
        "message": f"No data available for location {lat},{lon}",
        "data": None,
        "cached": False
    }

@router.get("/cache/status")
async def get_cache_status():
    """
    Get the current cache status.
    """
    return {
        "status": "ok",
        "locations_cached": len(_locations_cache),
        "last_updated": _cache_last_updated.isoformat() if _cache_last_updated else None,
        "next_update": "Updates every 5 minutes",
        "scheduler_running": scheduler.running if scheduler else False
    }

@router.post("/cache/refresh")
async def refresh_cache():
    """
    Manually trigger a cache refresh.
    Useful for testing or immediate updates.
    """
    try:
        import threading
        thread = threading.Thread(target=process_all_locations, daemon=True)
        thread.start()
        
        return {
            "status": "triggered",
            "message": "Cache refresh started in background",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering refresh: {str(e)}")