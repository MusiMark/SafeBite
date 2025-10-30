# test_predictions.py
# Test the trained model with multiple predictions

import sys
import os

# Import required classes first
from src.final2.pm25_predictor.train_model import OptimizedPM25Predictor, OptimizedGNNPredictor, OptimizedKGBuilder
from load_and_predict import load_model, predict_pm25
import pandas as pd

def test_multiple_predictions():
    """Test the model with multiple prediction scenarios"""
    
    print("=" * 70)
    print(" " * 20 + "PM2.5 Prediction Model Testing")
    print("=" * 70)
    
    # Load the model once
    model_path = "pm25_gnn_complete_complete_model.pkl"
    predictor = load_model(model_path)
    
    # Test cases: (latitude, longitude, datetime, description)
    test_cases = [
        # Kampala area at different times
        (0.3161, 32.5924, "2025-03-11 06:00:00", "Early Morning - Kampala"),
        (0.3161, 32.5924, "2025-03-11 08:00:00", "Morning Rush - Kampala"),
        (0.3161, 32.5924, "2025-03-11 12:00:00", "Noon - Kampala"),
        (0.3161, 32.5924, "2025-03-11 18:00:00", "Evening Rush - Kampala"),
        (0.3161, 32.5924, "2025-03-11 22:00:00", "Late Night - Kampala"),
        
        # Different locations
        (0.3000, 32.6000, "2025-03-11 10:00:00", "Location 2 - Morning"),
        (0.3500, 32.5500, "2025-03-11 10:00:00", "Location 3 - Morning"),
        (0.2800, 32.5800, "2025-03-11 14:00:00", "Location 4 - Afternoon"),
        
        # Different dates
        (0.3161, 32.5924, "2025-01-15 10:00:00", "January - Kampala"),
        (0.3161, 32.5924, "2025-06-15 10:00:00", "June - Kampala"),
    ]
    
    print("\n📊 Prediction Results:")
    print("-" * 70)
    print(f"{'Description':<30} {'DateTime':<20} {'PM2.5 (µg/m³)':>12}")
    print("-" * 70)
    
    results = []
    for lat, lon, dt_str, description in test_cases:
        pm25_value = predict_pm25(predictor, lat, lon, dt_str)
        results.append({
            'Latitude': lat,
            'Longitude': lon,
            'DateTime': dt_str,
            'Description': description,
            'PM2.5': pm25_value
        })
        
        # Color code based on air quality
        if pm25_value <= 12:
            quality = "🟢 Good"
        elif pm25_value <= 35:
            quality = "🟡 Moderate"
        elif pm25_value <= 55:
            quality = "🟠 Unhealthy (sensitive)"
        else:
            quality = "🔴 Unhealthy"
        
        print(f"{description:<30} {dt_str:<20} {pm25_value:>10.3f}  {quality}")
    
    print("-" * 70)
    
    # Convert to DataFrame for better statistics
    df_results = pd.DataFrame(results)
    
    print("\n� Summary Statistics:")
    print(f"   Mean PM2.5:      {df_results['PM2.5'].mean():8.3f} µg/m³")
    print(f"   Median PM2.5:    {df_results['PM2.5'].median():8.3f} µg/m³")
    print(f"   Min PM2.5:       {df_results['PM2.5'].min():8.3f} µg/m³")
    print(f"   Max PM2.5:       {df_results['PM2.5'].max():8.3f} µg/m³")
    print(f"   Std Dev:         {df_results['PM2.5'].std():8.3f} µg/m³")
    
    print("\n🏥 WHO Air Quality Guidelines:")
    print("   Good:                    0-12 µg/m³")
    print("   Moderate:               12-35 µg/m³")
    print("   Unhealthy (sensitive):  35-55 µg/m³")
    print("   Unhealthy:              55+ µg/m³")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_multiple_predictions()
