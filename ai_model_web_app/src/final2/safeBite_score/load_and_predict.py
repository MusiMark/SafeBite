from fastapi import HTTPException
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# Paths (adjust if needed)
def get_file_path(filename):
    path = os.path.join(os.path.dirname(__file__), '..', 'safeBite_score', filename)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=500, 
            detail=f"Sample data file not found at {path}"
        )
    return path

MODEL_PATH = get_file_path('eatsafe_model.keras')
PREPROCESSOR_PATH = get_file_path('preprocessor.pkl')


# Feature order must match training script
FEATURES = [
    'T','H','PMS1','PMS2_5','PMS10','CO2','NO2','CO','VoC','C2H5OH',
    'pm25_max_2min','pm25_mean_2min','co2_max_2min','co2_mean_1min',
    'co2_rise_1to2min','T_rise_1min','pm10_mean_30s','pm10_spike',
    'H_mean_2min','voc_mean_1min','ethanol_mean_1min','Label','Confidence'
]
NUMERIC_FEATURES = [
    'T','H','PMS1','PMS2_5','PMS10','CO2','NO2','CO','VoC','C2H5OH',
    'pm25_max_2min','pm25_mean_2min','co2_max_2min','co2_mean_1min',
    'co2_rise_1to2min','T_rise_1min','pm10_mean_30s','pm10_spike',
    'H_mean_2min','voc_mean_1min','ethanol_mean_1min','Confidence'
]
CATEGORICAL_FEATURES = ['Label']

# Load artifacts
model = tf.keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

def predict_eatsafe(input_dict):
    """
    Predict EatSafe_Score from a dict of feature values.
    Steps:
      1) Convert to pandas DataFrame with a single row.
      2) Apply preprocessor.transform
      3) Predict using loaded model
      4) Return a single float rounded to 2 decimals
    """
    # Ensure all required features are present
    missing = [f for f in FEATURES if f not in input_dict]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Build single-row DataFrame; enforce dtypes (float32 for numeric, category for Label)
    data = {k: [input_dict[k]] for k in FEATURES}
    df = pd.DataFrame(data)

    # Numeric to float32
    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype('float32')

    # Categorical
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype('category')

    # Transform
    X_proc = preprocessor.transform(df).astype('float32', copy=False)

    # Predict
    pred = model.predict(X_proc, verbose=0)
    value = float(pred[0, 0])

    # Round to 2 decimals
    return round(value, 2)

# if __name__ == '__main__':
def get_eatscore(row):

    # example = {
    #     'T': 29.3, 'H': 51.7, 'PMS1': 49, 'PMS2_5': 83, 'PMS10': 86,
    #     'CO2': 620, 'NO2': 368, 'CO': 195, 'VoC': 570, 'C2H5OH': 430,
    #     'pm25_max_2min': 83.0, 'pm25_mean_2min': 83.0, 'co2_max_2min': 620.0,
    #     'co2_mean_1min': 619.5, 'co2_rise_1to2min': 1.0, 'T_rise_1min': 0.0,
    #     'pm10_mean_30s': 86.0, 'pm10_spike': 0.0, 'H_mean_2min': 51.7,
    #     'voc_mean_1min': 571.0, 'ethanol_mean_1min': 430.5,
    #     'Label': 'normal', 'Confidence': 0.15
    # }

    example = {
        'T': row['T'], 'H': row['H'], 'PMS1': row['PMS1'], 'PMS2_5': row['PMS2_5'], 'PMS10': row['PMS10'],
        'CO2': row['CO2'], 'NO2': row['NO2'], 'CO': row['CO'], 'VoC': row['VoC'], 'C2H5OH': row['C2H5OH'],
        'pm25_max_2min': row['pm25_max_2min'], 'pm25_mean_2min': row['pm25_mean_2min'],
        'co2_max_2min': row['co2_max_2min'], 'co2_mean_1min': row['co2_mean_1min'], 'co2_rise_1to2min': row['co2_rise_1to2min'],
        'T_rise_1min': row['T_rise_1min'], 'pm10_mean_30s': row['pm10_mean_30s'], 'pm10_spike': row['pm10_spike'],
        'H_mean_2min': row['H_mean_2min'], 'voc_mean_1min': row['voc_mean_1min'], 'ethanol_mean_1min': row['ethanol_mean_1min'],
        'Label': row['Label'], 'Confidence': row['Confidence']
    }

    prediction = predict_eatsafe(example)
    # print(f"Predicted EatSafe_Score: {prediction}")

    return prediction

