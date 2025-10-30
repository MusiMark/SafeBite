# filepath: ai-model-web-app/src/final2/main.py
#imports
import pandas as pd

from anomaly_detector.infer_tgat import anomaly_detector
from data_processing import data_preprocessing, get_latest_row
from safeBite_score.load_and_predict import get_eatscore
from inside_air_predictor.infer_iterative import generate_future


# 1. First get coordinates from user
# coords = input("Enter coordinates (latitude,longitude): ")
# latitude, longitude = map(float, coords.split(','))

##
#  ==================================
# 2. Fetch location's outside pm2.5 values
# ==================================
##

##
#  ===================================
# 3.Check database for the location's inside air quality data
# ===================================
# TODO: Link to database
# fetch data for the given coordinates and return inside air quality of the last 24 hours
##
original_df = pd.read_csv("D:/AI_project/final2/inside_air_predictor/sample.csv")


##
#  ==================================
# 4. Get Future Air Data (30 min forecast)
# ==================================
##
future_df = generate_future(original_df)


##
#  ==================================
# 5. Inside air quality dataset processing
# ==================================
##
original_df = data_preprocessing(original_df)
future_df = data_preprocessing(future_df)


##
#  ==================================
# 6. Anomaly Detection
# ==================================
##

# Reset index to make 'ts' a column again
future_df = future_df.reset_index()
original_df = original_df.reset_index()

original_df = anomaly_detector(original_df)
future_df = anomaly_detector(future_df)


future_anomaly = get_latest_row(future_df, 'ts')
current_anomaly = get_latest_row(original_df, 'ts')


##
#  ==================================
# 7. EatSafe Score Calculation
# ==================================
##

eatsofe_now = get_eatscore(current_anomaly)
eatsofe_future = get_eatscore(future_anomaly)

print(f"This is the eat score 30 minutes from now {eatsofe_future}")
print(f"This is the eat score now {eatsofe_now}")