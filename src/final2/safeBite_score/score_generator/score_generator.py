import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (change file path to yours)
df = pd.read_csv('labeled_sensor_data_sensoronly.csv', parse_dates=['ts'])

# Label penalty dictionary
LABEL_PENALTY = {
    'normal': 0,
    'dishwashing': 5,
    'cooking_spill': 10,
    'possible_gas_or_chemical_spill': 20,
    'smoke': 25,
    'fire': 40
}

def calculate_eatsafe(row):
    # --- CO2 Score ---
    co2 = row['CO2']
    if co2 < 800:
        s_co2 = 100
    elif co2 < 1000:
        s_co2 = 80
    elif co2 < 1500:
        s_co2 = 50
    else:
        s_co2 = 20

    # --- PM2.5 Score ---
    pm = row['pm25_mean_2min']
    if pm < 12:
        s_pm = 100
    elif pm < 35:
        s_pm = 80
    elif pm < 55:
        s_pm = 60
    elif pm < 150:
        s_pm = 30
    else:
        s_pm = 10

    # --- VOC Score ---
    voc = row['VoC']
    s_voc = max(0, 100 - (voc - 200) * 0.1) if voc > 200 else 100

    # --- Comfort Score (Temperature + Humidity) ---
    t, h = row['T'], row['H']
    s_comfort = 100
    if not (20 <= t <= 26):
        s_comfort -= 10
    if not (40 <= h <= 60):
        s_comfort -= 10

    # --- Penalty based on Label ---
    label = str(row['Label']).strip()
    penalty = LABEL_PENALTY.get(label, 0)  # Default penalty = 0 if not found

    # --- Final Weighted Score ---
    score = (
        0.3 * s_co2 +
        0.3 * s_pm +
        0.2 * s_voc +
        0.1 * s_comfort +
        0.1 * (100 - penalty)
    )
    return max(0, min(100, score))

# Apply EatSafe score
df['EatSafe_Score'] = df.apply(calculate_eatsafe, axis=1)

# Plotting EatSafe over time
plt.figure(figsize=(10,5))
plt.plot(df['ts'], df['EatSafe_Score'], linewidth=2)
plt.title("EatSafe Score Over Time")
plt.xlabel("Timestamp")
plt.ylabel("EatSafe Score (0-100)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Optional: save results
df.to_csv('eatsafe_scored_data.csv', index=False)
print("✅ EatSafe scores calculated & saved to eatsafe_scored_data.csv")
