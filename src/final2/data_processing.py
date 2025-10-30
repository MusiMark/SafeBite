import numpy as np
import pandas as pd

def data_preprocessing(df):
    # Step 1: Convert the 'ts' column to datetime (if not already)
    df['ts'] = pd.to_datetime(df['ts'])

    # Step 2: Sort by the 'ts' column
    df = df.sort_values("ts").reset_index(drop=True)

    # drop irrelevant columns if present
    for col in ['ID', 'Loc', 'Customer', 'Ph']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ensure numeric columns exist
    numeric_cols = ['T', 'H', 'PMS1', 'PMS2_5', 'PMS10', 'CO2', 'NO2', 'CO', 'VoC', 'C2H5OH']
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # set datetime index for rolling time windows
    df = df.set_index('ts')

    # ---------- 2) compute rolling / temporal features ----------
    # 2-minute and 1-minute windows
    df['pm25_max_2min'] = df['PMS2_5'].rolling('2min').max()
    df['pm25_mean_2min'] = df['PMS2_5'].rolling('2min').mean()

    df['co2_max_2min'] = df['CO2'].rolling('2min').max()
    df['co2_mean_1min'] = df['CO2'].rolling('1min').mean()
    df['co2_rise_1to2min'] = df['CO2'].rolling('2min').apply(
        lambda x: np.nan if len(x) == 0 else (np.nanmax(x) - np.nanmin(x)), raw=True)

    # Temperature rapid rise (1 min)
    df['T_rise_1min'] = df['T'].rolling('1min').apply(
        lambda x: np.nan if len(x) == 0 else (np.nanmax(x) - np.nanmin(x)), raw=True)

    # PM10 spike detection: sudden jump vs previous 30s mean
    df['pm10_mean_30s'] = df['PMS10'].rolling('30s').mean()
    df['pm10_spike'] = df['PMS10'] - df['pm10_mean_30s'].shift(1)

    # Humidity rolling average
    df['H_mean_2min'] = df['H'].rolling('2min').mean()

    # VOC and ethanol short-term means
    df['voc_mean_1min'] = df['VoC'].rolling('1min').mean()
    df['ethanol_mean_1min'] = df['C2H5OH'].rolling('1min').mean()


    return df


import pandas as pd


def get_latest_row(df, date_column):

    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Find and return the row with the latest date
    latest_row = df.loc[df[date_column].idxmax()]

    return latest_row
