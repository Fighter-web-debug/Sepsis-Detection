import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

data = pd.read_parquet('data/processed/features.parquet')

ALL_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
                'Creatinine', 'Lactate', 'WBC', 'Glucose',
                'Potassium', 'Hgb', 'Age', 'ICULOS',
                'HR_roll_mean', 'O2Sat_roll_mean', 'SBP_roll_mean', 'Resp_roll_mean',
                'HR_roll_std', 'O2Sat_roll_std', 'SBP_roll_std', 'Resp_roll_std']

data[ALL_FEATURES] = data[ALL_FEATURES].fillna(data[ALL_FEATURES].median())

scaler = StandardScaler()
data[ALL_FEATURES] = scaler.fit_transform(data[ALL_FEATURES])

with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

data.to_parquet('data/processed/normalized.parquet', index=False)
print("Scaler saved to data/processed/scaler.pkl")
print("Normalized data saved to data/processed/normalized.parquet")
print(f"Feature means (should be ~0): {data[ALL_FEATURES].mean().round(3).values[:4]}")
print(f"Feature stds  (should be ~1): {data[ALL_FEATURES].std().round(3).values[:4]}")