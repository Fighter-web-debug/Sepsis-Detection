import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

data = pd.read_parquet('data/processed/clean.parquet')

DROP_COLS = [
    'Bilirubin_direct', 'Fibrinogen', 'TroponinI',
    'Bilirubin_total', 'Alkalinephos', 'AST',
    'EtCO2', 'SaO2', 'PTT'
]
data = data.drop(columns=[c for c in DROP_COLS if c in data.columns])

FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
            'Creatinine', 'Lactate', 'WBC', 'Glucose',
            'Potassium', 'Hgb', 'Age', 'ICULOS']

for col in ['HR', 'O2Sat', 'SBP', 'Resp']:
    data[f'{col}_roll_mean'] = (
        data.groupby('patient_id')[col]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )
    data[f'{col}_roll_std'] = (
        data.groupby('patient_id')[col]
        .transform(lambda x: x.rolling(4, min_periods=1).std().fillna(0))
    )

ROLLING_FEATURES = [f'{c}_roll_mean' for c in ['HR','O2Sat','SBP','Resp']] + \
                   [f'{c}_roll_std'  for c in ['HR','O2Sat','SBP','Resp']]

ALL_FEATURES = FEATURES + ROLLING_FEATURES
print(f"Total features: {len(ALL_FEATURES)}")
print(ALL_FEATURES)

data.to_parquet('data/processed/features.parquet', index=False)
print("Saved to data/processed/features.parquet")