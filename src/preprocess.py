import pandas as pd
import os

os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

data = pd.read_parquet('data/processed/combined_raw.parquet')

data = data.sort_values(['patient_id', 'ICULOS']).reset_index(drop=True)

data['hour'] = data.groupby('patient_id').cumcount()

vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
data[vitals] = data.groupby('patient_id')[vitals].transform(
    lambda x: x.ffill().bfill()
)

min_hours = 6
valid_patients = data.groupby('patient_id').size()
valid_patients = valid_patients[valid_patients >= min_hours].index
data = data[data['patient_id'].isin(valid_patients)]

print(f"Patients after filtering: {data['patient_id'].nunique():,}")
print(f"Total rows: {len(data):,}")
print(f"Sepsis patients: {data.groupby('patient_id')['SepsisLabel'].max().sum():.0f}")

data.to_parquet('data/processed/clean.parquet', index=False)
print("Saved to data/processed/clean.parquet")