import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

data = pd.read_parquet('data/processed/normalized.parquet')

ALL_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
                'Creatinine', 'Lactate', 'WBC', 'Glucose',
                'Potassium', 'Hgb', 'Age', 'ICULOS',
                'HR_roll_mean', 'O2Sat_roll_mean', 'SBP_roll_mean', 'Resp_roll_mean',
                'HR_roll_std', 'O2Sat_roll_std', 'SBP_roll_std', 'Resp_roll_std']

WINDOW = 12
PRED_GAP = 6

X_list, y_list = [], []
patients = data['patient_id'].unique()

print(f"Building sequences for {len(patients):,} patients...")

for i, pid in enumerate(patients):
    if i % 5000 == 0:
        print(f"  {i:,}/{len(patients):,}")

    pt = data[data['patient_id'] == pid].reset_index(drop=True)
    n = len(pt)

    if n < WINDOW + PRED_GAP:
        continue

    feats = pt[ALL_FEATURES].values
    labels = pt['SepsisLabel'].values

    for t in range(WINDOW, n - PRED_GAP + 1):
        window = feats[t-WINDOW:t]
        future_labels = labels[t:t+PRED_GAP]
        label = 1 if future_labels.max() == 1 else 0
        X_list.append(window)
        y_list.append(label)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"\nX shape: {X.shape}  (sequences, timesteps, features)")
print(f"y shape: {y.shape}")
print(f"Positive (sepsis) rate: {y.mean()*100:.2f}%")

np.save('data/processed/X.npy', X)
np.save('data/processed/y.npy', y)
print("Saved X.npy and y.npy")