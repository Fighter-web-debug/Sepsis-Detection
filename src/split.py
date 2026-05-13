import os

import numpy as np
from sklearn.model_selection import train_test_split

os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')

print(f"Loaded X: {X.shape}, y: {y.shape}")

n = len(X)
idx = np.arange(n)

train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

np.save('data/processed/X_train.npy', X[train_idx])
np.save('data/processed/X_val.npy',   X[val_idx])
np.save('data/processed/X_test.npy',  X[test_idx])
np.save('data/processed/y_train.npy', y[train_idx])
np.save('data/processed/y_val.npy',   y[val_idx])
np.save('data/processed/y_test.npy',  y[test_idx])

print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
print(f"Train sepsis rate: {y[train_idx].mean()*100:.2f}%")
print(f"Val sepsis rate:   {y[val_idx].mean()*100:.2f}%")
print(f"Test sepsis rate:  {y[test_idx].mean()*100:.2f}%")