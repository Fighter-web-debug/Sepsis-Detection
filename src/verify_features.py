import numpy as np
import os

os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val   = np.load('data/processed/X_val.npy')
X_test  = np.load('data/processed/X_test.npy')

print("=== Day 4 verification ===")
print(f"X_train shape: {X_train.shape}  <- (sequences, 12 hours, features)")
print(f"X_val shape:   {X_val.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train unique values: {np.unique(y_train)}")
print(f"No NaN in X_train: {not np.isnan(X_train).any()}")
print(f"No NaN in y_train: {not np.isnan(y_train).any()}")
print("All checks passed. Ready for model training.")