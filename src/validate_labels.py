import numpy as np
import os

os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

y_train = np.load('data/processed/y_train.npy')
y_val   = np.load('data/processed/y_val.npy')
y_test  = np.load('data/processed/y_test.npy')

print(f"Train sepsis rate: {y_train.mean()*100:.2f}%")
print(f"Val sepsis rate:   {y_val.mean()*100:.2f}%")
print(f"Test sepsis rate:  {y_test.mean()*100:.2f}%")

total = len(y_train) + len(y_val) + len(y_test)
print(f"\nTotal sequences: {total:,}")
print(f"Total positive:  {int(y_train.sum()+y_val.sum()+y_test.sum()):,}")

assert y_train.mean() > 0.01, "ERROR: Less than 1% sepsis in train — something is wrong"
assert y_train.mean() < 0.20, "ERROR: More than 20% sepsis — labels may be corrupted"
print("\nAll label checks passed.")