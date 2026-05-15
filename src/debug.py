import numpy as np
import os
os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

shap_arr = np.load('data/processed/shap_values.npy')
labels   = np.load('data/processed/shap_labels.npy')

print(f"shap_arr shape: {shap_arr.shape}")
print(f"labels shape:   {labels.shape}")