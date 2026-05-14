import os
import sys

import numpy as np
import shap
import torch

sys.path.append('src')
os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

from model import SepsisLSTM

FEATURE_NAMES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
    'Creatinine', 'Lactate', 'WBC', 'Glucose',
    'Potassium', 'Hgb', 'Age', 'ICULOS',
    'HR_roll_mean', 'O2Sat_roll_mean',
    'SBP_roll_mean', 'Resp_roll_mean',
    'HR_roll_std', 'O2Sat_roll_std',
    'SBP_roll_std', 'Resp_roll_std',
]

X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

model = SepsisLSTM(input_size=X_test.shape[2])
model.load_state_dict(
    torch.load('models/best_model.pt', map_location='cpu')
)
model.eval()

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.unsqueeze(1)

wrapped_model = ModelWrapper(model)
wrapped_model.eval()

sepsis_idx = np.where(y_test == 1)[0][:100]
normal_idx = np.where(y_test == 0)[0][:100]
sample_idx = np.concatenate([sepsis_idx, normal_idx])

X_sample    = torch.tensor(X_test[sample_idx], dtype=torch.float32)
X_background = torch.tensor(X_test[:50],       dtype=torch.float32)

print(f"Computing SHAP values for {len(X_sample)} samples...")
print("This takes ~5 minutes on CPU...")

explainer = shap.GradientExplainer(wrapped_model, X_background)

all_shap = []
for i in range(len(X_sample)):
    if i % 20 == 0:
        print(f"  Sample {i}/{len(X_sample)}...")
    sv = explainer.shap_values(X_sample[i:i+1])
    all_shap.append(np.array(sv[0]))

shap_arr = np.concatenate(all_shap, axis=0)
shap_arr = shap_arr.squeeze(-1)
shap_arr = shap_arr.reshape(200, 12, 21)
print(f"Final shap_arr shape: {shap_arr.shape}")

np.save('data/processed/shap_values.npy',  shap_arr)
np.save('data/processed/shap_samples.npy', X_test[sample_idx])
np.save('data/processed/shap_labels.npy',  y_test[sample_idx])
print("Saved SHAP values to data/processed/")