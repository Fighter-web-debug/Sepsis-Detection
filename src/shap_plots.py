import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

FEATURE_NAMES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
    'Creatinine', 'Lactate', 'WBC', 'Glucose',
    'Potassium', 'Hgb', 'Age', 'ICULOS',
    'HR_roll_mean', 'O2Sat_roll_mean',
    'SBP_roll_mean', 'Resp_roll_mean',
    'HR_roll_std', 'O2Sat_roll_std',
    'SBP_roll_std', 'Resp_roll_std',
]

shap_arr = np.load('data/processed/shap_values.npy')
if shap_arr.shape[0] == 12:
    shap_arr = shap_arr.transpose(1, 0, 2)
print(f"SHAP array shape after fix: {shap_arr.shape}")

X_sample = np.load('data/processed/shap_samples.npy')
labels   = np.load('data/processed/shap_labels.npy')

mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))
feat_imp = sorted(zip(FEATURE_NAMES, mean_abs_shap),
                  key=lambda x: x[1], reverse=True)
names, vals = zip(*feat_imp)

colors = []
for n in names:
    if n in ['HR','O2Sat','Temp','SBP','MAP','Resp']:
        colors.append('#534AB7')
    elif 'roll' in n:
        colors.append('#AFA9EC')
    else:
        colors.append('#D85A30')

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(range(len(names)), vals,
               color=colors, edgecolor='none')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=12)
ax.set_xlabel('Mean |SHAP value|', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

p1 = mpatches.Patch(color='#534AB7', label='Vitals')
p2 = mpatches.Patch(color='#AFA9EC', label='Rolling stats')
p3 = mpatches.Patch(color='#D85A30', label='Labs / demographics')
ax.legend(handles=[p1,p2,p3], frameon=False, fontsize=11)

plt.title('Global feature importance — mean |SHAP value|',
          fontweight='normal', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/plot10_shap_global.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot10_shap_global.png")

sepsis_sample_idx = np.where(labels == 1)[0][0]
patient_shap = shap_arr[sepsis_sample_idx]
patient_feat = X_sample[sepsis_sample_idx]

mean_shap_per_feat = patient_shap.mean(axis=0)
feat_pairs = sorted(zip(FEATURE_NAMES, mean_shap_per_feat),
                     key=lambda x: abs(x[1]), reverse=True)[:12]
f_names, f_vals = zip(*feat_pairs)

colors_wp = ['#D85A30' if v > 0 else '#534AB7' for v in f_vals]

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(range(len(f_names)), f_vals,
        color=colors_wp, edgecolor='none')
ax.set_yticks(range(len(f_names)))
ax.set_yticklabels(f_names, fontsize=12)
ax.axvline(0, color='.3', lw=0.8)
ax.set_xlabel('SHAP value (positive = increases sepsis risk)',
              fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

r_patch = mpatches.Patch(color='#D85A30', label='Increases risk')
b_patch = mpatches.Patch(color='#534AB7', label='Decreases risk')
ax.legend(handles=[r_patch, b_patch], frameon=False, fontsize=11)

plt.title('Single patient explanation — sepsis case',
          fontweight='normal', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/plot11_shap_patient.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot11_shap_patient.png")

sepsis_shap = shap_arr[labels == 1]

mean_over_patients = np.abs(sepsis_shap).mean(axis=0)

top_feat_idx = np.argsort(
    mean_over_patients.mean(axis=0)
)[-10:][::-1]
top_names = [FEATURE_NAMES[i] for i in top_feat_idx]
heatmap_data = mean_over_patients[:, top_feat_idx].T

# noinspection PyRedeclaration
fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(heatmap_data, aspect='auto',
               cmap='YlOrRd', interpolation='nearest')

ax.set_xticks(range(12))
ax.set_xticklabels([f'h-{12-i}' for i in range(12)],
                   fontsize=11)
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names, fontsize=11)
ax.set_xlabel('Hours before prediction (h-12 = oldest)',
              fontsize=11)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Mean |SHAP value|', fontsize=10)

plt.title('Temporal attribution — which hours matter most (sepsis cases)',
          fontweight='normal', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/plot12_shap_temporal.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot12_shap_temporal.png")