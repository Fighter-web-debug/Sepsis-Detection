import os
import sys

import numpy as np
import plt
import torch

sys.path.append('src')
os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

from model import SepsisLSTM
from dataset import SepsisDataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report, f1_score
)
from sklearn.metrics import confusion_matrix as conf_matrix

X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')
print(f"Test set: {X_test.shape} | Sepsis rate: {y_test.mean()*100:.2f}%")

model = SepsisLSTM(input_size=X_test.shape[2])
model.load_state_dict(
    torch.load('models/best_model.pt', map_location='cpu')
)
model.eval()
print("Model loaded successfully")

ds     = SepsisDataset(X_test, y_test)
loader = DataLoader(ds, batch_size=512, shuffle=False)
preds  = []

with torch.no_grad():
    for Xb, _ in loader:
        logits = model(Xb)
        probs  = torch.sigmoid(logits)
        preds.extend(probs.numpy())

preds  = np.array(preds)
print(f"Predictions: min={preds.min():.4f} max={preds.max():.4f} mean={preds.mean():.4f}")

auroc = roc_auc_score(y_test, preds)
auprc = average_precision_score(y_test, preds)
print(f"\nAUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")

np.save('data/processed/test_preds.npy', preds)

thresholds = np.arange(0.05, 0.95, 0.01)
f1s        = [f1_score(y_test, preds >= t, zero_division=0)
              for t in thresholds]
best_t     = thresholds[np.argmax(f1s)]
best_f1    = max(f1s)

print(f"\nBest threshold: {best_t:.2f}")
print(f"Best F1 score:  {best_f1:.4f}")
print(f"\nClassification report at threshold {best_t:.2f}:")
print(classification_report(
    y_test, preds >= best_t,
    target_names=['No sepsis', 'Sepsis'],
    zero_division=0
))

print("\nThreshold comparison:")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<10} {'F1':<8}")
print("-" * 44)
for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.92]:
    from sklearn.metrics import precision_score, recall_score
    p = precision_score(y_test, preds >= t, zero_division=0)
    r = recall_score(y_test, preds >= t, zero_division=0)
    f = f1_score(y_test, preds >= t, zero_division=0)
    print(f"{t:<12.2f} {p:<12.3f} {r:<10.3f} {f:<8.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fpr, tpr, _ = roc_curve(y_test, preds)
ax = axes[0]
ax.plot(fpr, tpr, color='#534AB7', lw=2,
        label=f'LSTM (AUROC={auroc:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random')
ax.fill_between(fpr, tpr, alpha=0.08, color='#534AB7')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('ROC curve', fontweight='normal')

prec, rec, _ = precision_recall_curve(y_test, preds)
ax = axes[1]
ax.plot(rec, prec, color='#D85A30', lw=2,
        label=f'LSTM (AUPRC={auprc:.3f})')
ax.axhline(y_test.mean(), color='k', linestyle='--',
           lw=1, alpha=0.5, label=f'Random ({y_test.mean():.3f})')
ax.fill_between(rec, prec, alpha=0.08, color='#D85A30')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Precision-recall curve', fontweight='normal')

plt.tight_layout()
plt.savefig('notebooks/plot8_roc_pr.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot8_roc_pr.png")

# noinspection PyRedeclaration
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = conf_matrix(y_test, (preds >= best_t).astype(int))
cm = cm.astype(int)
ax = axes[0]
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred: No sepsis','Pred: Sepsis'])
ax.set_yticklabels(['True: No sepsis','True: Sepsis'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}',
                ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else '#444441',
                fontsize=13, fontweight='500')
ax.set_title(f'Confusion matrix (threshold={best_t:.2f})',
             fontweight='normal')

ax = axes[1]
sep_scores = preds[y_test == 1]
nor_scores = preds[y_test == 0]
ax.hist(nor_scores, bins=60, alpha=0.6, color='#534AB7',
        label='No sepsis', density=True)
ax.hist(sep_scores, bins=60, alpha=0.7, color='#D85A30',
        label='Sepsis', density=True)
ax.axvline(best_t, color='#085041', lw=2, linestyle='--',
           label=f'Threshold={best_t:.2f}')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Density')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Score distribution', fontweight='normal')

plt.tight_layout()
plt.savefig('notebooks/plot9_confusion_scores.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot9_confusion_scores.png")