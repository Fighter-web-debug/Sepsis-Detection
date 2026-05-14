import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_test  = np.load('data/processed/X_test.npy')
y_test  = np.load('data/processed/y_test.npy')

X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat  = X_test.reshape(len(X_test),  -1)
print(f"Flattened shape: {X_train_flat.shape}")

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                         random_state=42, n_jobs=-1)
lr.fit(X_train_flat, y_train)
lr_preds = lr.predict_proba(X_test_flat)[:,1]
lr_auroc = roc_auc_score(y_test, lr_preds)
lr_auprc = average_precision_score(y_test, lr_preds)
print(f"LR  AUROC={lr_auroc:.4f}  AUPRC={lr_auprc:.4f}")

print("\nTraining Random Forest (this takes ~5 min)...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                             random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X_train_flat, y_train)
rf_preds = rf.predict_proba(X_test_flat)[:,1]
rf_auroc = roc_auc_score(y_test, rf_preds)
rf_auprc = average_precision_score(y_test, rf_preds)
print(f"RF  AUROC={rf_auroc:.4f}  AUPRC={rf_auprc:.4f}")

lstm_preds = np.load('data/processed/test_preds.npy')
lstm_auroc = roc_auc_score(y_test, lstm_preds)
lstm_auprc = average_precision_score(y_test, lstm_preds)
print(f"LSTM AUROC={lstm_auroc:.4f}  AUPRC={lstm_auprc:.4f}")

print("\n=== Summary ===")
print(f"{'Model':<20} {'AUROC':>8} {'AUPRC':>8}")
print("-" * 38)
print(f"{'Logistic Regression':<20} {lr_auroc:>8.4f} {lr_auprc:>8.4f}")
print(f"{'Random Forest':<20} {rf_auroc:>8.4f} {rf_auprc:>8.4f}")
print(f"{'LSTM (ours)':<20} {lstm_auroc:>8.4f} {lstm_auprc:>8.4f}")