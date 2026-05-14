import numpy as np
import os
import sys
import torch

sys.path.append('src')
os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

from model import SepsisLSTM
from dataset import make_loaders
import torch.nn as nn

X_train = np.load('data/processed/X_train.npy')[:2000]
y_train = np.load('data/processed/y_train.npy')[:2000]
X_val   = np.load('data/processed/X_val.npy')[:500]
y_val   = np.load('data/processed/y_val.npy')[:500]

print(f"Mini-train: {X_train.shape} | Mini-val: {X_val.shape}")

train_loader, val_loader = make_loaders(
    X_train, y_train, X_val, y_val, batch_size=64
)

DEVICE = torch.device('cpu')
model  = SepsisLSTM(input_size=X_train.shape[2]).to(DEVICE)
pos_w  = torch.tensor([46.6])
crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 4):
    model.train()
    tr = 0
    for Xb, yb in train_loader:
        opt.zero_grad()
        loss = crit(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr += loss.item()
    model.eval()
    vl = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            vl += crit(model(Xb), yb).item()
    print(f"Epoch {epoch} | train={tr/len(train_loader):.4f} "
          f"| val={vl/len(val_loader):.4f}")

print("\nLocal test passed. Ready for Colab.")