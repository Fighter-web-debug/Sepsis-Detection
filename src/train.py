import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os, sys, time
sys.path.append('src')

BASE = r'C:\Users\KIIT0001\sepsis-detection'
os.chdir(BASE)

from model import SepsisLSTM, count_parameters
from dataset import make_loaders

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 1e-3
POS_WEIGHT = 46.6
PATIENCE   = 5
MODEL_PATH = 'models/best_model.pt'

print(f"Device: {DEVICE}")
print(f"Loading data...")

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val   = np.load('data/processed/X_val.npy')
y_val   = np.load('data/processed/y_val.npy')

print(f"Train: {X_train.shape} | Val: {X_val.shape}")

train_loader, val_loader = make_loaders(
    X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE
)

model     = SepsisLSTM(input_size=X_train.shape[2]).to(DEVICE)
pos_w     = torch.tensor([POS_WEIGHT]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                              weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=3
)

print(f"Parameters: {count_parameters(model):,}")
print(f"Starting training for {EPOCHS} epochs...\n")

train_losses, val_losses = [], []
best_val   = float('inf')
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    model.train()
    tr_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)

    model.eval()
    vl_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            vl_loss += criterion(model(Xb), yb).item()
    vl_loss /= len(val_loader)

    scheduler.step(vl_loss)
    train_losses.append(tr_loss)
    val_losses.append(vl_loss)

    elapsed = time.time() - t0
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train={tr_loss:.4f} | val={vl_loss:.4f} | "
          f"{elapsed:.1f}s")

    if vl_loss < best_val:
        best_val   = vl_loss
        no_improve = 0
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  Saved best model (val={best_val:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label='Train loss', color='#534AB7')
ax.plot(val_losses,   label='Val loss',   color='#D85A30')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Training curve', fontweight='normal')
plt.tight_layout()
plt.savefig('notebooks/plot7_training_curve.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(f"\nDone. Best val loss: {best_val:.4f}")