import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SepsisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, y_train, X_val, y_val,
                 batch_size=256, num_workers=0):
    train_ds = SepsisDataset(X_train, y_train)
    val_ds   = SepsisDataset(X_val,   y_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader