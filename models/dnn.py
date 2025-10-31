from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def _infer_output(y: np.ndarray) -> Tuple[str, int]:
    # Heuristic: classification if discrete with small number of classes
    if y.dtype.kind in ('i', 'b') or (np.unique(y).size <= max(20, int(0.05 * y.size))):
        num_classes = int(np.unique(y).size)
        if num_classes <= 2:
            return 'binary', 1
        return 'multiclass', num_classes
    return 'regression', 1



class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, task_type: str):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self.task_type = task_type

    def forward(self, x):
        return self.net(x)


@dataclass
class DNNConfig:
    hidden: List[int]
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42


def train_dnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
    cfg: DNNConfig
):
    torch.manual_seed(cfg.seed)
    task_type, out_dim = _infer_output(y_train)

    # Prepare tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train)
    Xv = torch.tensor(X_val, dtype=torch.float32) if X_val is not None else None
    yv = torch.tensor(y_val) if y_val is not None else None

    in_dim = Xtr.shape[1]
    model = MLP(in_dim, cfg.hidden, out_dim, task_type)

    if task_type == 'regression':
        ytr = ytr.float().view(-1, 1)
        if yv is not None:
            yv = yv.float().view(-1, 1)
        criterion = nn.MSELoss()
    elif task_type == 'binary':
        ytr = ytr.float().view(-1, 1)
        if yv is not None:
            yv = yv.float().view(-1, 1)
        criterion = nn.BCEWithLogitsLoss()
    else:  # multiclass
        ytr = ytr.long().view(-1)
        if yv is not None:
            yv = yv.long().view(-1)
        criterion = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optim.zero_grad(); loss.backward(); optim.step()

    model.eval()

    def predict_proba(X):
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            if task_type == 'binary':
                proba = torch.sigmoid(logits).numpy()
                return np.hstack([1 - proba, proba])
            if task_type == 'multiclass':
                return torch.softmax(logits, dim=1).numpy()
            return logits.numpy()

    def predict(X):
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            if task_type == 'regression':
                return logits.numpy().reshape(-1)
            if task_type == 'binary':
                return (torch.sigmoid(logits) > 0.5).numpy().astype(int).reshape(-1)
            return torch.argmax(logits, dim=1).numpy()

    return model, predict, predict_proba, task_type


