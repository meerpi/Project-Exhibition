# model definitions - pytorch + sklearn fallback
# static MLP for fingerspelling (A-Z)

import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
import os

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

from configs.config import (
    CLASSIFIER_PATH,
    PYTORCH_MODEL_PATH,
    MLP_HIDDEN_LAYERS,
    MLP_MAX_ITER,
    MLP_LEARNING_RATE_INIT,
    RANDOM_SEED,
    NUM_CLASSES,
    LANDMARK_DIM,
    PYTORCH_EPOCHS,
    PYTORCH_BATCH_SIZE,
    PYTORCH_LR,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_device():
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if HAS_TORCH else "cpu"


if HAS_TORCH:
    class PyTorchHandSignMLP(nn.Module):
        """static hand sign classifier - 63 -> 128 -> 64 -> 26"""
        def __init__(self, input_dim=LANDMARK_DIM, num_classes=NUM_CLASSES):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)

        def predict(self, X_numpy):
            self.eval()
            device = next(self.parameters()).device
            with torch.no_grad():
                tensor_x = torch.tensor(X_numpy, dtype=torch.float32, device=device)
                outputs = self.net(tensor_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            return preds


def create_model():
    return MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        learning_rate_init=MLP_LEARNING_RATE_INIT,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        verbose=True,
    )


def train_pytorch_gpu(X, y, epochs=PYTORCH_EPOCHS,
                      batch_size=PYTORCH_BATCH_SIZE, lr=PYTORCH_LR):
    """train pytorch MLP on gpu"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed.")

    device = get_device()
    print(f"  Training on: {device} " + (f"({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))

    model = PyTorchHandSignMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    print(len(tensor_x), "training samples")

    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs:3d}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    return model


def train_model(X, y, use_gpu=True):
    """train gpu (pytorch) or cpu (sklearn) depending on whats available"""
    if use_gpu and HAS_TORCH and torch.cuda.is_available():
        return train_pytorch_gpu(X, y)
    else:
        model = create_model()
        model.fit(X, y)
        return model


def save_model(model, path=None):
    if HAS_TORCH and isinstance(model, PyTorchHandSignMLP):
        path = path or PYTORCH_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"PyTorch model saved -> {path}")
        return path
    else:
        path = path or CLASSIFIER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        print(f"sklearn model saved -> {path}")
        return path


def load_model(path=None):
    """load from .pt or .joblib"""
    pt_path = PYTORCH_MODEL_PATH
    jl_path = CLASSIFIER_PATH

    if (path is None and pt_path.exists()) or (path and str(path).endswith(".pt")):
        load_path = path or pt_path
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required to load .pt model.")
        device = get_device()
        model = PyTorchHandSignMLP().to(device)
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.eval()
        print(f"PyTorch model loaded from {load_path} on {device}")
        return model

    load_path = path or jl_path
    if not load_path.exists():
        raise FileNotFoundError(
            f"No trained model at {load_path}. Run 'python src/train.py' first."
        )
    model = joblib.load(load_path)
    print(f"Model loaded from {load_path}")
    return model
