# model definitions - pytorch + sklearn fallback
# supports static MLP (fingerspelling), temporal GRU, ST-GCN, and transformer

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
    WORD_MODEL_PATH,
    STGCN_MODEL_PATH,
    MLP_HIDDEN_LAYERS,
    MLP_MAX_ITER,
    MLP_LEARNING_RATE_INIT,
    RANDOM_SEED,
    NUM_CLASSES,
    NUM_WORD_CLASSES,
    LANDMARK_DIM,
    MULTI_LANDMARK_DIM,
    PYTORCH_EPOCHS,
    PYTORCH_BATCH_SIZE,
    PYTORCH_LR,
    NUM_GRAPH_NODES,
    COORD_DIM,
    STGCN_HIDDEN_DIMS,
    STGCN_TEMPORAL_KERNEL,
    STGCN_DROPOUT,
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


    class PyTorchTemporalASLModel(nn.Module):
        """1D-Conv + BiGRU for temporal word sign sequences (30 x 126)"""
        def __init__(self, input_dim=MULTI_LANDMARK_DIM,
                     hidden_dim=64,
                     num_classes=NUM_WORD_CLASSES):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.gru = nn.GRU(
                input_size=128,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            # x: (B, 30, 126) -> conv needs (B, 126, 30)
            x_conv = x.transpose(1, 2)
            x_conv = self.conv(x_conv)
            x_conv = x_conv.transpose(1, 2)

            gru_out, _ = self.gru(x_conv)
            last_timestep = gru_out[:, -1, :]
            logits = self.fc(last_timestep)
            return logits

        def predict(self, seq_numpy):
            self.eval()
            device = next(self.parameters()).device
            if seq_numpy.ndim == 2:
                seq_numpy = np.expand_dims(seq_numpy, axis=0)
            with torch.no_grad():
                tensor_x = torch.tensor(seq_numpy, dtype=torch.float32, device=device)
                logits = self.forward(tensor_x)
                pred_idx = torch.argmax(logits, dim=1).item()
            return pred_idx


    # ── ST-GCN stuff ──

    class SpatialGraphConv(nn.Module):
        """single spatial graph conv: H' = A_norm @ H @ W"""
        def __init__(self, in_channels, out_channels, adj_matrix):
            super().__init__()
            self.register_buffer("adj", adj_matrix)
            self.fc = nn.Linear(in_channels, out_channels, bias=True)

        def forward(self, x):
            # x: (B, T, V, C_in) -> (B, T, V, C_out)
            x = torch.einsum("vw, btwc -> btvc", self.adj, x)
            x = self.fc(x)
            return x


    class STGCNBlock(nn.Module):
        """spatial graph conv + temporal conv + residual"""
        def __init__(self, in_channels, out_channels,
                     adj_matrix, temporal_kernel=9, dropout=0.3):
            super().__init__()
            self.spatial_conv = SpatialGraphConv(in_channels, out_channels, adj_matrix)
            self.bn_spatial = nn.BatchNorm2d(out_channels)

            temporal_pad = (temporal_kernel - 1) // 2
            self.temporal_conv = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(temporal_kernel, 1),
                padding=(temporal_pad, 0),
            )
            self.bn_temporal = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(dropout)

            if in_channels != out_channels:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual = nn.Identity()

        def forward(self, x):
            # x: (B, C_in, T, V)
            res = self.residual(x)

            # spatial conv needs (B, T, V, C)
            x_perm = x.permute(0, 2, 3, 1)
            x_perm = self.spatial_conv(x_perm)
            x_out = x_perm.permute(0, 3, 1, 2)  # back to (B, C, T, V)
            x_out = self.bn_spatial(x_out)
            x_out = self.relu(x_out)

            x_out = self.temporal_conv(x_out)
            x_out = self.bn_temporal(x_out)

            x_out = self.relu(x_out + res)
            x_out = self.dropout(x_out)
            return x_out


    class SpatioTemporalGCN(nn.Module):
        """
        ST-GCN for isolated sign recognition.
        input: (B, T=30, 126) flat or (B, T, 42, 3) graph
        output: (B, num_classes)
        """
        def __init__(self, num_classes=NUM_WORD_CLASSES,
                     num_nodes=NUM_GRAPH_NODES,
                     in_channels=COORD_DIM,
                     hidden_dims=None,
                     temporal_kernel=STGCN_TEMPORAL_KERNEL,
                     dropout=STGCN_DROPOUT):
            super().__init__()
            from src.hand_graph import get_adjacency_matrix

            self.num_nodes = num_nodes
            self.in_channels = in_channels

            hidden_dims = hidden_dims or list(STGCN_HIDDEN_DIMS)
            adj = get_adjacency_matrix(num_hands=2, add_cross_hand=True)

            channels = [in_channels] + hidden_dims
            self.blocks = nn.ModuleList()
            for i in range(len(hidden_dims)):
                self.blocks.append(
                    STGCNBlock(channels[i], channels[i + 1], adj,
                               temporal_kernel=temporal_kernel, dropout=dropout)
                )

            self.fc = nn.Linear(hidden_dims[-1], num_classes)

        def forward(self, x):
            # auto reshape flat (B, T, 126) -> (B, T, 42, 3)
            if x.ndim == 3 and x.shape[-1] == self.num_nodes * self.in_channels:
                B, T, _ = x.shape
                x = x.view(B, T, self.num_nodes, self.in_channels)

            x = x.permute(0, 3, 1, 2)  # (B, C, T, V)

            for block in self.blocks:
                x = block(x)

            # global avg pool over time + joints
            x = x.mean(dim=[2, 3])
            logits = self.fc(x)
            return logits

        def predict(self, seq_numpy):
            self.eval()
            device = next(self.parameters()).device
            if seq_numpy.ndim == 2:
                seq_numpy = np.expand_dims(seq_numpy, axis=0)
            with torch.no_grad():
                tensor_x = torch.tensor(seq_numpy, dtype=torch.float32, device=device)
                logits = self.forward(tensor_x)
                pred_idx = torch.argmax(logits, dim=1).item()
            return pred_idx


    class SignTransformer(nn.Module):
        """
        1D-CNN stem + transformer encoder. based on kaggle ISLR 1st place.
        input: (B, 30, 252) pos+vel or (B, 30, 126) position only
        """
        def __init__(self, in_features=252, d_model=192,
                     nhead=8, num_layers=4,
                     num_classes=NUM_WORD_CLASSES, dropout=0.2):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_features, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.SiLU(),
            )

            self.pos_embed = nn.Parameter(torch.zeros(1, 30, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.attn_pool = nn.Linear(d_model, 1)
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )

        def forward(self, x):
            if x.ndim == 2:
                x = x.unsqueeze(0)
            # compute velocity if only position vector given
            if x.shape[-1] == 126:
                vel = torch.zeros_like(x)
                vel[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
                x = torch.cat([x, vel], dim=-1)

            x_stem = self.stem(x.transpose(1, 2)).transpose(1, 2)
            x_stem = x_stem + self.pos_embed
            trans_out = self.transformer(x_stem)
            weights = torch.softmax(self.attn_pool(trans_out), dim=1)
            pooled = (trans_out * weights).sum(dim=1)
            return self.fc(pooled)

        def predict(self, seq_numpy):
            self.eval()
            device = next(self.parameters()).device
            if seq_numpy.ndim == 2:
                seq_numpy = np.expand_dims(seq_numpy, axis=0)
            with torch.no_grad():
                tensor_x = torch.tensor(seq_numpy, dtype=torch.float32, device=device)
                logits = self.forward(tensor_x)
                pred_idx = torch.argmax(logits, dim=1).item()
            return pred_idx


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


def load_word_model(path=None):
    """
    load word model - auto detects architecture from checkpoint keys.
    returns (model, type_string)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for word models.")

    device = get_device()

    if path is not None:
        load_path = path
    elif STGCN_MODEL_PATH.exists():
        load_path = STGCN_MODEL_PATH
    elif WORD_MODEL_PATH.exists():
        load_path = WORD_MODEL_PATH
    else:
        raise FileNotFoundError(
            f"No word model found. Run 'python src/train_word_model.py' first."
        )

    checkpoint = torch.load(load_path, map_location=device, weights_only=True)

    # figure out which architecture this is
    keys = list(checkpoint.keys())
    print(len(keys), "keys in checkpoint")  # TODO remove this
    is_transformer = any(k.startswith("stem.") or k.startswith("transformer.") for k in keys)
    is_stgcn = any(k.startswith("blocks.") for k in keys)

    if is_transformer:
        model = SignTransformer().to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"  ✓ SOTA SignTransformer word model loaded from {load_path} on {device}")
        return model, "transformer"
    elif is_stgcn:
        model = SpatioTemporalGCN().to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"  ✓ ST-GCN word model loaded from {load_path} on {device}")
        return model, "stgcn"
    else:
        model = PyTorchTemporalASLModel().to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"  ✓ GRU word model loaded from {load_path} on {device}")
        return model, "gru"
