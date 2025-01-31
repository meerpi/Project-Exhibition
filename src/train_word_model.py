# training for whole-word ASL sign models (GRU or ST-GCN)
# uses sequence data from data/word_landmarks.npz

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from configs.config import (
    WORD_CLASSES,
    DATA_DIR,
    MODEL_DIR,
    WORD_MODEL_PATH,
    STGCN_MODEL_PATH,
    IDX_TO_WORD,
    RANDOM_SEED,
    TEST_SIZE,
    STGCN_EPOCHS,
    STGCN_LR,
    STGCN_LABEL_SMOOTHING,
    STGCN_WEIGHT_DECAY,
    MULTI_LANDMARK_DIM,
    LANDMARK_DIM,
    AUG_SPATIAL_JITTER,
    AUG_SCALE_RANGE,
    AUG_ROTATE_MAX_DEG,
    AUG_TEMPORAL_WARP,
    AUG_MIRROR_PROB,
)
from src.model import (
    PyTorchTemporalASLModel,
    SpatioTemporalGCN,
    get_device,
)


class ASLAugmentDataset(Dataset):
    """on-the-fly augmentation: jitter, scale, rotate, temporal warp, mirror"""

    def __init__(self, X, y, augment=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x):
        T, D = x.shape

        # jitter
        if AUG_SPATIAL_JITTER > 0:
            x = x + torch.randn_like(x) * AUG_SPATIAL_JITTER

        # scale
        scale = torch.empty(1).uniform_(*AUG_SCALE_RANGE).item()
        x = x * scale

        # 2d rotation in XY plane
        if AUG_ROTATE_MAX_DEG > 0:
            angle = torch.empty(1).uniform_(-AUG_ROTATE_MAX_DEG, AUG_ROTATE_MAX_DEG).item()
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)

            x_3d = x.view(T, -1, 3)
            xc = x_3d[:, :, 0].clone()
            yc = x_3d[:, :, 1].clone()
            x_3d[:, :, 0] = cos_a * xc - sin_a * yc
            x_3d[:, :, 1] = sin_a * xc + cos_a * yc
            x = x_3d.view(T, D)

        # temporal warp - resample at random speed
        if AUG_TEMPORAL_WARP[0] < AUG_TEMPORAL_WARP[1]:
            warp = torch.empty(1).uniform_(*AUG_TEMPORAL_WARP).item()
            warped_len = max(3, int(T * warp))
            if warped_len != T:
                x_np = x.numpy()
                old_idx = np.arange(warped_len)
                new_idx = np.linspace(0, warped_len - 1, T)

                src_idx = np.linspace(0, T - 1, warped_len)
                warped = np.zeros((warped_len, D), dtype=np.float32)
                for d in range(D):
                    warped[:, d] = np.interp(src_idx, np.arange(T), x_np[:, d])

                result = np.zeros((T, D), dtype=np.float32)
                for d in range(D):
                    result[:, d] = np.interp(new_idx, old_idx, warped[:, d])
                x = torch.from_numpy(result)

        # mirror hands (swap left/right)
        if torch.rand(1).item() < AUG_MIRROR_PROB:
            hand1 = x[:, :LANDMARK_DIM].clone()
            hand2 = x[:, LANDMARK_DIM:].clone()
            x[:, :LANDMARK_DIM] = hand2
            x[:, LANDMARK_DIM:] = hand1

        return x


def train_word_model_gpu(model, X_train, y_train, X_val, y_val,
                         epochs=60, batch_size=32, lr=0.001,
                         use_scheduler=False, label_smoothing=0.0,
                         weight_decay=1e-4):
    device = get_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print(f"\n  training {model.__class__.__name__} on {device} ({gpu_name})")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {param_count:,}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_ds = ASLAugmentDataset(X_train, y_train, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))

    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_acc = 0.0
    best_state = None
    patience = 25
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100

        # validation (batched for 6gb gpu)
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for vi in range(0, len(val_x), batch_size):
                vb = val_x[vi:vi+batch_size]
                vp = torch.argmax(model(vb), dim=1)
                val_preds_list.append(vp)
            val_preds = torch.cat(val_preds_list)
            val_acc = (val_preds == val_y).float().mean().item() * 100

        if scheduler:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [{epoch:3d}/{epochs}] loss={train_loss:.4f} "
                  f"train={train_acc:.1f}% val={val_acc:.1f}% "
                  f"best={best_val_acc:.1f}% lr={lr_now:.6f}")

        if patience_counter >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print(f"  restored best model ({best_val_acc:.1f}%)")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train word sign model")
    parser.add_argument("--model", type=str, default="stgcn", choices=["gru", "stgcn"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if args.model == "stgcn":
        epochs = args.epochs or STGCN_EPOCHS
        lr = args.lr or STGCN_LR
        save_path = STGCN_MODEL_PATH
    else:
        epochs = args.epochs or 60
        lr = args.lr or 0.001
        save_path = WORD_MODEL_PATH

    npz_path = DATA_DIR / "word_landmarks.npz"
    if not npz_path.exists():
        print("dataset not found, running extractor...")
        from src.extract_word_landmarks import generate_word_dataset
        generate_word_dataset()

    data = np.load(npz_path)
    X, y, signers = data["X"], data["y"], data["signers"]
    print(X.shape)  # sanity check

    print(f"\n  {args.model.upper()} training")
    print(f"  sequences: {len(X)} shape: {X.shape[1:]}")
    print(f"  signers: {len(set(signers))}")

    # class distribution
    for idx, count in sorted(Counter(y).items()):
        word = IDX_TO_WORD.get(idx, f"?{idx}")
        print(f"    {word:12s}: {count:5d}")

    # filter garbage sequences (>60% zero frames)
    zero_counts = np.array([np.sum(np.all(seq == 0, axis=1)) for seq in X])
    max_zeros = int(X.shape[1] * 0.6)
    good = zero_counts <= max_zeros
    if not np.all(good):
        n_bad = np.sum(~good)
        print(f"\n  dropped {n_bad} bad sequences (>{max_zeros} zero frames)")
        X, y, signers = X[good], y[good], signers[good]
        print("after filter:", len(X))

    unique_signers = set(signers)
    if len(unique_signers) > 1:
        print(f"  signer-independent split")
        split_type = "signer-independent"
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        train_idx, test_idx = next(gss.split(X, y, groups=signers))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        print("  single signer - random split")
        split_type = "random"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

    print(f"  train: {len(X_train)} test: {len(X_test)}")

    num_classes = len(WORD_CLASSES)
    present = set(y_train) | set(y_test)
    if max(present) >= num_classes:
        num_classes = max(present) + 1

    if args.model == "stgcn":
        model = SpatioTemporalGCN(num_classes=num_classes)
    else:
        model = PyTorchTemporalASLModel(num_classes=num_classes)

    model = train_word_model_gpu(
        model, X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=args.batch_size, lr=lr,
        use_scheduler=(args.model == "stgcn"),
        label_smoothing=STGCN_LABEL_SMOOTHING if args.model == "stgcn" else 0.0,
        weight_decay=STGCN_WEIGHT_DECAY if args.model == "stgcn" else 1e-4,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n  saved to {save_path}")

    # eval
    device = get_device()
    model.eval()
    y_pred_parts = []
    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            batch = torch.tensor(X_test[i:i+64], dtype=torch.float32, device=device)
            preds = torch.argmax(model(batch), dim=1).cpu().numpy()
            y_pred_parts.append(preds)
    y_pred = np.concatenate(y_pred_parts)

    acc = accuracy_score(y_test, y_pred)
    present_classes = sorted(set(y_test) | set(y_pred))
    target_names = [IDX_TO_WORD.get(i, f"class_{i}") for i in present_classes]

    print(f"\n  accuracy: {acc*100:.2f}% ({split_type})")

    report = classification_report(
        y_test, y_pred, labels=present_classes, target_names=target_names, zero_division=0
    )
    print(report)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Purples", values_format="d")
    ax.set_title(f"Word Signs ({args.model.upper()}) - {acc*100:.1f}%")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    cm_path = MODEL_DIR / "confusion_matrix_words.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  confusion matrix -> {cm_path}")

    # top confusion pairs
    print("\n  worst confusions:")
    confused = []
    for i in range(len(present_classes)):
        for j in range(len(present_classes)):
            if i != j and cm[i, j] > 0:
                confused.append((target_names[i], target_names[j], cm[i, j]))

    for tw, pw, cnt in sorted(confused, key=lambda x: x[2], reverse=True)[:5]:
        print(f"    '{tw}' -> '{pw}' ({cnt}x)")

    print()


if __name__ == "__main__":
    main()
