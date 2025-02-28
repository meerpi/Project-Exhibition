# training script for A-Z fingerspelling model
# loads landmark CSVs, trains MLP, saves confusion matrix + metrics

import argparse
import csv
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
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from configs.config import (
    CLASSES,
    DATA_DIR,
    MODEL_DIR,
    LANDMARK_DIM,
    LETTER_TO_IDX,
    IDX_TO_LETTER,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.model import train_model, save_model


def load_csv_data(data_dir):
    """load all landmark CSVs from data_dir, returns X, y, signers"""
    csv_files = sorted(data_dir.glob("landmarks_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No landmark CSVs in {data_dir}. Run collect_data.py first."
        )

    all_features = []
    all_labels = []
    all_signers = []

    for csv_path in csv_files:
        print(f"  loading {csv_path.name}...")
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                if len(row) < 2 + LANDMARK_DIM:
                    continue

                letter = row[0].upper()
                signer = row[1]

                if letter not in LETTER_TO_IDX:
                    continue

                features = [float(x) for x in row[2: 2 + LANDMARK_DIM]]
                all_features.append(features)
                all_labels.append(LETTER_TO_IDX[letter])
                all_signers.append(signer)

    if not all_features:
        raise ValueError("no valid samples found in CSVs")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    signers = np.array(all_signers)

    return X, y, signers


def evaluate_and_report(model, X_test, y_test, output_dir, split_type):
    """run eval, print report, save confusion matrix"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    present_classes = sorted(set(y_test) | set(y_pred))
    target_names = [IDX_TO_LETTER[i] for i in present_classes]

    print(f"\n  accuracy: {accuracy * 100:.2f}% ({split_type})")

    if split_type == "random":
        print(f"  ⚠  random split - accuracy likely inflated")

    report = classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=target_names,
        zero_division=0,
    )
    print(report)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix ({split_type})\nAcc: {accuracy*100:.1f}%")
    plt.tight_layout()

    cm_path = output_dir / f"confusion_matrix_{split_type}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  confusion matrix -> {cm_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train hand sign classifier")
    parser.add_argument("--data", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data)

    np.random.seed(RANDOM_SEED)

    print(f"\n  loading data from {data_dir}...")
    X, y, signers = load_csv_data(data_dir)

    print(f"\n  samples: {len(X)}, features: {X.shape[1]}, classes: {len(set(y))}")

    # show class distribution
    class_counts = Counter(y)
    for idx in sorted(class_counts):
        letter = IDX_TO_LETTER[idx]
        count = class_counts[idx]
        bar = "█" * min(count // 2, 40)
        print(f"    {letter}: {count:4d} {bar}")

    unique_signers = sorted(set(signers))
    has_multiple_signers = len(unique_signers) > 1
    print(f"\n  signers: {unique_signers}")

    if has_multiple_signers:
        print(f"  using signer-independent split")
        split_type = "signer-independent"

        gss = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        train_idx, test_idx = next(gss.split(X, y, groups=signers))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_signers = set(signers[train_idx])
        test_signers = set(signers[test_idx])
        print(f"  train signers: {sorted(train_signers)}")
        print(f"  test signers: {sorted(test_signers)}")
        assert train_signers.isdisjoint(test_signers), "signer leakage!"
    else:
        print(f"  only 1 signer - random split (accuracy will be inflated)")
        split_type = "random"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

    print(f"  train: {len(X_train)} | test: {len(X_test)}")

    print(f"\n  training...")
    model = train_model(X_train, y_train, use_gpu=True)
    if hasattr(model, 'n_iter_'):
        print(f"  iterations: {model.n_iter_}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model)

    accuracy = evaluate_and_report(model, X_test, y_test, MODEL_DIR, split_type)

    # save loss curve if sklearn
    if hasattr(model, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(model.loss_curve_, label="Training Loss")
        if hasattr(model, "validation_scores_") and model.validation_scores_:
            ax2 = ax.twinx()
            ax2.plot(model.validation_scores_, color="orange", label="Val Acc")
            ax2.set_ylabel("Validation Accuracy")
            ax2.legend(loc="center right")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend(loc="upper right")
        plt.tight_layout()
        loss_path = MODEL_DIR / "training_loss.png"
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"  loss curve -> {loss_path}")

    print(f"\n  done! {accuracy*100:.2f}% ({split_type})\n")


if __name__ == "__main__":
    main()
