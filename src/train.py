# training script for A-Z fingerspelling model
# trains MLP on landmark CSVs, saves model + confusion matrix

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
from sklearn.model_selection import train_test_split

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
    """load all landmark CSVs from data_dir"""
    csv_files = sorted(data_dir.glob("landmarks_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No landmark CSVs in {data_dir}. Run collect_data.py first."
        )

    all_features = []
    all_labels = []

    for csv_path in csv_files:
        print(f"  loading {csv_path.name}...")
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) < 2 + LANDMARK_DIM:
                    continue
                letter = row[0].upper()
                if letter not in LETTER_TO_IDX:
                    continue
                features = [float(x) for x in row[2: 2 + LANDMARK_DIM]]
                all_features.append(features)
                all_labels.append(LETTER_TO_IDX[letter])

    if not all_features:
        raise ValueError("no valid samples found")

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Train hand sign classifier")
    parser.add_argument("--data", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data)
    np.random.seed(RANDOM_SEED)

    print(f"\n  loading data from {data_dir}...")
    X, y = load_csv_data(data_dir)
    print(f"  samples: {len(X)}, features: {X.shape[1]}, classes: {len(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  train: {len(X_train)} | test: {len(X_test)}")

    print(f"\n  training...")
    model = train_model(X_train, y_train, use_gpu=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  accuracy: {accuracy * 100:.2f}%")

    # confusion matrix
    present_classes = sorted(set(y_test) | set(y_pred))
    target_names = [IDX_TO_LETTER[i] for i in present_classes]
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - Acc: {accuracy*100:.1f}%")
    plt.tight_layout()
    cm_path = MODEL_DIR / "confusion_matrix_random.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  confusion matrix -> {cm_path}")
    print(f"\n  done! {accuracy*100:.2f}%\n")


if __name__ == "__main__":
    main()
