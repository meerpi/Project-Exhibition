# extract landmarks from image folders instead of webcam
# useful if you have a dataset of hand sign images already
#
# expects: dataset_dir/A/*.jpg, dataset_dir/B/*.png, etc

import argparse
import csv
import sys
from pathlib import Path

import cv2

_PROJ_DIR = Path(__file__).resolve().parent.parent
if str(_PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJ_DIR))

from configs.config import CLASSES, DATA_DIR, LANDMARK_DIM, MEDIAPIPE_MODEL_PATH
from src.landmarks import HandLandmarkExtractor


def process_image_directory(dataset_dir, signer_name):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"not found: {dataset_dir}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"landmarks_{signer_name}.csv"

    print(f"\n  extracting landmarks from {dataset_dir}")
    print(f"  output: {csv_path}")

    extractor = HandLandmarkExtractor(MEDIAPIPE_MODEL_PATH)

    total = 0
    detected = 0
    failed = 0
    skipped_letters = []  # was going to track these but never used it

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["letter", "signer"] + [f"lm_{i}" for i in range(LANDMARK_DIM)]
        writer.writerow(header)

        for letter in CLASSES:
            letter_dir = dataset_dir / letter
            if not letter_dir.exists():
                letter_dir = dataset_dir / letter.lower()

            if not letter_dir.exists():
                skipped_letters.append(letter)
                continue

            imgs = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.png")) + list(letter_dir.glob("*.jpeg"))
            print(f"  {letter}: {len(imgs)} images")

            for img_path in imgs:
                total += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    failed += 1
                    continue

                landmarks = extractor.extract(img)
                if landmarks is not None:
                    row = [letter, signer_name] + landmarks.tolist()
                    writer.writerow(row)
                    detected += 1
                else:
                    failed += 1

    extractor.close()

    print(f"\n  done: {detected}/{total} extracted, {failed} failed")
    print(f"  saved to {csv_path}\n")

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from image dataset")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--signer", type=str, default="dataset")
    args = parser.parse_args()

    process_image_directory(Path(args.dataset_dir), args.signer)


if __name__ == "__main__":
    main()
