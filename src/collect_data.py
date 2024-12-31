# webcam data collection for hand sign landmarks
# saves to CSV in data/ with signer ID for proper eval later

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    CLASSES,
    DATA_DIR,
    LANDMARK_DIM,
    CAMERA_INDEX,
)
from src.landmarks import HandLandmarkExtractor


def get_csv_path(signer):
    return DATA_DIR / f"landmarks_{signer}.csv"


def init_csv(csv_path):
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["letter", "signer"] + [f"lm_{i}" for i in range(LANDMARK_DIM)]
        writer.writerow(header)


def append_sample(csv_path, letter, signer, landmarks):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [letter, signer] + landmarks.tolist()
        writer.writerow(row)


def collect_interactive(signer, camera_index=CAMERA_INDEX):
    """
    interactive mode - press A-Z to pick letter, SPACE to capture,
    R for rapid mode, Q to quit
    """
    csv_path = get_csv_path(signer)
    init_csv(csv_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: cant open camera {camera_index}")
        return

    extractor = HandLandmarkExtractor()

    current_letter = "A"
    rapid_mode = False
    rapid_counter = 0
    rapid_interval = 5
    sample_counts = {c: 0 for c in CLASSES}

    # count existing samples
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0] in sample_counts:
                    sample_counts[row[0]] += 1

    print(f"\nData Collection - signer: {signer}")
    print(f"  A-Z = select letter | SPACE = capture | R = rapid | Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        landmarks = extractor.extract(frame)

        display = frame.copy()
        hand_detected = landmarks is not None

        color = (0, 200, 0) if hand_detected else (0, 0, 200)
        status = "HAND DETECTED" if hand_detected else "NO HAND"
        cv2.rectangle(display, (0, 0), (display.shape[1], 80), (30, 30, 30), -1)
        cv2.putText(display, f"Letter: {current_letter}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display, f"Samples: {sample_counts[current_letter]}", (250, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(display, status, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        mode_text = "RAPID ON" if rapid_mode else "Manual"
        mode_color = (0, 200, 255) if rapid_mode else (180, 180, 180)
        cv2.putText(display, mode_text, (450, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        captured = False
        if rapid_mode and hand_detected:
            rapid_counter += 1
            if rapid_counter >= rapid_interval:
                rapid_counter = 0
                append_sample(csv_path, current_letter, signer, landmarks)
                sample_counts[current_letter] += 1
                captured = True

        if captured:
            cv2.rectangle(display, (0, 80), (display.shape[1], 110), (0, 200, 0), -1)
            cv2.putText(display, "CAPTURED!", (250, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Data Collection", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            rapid_mode = not rapid_mode
            rapid_counter = 0
            print(f"rapid mode: {'ON' if rapid_mode else 'OFF'}")
        elif key == ord(" "):
            if hand_detected:
                append_sample(csv_path, current_letter, signer, landmarks)
                sample_counts[current_letter] += 1
                print(f"  captured {current_letter} - total: {sample_counts[current_letter]}")
            else:
                print("  no hand - skipped")
        elif chr(key).upper() in CLASSES:
            current_letter = chr(key).upper()
            rapid_counter = 0
            print(f"  switched to: {current_letter} ({sample_counts[current_letter]} samples)")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    # summary
    total = sum(sample_counts.values())
    print(f"\n  total samples: {total}")
    print(f"  saved to: {csv_path}")


def collect_batch(signer, letter, count, camera_index=CAMERA_INDEX):
    """collect N samples for one letter automatically"""
    csv_path = get_csv_path(signer)
    init_csv(csv_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: cant open camera {camera_index}")
        return

    extractor = HandLandmarkExtractor()

    collected = 0
    skipped = 0
    print(f"collecting {count} samples of '{letter}' for '{signer}'...")

    while collected < count:
        ret, frame = cap.read()
        if not ret:
            print("frame capture failed")
            break

        frame = cv2.flip(frame, 1)
        landmarks = extractor.extract(frame)
        # print(type(landmarks))  # was using this to debug None returns

        display = frame.copy()
        cv2.putText(display, f"Letter: {letter}  [{collected}/{count}]",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Batch Collection", display)

        if landmarks is not None:
            append_sample(csv_path, letter, signer, landmarks)
            collected += 1
        else:
            skipped += 1

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            print("interrupted")
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    print(f"done: {collected}/{count} collected, {skipped} skipped (no hand)")


def main():
    parser = argparse.ArgumentParser(description="Collect hand sign data from webcam")
    parser.add_argument("--signer", type=str, default="default",
                        help="signer name for evaluation splits")
    parser.add_argument("--letter", type=str, default=None,
                        help="single letter to collect (batch mode)")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)

    args = parser.parse_args()

    if args.letter:
        letter = args.letter.upper()
        if letter not in CLASSES:
            print(f"ERROR: '{letter}' not valid, use A-Z")
            sys.exit(1)
        collect_batch(args.signer, letter, args.count, args.camera)
    else:
        collect_interactive(args.signer, args.camera)


if __name__ == "__main__":
    main()
