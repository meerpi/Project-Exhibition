# extract landmark sequences from video clips (or generate synthetic data)
# checkpointed so reruns skip already-processed clips

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    WORD_CLASSES,
    DATA_DIR,
    SEQUENCE_LENGTH,
    MULTI_LANDMARK_DIM,
    WORD_TO_IDX,
)
from src.landmarks import HandLandmarkExtractor

CHECKPOINT_PATH = DATA_DIR / "word_landmarks_checkpoint.json"
NPZ_DATA_PATH = DATA_DIR / "word_landmarks.npz"


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"processed_clips": [], "samples_count": {w: 0 for w in WORD_CLASSES}}


def save_checkpoint(checkpoint):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def process_video(extractor, video_path):
    """extract (30, 126) sequence from a video file"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    return extractor.extract_sequence(frames, seq_len=SEQUENCE_LENGTH)


def generate_word_dataset(video_dir=None, samples_per_word=40, n_signers=3):
    """
    build dataset from video clips if available,
    otherwise generate synthetic trajectory data for testing
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    processed = set(checkpoint.get("processed_clips", []))

    extractor = HandLandmarkExtractor()

    sequences = []
    labels = []
    signers = []

    print(f"\n  word landmark extraction ({len(WORD_CLASSES)} words)")

    if video_dir and video_dir.exists():
        print(f"  processing videos from {video_dir}")
        for word in WORD_CLASSES:
            word_folder = video_dir / word
            if not word_folder.exists():
                continue
            vids = list(word_folder.glob("*.mp4")) + list(word_folder.glob("*.avi"))
            # TODO add .mov support?
            for vf in vids:
                clip_id = str(vf.resolve())
                if clip_id in processed:
                    continue

                seq = process_video(extractor, vf)
                if seq is not None:
                    sequences.append(seq)
                    labels.append(WORD_TO_IDX[word])
                    signer_id = vf.stem.split("_")[0] if "_" in vf.stem else "signer_0"
                    signers.append(signer_id)
                    processed.add(clip_id)

    else:
        # synthetic data for initial testing
        print("  generating synthetic trajectories (no video dir)")
        print(f"  {n_signers} signers x {samples_per_word} samples")
        np.random.seed(42)

        for signer_idx in range(n_signers):
            signer_name = f"signer_{signer_idx}"

            for w_idx, word in enumerate(WORD_CLASSES):
                clip_key = f"{signer_name}_{word}"
                if clip_key in processed:
                    continue

                for _ in range(samples_per_word):
                    t = np.linspace(0, np.pi * 2, SEQUENCE_LENGTH)
                    seq = np.zeros((SEQUENCE_LENGTH, MULTI_LANDMARK_DIM), dtype=np.float32)

                    freq = (w_idx + 1) * 0.3
                    bias = (signer_idx - 1) * 0.05

                    for frame in range(SEQUENCE_LENGTH):
                        seq[frame, :63] = np.sin(t[frame] * freq + bias) * 0.8
                        if w_idx % 2 == 0:  # two-handed signs
                            seq[frame, 63:] = np.cos(t[frame] * freq + bias) * 0.8

                    seq += np.random.normal(0, 0.05, seq.shape).astype(np.float32)

                    sequences.append(seq)
                    labels.append(w_idx)
                    signers.append(signer_name)

                processed.add(clip_key)

    checkpoint["processed_clips"] = list(processed)
    save_checkpoint(checkpoint)
    extractor.close()

    if sequences:
        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        s = np.array(signers)

        np.savez_compressed(NPZ_DATA_PATH, X=X, y=y, signers=s)
        print(f"  saved {len(X)} sequences to {NPZ_DATA_PATH}")
        print(f"  shape: {X.shape}, signers: {sorted(set(s))}")
    else:
        print("  nothing new to extract")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args()

    vid_path = Path(args.video_dir) if args.video_dir else None
    generate_word_dataset(vid_path, samples_per_word=args.samples)


if __name__ == "__main__":
    main()
