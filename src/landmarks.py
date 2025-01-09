# hand landmark extraction with mediapipe
# this is the main extraction module - single hand, multi hand, sequences

import sys
from pathlib import Path

import numpy as np

from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    MEDIAPIPE_MODEL_PATH,
    NUM_LANDMARKS,
    LANDMARK_DIM,
    MAX_HANDS,
    MULTI_LANDMARK_DIM,
    SEQUENCE_LENGTH,
    WRIST_IDX,
    SCALE_REF_IDX,
)


class HandLandmarkExtractor:
    """wraps mediapipe HandLandmarker for 1 or 2 hand extraction"""

    def __init__(self, model_path=None, num_hands=MAX_HANDS):
        model_path = model_path or MEDIAPIPE_MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand_landmarker.task not found at {model_path}."
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def extract(self, bgr_frame):
        """single hand 63-dim vector for fingerspelling"""
        rgb_frame = bgr_frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]
        if len(hand) != NUM_LANDMARKS:
            return None

        return _normalize_landmarks(hand)

    def extract_multi_hand(self, bgr_frame):
        """2-hand normalized 126-dim vector for word signs"""
        rgb_frame = bgr_frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result: HandLandmarkerResult = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        combined = np.zeros(MULTI_LANDMARK_DIM, dtype=np.float32)

        for i, hand in enumerate(result.hand_landmarks[:MAX_HANDS]):
            if len(hand) == NUM_LANDMARKS:
                normed = _normalize_landmarks(hand)
                if normed is not None:
                    # left -> slot 0, right -> slot 1
                    slot = 0
                    if i < len(result.handedness) and result.handedness[i]:
                        label = result.handedness[i][0].category_name.lower()
                        if "right" in label:
                            slot = 1
                    combined[slot * LANDMARK_DIM : (slot + 1) * LANDMARK_DIM] = normed

        if np.all(combined == 0):
            return None

        return combined

    def extract_raw_multi_hand(self, bgr_frame):
        """raw (unnormalized) 126-dim - keeps absolute positions for trajectories"""
        rgb_frame = bgr_frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        combined = np.zeros(MULTI_LANDMARK_DIM, dtype=np.float32)

        for i, hand in enumerate(result.hand_landmarks[:MAX_HANDS]):
            if len(hand) == NUM_LANDMARKS:
                raw = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand],
                    dtype=np.float32,
                )
                slot = 0
                if i < len(result.handedness) and result.handedness[i]:
                    label = result.handedness[i][0].category_name.lower()
                    if "right" in label:
                        slot = 1
                combined[slot * LANDMARK_DIM : (slot + 1) * LANDMARK_DIM] = raw.flatten()

        if np.all(combined == 0):
            return None

        return combined

    def extract_sequence(self, bgr_frames, seq_len=SEQUENCE_LENGTH):
        """
        trajectory-preserving sequence extraction (30 frames x 126 dim).
        unlike per-frame normalization, this anchors the whole sequence
        to the first frame's wrist so hand movement is preserved.
        """
        if not bgr_frames:
            return None

        # get raw landmarks per frame
        raw_vectors = []
        for frame in bgr_frames:
            vector = self.extract_raw_multi_hand(frame)
            if vector is None:
                vector = np.zeros(MULTI_LANDMARK_DIM, dtype=np.float32)
            raw_vectors.append(vector)

        raw_arr = np.array(raw_vectors, dtype=np.float32)  # (T, 126)
        # print(raw_arr.shape, np.count_nonzero(raw_arr))

        # fill in missing frames
        raw_arr = _interpolate_missing(raw_arr)

        # normalize whole sequence at once
        raw_arr = _normalize_sequence(raw_arr)

        if raw_arr is None:
            return None

        return _resample(raw_arr, target_len=seq_len)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _normalize_landmarks(hand_landmarks):
    """wrist-center + scale by wrist-to-MCP9 distance"""
    raw = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )

    wrist = raw[WRIST_IDX]
    centered = raw - wrist

    scale_ref = centered[SCALE_REF_IDX]
    scale_dist = np.linalg.norm(scale_ref)

    if scale_dist < 1e-6:
        return None

    normalized = centered / scale_dist
    return normalized.flatten()


def _interpolate_missing(seq):
    """fill zero-frames by interpolating from nearest valid ones"""
    T, D = seq.shape
    valid_mask = np.any(seq != 0, axis=1)

    if not np.any(valid_mask):
        return seq
    if np.all(valid_mask):
        return seq

    valid_indices = np.where(valid_mask)[0]

    for dim in range(D):
        values = seq[:, dim]
        valid_vals = values[valid_indices]
        seq[:, dim] = np.interp(
            np.arange(T), valid_indices, valid_vals
        )

    return seq


def _normalize_sequence(seq):
    """
    normalize entire sequence with single anchor (first frame wrist)
    and single scale (median wrist-MCP9 dist). preserves trajectory.
    """
    T = seq.shape[0]
    seq_3d = seq.reshape(T, -1, 3)  # (T, 42, 3)

    # find anchor wrist from first non-zero frame
    anchor_wrist = None
    for t in range(T):
        wrist = seq_3d[t, WRIST_IDX]
        if np.linalg.norm(wrist) > 1e-8:
            anchor_wrist = wrist.copy()
            break

    if anchor_wrist is None:
        return None
    # print("anchor:", anchor_wrist)

    for t in range(T):
        seq_3d[t] -= anchor_wrist

    # scale: median wrist-to-MCP9 distance
    scales = []
    for t in range(T):
        wrist = seq_3d[t, WRIST_IDX]
        mcp9 = seq_3d[t, SCALE_REF_IDX]
        dist = np.linalg.norm(mcp9 - wrist)
        if dist > 1e-6:
            scales.append(dist)

    if not scales:
        return None

    median_scale = np.median(scales)
    if median_scale < 1e-6:
        return None

    seq_3d /= median_scale

    return seq_3d.reshape(T, -1)


def _resample(seq, target_len=SEQUENCE_LENGTH):
    """resample to target_len frames"""
    current_len = len(seq)
    if current_len == target_len:
        return seq

    if current_len == 0:
        return np.zeros((target_len, MULTI_LANDMARK_DIM), dtype=np.float32)

    indices = np.linspace(0, current_len - 1, num=target_len)
    resampled = np.zeros((target_len, seq.shape[1]), dtype=np.float32)

    for dim in range(seq.shape[1]):
        resampled[:, dim] = np.interp(indices, np.arange(current_len), seq[:, dim])

    return resampled
