# hand landmark extraction with mediapipe
# now with wrist centering and scale normalization

import sys
from pathlib import Path

import numpy as np

from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
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
    WRIST_IDX,
    SCALE_REF_IDX,
)


class HandLandmarkExtractor:
    """wraps mediapipe HandLandmarker for single hand extraction"""

    def __init__(self, model_path=None, num_hands=1):
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
