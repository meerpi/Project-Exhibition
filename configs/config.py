# config for the project - paths, constants etc

from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

MEDIAPIPE_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

# model checkpoints
CLASSIFIER_PATH = MODEL_DIR / "hand_sign_mlp.joblib"
PYTORCH_MODEL_PATH = MODEL_DIR / "hand_sign_pytorch.pt"

# --- classes ---
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = len(CLASSES)

IDX_TO_LETTER = {i: c for i, c in enumerate(CLASSES)}
LETTER_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# landmark stuff
NUM_LANDMARKS = 21
LANDMARK_DIM = NUM_LANDMARKS * 3  # 63 per hand

# normalization refs
WRIST_IDX = 0
SCALE_REF_IDX = 9  # middle finger MCP

# --- training ---
RANDOM_SEED = 42
TEST_SIZE = 0.2
MLP_HIDDEN_LAYERS = (128, 64)
MLP_MAX_ITER = 500
MLP_LEARNING_RATE_INIT = 0.001

# pytorch
PYTORCH_EPOCHS = 100
PYTORCH_BATCH_SIZE = 32
PYTORCH_LR = 0.001

# demo settings
SMOOTHING_WINDOW = 10
STABLE_THRESHOLD = 6
LETTER_HOLD_FRAMES = 20
CAMERA_INDEX = 0
