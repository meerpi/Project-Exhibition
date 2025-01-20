# config for the whole project - paths, constants, hyperparams etc

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
WORD_MODEL_PATH = MODEL_DIR / "asl_word_model.pt"
STGCN_MODEL_PATH = MODEL_DIR / "asl_word_stgcn.pt"

# --- classes ---
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = len(CLASSES)

# 20 word vocabulary - TODO maybe add more words later
WORD_CLASSES = [
    "hello", "thankyou", "please", "yes", "no",
    "sorry", "help", "water", "more", "stop",
    "good", "bad", "friend", "family", "eat",
    "drink", "again", "want", "need", "love"
]
NUM_WORD_CLASSES = len(WORD_CLASSES)

IDX_TO_LETTER = {i: c for i, c in enumerate(CLASSES)}
LETTER_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_WORD = {i: w for i, w in enumerate(WORD_CLASSES)}
WORD_TO_IDX = {w: i for i, w in enumerate(WORD_CLASSES)}

# landmark stuff
NUM_LANDMARKS = 21
LANDMARK_DIM = NUM_LANDMARKS * 3  # 63 per hand
MAX_HANDS = 2
MULTI_LANDMARK_DIM = LANDMARK_DIM * MAX_HANDS  # 126
SEQUENCE_LENGTH = 30

# normalization refs
WRIST_IDX = 0
SCALE_REF_IDX = 9  # middle finger MCP

# graph structure for STGCN
NUM_HAND_JOINTS = 21
NUM_GRAPH_NODES = NUM_HAND_JOINTS * MAX_HANDS  # 42
COORD_DIM = 3

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

# stgcn hyperparams
STGCN_HIDDEN_DIMS = [32, 64, 128]
STGCN_TEMPORAL_KERNEL = 9
STGCN_DROPOUT = 0.5
STGCN_EPOCHS = 150
# STGCN_LR = 0.0005  # tried this, worse
STGCN_LR = 0.001
STGCN_LABEL_SMOOTHING = 0.1
STGCN_WEIGHT_DECAY = 5e-4

# augmentation
AUG_SPATIAL_JITTER = 0.02
AUG_SCALE_RANGE = (0.85, 1.15)
AUG_ROTATE_MAX_DEG = 15.0
AUG_TEMPORAL_WARP = (0.8, 1.2)
AUG_MIRROR_PROB = 0.5

# demo settings
SMOOTHING_WINDOW = 10
STABLE_THRESHOLD = 6   # votes needed
LETTER_HOLD_FRAMES = 20
CAMERA_INDEX = 0
