# ASL Hand Sign Recognition

Real-time ASL fingerspelling (A–Z) and whole-word gesture recognition using MediaPipe + PyTorch.

## What It Does

- Recognizes 26 static fingerspelling letters (A–Z)
- Recognizes 20 whole-word ASL signs: `hello`, `thankyou`, `please`, `yes`, `no`, `sorry`, `help`, `water`, `more`, `stop`, `good`, `bad`, `friend`, `family`, `eat`, `drink`, `again`, `want`, `need`, `love`
- Text buffer with space/backspace + TTS output

**Note:** only works with isolated signs (pauses between them), closed vocabulary, no facial expressions.

## How It Works

```
                     ┌──> Static MLP (63-dim) ────> Letter (A-Z) ──┐
Webcam ──> MediaPipe ┤                                              ├──> Text Buffer -> TTS
                     └──> 1D-Conv + BiGRU (30x126) ──> Word (20) ──┘
```

1. MediaPipe extracts 2×21 hand landmarks (126 dims)
2. Wrist-centered + scale-normalized
3. Engine 1: MLP classifies static single-hand poses → letters
4. Engine 2: Temporal model classifies 30-frame sequences → words
5. Auto mode switches based on hand count

## Setup

```bash
source .venv/bin/activate
uv pip install -r requirements.txt

# mediapipe model (one-time)
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

## Usage

```bash
# live demo
python src/demo.py

# train fingerspelling model
python src/train.py

# train word model (needs data/word_landmarks.npz)
python src/extract_word_landmarks.py
python src/train_word_model.py
```

**Demo keys**: M=mode, S=speak, SPACE=space, BACKSPACE=delete, C=clear, Q=quit

## Project Structure

```
Project-Exhibition/
├── src/
│   ├── landmarks.py            # mediapipe extraction
│   ├── model.py                # all the pytorch models
│   ├── hand_graph.py           # skeleton graph for stgcn
│   ├── demo.py                 # live demo
│   ├── collect_data.py         # webcam collection
│   ├── extract_from_images.py  # image folder extraction
│   ├── extract_word_landmarks.py
│   ├── train.py                # A-Z training
│   └── train_word_model.py     # word model training
├── configs/config.py
├── data/
├── models/
├── requirements.txt
└── README.md
```

## Requirements

- mediapipe, torch, opencv-python-headless, scikit-learn, numpy, matplotlib
