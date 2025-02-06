# real-time ASL demo - fingerspelling + word signs + TTS
#
# modes: AUTO (switches based on hand count), LETTERS, WORDS
# keys: M=mode, S=speak, SPACE/BACKSPACE/C=text editing, Q=quit

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT))  # need this for configs import

from configs.config import (
    IDX_TO_LETTER,
    IDX_TO_WORD,
    CLASSES,
    WORD_CLASSES,
    SMOOTHING_WINDOW,
    STABLE_THRESHOLD,
    LETTER_HOLD_FRAMES,
    SEQUENCE_LENGTH,
    MULTI_LANDMARK_DIM,
    CAMERA_INDEX,
    PYTORCH_MODEL_PATH,
    WORD_MODEL_PATH,
)
from src.landmarks import HandLandmarkExtractor
from src.model import load_model, load_word_model, get_device, HAS_TORCH

if HAS_TORCH:
    import torch


class PredictionSmoother:
    """majority vote over a sliding window"""
    def __init__(self, window_size=SMOOTHING_WINDOW, threshold=STABLE_THRESHOLD):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold

    def update(self, prediction):
        self.window.append(prediction)
        if prediction is None:
            return None
        votes = sum(1 for p in self.window if p == prediction)
        if votes >= self.threshold:
            return IDX_TO_LETTER.get(prediction)
        return None


class WordBuilder:
    """text buffer that accumulates letters and words"""
    def __init__(self, hold_frames=LETTER_HOLD_FRAMES):
        self.sentence = ""
        self.last_letter = None
        self.hold_counter = 0
        self.hold_frames = hold_frames
        self.letter_added = False

    def update_letter(self, stable_letter):
        if stable_letter is None:
            self.hold_counter = 0
            self.last_letter = None
            self.letter_added = False
            return

        if stable_letter == self.last_letter:
            if not self.letter_added:
                self.hold_counter += 1
                if self.hold_counter >= self.hold_frames:
                    self.sentence += stable_letter
                    self.letter_added = True
        else:
            self.last_letter = stable_letter
            self.hold_counter = 1
            self.letter_added = False

    def add_word(self, word):
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " " + word.upper() + " "
        else:
            self.sentence += word.upper() + " "
        self.last_letter = None
        self.hold_counter = 0
        self.letter_added = False

    def add_space(self):
        self.sentence += " "
        self.last_letter = None

    def backspace(self):
        if self.sentence:
            self.sentence = self.sentence[:-1]

    def clear(self):
        self.sentence = ""
        self.last_letter = None


class TTSEngine:
    def __init__(self):
        self._engine = None
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 150)
            print("  TTS ready (pyttsx3)")
        except Exception:
            print("  TTS not available - will print to console instead")

    def speak(self, text):
        if not text.strip():
            return
        print(f"\n[Speaking]: \"{text}\"")
        if self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")


def run_demo(camera_index=CAMERA_INDEX):
    print("\n" + "="*60)
    print("  ASL DEMO - fingerspelling + word signs")
    print("="*60)

    # load models
    print("\nloading models...")
    letter_model = load_model()

    device = get_device()
    word_model = None
    word_model_type = "gru"  # default fallback
    try:
        word_model, word_model_type = load_word_model()
        print("loaded:", word_model_type)  # debug
    except FileNotFoundError:
        print(f"  no word model found - word mode disabled")
    except Exception as e:
        print(f"  couldn't load word model: {e}")

    extractor = HandLandmarkExtractor(num_hands=2)
    smoother = PredictionSmoother()
    builder = WordBuilder()
    tts = TTSEngine()

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    last_word_time = 0.0

    # 0=auto, 1=letters, 2=words
    modes = ["AUTO", "LETTERS", "WORDS"]
    mode_idx = 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: cant open camera {camera_index}")
        return

    frame_times = deque(maxlen=30)
    prev_time = time.perf_counter()

    print(f"\n  M=mode  S=speak  SPACE=space  BACKSPACE=delete  C=clear  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        t_start = time.perf_counter()

        landmarks_single = extractor.extract(frame)

        if word_model_type == "transformer":
            landmarks_multi = extractor.extract_raw_multi_hand(frame)
        else:
            landmarks_multi = extractor.extract_multi_hand(frame)

        if landmarks_multi is None:
            landmarks_multi = np.zeros(MULTI_LANDMARK_DIM, dtype=np.float32)

        frame_buffer.append(landmarks_multi)

        active_mode = modes[mode_idx]
        current_pred_text = ""
        current_pred_type = ""

        has_two_hands = np.any(landmarks_multi[63:] != 0)
        is_word_mode = (active_mode == "WORDS") or (active_mode == "AUTO" and has_two_hands)

        if is_word_mode and word_model is not None and len(frame_buffer) == SEQUENCE_LENGTH:
            current_pred_type = "WORD"
            now = time.perf_counter()
            if now - last_word_time > 1.5:
                seq_arr = np.array(frame_buffer, dtype=np.float32)
                word_pred_idx = word_model.predict(seq_arr)
                predicted_word = IDX_TO_WORD.get(word_pred_idx, "?")
                current_pred_text = predicted_word.upper()
                builder.add_word(predicted_word)
                last_word_time = now
        else:
            current_pred_type = "LETTER"
            raw_pred = None
            if landmarks_single is not None:
                feats = landmarks_single.reshape(1, -1)
                if hasattr(letter_model, "predict"):
                    raw_pred = int(letter_model.predict(feats)[0])

            stable_letter = smoother.update(raw_pred)
            if stable_letter:
                current_pred_text = stable_letter
            builder.update_letter(stable_letter)

        t_end = time.perf_counter()
        pipeline_ms = (t_end - t_start) * 1000

        now = time.perf_counter()
        frame_times.append(now - prev_time)
        prev_time = now
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

        # --- draw UI ---
        display = frame.copy()
        h, w = display.shape[:2]

        # top bar
        cv2.rectangle(display, (0, 0), (w, 90), (20, 20, 20), -1)

        cv2.putText(display, f"Mode: {active_mode}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(display, f"Type: {current_pred_type}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        if current_pred_text:
            color = (0, 255, 0) if current_pred_type == "LETTER" else (255, 200, 0)
            cv2.putText(display, current_pred_text, (w // 2 - 40, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
        else:
            cv2.putText(display, "...", (w // 2 - 20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)

        cv2.putText(display, f"FPS: {fps:.0f}", (w - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
        cv2.putText(display, f"{pipeline_ms:.1f}ms", (w - 140, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

        # bottom bar - sentence
        cv2.rectangle(display, (0, h - 60), (w, h), (15, 15, 15), -1)
        sentence_text = builder.sentence if builder.sentence else "(sign to build text)"
        text_color = (255, 255, 255) if builder.sentence else (120, 120, 120)
        cv2.putText(display, sentence_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        cv2.imshow("ASL Demo", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            mode_idx = (mode_idx + 1) % len(modes)
            print(f"mode -> {modes[mode_idx]}")
        elif key == ord("s"):
            tts.speak(builder.sentence)
        elif key == ord(" "):
            builder.add_space()
        elif key == 8:
            builder.backspace()
        elif key == ord("c"):
            builder.clear()

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    print(f"\nFinal sentence: \"{builder.sentence}\"")


def main():
    parser = argparse.ArgumentParser(description="ASL demo")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    args = parser.parse_args()
    run_demo(camera_index=args.camera)


if __name__ == "__main__":
    main()
