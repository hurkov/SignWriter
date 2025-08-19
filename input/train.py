#!/usr/bin/env python3
import os
import sys
import time
import argparse
import pickle
import csv
from pathlib import Path
import shutil

import cv2
import numpy as np
import mediapipe as mp
import platform
from typing import Any
import threading
import contextlib
import sys as _sys
# Ensure project root is importable when running script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from conf.config import config

# Paths relative to project root (this file is input/train.py)
# ROOT already set above
IMAGES_DIR = ROOT / "data" / "images"
DATASETS_DIR = ROOT / "data" / "datasets"
MODELS_DIR = ROOT / "data" / "models"
LANDMARKS_DIR = ROOT / "output" / "landmarks"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands

# Reduce noisy logs from TF/ABSL where possible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
try:
    from absl import logging as absl_logging  # type: ignore
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass


@contextlib.contextmanager
def _quiet_cpp_stderr():
    """Temporarily redirect C/C++ stderr to /dev/null to hide Mediapipe/TFLite warnings."""
    try:
        old_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            os.dup2(old_fd, 2)
        except Exception:
            pass
        try:
            os.close(devnull)
        except Exception:
            pass
        try:
            os.close(old_fd)
        except Exception:
            pass


def _draw_hand_overlays(frame: np.ndarray, hand_landmarks: Any) -> None:
    """Draw colored overlays for palm (gray) and each finger (distinct color)."""
    h, w = frame.shape[:2]
    # Colors (BGR)
    PALM = (160, 160, 160)
    THUMB = (255, 0, 0)      # Blue
    INDEX = (0, 255, 0)      # Green
    MIDDLE = (0, 140, 255)   # Orange
    RING = (211, 0, 148)     # Purple
    PINKY = (255, 0, 255)    # Magenta
    # Slimmer strokes and smaller points (scale lightly with frame size)
    base = max(1, int(round(min(h, w) * 0.003)))
    STROKE = max(1, base)            # typically 1-3 px
    PALM_JOINT_R = max(2, base)      # typically 2-3 px
    FINGER_JOINT_R = max(2, base+1)  # typically 3-4 px

    # Helper to get pixel coords
    def p(i: int) -> tuple[int, int]:
        lm = hand_landmarks.landmark[i]
        return int(lm.x * w), int(lm.y * h)

    # Palm points and lines
    palm_points = [0, 1, 5, 9, 13, 17]
    palm_lines = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17)]
    for a, b in palm_lines:
        cv2.line(frame, p(a), p(b), PALM, STROKE, lineType=cv2.LINE_AA)
    for idx in palm_points:
        cv2.circle(frame, p(idx), PALM_JOINT_R, PALM, -1, lineType=cv2.LINE_AA)

    # Fingers connections
    fingers = {
        'thumb': ([1, 2, 3, 4], THUMB),
        'index': ([5, 6, 7, 8], INDEX),
        'middle': ([9, 10, 11, 12], MIDDLE),
        'ring': ([13, 14, 15, 16], RING),
        'pinky': ([17, 18, 19, 20], PINKY),
    }
    for _, (pts, color) in fingers.items():
        # draw bones
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(frame, p(a), p(b), color, STROKE, lineType=cv2.LINE_AA)
        # draw joints
        for i in pts:
            cv2.circle(frame, p(i), FINGER_JOINT_R, color, -1, lineType=cv2.LINE_AA)


def collect_images(
    label: str,
    samples: int,
    camera_index: int = 0,
    width: int | None = None,
    height: int | None = None,
    display_width: int | None = 640,
    display_scale: float | None = None,
) -> None:
    """Collect images from webcam and save under data/images/<label>.
    Shows a live OpenCV window while capturing.
    """
    label_dir = IMAGES_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    sysname = platform.system()
    if sysname == 'Darwin':
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        backend = "AVFOUNDATION"
    elif sysname == 'Windows':
        # Try DirectShow first, then fallback to default
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        backend = "DSHOW"
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
            backend = "MSMF"
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(camera_index)
            backend = "DEFAULT"
    else:
        # Linux: prefer V4L2
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        backend = "V4L2"
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(camera_index)
            backend = "DEFAULT"
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Prefer higher FPS / lower latency when previewing during collection
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    try:
        # Ask for 60 FPS when the camera supports it
        cap.set(cv2.CAP_PROP_FPS, 60)
    except Exception:
        pass
    try:
        # Reduce internal buffering to keep latency low
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print(f"Using camera index {camera_index} with backend: {backend}", flush=True)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    existing = list(label_dir.glob("*.jpg"))
    start_idx = len(existing)

    print(f"Collecting {samples} samples for label '{label}'. Press 'q' to abort.", flush=True)
    collected = 0

    # Slow down sampling rate: ~10 samples per 5 seconds => 1 every 0.5s
    sample_interval = 0.5
    last_saved = time.monotonic()

    # Prepare MediaPipe Hands for live overlay (favor FPS during collection)
    with _quiet_cpp_stderr():
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,  # faster inference, good enough for preview while collecting
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            # Window management: position once after first show
            positioned = False
            # Cache desired anchor once (avoid repeated config lookups)
            try:
                desired_anchor = config.get("ui", "window_position", "top_left")
            except Exception:
                desired_anchor = "top_left"
            # Match recognition preview sizing: prefer fixed display width if provided
            target_display_width = int(display_width) if display_width else None
            fixed_display_scale = float(display_scale) if display_scale else None
            try:
                cv2.namedWindow("Collecting Samples", cv2.WINDOW_AUTOSIZE)
            except Exception:
                pass
            # Warmup frames (brief)
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    continue
                h, w = frame.shape[:2]
                # Downscale before MediaPipe to speed up processing, keep aspect ratio
                target_w = 960 if w > 1280 else (640 if w > 960 else None)
                if target_w:
                    scale = float(target_w) / float(w)
                    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                    proc_img = small
                else:
                    proc_img = frame
                rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                # Draw overlays on the display image (proc_img) to avoid copying full-res frames
                out = proc_img
                if results and results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        _draw_hand_overlays(out, hand_lms)
                # Downscale for display only (save remains full-res)
                if target_display_width:
                    scale = max(0.05, min(target_display_width / float(w), 2.0))
                else:
                    scale = float(fixed_display_scale or 1.0)
                if scale != 1.0:
                    display = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                else:
                    display = out
                cv2.imshow("Collecting Samples", display)
                # Position the window once using display size
                if not positioned:
                    try:
                        oh, ow = display.shape[:2]
                        from input.recognize import _compute_window_xy  # reuse helper
                        x, y = _compute_window_xy(desired_anchor, ow, oh)
                        cv2.moveWindow("Collecting Samples", x, y)
                    except Exception:
                        pass
                    positioned = True
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Aborted during warmup.", flush=True)
                    return

            # Capture loop
            while collected < samples:
                ret, frame = cap.read()
                if not ret:
                    # keep trying to read frames without spamming messages
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Aborted by user.", flush=True)
                        break
                    continue

                h, w = frame.shape[:2]
                # Downscale before MediaPipe to boost FPS, keep aspect
                target_w = 960 if w > 1280 else (640 if w > 960 else None)
                if target_w:
                    scale = float(target_w) / float(w)
                    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                    proc_img = small
                else:
                    proc_img = frame
                rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                # Show overlayed frame (on display image only)
                out = proc_img
                if results and results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        _draw_hand_overlays(out, hand_lms)

                # Downscale for display only
                if target_display_width:
                    scale = max(0.05, min(target_display_width / float(w), 2.0))
                else:
                    scale = float(fixed_display_scale or 1.0)
                if scale != 1.0:
                    display = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                else:
                    display = out
                cv2.imshow("Collecting Samples", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Aborted by user.", flush=True)
                    break

                now = time.monotonic()
                if now - last_saved >= sample_interval:
                    img_path = label_dir / f"{label}_{start_idx + collected:04d}.jpg"
                    # Save the raw frame (not overlay) for clean dataset
                    cv2.imwrite(str(img_path), frame)
                    collected += 1
                    last_saved = now
                    print(f"Collecting {collected}/{samples}", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {collected} images for '{label}'.", flush=True)


def extract_hand_landmarks(image_bgr: np.ndarray) -> np.ndarray:
    """Extract 3D hand landmarks (left + right) using MediaPipe Hands.
    Returns a fixed-length feature vector of shape (126,) = 2 hands * 21 points * 3 coords.
    Missing hands are zero-padded.
    """
    with _quiet_cpp_stderr():
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

    # Prepare zero arrays for two hands
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        # Pair handedness with landmarks
        hands_info = list(zip(results.multi_handedness, results.multi_hand_landmarks))
        for hand_info, hand_lms in hands_info:
            label = hand_info.classification[0].label.lower()  # 'left' or 'right'
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            if label == 'left':
                left = coords
            else:
                right = coords

    # Flatten to 126-dim vector
    features = np.concatenate([left.flatten(), right.flatten()], axis=0)
    return features


def build_dataset() -> tuple[list[list[float]], list[str]]:
    """Create dataset (X, y) from all images under data/images/* using MediaPipe Hands.
    Also prints progress markers as TRAIN_PROGRESS <0-100> for UI consumption.
    """
    X: list[list[float]] = []
    y: list[str] = []

    labels = [d.name for d in IMAGES_DIR.iterdir() if d.is_dir()]
    if not labels:
        print("No labels found in images directory. Nothing to build.")
        return X, y

    # Count total images for progress
    total_images = 0
    for label in labels:
        total_images += len(list((IMAGES_DIR / label).glob("*.jpg")))
    processed = 0
    last_pct = -1

    if total_images == 0:
        print("No images found to build dataset.", flush=True)
        return X, y

    print("TRAIN_PHASE building", flush=True)
    for label in sorted(labels):
        img_files = sorted((IMAGES_DIR / label).glob("*.jpg"))
        print(f"Processing {len(img_files)} images for label '{label}'...")
        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                processed += 1
                continue
            feats = extract_hand_landmarks(img).astype(np.float32)
            X.append(feats.tolist())
            y.append(label)
            processed += 1
            pct = int((processed / max(1, total_images)) * 80)  # building up to 80%
            if pct != last_pct:
                print(f"TRAIN_PROGRESS {pct}", flush=True)
                last_pct = pct

    return X, y


def save_landmarks_csv(X: list[list[float]], y: list[str], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["label"] + [f"f{i}" for i in range(len(X[0]))] if X else ["label"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for feats, label in zip(X, y):
            writer.writerow([label] + feats)


def train_classifier(X: list[list[float]], y: list[str], out_model: Path) -> bool:
    from sklearn.ensemble import RandomForestClassifier

    classes = sorted(set(y))
    if len(classes) < 2:
        print("Warning: Need at least 2 classes to train a classifier. Skipping model training.")
        return False

    print("TRAIN_PHASE training", flush=True)
    # Emit smooth progress during fit (85..99)
    stop_evt = threading.Event()
    def _emit():
        p = 85
        while not stop_evt.is_set() and p < 99:
            print(f"TRAIN_PROGRESS {p}", flush=True)
            p += 1
            time.sleep(0.2)
    t = threading.Thread(target=_emit, daemon=True)
    t.start()
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    with _quiet_cpp_stderr():
        clf.fit(X, y)
    stop_evt.set()
    print("TRAIN_PROGRESS 99", flush=True)

    with open(out_model, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to: {out_model}")
    print("TRAIN_PROGRESS 100", flush=True)
    return True


def cleanup_images() -> int:
    """Delete all collected image samples under data/images. Returns number of files removed."""
    removed = 0
    try:
        for label_dir in IMAGES_DIR.iterdir():
            if not label_dir.is_dir():
                continue
            for img in label_dir.glob("*.jpg"):
                try:
                    img.unlink()
                    removed += 1
                except Exception:
                    pass
            # Remove empty label dir if no other files left
            try:
                if not any(label_dir.iterdir()):
                    label_dir.rmdir()
            except Exception:
                pass
    except Exception:
        pass
    return removed


def main():
    parser = argparse.ArgumentParser(description="Unified training pipeline")
    parser.add_argument("--label", required=True, help="Class label (e.g., bound key)")
    parser.add_argument("--samples", type=int, required=True, help="Number of samples to capture")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=None, help="Camera width (optional)")
    parser.add_argument("--height", type=int, default=None, help="Camera height (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing samples and per-label datasets for this label")
    parser.add_argument("--display-width", type=int, default=640, help="Target preview width in pixels; matches recognition window size")
    parser.add_argument("--display-scale", type=float, default=None, help="Alternative scale factor if width is not desired; 1.0 = native size")
    args = parser.parse_args()

    # Optional overwrite of existing label data (images and per-label dataset/landmarks)
    label_dir = IMAGES_DIR / args.label
    if args.overwrite and label_dir.exists():
        # Remove only images for this label
        for p in label_dir.glob("*.jpg"):
            try:
                p.unlink()
            except Exception:
                pass
        # Remove per-label dataset/landmarks files if present
        try:
            (DATASETS_DIR / f"{args.label}.pickle").unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            (LANDMARKS_DIR / f"{args.label}.csv").unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

    # Step 1: Collect images for this label
    collect_images(
        args.label,
        args.samples,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        display_width=args.display_width,
        display_scale=args.display_scale,
    )

    # Step 2: Build full dataset from all labels
    X, y = build_dataset()
    if not X:
        print("No data to train on. Exiting.")
        return

    # Save combined dataset
    dataset_path = DATASETS_DIR / "dataset.pickle"
    with open(dataset_path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    print(f"Dataset saved to: {dataset_path}")

    # Save combined landmarks CSV for inspection
    csv_path = LANDMARKS_DIR / "landmarks.csv"
    save_landmarks_csv(X, y, csv_path)
    print(f"Landmarks CSV saved to: {csv_path}")

    # Also save per-label dataset and landmarks for the current label
    X_label = [feats for feats, lbl in zip(X, y) if lbl == args.label]
    y_label = [lbl for lbl in y if lbl == args.label]
    if X_label:
        per_label_dataset = DATASETS_DIR / f"{args.label}.pickle"
        with open(per_label_dataset, "wb") as f:
            pickle.dump({"X": X_label, "y": y_label}, f)
        print(f"Per-label dataset saved to: {per_label_dataset}")
        per_label_csv = LANDMARKS_DIR / f"{args.label}.csv"
        save_landmarks_csv(X_label, y_label, per_label_csv)
        print(f"Per-label landmarks CSV saved to: {per_label_csv}")
    else:
        print(f"Warning: No features generated for label '{args.label}'.")

    # Step 3: Train model
    model_path = MODELS_DIR / "model.pickle"
    success = train_classifier(X, y, model_path)

    # Step 4: Cleanup collected samples if training succeeded
    if success:
        print("Cleaning up collected image samples...", flush=True)
        removed = cleanup_images()
        print(f"Deleted {removed} sample image(s) from {IMAGES_DIR}", flush=True)


if __name__ == "__main__":
    main()
