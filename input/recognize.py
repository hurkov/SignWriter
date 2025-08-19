#!/usr/bin/env python3
from __future__ import annotations
import argparse
import math
import platform
import pickle
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import mediapipe as mp
import sys as _sys

# Compute project root and ensure it's on sys.path so `conf` and other top-level
# packages can be imported even when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

# Import configuration after ensuring sys.path includes project root
from conf.config import config
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ROOT is already defined above
MODEL_PATH = ROOT / "data" / "models" / "model.pickle"
DATASET_PATH = ROOT / "data" / "datasets" / "dataset.pickle"

mp_hands = mp.solutions.hands


def _get_screen_size() -> tuple[int, int]:
    """Best-effort screen size without risking native Tk crashes on macOS.
    - On macOS, use AppleScript only.
    - On other platforms, try tkinter; fallback to 1920x1080.
    """
    if platform.system() == 'Darwin':
        try:
            import subprocess  # nosec - local system query only
            out = subprocess.check_output([
                'osascript',
                '-e', 'tell application "Finder" to get bounds of window of desktop'
            ], text=True).strip()
            parts = [int(p.strip()) for p in out.split(',')]
            if len(parts) == 4:
                return int(parts[2]), int(parts[3])
        except Exception:
            return 1920, 1080
    # Non-macOS: tkinter is acceptable
    try:
        import tkinter as tk  # type: ignore
        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


def _compute_window_xy(anchor: str, win_w: int, win_h: int, margin: int = 20) -> tuple[int, int]:
    sw, sh = _get_screen_size()
    anchor = (anchor or "top_left").lower()
    if anchor == "center":
        x = max(0, (sw - win_w) // 2)
        y = max(0, (sh - win_h) // 2)
    elif anchor == "top_right":
        x = max(0, sw - win_w - margin)
        y = margin
    elif anchor == "bottom_left":
        x = margin
        y = max(0, sh - win_h - margin)
    elif anchor == "bottom_right":
        x = max(0, sw - win_w - margin)
        y = max(0, sh - win_h - margin)
    else:  # top_left
        x = margin
        y = margin
    return x, y

def _normalize_pred_label(pred: object) -> str:
    """Return a cleaned string label for a prediction.
    - Decodes numpy scalars and bytes
    - Normalizes common token names (space, dot/period, enter/return, exclamation, question)
    - Leaves single letters as-is
    """
    try:
        # numpy scalar
        if hasattr(pred, 'item'):
            pred = pred.item()  # type: ignore[attr-defined]
        # bytes-like
        if isinstance(pred, (bytes, bytearray)):
            try:
                pred = pred.decode('utf-8', errors='ignore')  # type: ignore[attr-defined]
            except Exception:
                pred = str(pred)
    except Exception:
        pass
    s = str(pred).strip()
    # Normalize common tokens to upper canonical forms
    low = s.lower()
    if low in ("space", " "):
        return "SPACE"
    if low in ("dot", "period", "."):
        return "DOT"
    if low in ("enter", "return", "newline"):
        return "ENTER"
    if low in ("!", "excl", "exclamation"):
        return "EXCL"
    if low in ("?", "qmark", "question"):
        return "QMARK"
    return s

def features_from_results(results) -> np.ndarray:
    """Build a 126-dim feature vector (left+right hand landmarks) from MediaPipe results."""
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if results and results.multi_hand_landmarks and results.multi_handedness:
        hands_info = list(zip(results.multi_handedness, results.multi_hand_landmarks))
        for hand_info, hand_lms in hands_info:
            label = hand_info.classification[0].label.lower()
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            if label == 'left':
                left = coords
            else:
                right = coords
    return np.concatenate([left.flatten(), right.flatten()], axis=0)


def draw_hand_overlays(frame: np.ndarray, hand_landmarks, preview_scale: float = 1.0, min_point_diameter: int = 22) -> None:
    """Draw colored palm/finger bones and joints like in training preview."""
    h, w = frame.shape[:2]
    PALM = (160, 160, 160)
    THUMB = (255, 0, 0)
    INDEX = (0, 255, 0)
    MIDDLE = (0, 140, 255)
    RING = (211, 0, 148)
    PINKY = (255, 0, 255)
    # Scale line/joint sizes so that on the displayed (possibly downscaled) preview
    # the landmark diameter is at least `min_point_diameter` pixels.
    s = max(preview_scale, 1e-3)  # display scale factor (e.g., 0.4)
    inv = 1.0 / s
    target_radius = int(math.ceil((min_point_diameter / 2.0) * inv))
    # Make dots smaller while keeping line thickness the same
    dot_scale = 0.6  # 60% of target size
    min_dot = max(3, int(round(min(h, w) * 0.004)))
    FINGER_JOINT_R = max(int(target_radius * dot_scale), min_dot)
    PALM_JOINT_R = max(int(target_radius * dot_scale * 0.8), min_dot)
    STROKE = max(int(math.ceil((min_point_diameter / 6.0) * inv)), int(round(min(h, w) * 0.004)))

    def p(i: int) -> tuple[int, int]:
        lm = hand_landmarks.landmark[i]
        return int(lm.x * w), int(lm.y * h)

    palm_points = [0, 1, 5, 9, 13, 17]
    palm_lines = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17)]
    for a, b in palm_lines:
        cv2.line(frame, p(a), p(b), PALM, STROKE, lineType=cv2.LINE_AA)
    for idx in palm_points:
        cv2.circle(frame, p(idx), PALM_JOINT_R, PALM, -1, lineType=cv2.LINE_AA)

    fingers = {
        'thumb': ([1, 2, 3, 4], THUMB),
        'index': ([5, 6, 7, 8], INDEX),
        'middle': ([9, 10, 11, 12], MIDDLE),
        'ring': ([13, 14, 15, 16], RING),
        'pinky': ([17, 18, 19, 20], PINKY),
    }
    for _, (pts, color) in fingers.items():
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(frame, p(a), p(b), color, STROKE, lineType=cv2.LINE_AA)
        for i in pts:
            cv2.circle(frame, p(i), FINGER_JOINT_R, color, -1, lineType=cv2.LINE_AA)


@lru_cache(maxsize=1)
def _find_jetbrains_mono(bold: bool = False) -> str | None:
    if not PIL_AVAILABLE:
        return None
    candidates = []
    names = [
        "JetBrainsMono-Bold.ttf" if bold else "JetBrainsMono-Regular.ttf",
        "JetBrainsMonoNL-Bold.ttf" if bold else "JetBrainsMonoNL-Regular.ttf",
    ]
    search_dirs = [
        ROOT / "assets" / "fonts",
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/System/Library/Fonts"),
    ]
    for d in search_dirs:
        try:
            if d and d.exists():
                # Exact names first
                for n in names:
                    fp = d / n
                    if fp.exists():
                        return str(fp)
                # Fuzzy search
                for p in d.glob("*JetBrains*Mono*.*ttf"):
                    if bold and ("Bold" in p.name or "Heavy" in p.name):
                        candidates.append(str(p))
                    elif not bold and ("Regular" in p.name or "Medium" in p.name):
                        candidates.append(str(p))
        except Exception:
            pass
    return candidates[0] if candidates else None


def _measure_text_pil(text: str, font_path: str, px_size: int) -> tuple[int, int, ImageFont.FreeTypeFont | None]:
    try:
        font = ImageFont.truetype(font_path, px_size)
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return int(w), int(h), font
    except Exception:
        return 0, 0, None


def draw_text(out_bgr: np.ndarray, text: str, org: tuple[int, int], color_bgr: tuple[int, int, int], desired_display_px: int, preview_scale: float, bold: bool = False, fast: bool = False) -> tuple[int, int]:
    """Draw text using JetBrains Mono if available; fallback to cv2.putText.
    Returns drawn text size (w,h) in original image pixels.
    org is top-left corner.
    """
    inv = 1.0 / max(preview_scale, 1e-3)
    target_px = max(8, int(math.ceil(desired_display_px * inv)))
    if PIL_AVAILABLE and not fast:
        font_path = _find_jetbrains_mono(bold)
        if font_path:
            w, h, font = _measure_text_pil(text, font_path, target_px)
            if font is not None and w > 0 and h > 0:
                # Convert and draw
                img = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)
                # PIL uses RGB
                color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
                draw.text(org, text, font=font, fill=color_rgb)
                out_bgr[:,:,:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                return w, h
    # Fallback to cv2 fonts
    # Approximate scale so height ~ target_px
    scale = max(0.5, target_px / 22.0)
    thickness = max(2, int(round(scale)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(out_bgr, text, (org[0], org[1] + th), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness)
    return tw, th


def measure_text(text: str, desired_display_px: int, preview_scale: float, bold: bool = False, fast: bool = False) -> tuple[int, int]:
    """Measure text size matching draw_text rendering without drawing."""
    inv = 1.0 / max(preview_scale, 1e-3)
    target_px = max(8, int(math.ceil(desired_display_px * inv)))
    if PIL_AVAILABLE and not fast:
        font_path = _find_jetbrains_mono(bold)
        if font_path:
            w, h, font = _measure_text_pil(text, font_path, target_px)
            if font is not None and w > 0 and h > 0:
                return w, h
    # Fallback: approximate with cv2 font metrics
    scale = max(0.5, target_px / 22.0)
    thickness = max(2, int(round(scale)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return tw, th


def main():
    parser = argparse.ArgumentParser(description="Live hand sign recognition")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--display-scale", type=float, default=0.4,
                        help="Preview scale factor; 0.5 makes the cv2 window half-size without changing capture/processing resolution.")
    parser.add_argument("--display-width", type=int, default=None,
                        help="Target preview width in pixels; overrides --display-scale if provided (height keeps aspect ratio).")
    parser.add_argument("--max-res", action="store_true",
                        help="Try to set the camera to the highest supported resolution.")
    parser.add_argument("--prefer-fps", action="store_true",
                        help="Prefer higher FPS over maximum resolution (tries 720p/60, 480p/60, etc.)")
    parser.add_argument("--proc-width", type=int, default=None,
                        help="Max processing width; frames are downscaled before MediaPipe to this width (keeping aspect). Improves FPS.")
    parser.add_argument("--minimal-overlay", action="store_true",
                        help="Draw only lightweight bounding boxes and labels (skip landmark bones/dots) for higher FPS.")
    parser.add_argument("--overlay-min-px", type=int, default=11,
                        help="Minimum on-screen size (in pixels) for landmarks and label text; default 11 (about half of 22).")
    parser.add_argument("--show-global-label", action="store_true",
                        help="Show a small prediction text in the window corner (hidden by default).")
    parser.add_argument("--fast-start", action="store_true",
                        help="Start faster: skip dataset-based KNN and use the trained model directly; limit resolution probing.")
    parser.add_argument("--use-model", action="store_true", help="Force using trained model instead of saved landmarks dataset")
    parser.add_argument("--type", action="store_true",
                        help="Type recognized labels into the active app (macOS Accessibility permission required; Wayland may restrict on Linux)")
    args = parser.parse_args()
    model = None
    knn = None
    classes = None
    # Prefer using saved landmarks dataset (KNN) unless --use-model or --fast-start is set
    if DATASET_PATH.exists() and not args.use_model and not args.fast_start:
        try:
            with open(DATASET_PATH, "rb") as f:
                data = pickle.load(f)
            X = np.array(data.get("X", []), dtype=np.float32)
            y = np.array(data.get("y", []))
            if len(X) and len(y):
                try:
                    from sklearn.neighbors import KNeighborsClassifier  # type: ignore
                except Exception:
                    print("scikit-learn is required to use the saved dataset (KNN). Please install scikit-learn.", flush=True)
                    X = np.empty((0, 1), dtype=np.float32)
                    y = np.empty((0,), dtype=object)
                if len(X) and len(y):
                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(X, y)
                    classes = np.unique(y)
                    print(f"Using saved landmarks dataset with {len(y)} samples and {len(classes)} classes.", flush=True)
                else:
                    print("Saved dataset unavailable or scikit-learn missing; falling back to trained model if available.", flush=True)
            else:
                print("Saved dataset is empty; falling back to trained model if available.", flush=True)
        except Exception as e:
            print(f"Failed to load dataset: {e}; falling back to model.", flush=True)

    if knn is None:
        if not MODEL_PATH.exists():
            print(f"Model not found at {MODEL_PATH}, and no usable dataset at {DATASET_PATH}. Train first.", flush=True)
            return 1
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("Using trained model for recognition.", flush=True)
        except ModuleNotFoundError as e:
            print(f"Failed to load model: {e}. scikit-learn is required. Please install scikit-learn.", flush=True)
            return 1
        except Exception as e:
            print(f"Failed to load model: {e}", flush=True)
            return 1

    sysname = platform.system()

    def _try_open(idx: int, backend: int | None, label: str) -> tuple[cv2.VideoCapture, str] | None:
        cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        # Validate by attempting a frame read quickly
        ok, _ = cap.read()
        if not ok:
            try:
                cap.release()
            except Exception:
                pass
            return None
        return cap, label

    # Build candidate indices: start with requested, then common fallbacks.
    req = int(args.camera) if args.camera is not None else -1
    candidates = []
    if req >= 0:
        candidates.append(req)
    for c in [0, 1, 2, 3]:
        if c not in candidates:
            candidates.append(c)

    cap = None
    backend = "DEFAULT"
    for idx in candidates:
        # Backend order by OS
        if sysname == 'Darwin':
            backend_order = [
                (cv2.CAP_AVFOUNDATION, 'AVFOUNDATION'),
                (None, 'DEFAULT'),
            ]
        elif sysname == 'Windows':
            backend_order = [
                (cv2.CAP_DSHOW, 'DSHOW'),
                (cv2.CAP_MSMF, 'MSMF'),
                (None, 'DEFAULT'),
            ]
        else:  # Linux and others
            backend_order = [
                (cv2.CAP_V4L2, 'V4L2'),
                (None, 'DEFAULT'),
            ]
        for bconst, blabel in backend_order:
            res = _try_open(idx, bconst, blabel)
            if res is not None:
                cap, backend = res
                args.camera = idx
                break
        if cap is not None:
            break

    if cap is None:
        print(f"Error: Could not open camera. Tried indices: {candidates}", flush=True)
        return 1

    # Prefer explicit width/height if given; otherwise, apply FPS-first or resolution-first strategy.
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Try to improve throughput via MJPG when available
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    # Reduce buffering to keep latency low
    try:
        if args.prefer_fps:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if (not args.width and not args.height) and args.prefer_fps:
        # Try high-FPS friendly modes first (favor 720p/60 or 480p/60 which many external cams support)
        fps_targets = [120.0, 60.0, 90.0, 30.0]
        size_candidates = [
            (1280, 720),  # 720p
            (640, 480),   # 480p
            (1920, 1080), # 1080p fallback
        ]
        selected = None
        for tgt_fps in fps_targets:
            try:
                cap.set(cv2.CAP_PROP_FPS, tgt_fps)
            except Exception:
                pass
            for w_req, h_req in size_candidates:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_req)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_req)
                w_eff = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h_eff = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps_eff = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                if abs(w_eff - w_req) <= 16 and abs(h_eff - h_req) <= 16:
                    selected = (w_eff, h_eff, fps_eff)
                    break
            if selected is not None:
                break
        if selected is None:
            selected = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0), float(cap.get(cv2.CAP_PROP_FPS) or 0))
    elif (not args.width and not args.height) and args.max_res:
        # Try common high resolutions from highest to lower
        # Include 4:3 and 16:9 options to match various webcams
        candidates = (
            [(1920, 1080), (1280, 720)] if args.fast_start else
            [
                (3840, 2160), (2560, 1440), (2560, 1600),
                (2048, 1536), (1920, 1200), (1920, 1080),
                (1600, 1200), (1440, 1080), (1280, 960), (1280, 800), (1280, 720),
                (1024, 768), (800, 600), (640, 480),
            ]
        )
        # Try MJPG for better throughput
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        selected = None
        for w_req, h_req in candidates:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_req)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_req)
            w_eff = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h_eff = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if abs(w_eff - w_req) <= 8 and abs(h_eff - h_req) <= 8:
                selected = (w_eff, h_eff)
                break
        # Fallback: accept whatever camera opened with
        if selected is None:
            selected = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))

    eff_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    eff_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    eff_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    print(f"Using camera index {args.camera} with backend: {backend} | {eff_w}x{eff_h} @ {eff_fps:.1f}fps", flush=True)
    # cap validity already checked

    # Show a window immediately so UI appears fast
    try:
        cv2.namedWindow("Recognition", cv2.WINDOW_AUTOSIZE)
        placeholder = np.zeros((max(1, eff_h if eff_h > 0 else 480), max(1, eff_w if eff_w > 0 else 640), 3), dtype=np.uint8)
        cv2.putText(placeholder, "Starting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        cv2.imshow("Recognition", placeholder)
        # Position the window based on settings
        try:
            pos = config.get("ui", "window_position", "top_left")
            ph_h, ph_w = placeholder.shape[:2]
            x, y = _compute_window_xy(pos, ph_w, ph_h)
            cv2.moveWindow("Recognition", x, y)
        except Exception:
            pass
        cv2.waitKey(1)
    except Exception:
        pass

    print("Recognition started. Press 'q' to quit.", flush=True)
    # On macOS, if typing is requested and permission isn't granted yet, open Accessibility settings
    if args.type and platform.system() == 'Darwin':
        try:
            from output.keyboard_simulator import has_access as _kb_has, request_access as _kb_req
            if not _kb_has():
                _kb_req()
        except Exception:
            pass
    last_pred = None
    last_typed = None
    # Confidence threshold for typing: use configured sensitivity (0..1)
    try:
        type_thresh = float(config.get("detection", "confidence_threshold", 0.75))
    except Exception:
        type_thresh = 0.75
    # Minimal time between typing any label again (global cooldown)
    type_cooldown_s = 1.0
    last_type_time = 0.0
    # Smart casing settings
    try:
        smart_case = bool(config.get("keyboard", "smart_case", True))
        title_case_each = bool(config.get("keyboard", "title_case_each_word", False))
        pn = config.get("keyboard", "proper_nouns", [])
        proper_nouns = [str(x).strip().lower() for x in (pn or []) if str(x).strip()]
    except Exception:
        smart_case = True
        title_case_each = False
        proper_nouns = []
    # Casing state across typed characters
    sentence_start = True
    current_word = ""
    # Use persistent MediaPipe Hands for video
    mp_model_complexity = 0 if args.prefer_fps else 1
    # Track up to 2 hands like before; FPS improvements remain via model_complexity and scaling
    mp_max_hands = 2
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=mp_max_hands,
        model_complexity=mp_model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        try:
            positioned = False
            had_hand_prev = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                h, w = frame.shape[:2]
                # Optional downscale before MediaPipe to boost FPS
                if args.proc_width:
                    target_w = max(160, int(args.proc_width))
                elif args.prefer_fps:
                    target_w = 960 if w > 1280 else (640 if w > 960 else None)
                else:
                    target_w = None
                if target_w and w > target_w:
                    scale = float(target_w) / float(w)
                    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                    image_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                has_hand = bool(results and getattr(results, 'multi_hand_landmarks', None))

                # Build features for prediction
                feats = features_from_results(results).astype(np.float32)
                X = np.array([feats])
                try:
                    if has_hand:
                        if knn is not None:
                            pred = knn.predict(X)[0]
                            dists, idxs = knn.kneighbors(X, n_neighbors=1, return_distance=True)
                            dist = float(dists[0][0])
                            conf = 1.0 / (1.0 + dist)
                        else:
                            proba = None
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(X)[0]
                                pred_idx = int(np.argmax(proba))
                                pred = model.classes_[pred_idx]
                                conf = float(proba[pred_idx])
                            else:
                                pred = model.predict(X)[0]
                                conf = 1.0
                    else:
                        pred = "?"
                        conf = 0.0
                except Exception:
                    pred = "?"
                    conf = 0.0

                # Draw overlays on the frame directly to avoid extra copies
                out = frame
                # Determine preview scale now (used to ensure min on-screen sizes)
                if args.display_width:
                    preview_scale = max(0.05, min(args.display_width / float(w), 2.0))
                else:
                    preview_scale = float(args.display_scale or 1.0)
                inv = 1.0 / max(preview_scale, 1e-3)
                min_on_screen = max(6, min(int(args.overlay_min_px), 64))

                # Optional small corner label; hidden by default
                fast_draw = bool(getattr(args, 'prefer_fps', False))
                if args.show_global_label:
                    base_text = f"{pred} ({conf:.2f})"
                    # Draw at top-left with margin
                    margin = int(math.ceil(12 * inv))
                    draw_text(out, base_text, (margin, margin), (0, 165, 255), min_on_screen, preview_scale, bold=True, fast=fast_draw)

                # Show palm + fingers overlays by default (even in FPS mode). Only skip when explicitly requested.
                minimal = bool(args.minimal_overlay)
                if results and results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        # Optionally skip detailed bones/dots to save time
                        if not minimal:
                            # Training-style landmarks, upscaled so points appear >= min_on_screen px on display
                            draw_hand_overlays(out, hand_lms, preview_scale=preview_scale, min_point_diameter=min_on_screen)

                        # Compute square bounding box around landmarks
                        xs = np.array([lm.x for lm in hand_lms.landmark], dtype=np.float32)
                        ys = np.array([lm.y for lm in hand_lms.landmark], dtype=np.float32)
                        min_x, max_x = float(xs.min()), float(xs.max())
                        min_y, max_y = float(ys.min()), float(ys.max())
                        cx = (min_x + max_x) / 2.0
                        cy = (min_y + max_y) / 2.0
                        side = max(max_x - min_x, max_y - min_y)
                        side *= 1.2  # add padding
                        # Convert to pixel coords ensuring in-bounds
                        half_px = int(round((side * w) / 2.0))
                        cx_px = int(round(cx * w))
                        cy_px = int(round(cy * h))
                        x1 = max(0, cx_px - half_px)
                        y1 = max(0, cy_px - half_px)
                        x2 = min(w - 1, cx_px + half_px)
                        y2 = min(h - 1, cy_px + half_px)
                        color = (204, 102, 255)  # pink-ish for box
                        thickness = max(int(math.ceil((min_on_screen / 6.0) * inv)), int(round(min(h, w) * 0.004)))

                        # Label banner on top of box: make percentage text >= min_on_screen px tall on display
                        label = f"{str(pred)} {int(conf * 100)}%"
                        pad_y = int(math.ceil(12 * inv))
                        pad_x = int(math.ceil(12 * inv))
                        # Make the banner text a little bigger on-screen while staying readable
                        banner_target = max(min_on_screen + 2, int(round(min_on_screen * 1.25)))
                        # Measure with target on-screen pixels
                        tw, th = measure_text(label, banner_target, preview_scale, bold=True, fast=fast_draw)
                        rx1, ry1 = x1, max(0, y1 - th - pad_y)
                        rx2, ry2 = x1 + tw + pad_x, y1
                        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (211, 0, 148), -1)
                        # Draw text with a small inner padding
                        text_x = rx1 + int(6 * inv)
                        text_y = ry1 + int(4 * inv)
                        draw_text(out, label, (text_x, text_y), (255, 255, 255), banner_target, preview_scale, bold=True, fast=fast_draw)
                        # Square box
                        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

                # Optionally scale display only (keep processing at full resolution)
                # Compute final preview scale and resize for display
                if args.display_width:
                    scale = max(0.05, min(args.display_width / float(w), 2.0))
                else:
                    scale = float(args.display_scale or 1.0)
                if scale != 1.0:
                    # Clamp to sane range
                    scale = max(0.05, min(float(scale), 2.0))
                    display = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                else:
                    display = out

                cv2.imshow("Recognition", display)
                # Position once with the final display size
                if not positioned:
                    try:
                        pos = config.get("ui", "window_position", "top_left")
                        dh, dw = display.shape[:2]
                        x, y = _compute_window_xy(pos, dw, dh)
                        cv2.moveWindow("Recognition", x, y)
                    except Exception:
                        pass
                    positioned = True

                # Print when prediction changes to avoid spamming
                if has_hand and pred != last_pred:
                    print(f"Prediction: {pred} conf={conf:.3f}", flush=True)
                    last_pred = pred
                # If hand disappeared, reset sentence/word state and last_pred marker
                if (not has_hand) and had_hand_prev:
                    sentence_start = True
                    current_word = ""
                    last_pred = None
                    # also reset last_typed cooldown timer so next visible char can type promptly
                    try:
                        import time as _time
                        last_type_time = _time.time()
                    except Exception:
                        pass

                # Optionally type recognized label when sufficiently confident
                if args.type and has_hand:
                    try:
                        import time as _time
                        from output.keyboard_simulator import type_text as _type_text
                        now = _time.time()
                        val = _normalize_pred_label(pred)
                        is_alpha_char = len(val) == 1 and val.isalpha()
                        is_space = val in ("SPACE", " ")
                        is_period = (val == "DOT")
                        is_exclaim = (val == "EXCL")
                        is_question = (val == "QMARK")
                        is_enter = (val == "ENTER")
                        should_type = conf >= type_thresh and (now - last_type_time) >= type_cooldown_s

                        to_type: str | None = None
                        if should_type:
                            if is_enter:
                                to_type = "\n"
                                sentence_start = True
                                current_word = ""
                            elif is_space:
                                to_type = " "
                                current_word = ""
                                # don't reset sentence_start on space
                            elif is_period or is_exclaim or is_question:
                                to_type = "." if is_period else ("!" if is_exclaim else "?")
                                sentence_start = True
                                current_word = ""
                            elif is_alpha_char:
                                ch = val.lower()
                                if smart_case:
                                    start_of_word = len(current_word) == 0
                                    pn_hint = start_of_word and any(pn.startswith(ch) for pn in proper_nouns)
                                    if sentence_start or title_case_each or pn_hint:
                                        out_ch = ch.upper()
                                        sentence_start = False
                                    else:
                                        out_ch = ch.lower()
                                    current_word += ch
                                else:
                                    out_ch = val
                                to_type = out_ch

                        if to_type is not None:
                            if _type_text(to_type):
                                last_typed = to_type
                                last_type_time = now
                    except Exception:
                        # Typing is best-effort; ignore any failures
                        pass

                # Track hand presence for next iteration
                had_hand_prev = has_hand

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
