import os
import cv2
import math
import glob
import json
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Video feature extraction

def sample_frames(cap: cv2.VideoCapture, max_frames: int = 48, stride: int = 5) -> List[np.ndarray]:
    """Uniformly sample frames with a stride to reduce compute."""
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        if len(frames) >= max_frames:
            break
        idx += 1
    return frames


def compute_resolution_stats(frames: List[np.ndarray]) -> Tuple[float, float]:
    """Return average area and aspect ratio variability."""
    if not frames:
        return 0.0, 0.0
    areas = []
    ratios = []
    for f in frames:
        h, w = f.shape[:2]
        areas.append(w * h)
        ratios.append(w / max(h, 1))
    avg_area = float(np.mean(areas))
    ratio_std = float(np.std(ratios))
    return avg_area, ratio_std


def compute_motion_intensity(frames: List[np.ndarray]) -> float:
    """Estimate motion via mean optical flow magnitude across sampled pairs."""
    if len(frames) < 2:
        return 0.0
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    flow_mags = []
    for i in range(1, len(gray_frames)):
        prev = cv2.GaussianBlur(gray_frames[i - 1], (5, 5), 0)
        curr = cv2.GaussianBlur(gray_frames[i], (5, 5), 0)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(np.mean(mag))
    return float(np.mean(flow_mags)) if flow_mags else 0.0


def variance_of_laplacian(image: np.ndarray) -> float:
    """Focus/blur measure; lower means more blur/compression."""
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def compute_blockiness(image: np.ndarray, block: int = 8) -> float:
    """
    Simple blockiness metric: average gradient magnitude along block boundaries.
    Higher values suggest stronger block artifacts.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Collect gradients along 8-pixel grid lines
    vertical_lines = list(range(0, w, block))
    horizontal_lines = list(range(0, h, block))

    grads = []
    for x in vertical_lines:
        if 1 <= x < w - 1:
            grads.append(np.mean(np.abs(sobelx[:, x])))
    for y in horizontal_lines:
        if 1 <= y < h - 1:
            grads.append(np.mean(np.abs(sobely[y, :])))

    return float(np.mean(grads)) if grads else 0.0


def compute_compression_proxy(frames: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compression proxy from blur (Laplacian variance) and blockiness.
    Returns: (avg_blur_inverse, avg_blockiness)
    """
    if not frames:
        return 0.0, 0.0
    blurs = []
    blocks = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        blurs.append(variance_of_laplacian(gray))
        blocks.append(compute_blockiness(f))
    # Invert blur: lower blur variance -> higher compression indicator
    blur_inv = [1.0 / (b + 1e-6) for b in blurs]
    return float(np.mean(blur_inv)), float(np.mean(blocks))


def extract_video_features(path: str,
                           max_frames: int = 48,
                           stride: int = 5,
                           resize_to: Optional[Tuple[int, int]] = (360, 640)) -> Optional[np.ndarray]:
    """Open a video, sample frames, compute features, return feature vector."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    file_size = os.path.getsize(path) if os.path.exists(path) else 0
    bitrate = (file_size * 8) / max(frames_total / max(fps, 1e-6), 1e-6) if frames_total > 0 else 0.0

    frames = sample_frames(cap, max_frames=max_frames, stride=stride)
    cap.release()

    # Optional resize to normalize spatial scale
    if resize_to and frames:
        frames = [cv2.resize(f, (resize_to[1], resize_to[0])) for f in frames]

    avg_area, aspect_std = compute_resolution_stats(frames)
    motion = compute_motion_intensity(frames)
    blur_inv, blockiness = compute_compression_proxy(frames)

    features = np.array([
        avg_area,
        aspect_std,
        motion,
        blur_inv,
        blockiness,
        fps,
        bitrate
    ], dtype=np.float32)

    # Normalize to reduce scale variance (simple log / z-ish)
    features_norm = np.array([
        math.log1p(avg_area),         # large range
        aspect_std,
        motion,
        blur_inv,
        blockiness,
        math.log1p(fps),
        math.log1p(bitrate)
    ], dtype=np.float32)

    return features_norm



# Rule-based fallback (optional)

def rule_based_decision(feats: np.ndarray,
                        area_thresh: float = 12.0,     # log1p(area)
                        motion_thresh: float = 0.8,
                        blur_thresh: float = 0.8,
                        block_thresh: float = 3.0) -> str:
    """Return 'ViT', 'CNN', or 'ViT + CNN' from simple thresholds."""
    area_log, aspect_std, motion, blur_inv, blockiness, fps_log, bitrate_log = feats

    high_res = area_log >= area_thresh
    high_motion = motion >= motion_thresh
    heavy_compress = (blur_inv >= blur_thresh) or (blockiness >= block_thresh)

    if high_motion and heavy_compress:
        return "CNN"
    if high_res and not high_motion and not heavy_compress:
        return "ViT"
    # Mixed signals
    return "ViT + CNN"



# Training utilities

CLASS_TO_ID = {"ViT": 0, "CNN": 1, "ViT + CNN": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def load_labels_from_csv(csv_path: str) -> dict:
    """
    CSV format:
    video_path,label
    /path/to/video1.mp4,ViT
    /path/to/video2.mp4,CNN
    /path/to/video3.mp4,ViT + CNN
    """
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f.read().strip().splitlines()[1:]:
            p, lbl = line.split(",", 1)
            mapping[p.strip()] = lbl.strip()
    return mapping


def build_dataset(video_dir: str,
                  labels_csv: Optional[str] = None,
                  pattern: str = "**/*.mp4") -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y from a directory of videos and either:
    - labels CSV (preferred), or
    - infer labels from folder names (ViT/CNN/ViT + CNN) for quick testing.
    """
    paths = glob.glob(os.path.join(video_dir, pattern), recursive=True)
    if not paths:
        raise RuntimeError(f"No videos found under {video_dir}")

    if labels_csv and os.path.exists(labels_csv):
        label_map = load_labels_from_csv(labels_csv)
    else:
        label_map = {}

    X, y = [], []
    for p in tqdm(paths, desc="Extracting features"):
        feats = extract_video_features(p)
        if feats is None:
            continue

        # Determine label
        if p in label_map:
            lbl = label_map[p]
        else:
            # Infer from parent folder name
            parent = os.path.basename(os.path.dirname(p)).strip().lower()
            if "vit" in parent:
                lbl = "ViT"
            elif "cnn" in parent:
                lbl = "CNN"
            elif "both" in parent or "vit+cnn" in parent or "vit_cnn" in parent:
                lbl = "ViT + CNN"
            else:
                # If unknown, use rule-based decision to bootstrap
                lbl = rule_based_decision(feats)

        X.append(feats)
        y.append(CLASS_TO_ID[lbl])

    return np.vstack(X), np.array(y, dtype=np.int64)


def train_classifier(video_dir: str,
                     labels_csv: Optional[str],
                     model_out: str = "gating_rf.joblib") -> None:
    X, y = build_dataset(video_dir, labels_csv)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=["ViT", "CNN", "ViT + CNN"]))

    joblib.dump(clf, model_out)
    print(f"[INFO] Saved model to {model_out}")


# Inference

def predict_video(video_path: str,
                  model_path: Optional[str] = None) -> str:
    feats = extract_video_features(video_path)
    if feats is None:
        raise RuntimeError("Failed to extract features from video.")

    # If model is provided, use it; else rule-based fallback
    if model_path and os.path.exists(model_path):
        clf = joblib.load(model_path)
        pred_id = int(clf.predict(feats.reshape(1, -1))[0])
        return ID_TO_CLASS[pred_id]
    else:
        return rule_based_decision(feats)



# CLI

def main():
    parser = argparse.ArgumentParser(description="Gating classifier for model selection (ViT vs CNN)")
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="Train classifier")
    t.add_argument("--video_dir", required=True, help="Directory containing videos")
    t.add_argument("--labels_csv", default=None, help="CSV with labels (optional)")
    t.add_argument("--model_out", default="gating_rf.joblib", help="Output path for the trained model")

    p = sub.add_parser("predict", help="Predict preferred model for a video")
    p.add_argument("--video_path", required=True, help="Path to input video file")
    p.add_argument("--model_path", default=None, help="Path to trained model (optional)")

    args = parser.parse_args()

    if args.cmd == "train":
        train_classifier(args.video_dir, args.labels_csv, args.model_out)
    elif args.cmd == "predict":
        label = predict_video(args.video_path, args.model_path)
        print(label)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
