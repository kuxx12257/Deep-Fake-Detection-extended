mport os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
IMG_SIZE = 224
SEQUENCE_LENGTH = 30
MODEL_PATH = "deepfake_efficientnet_lstm.h5"
VAL_FOLDER = "dataset/val/"
def extract_frames(video_path, sequence_length):
    """
    Extracts 'sequence_length' frames from a video, resized and normalized.
    Handles empty or unreadable videos safely.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"ERROR: Video {video_path} has no frames")
        return None

    indices = np.linspace(0, max(0, total_frames - 1), sequence_length).astype(int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(f"ERROR: No frames could be read from {video_path}")
        return None

    while len(frames) < sequence_length:
        frames.append(frames[-1])

    return np.array(frames)

def predict_video(video_path, model, sequence_length):
    """
    Predicts whether the video is REAL or FAKE.
    Returns label and confidence score.
    """
    frames = extract_frames(video_path, sequence_length)
    if frames is None:
        return "ERROR", 0.0

    frames = np.expand_dims(frames, axis=0)
    prediction = model.predict(frames)[0][0]
    label = "FAKE" if prediction > 0.5 else "REAL"
    return label, prediction


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("Model loaded successfully!")


if not os.path.exists(VAL_FOLDER):
    raise FileNotFoundError(f"Validation folder not found: {VAL_FOLDER}")

video_files = [f for f in os.listdir(VAL_FOLDER) if f.endswith((".mp4", ".avi", ".mov"))]

if len(video_files) == 0:
    print("No videos found in validation folder.")
else:
    for video_file in video_files:
        video_path = os.path.join(VAL_FOLDER, video_file)
        label, confidence = predict_video(video_path, model, SEQUENCE_LENGTH)
        if label != "ERROR":
            print(f"Video: {video_file} | Prediction: {label} | Confidence: {confidence:.4f}")
        else:
            print(f"Video: {video_file} | Could not process video")

