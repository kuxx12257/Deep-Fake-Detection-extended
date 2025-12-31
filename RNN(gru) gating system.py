import cv2
import torch
import torch.nn as nn
import numpy as np

# RNN Gating Model 
class RNNGatingSystem(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(RNNGatingSystem, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):  # x: (B, T, D)
        _, h_n = self.gru(x)      # h_n: (1, B, H)
        h_last = h_n[-1]          # (B, H)
        logits = self.fc(h_last)  # (B, 2)
        weights = torch.softmax(logits, dim=-1)
        return weights            # (B, 2)

# Feature Extraction from Frame 
def extract_features(frame, prev_gray=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    face_size = min(frame.shape[0], frame.shape[1]) / 2

    # Motion estimation
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion = np.mean(np.linalg.norm(flow, axis=2))
    else:
        motion = 0.0

    return [blur, brightness, face_size, motion], gray

# Video to Feature Sequence
def video_to_feature_sequence(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    features = []
    prev_gray = None
    frame_count = 0

    print("\n🎥 Processing video and extracting features...\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        # Simulated model predictions (replace with real ones)
        p_cnn = np.random.uniform(0.3, 0.7)
        p_vit = np.random.uniform(0.3, 0.7)

        f_vec, gray = extract_features(frame, prev_gray)
        prev_gray = gray

        # Combine all features
        x_t = [p_cnn, p_vit] + f_vec
        features.append(x_t)

        print(f"Frame {frame_count+1}:")
        print(f"   - CNN pred     : {p_cnn:.3f}")
        print(f"   - ViT pred     : {p_vit:.3f}")
        print(f"   - Blur         : {f_vec[0]:.2f}")
        print(f"   - Brightness   : {f_vec[1]:.2f}")
        print(f"   - Face size    : {f_vec[2]:.2f}")
        print(f"   - Motion       : {f_vec[3]:.2f}\n")

        frame_count += 1

    cap.release()
    return torch.tensor([features], dtype=torch.float32)  # (1, T, D)

# Main Execution
if __name__ == "__main__":
    video_path = "sample_video.mp4"  # Replace with your video file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = RNNGatingSystem(input_dim=5, hidden_dim=64).to(device)
    model.eval()

    # Extract features from video
    feature_seq = video_to_feature_sequence(video_path)  # (1, T, 5)
    feature_seq = feature_seq.to(device)

    # Run gating model
    with torch.no_grad():
        weights = model(feature_seq)[0].cpu().numpy()
        print(" Final Gating Weights:")
        print(f"   - CNN: {weights[0]:.2f}")
        print(f"   - ViT: {weights[1]:.2f}")
