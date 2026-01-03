import cv2  # OpenCV for video processing
import numpy as np  # For numerical operations
from statistics import mean  # To compute average of feature values

# Extract features from the video
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    blur_scores, brightness_scores, face_sizes = [], [], []  # Lists to store frame-wise features
    frame_count = 0  # Counter to limit number of frames processed

    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret or frame_count > 30:  # Stop after 30 frames or end of video
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()  # Compute blur using Laplacian variance
        brightness = np.mean(gray)  # Compute average brightness of the frame

        blur_scores.append(blur)  # Store blur score
        brightness_scores.append(brightness)  # Store brightness score

        # Simulate face size as half of the smaller dimension (replace with real face detection if needed)
        face_sizes.append(min(frame.shape[0], frame.shape[1]) // 2)

        frame_count += 1  # Move to next frame

    cap.release()  # Release the video file

    # Return average values of the features
    return {
        "avg_blur": mean(blur_scores),
        "avg_brightness": mean(brightness_scores),
        "avg_face_size": mean(face_sizes)
    }

# Step 2: Apply voting rules to select the model 
def voting_based_model_selector(features):
    votes = {"CNN": 0, "ViT": 0}  # Initialize vote counters

    # Rule 1: Blur-based decision
    if features["avg_blur"] < 100:  # If blur is high (low variance), prefer CNN
        votes["CNN"] += 1
    else:  # If image is sharp, ViT may perform better
        votes["ViT"] += 1

    # Rule 2: Brightness-based decision
    if features["avg_brightness"] < 80:  # Low brightness favors CNN
        votes["CNN"] += 1
    else:  # High brightness favors ViT
        votes["ViT"] += 1

    # Rule 3: Face size-based decision
    if features["avg_face_size"] < 100:  # Small face → CNN
        votes["CNN"] += 1
    else:  # Large face → ViT
        votes["ViT"] += 1

    # Final decision based on majority vote
    if votes["CNN"] > votes["ViT"]:
        return "CNN"
    elif votes["ViT"] > votes["CNN"]:
        return "ViT"
    else:
        return "Both"  # Equal votes → use both models

# Step 3: Main function to run the pipeline 
def main(video_path):
    features = extract_video_features(video_path)  # Step 1: Extract features
    selected_model = voting_based_model_selector(features)  # Step 2: Apply voting
    print(f"Selected model based on voting: {selected_model}")  # Output decision
    print(f"Extracted features: {features}")  # Output feature values

# Step 4: Run the script with a sample video
if __name__ == "__main__":
    video_file = "sample_video.mp4"  # Replace with your actual video file path
    main(video_file)
