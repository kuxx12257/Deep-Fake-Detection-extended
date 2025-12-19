import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from vit_keras import vit
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score
import warnings

gpus = tf.config.list_physical_devices("GPU")
device = "/GPU:0" if gpus else "/CPU:0"
print("Using device:", device)

IMG_SIZE = (224, 224)
SEQ_LEN = 8
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42
DATASET_DIR = "frames_dataset"

AUTOTUNE = tf.data.AUTOTUNE

def sample_frames(frame_list, seq_len):
    if len(frame_list) >= seq_len:
        idx = np.linspace(0, len(frame_list) - 1, seq_len).astype(int)
        return [frame_list[i] for i in idx]
    else:
        return frame_list + [frame_list[-1]] * (seq_len - len(frame_list))

def build_video_sequences(dataset_dir, split="training", val_ratio=0.2):
    videos, labels = [], []
    class_map = {"real": 0, "manipulated": 1}

    for class_name, class_label in class_map.items():
        frame_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(frame_dir):
            continue

        video_groups = defaultdict(list)
        for fname in os.listdir(frame_dir):
            if "_frame" in fname:
                vid = fname.split("_frame")[0]
                video_groups[vid].append(fname)

        video_ids = sorted(video_groups.keys())
        split_idx = int(len(video_ids) * (1 - val_ratio))

        if split == "training":
            video_ids = video_ids[:split_idx]
        else:
            video_ids = video_ids[split_idx:]

        for vid in video_ids:
            frame_files = sample_frames(sorted(video_groups[vid]), SEQ_LEN)
            frames = []

            for f in frame_files:
                img = cv2.imread(os.path.join(frame_dir, f))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = tf.image.resize(img, IMG_SIZE)
                img = tf.cast(img, tf.float32) / 255.0
                frames.append(img)

            if len(frames) == SEQ_LEN:
                videos.append(tf.stack(frames))
                labels.append(class_label)

    return tf.data.Dataset.from_tensor_slices((videos, labels))

train_ds = build_video_sequences(DATASET_DIR, "training")
val_ds   = build_video_sequences(DATASET_DIR, "validation")

train_ds = train_ds.shuffle(128).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds   = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

with tf.device(device):
    vit_backbone = vit.vit_b16(
        image_size=IMG_SIZE[0],
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        activation="linear",
    )
    vit_backbone.trainable = False

    frame_input = layers.Input(shape=(*IMG_SIZE, 3))
    frame_features = vit_backbone(frame_input)
    frame_encoder = tf.keras.Model(frame_input, frame_features)

    video_input = layers.Input(shape=(SEQ_LEN, *IMG_SIZE, 3))
    x = layers.TimeDistributed(frame_encoder)(video_input)

    mean_pool = tf.reduce_mean(x, axis=1)
    max_pool  = tf.reduce_max(x, axis=1)

    x = layers.Concatenate()([mean_pool, max_pool])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(video_input, outputs)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

num_real = 400
num_fake = 3000

class_weights = {
    0: num_fake / num_real,
    1: 1.0
}

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1
)

model.save("vit_temporal_pooling_model")

def preprocess_frame(img):
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def predict_video(model, frames):
    frames = sample_frames(frames, SEQ_LEN)
    frames = tf.stack(frames)
    frames = tf.expand_dims(frames, axis=0)

    prob = model(frames, training=False)[0][0].numpy()
    label = 1 if prob > 0.5 else 0
    confidence = prob if label == 1 else 1 - prob
    return label, confidence

print("\nRunning Video-Level Evaluation...")

video_preds, video_labels = [], []

for class_name, class_label in [("real", 0), ("manipulated", 1)]:
    frame_dir = os.path.join(DATASET_DIR, class_name)
    video_groups = defaultdict(list)

    for fname in os.listdir(frame_dir):
        if "_frame" in fname:
            vid = fname.split("_frame")[0]
            video_groups[vid].append(fname)

    for vid, frame_files in video_groups.items():
        frames = []
        for f in frame_files:
            img = cv2.imread(os.path.join(frame_dir, f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_frame(img)
            frames.append(img)

        if len(frames) == 0:
            continue

        pred, conf = predict_video(model, frames)
        video_preds.append(pred)
        video_labels.append(class_label)

        label_str = "Real" if pred == 0 else "Manipulated"
        print(f"{vid} → {label_str} ({conf:.3f})")

print("\nVIDEO-LEVEL CLASSIFICATION REPORT")
print(classification_report(video_labels, video_preds, target_names=["Real", "Manipulated"]))

acc = accuracy_score(video_labels, video_preds)
print(f"\nVideo-Level Accuracy: {acc * 100:.2f}%")
