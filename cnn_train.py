import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    TimeDistributed,
    GlobalAveragePooling2D,
    LSTM,
    Dense,
    Dropout,
    Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from random import shuffle

os.makedirs("dataset", exist_ok=True)
os.system("kaggle datasets download -d username/dataset-name -p dataset/ --unzip")

IMG_SIZE = 224
SEQUENCE_LENGTH = 30
BATCH_SIZE = 4
EPOCHS = 10
DATASET_PATH = "dataset"

def extract_frames(video_path, sequence_length):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    while len(frames) < sequence_length:
        frames.append(frames[-1])

    return np.array(frames)

class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, batch_size, sequence_length):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = np.array([extract_frames(path, self.sequence_length) for path in batch_paths])
        y = np.array(batch_labels)
        return X, y
video_paths_real = []
video_paths_fake = []

for cls, label in zip(["real", "fake"], [0, 1]):
    folder = os.path.join(DATASET_PATH, cls)
    videos = [os.path.join(folder, v) for v in os.listdir(folder) if v.endswith((".mp4", ".avi", ".mov"))]
    videos.sort()
    videos = videos[:200] 
    if label == 0:
        video_paths_real = videos
    else:
        video_paths_fake = videos


video_paths = video_paths_real + video_paths_fake
labels = [0]*len(video_paths_real) + [1]*len(video_paths_fake)

combined = list(zip(video_paths, labels))
shuffle(combined)
video_paths, labels = zip(*combined)
video_paths, labels = list(video_paths), list(labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    video_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

train_generator = VideoDataGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)
val_generator = VideoDataGenerator(val_paths, val_labels, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable =  False

inputs = Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3))
x = TimeDistributed(base_model)(inputs)
x = TimeDistributed(GlobalAveragePooling2D())(x)
x = LSTM(128)(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("deepfake_efficientnet_lstm.h5", save_best_only=True)
]
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)
