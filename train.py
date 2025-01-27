import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import class_weight
from tensorflow.keras.experimental import CosineDecay

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
JOINTS_SHAPE = (16, 14, 2)  # 16 frames, 14 joints, 2 coordinates (x, y)
VIDEO_SHAPE = (16, 128, 171, 3)  # 16 frames, 128x171 resolution, 3 channels (RGB)
NUM_CLASSES = 10  # Number of action classes
BATCH_SIZE_JOINTFC = 16
BATCH_SIZE_C3D = 12
EPOCHS_JOINTFC = 50
EPOCHS_C3D = 10

# Load labels and annotations
with open("labels_dict.json", "r") as f:
    labels_dict = json.load(f)

with open("annotation_dict.json", "r") as f:
    annotation_dict = json.load(f)

# Limit to the first 1000 data points
annotation_dict = dict(list(annotation_dict.items())[:37085])

# Joint Data Generator
class JointDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_ids, labels, batch_size, joint_dir):
        super().__init__()
        self.file_ids = file_ids
        self.labels = labels
        self.batch_size = batch_size
        self.joint_dir = joint_dir

    def __len__(self):
        return int(np.ceil(len(self.file_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.file_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_joints = []
        for file_id in batch_ids:
            joint_file = os.path.join(self.joint_dir, f"{file_id}.npy")
            frames = np.load(joint_file, allow_pickle=True)
            # Pad or truncate to 16 frames
            if len(frames) < 16:
                frames.extend([np.full((14, 2), -1)] * (16 - len(frames)))
            frames = frames[:16]
            joints_list = []
            for frame in frames:
                joints = np.array([frame.get(j, (-1, -1)) for j in range(14)])
                joints = joints.astype(np.float32)
                joints[..., 0] /= 128  # Normalize x
                joints[..., 1] /= 171  # Normalize y
                # Replace invalid joints (where both x and y are < 0) with [-1, -1]
                invalid_mask = (joints[..., 0] < 0) & (joints[..., 1] < 0)
                joints[invalid_mask] = [-1, -1]
                joints_list.append(joints)
            joints_array = np.array(joints_list).reshape(-1)
            batch_joints.append(joints_array)
        return np.array(batch_joints), np.array(batch_labels)

# Video Data Generator
class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_ids, labels, batch_size, video_dir):
        super().__init__()
        self.file_ids = file_ids
        self.labels = labels
        self.batch_size = batch_size
        self.video_dir = video_dir

    def __len__(self):
        return int(np.ceil(len(self.file_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.file_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_videos = []
        for file_id in batch_ids:
            video_file = os.path.join(self.video_dir, f"{file_id}.mp4")
            frames = []
            cap = cv2.VideoCapture(video_file)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (171, 128))  # Resize to 128x171
                frame = frame / 255.0  # Normalize to [0, 1]
                frames.append(frame)
            cap.release()
            # Pad or truncate to 16 frames
            if len(frames) < 16:
                frames.extend([np.zeros((128, 171, 3))] * (16 - len(frames)))
            frames = frames[:16]
            batch_videos.append(np.array(frames))
        return np.array(batch_videos), np.array(batch_labels)

# Build jointFC model (modified to match the document)
def build_jointFC(input_shape=(448,), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Dense(512, activation='relu', kernel_initializer='he_normal', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Build C3D model (modified to match the document)
def build_C3D(input_shape=(16, 128, 171, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling3D((1, 2, 2), strides=(1, 2, 2)),
        layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2)),
        layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2)),
        layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2)),
        layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close()

# Main function
def main():
    # Load file IDs and labels
    file_ids = list(annotation_dict.keys())
    labels = list(annotation_dict.values())

    # Split data into train, validation, and test sets
    train_ids, test_ids, train_labels, test_labels = train_test_split(file_ids, labels, test_size=0.15, random_state=42)
    train_ids, val_ids, train_labels, val_labels = train_test_split(train_ids, train_labels, test_size=0.15, random_state=42)

    # Create data generators
    train_joint_gen = JointDataGenerator(train_ids, train_labels, BATCH_SIZE_JOINTFC, "examples")
    val_joint_gen = JointDataGenerator(val_ids, val_labels, BATCH_SIZE_JOINTFC, "examples")
    test_joint_gen = JointDataGenerator(test_ids, test_labels, BATCH_SIZE_JOINTFC, "examples")

    train_video_gen = VideoDataGenerator(train_ids, train_labels, BATCH_SIZE_C3D, "examples")
    val_video_gen = VideoDataGenerator(val_ids, val_labels, BATCH_SIZE_C3D, "examples")
    test_video_gen = VideoDataGenerator(test_ids, test_labels, BATCH_SIZE_C3D, "examples")

    # Class weights to address imbalance
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = dict(enumerate(class_weights))

    # Compile model
    jointFC_model = build_jointFC()
    jointFC_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    initial_learning_rate = 1e-3
    decay_steps = EPOCHS_JOINTFC * len(train_joint_gen)
    cosine_decay = CosineDecay(initial_learning_rate, decay_steps)
    cosine_lr_callback = tf.keras.callbacks.LearningRateScheduler(cosine_decay)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    )

    # Train model
    jointFC_history = jointFC_model.fit(
        train_joint_gen,
        validation_data=val_joint_gen,
        epochs=EPOCHS_JOINTFC,
        callbacks=[reduce_lr_callback, early_stopping]
    )

    # # Train C3D model
    # C3D_model = build_C3D()
    # C3D_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    #                   loss='sparse_categorical_crossentropy',
    #                   metrics=['accuracy'])
    # C3D_history = C3D_model.fit(
    #     train_video_gen,
    #     validation_data=val_video_gen,
    #     epochs=EPOCHS_C3D,
    #     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    # )

    # Evaluate models
    jointFC_test_loss, jointFC_test_acc = jointFC_model.evaluate(test_joint_gen)
    print(f"JointFC Test Accuracy: {jointFC_test_acc}")

    # C3D_test_loss, C3D_test_acc = C3D_model.evaluate(test_video_gen)
    # print(f"C3D Test Accuracy: {C3D_test_acc}")

if __name__ == "__main__":
    main()
