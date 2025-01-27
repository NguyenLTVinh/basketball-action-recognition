import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
    Normalize,
)
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.models.hub import slow_r50
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Class Names
class_names = [
    "block", "pass", "run", "dribble", "shoot", "ball in hand", "defense", "pick", "no_action", "walk"
]

# Constants
NUM_CLASSES = 10
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_FRAMES = 16
SIDE_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
class BasketballActionDataset(Dataset):
    def __init__(self, video_dir, annotation_dict, transform=None):
        self.video_dir = video_dir
        self.annotation_dict = annotation_dict
        self.video_files = list(annotation_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file + ".mp4")
        label = self.annotation_dict[video_file]

        # Load video with progress bar
        with tqdm(total=1, desc=f"Loading {video_file}", leave=False) as pbar:
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=video.duration)
            pbar.update(1)

        # Apply transformations
        if self.transform:
            video_data = self.transform(video_data)

        return video_data["video"], label

# Transformations
transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        ShortSideScale(size=SIDE_SIZE),
        Lambda(lambda x: x / 255.0),
        Normalize(mean=MEAN, std=STD),
    ]),
)

# Load Annotation Dictionary
with open("annotation_dict.json", "r") as f:
    annotation_dict = json.load(f)

# Convert annotation_dict to lists for stratified sampling
video_files = list(annotation_dict.keys())
labels = [annotation_dict[video] for video in video_files]

# Calculate the class distribution
class_counts = pd.Series(labels).value_counts().sort_index()
print("Original Class Distribution:")
print(class_counts)

# Find the minimum class size
min_class_size = class_counts.min()

# Sample an equal number from each class
sampled_files = []
sampled_labels = []
for label in class_counts.index:
    # Get all video files corresponding to this label
    class_videos = [video for video, video_label in zip(video_files, labels) if video_label == label]
    # Randomly sample 'min_class_size' number of videos from this class
    sampled_class_files = np.random.choice(class_videos, size=min_class_size, replace=False)
    sampled_files.extend(sampled_class_files)
    sampled_labels.extend([label] * min_class_size)

# Verify the class distribution in the sampled dataset
sampled_class_counts = pd.Series(sampled_labels).value_counts().sort_index()
print("\nSampled Class Distribution (equal samples from each class):")
print(sampled_class_counts)

# Split the sampled dataset into train, validation, and test sets
train_files, test_files, train_labels, test_labels = train_test_split(
    sampled_files, sampled_labels, test_size=0.15, stratify=sampled_labels, random_state=42
)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.1765, stratify=train_labels, random_state=42
)  # 0.1765 * 0.85 ≈ 0.15 (15% validation)

# Create dictionaries for train, validation, and test sets
train_dict = {video: label for video, label in zip(train_files, train_labels)}
val_dict = {video: label for video, label in zip(val_files, val_labels)}
test_dict = {video: label for video, label in zip(test_files, test_labels)}

# Create Datasets and DataLoaders
train_dataset = BasketballActionDataset("examples", train_dict, transform=transform)
val_dataset = BasketballActionDataset("examples", val_dict, transform=transform)
test_dataset = BasketballActionDataset("examples", test_dict, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model Setup
model = slow_r50(pretrained=True)
model.blocks[5].proj = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, NUM_CLASSES),
)
model = model.to(DEVICE)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # Wrap train_loader with tqdm for progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar description with current loss and accuracy
            train_loader_tqdm.set_postfix({
                "Loss": train_loss / len(train_loader),
                "Accuracy": train_correct / train_total
            })

        train_accuracy = train_correct / train_total
        val_accuracy, val_loss = evaluate_model(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

        scheduler.step()

# Evaluation Function
def evaluate_model(model, data_loader, criterion):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    # Wrap data_loader with tqdm for progress bar
    data_loader_tqdm = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for inputs, labels in data_loader_tqdm:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            # Update progress bar description with current loss and accuracy
            data_loader_tqdm.set_postfix({
                "Loss": val_loss / len(data_loader),
                "Accuracy": val_correct / val_total
            })

    val_accuracy = val_correct / val_total
    return val_accuracy, val_loss / len(data_loader)

# Test Function
def test_model(model, test_loader, class_names):
    model.eval()
    y_true, y_pred = [], []

    # Wrap test_loader with tqdm for progress bar
    test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)

    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("Test Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, class_names)

# Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

# Main Execution
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS)
    model.load_state_dict(torch.load("best_model.pth"))
    test_model(model, test_loader, class_names)
