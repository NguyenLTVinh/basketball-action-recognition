import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    Normalize,
)
from torchvision.transforms import Compose, Lambda
from pytorchvideo.models.hub import x3d_m
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# Class Names
class_names = [
    "block", "pass", "run", "dribble", "shoot", "ball in hand", "defense", "pick", "no_action", "walk"
]

# Constants
NUM_CLASSES = 10
BATCH_SIZE = 16
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

# Sample 1000 data points from the annotation dictionary
sampled_files = random.sample(list(annotation_dict.keys()), 1000)
sampled_dict = {video: annotation_dict[video] for video in sampled_files}

# Create Dataset and DataLoader
test_dataset = BasketballActionDataset("examples", sampled_dict, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Load the Best Model
model = x3d_m(pretrained=True)
model.blocks[5].proj = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, NUM_CLASSES),
)
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(DEVICE)

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

# Normalized Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

# Main Execution
if __name__ == "__main__":
    test_model(model, test_loader, class_names)