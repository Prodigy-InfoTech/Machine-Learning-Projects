import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_df = pd.read_csv('train.csv')
train_ids = train_df['id'].tolist()
train_labels = train_df['species']

image_directory = 'images'
image_list = []

for filename in os.listdir(image_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        label = int(os.path.splitext(filename)[0])
        if label in train_ids:
            image_list.append(f"images/{label}.jpg")

# Training transformations with augmentation
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Testing transformations without augmentation
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(image_path) as img:
            if self.transform:
                img = self.transform(img)

        return img, label

train_labels = LabelEncoder().fit_transform(train_labels)
dataset = CustomDataset(image_list, train_labels, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=64)

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier[1] = nn.Linear(model.last_channel, 99)
state_dict = torch.load('model_final.pth', map_location=device, weights_only=True)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

# Move the model to the appropriate device (GPU/CPU)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total_samples += labels.size(0)  # Update total samples
        total_correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = total_correct / total_samples * 100
    print(f'Accuracy on training set: {accuracy:.2f}%')
