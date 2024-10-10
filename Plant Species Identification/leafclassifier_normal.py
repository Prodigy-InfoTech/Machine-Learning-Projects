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

train_labels.head()

image_directory = 'images'
image_list = []

for filename in os.listdir(image_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        label = int(os.path.splitext(filename)[0])  
        if label in train_ids:
            image_list.append(f"images/{label}.jpg") 

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((64, 64)),  # Resize to 64x64 during transformation
    transforms.RandomRotation(10),  # Random rotations
    transforms.RandomHorizontalFlip(),  # Horizontal flipping for augmentation
    transforms.ToTensor(),
])

def add_gaussian_noise(image, noise_factor=0.05):
    """Adds Gaussian noise to the image."""
    noise = torch.randn(image.size()) * noise_factor
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)  #

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

        img = add_gaussian_noise(img)  # Add Gaussian noise to the image
        label = torch.tensor(label).long()

        return img, label


# Create dataset and dataloader
train_labels = LabelEncoder().fit_transform(train_labels)
dataset = CustomDataset(image_list, train_labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=64)


# Load pre-trained MobileNet model
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
num_classes = 99  # Change this to your number of classes
# Modify the first convolutional layer to accept 1-channel input instead of 3
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

# Replace the classifier head
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Move the entire model to the device after modifications
model = model.to(device)

# Set the model to training mode
model.train()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)


num_epochs = 50
patience = 5
best_loss = float('inf')
epochs_without_improvement = 0
early_stop = False

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        total_loss += loss.item()
        accuracy = total_correct / total_samples * 100

    avg_loss = total_loss / len(dataloader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print("Early stopping triggered!")
        early_stop = True
        break

if early_stop:
    print(f"Stopped early at epoch {epoch+1}")
    
torch.save(model.state_dict(), 'model_final.pth')
print("Model saved successfully!")
    
