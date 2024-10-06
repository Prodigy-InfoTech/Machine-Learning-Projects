import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset

train_df = pd.read_csv('train.csv')
print(train_df['species'])

train_ids = train_df['id'].tolist()
train_labels = train_df['species']

# Directory containing the images
image_directory = 'images'

# List to store the image_list
image_list = []

# Loop through each file in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        label = int(os.path.splitext(filename)[0])  # Get the filename without the extension
        if label in train_ids:
            image_list.append(f"images/{label}.jpg")  # Convert to integer and add to the list
len(image_list)
image_list[:10]

smallest_image = float('inf')
largest_image = float('-inf')
smallest_path = ''
largest_path = ''
image_width = []
image_height = []
images = []
for image_path in image_list:
    image = Image.open(f"{image_path}")
    image_width.append(image.size[0])
    image_height.append(image.size[1])
    images.append(image)
    image_dims = image.size[0]*image.size[1]
    if image_dims < smallest_image:
        smallest_image = image_dims
        smallest_path = image
    elif image_dims > largest_image:
        largest_image = image_dims
        largest_path = image

print(smallest_image, largest_image)
print(smallest_path, largest_path)

median_width = median_height = max(int(np.median(image_width)), int(np.median(image_height))) 
resize_shape = (median_width, median_height)
print(resize_shape)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input: 684x684x1, Output: 684x684x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 342x342x32
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 342x342x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 171x171x64
        )

        # Third convolutional block (bottleneck)
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),  # Output: 171x171x128
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),   # Output: 171x171x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)   # Output: 171x171x128
        )

        # Fourth convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 171x171x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 85x85x256
        )

        # Fifth convolutional block (bottleneck)
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),  # Output: 85x85x512
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),   # Output: 85x85x256
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)   # Output: 85x85x512
        )

        # Global Average Pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce to 1x1 feature map
        self.fc = nn.Linear(512, num_classes)  # Final output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bottleneck1(x)
        x = self.conv3(x)
        x = self.bottleneck2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x

# Example usage
num_classes = 99  # Change this to your number of classes
model = CustomCNN(num_classes=num_classes)

# Print model summary
print(model)
print(resize_shape)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),  # Resizing to 64x64 during transformation
    transforms.RandomRotation(10),  # 10 random rotations
    transforms.RandomHorizontalFlip(),  # Horizontal flipping for augmentation
    transforms.ToTensor()
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
        
        label = torch.tensor(label).long()

        return img, label

# Create dataset and dataloader
train_labels = LabelEncoder().fit_transform(train_labels)
dataset = CustomDataset(image_list, train_labels, transform=transform)
len(dataset)

dataloader = DataLoader(dataset, batch_size=32)

model = CustomCNN(num_classes=99)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

num_epochs = 50  # Increased number of epochs
for epoch in range(num_epochs):
    for images, labels in dataloader:
        images, labels = images, labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
torch.save(model.state_dict(), 'model1.pth')


