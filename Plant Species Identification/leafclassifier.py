import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

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

class AdaptiveConvNet(nn.Module):
    def __init__(self):
        super(AdaptiveConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool2d((4, 4))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 99)  # Assuming 99 classes

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

images_path = 'images'
image_files = os.listdir(images_path)
image_files = [f for f in image_files if f.endswith(('png', 'jpg', 'jpeg'))]

indices = train_df.index.tolist()
one_based_indices = [str(idx + 1) + '.jpg' for idx in indices]

# Create a list of valid image paths
image_paths = [os.path.join(images_path, f) for f in image_files if f in one_based_indices]

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),  # Resizing to 64x64 during transformation
    transforms.RandomRotation(20),  # 20 random rotations
    transforms.RandomHorizontalFlip(),  # Horizontal flipping for augmentation
    transforms.ToTensor()
])

labels = train_df['species'].astype('category').cat.codes

# Create dataset and dataloader
dataset = CustomDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, and optimizer
model = AdaptiveConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

num_epochs = 50  # Increased number of epochs
for epoch in range(num_epochs):
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Adding Gaussian noise
        noise = torch.randn_like(images) * 0.1
        noisy_images = images + noise
        
        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model.pth')

# Evaluate model
correct = 0
total = 0
model.eval() 
with torch.no_grad(): 
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy after training: {accuracy:.2f}%')

torch.save(model.state_dict(), 'model.pth')
