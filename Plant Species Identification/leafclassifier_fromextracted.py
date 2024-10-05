import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset

# training using input features

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(192, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 99)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return torch.softmax(x, dim=1)

# Loading CSV
train_df = pd.read_csv('train.csv')
train_features = train_df.drop(columns=['species', 'id'])
train_labels = train_df['species']

# Converting to tensor + encoding
train_features = torch.tensor(train_features.values).float()
train_labels = torch.tensor(LabelEncoder().fit_transform(train_labels)).long()

# Creating tensor dataset and dataloader
train_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
print("Beginning Training")
model.train()
for epoch in range(5000):
    for i, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print("Training Finished")

# Save model
torch.save(model.state_dict(), 'model.pth')

print("Testing on Training Dataset")

model.eval()  # Set model to evaluation mode (disables dropout, batchnorm, etc.)
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for testing
    for features, labels in train_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Training Accuracy: {accuracy:.2f}%')

# Latest model - 73.64% accuracy
