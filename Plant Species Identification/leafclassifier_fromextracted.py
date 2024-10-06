import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')
# Define the neural network
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

# Move the model to the GPU if available
model = NeuralNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Training
print("Beginning Training")
model.train()
for epoch in range(6000):
    for i, (features, labels) in enumerate(train_loader):
        # Move features and labels to the GPU if available
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print("Training Finished")



print("Testing on Training Dataset")

model.eval()  # Set model to evaluation mode (disables dropout, batchnorm, etc.)
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for testing
    for features, labels in train_loader:
        # Move features and labels to the GPU if available
        features, labels = features.to(device), labels.to(device)
        
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Training Accuracy: {accuracy:.2f}%')

# Save model
if accuracy > 91:
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved")


# Latest model - 90.91% accuracy
# LR - 0.001, 5000 epochs, 32 batch size, 256, 128, 64 hidden layers - 73.71
# LR - 0.001 and layers 256,128,64 is steadily decreasing the loss
# LR - 0.002 , 10k epochs, 32 batch size, 256, 128, 64 hidden layers - 90.91% accuracy

"""
Using cuda
Beginning Training
Epoch 500, Loss: 4.310931205749512
Epoch 1000, Loss: 4.210741996765137
Epoch 1500, Loss: 4.209359169006348
Epoch 2000, Loss: 4.0443196296691895
Epoch 2500, Loss: 3.944333553314209
Epoch 3000, Loss: 3.7781193256378174
Epoch 3500, Loss: 3.811099052429199
Epoch 4000, Loss: 3.74504017829895
Epoch 4500, Loss: 3.8110392093658447
Epoch 5000, Loss: 3.7448220252990723
Epoch 5500, Loss: 3.711667537689209
Epoch 6000, Loss: 3.6454598903656006
Epoch 6500, Loss: 3.678544759750366
Epoch 7000, Loss: 3.7118730545043945
Epoch 7500, Loss: 3.7116472721099854
Epoch 8000, Loss: 3.645430326461792
Epoch 8500, Loss: 3.711684465408325
Epoch 9000, Loss: 3.810967206954956
Epoch 9500, Loss: 3.6454358100891113
Epoch 10000, Loss: 3.6786487102508545
Training Finished
Testing on Training Dataset
Training Accuracy: 90.91%
Model saved
ahaandesai@DES
"""