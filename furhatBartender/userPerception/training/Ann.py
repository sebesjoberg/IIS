import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

# Define transforms for data preprocessing and augmentation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Random horizontal flip with probability 0.5
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # Color jitter
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.473], std=[0.285]
        ),  # For grayscale, mean and std are single values
    ]
)

# Path to the root folder containing subfolders for each emotion
data_path = "../../../data/MyDiffusion"


# Create a dictionary to hold file paths grouped by class
class_to_files = defaultdict(list)

# Iterate through the dataset directory to collect file paths for each class
for root, dirs, files in os.walk(data_path):
    for file in files:
        class_name = os.path.basename(root)
        file_path = os.path.join(root, file)
        class_to_files[class_name].append(file_path)

# Perform stratified split based on class distributions
train_files = []
val_files = []
test_files = []
for class_files in class_to_files.values():
    random.shuffle(class_files)
    train_idx = int(0.8 * len(class_files))  # 80% train
    val_idx = int(0.9 * len(class_files))  # 10% validation
    train_files.extend(class_files[:train_idx])
    val_files.extend(class_files[train_idx:val_idx])
    test_files.extend(class_files[val_idx:])


# Define your custom dataset class using the file paths
class CustomDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
        label = os.path.basename(
            os.path.dirname(img_path)
        )  # Get label from folder name

        if self.transform:
            image = self.transform(image)

        return image, label


# Create CustomDataset instances for train, validation, and test
train_dataset = CustomDataset(train_files, transform=transform)
val_dataset = CustomDataset(val_files, transform=transform)
test_dataset = CustomDataset(test_files, transform=transform)

# Create DataLoader for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        U1 = 16
        U2 = 28
        U3 = 40
        self.U3flat = U3 * 7 * 7
        U4 = 30
        U5 = 7
        self.W1 = nn.Parameter(0.1 * torch.randn(U1, 1, 5, 5))
        self.b1 = nn.Parameter(torch.ones(U1) / 10)

        self.W2 = nn.Parameter(0.1 * torch.randn(U2, U1, 5, 5))
        self.b2 = nn.Parameter(torch.ones(U2) / 10)

        self.W3 = nn.Parameter(0.1 * torch.randn(U3, U2, 4, 4))
        self.b3 = nn.Parameter(torch.ones(U3) / 10)

        self.W4 = nn.Parameter(0.1 * torch.randn(self.U3flat, U4))
        self.b4 = nn.Parameter(torch.ones(U4) / 10)

        self.W5 = nn.Parameter(0.1 * torch.randn(U4, U5))
        self.b5 = nn.Parameter(torch.ones(U5) / 10)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        Q1 = F.relu(F.conv2d(x, self.W1, bias=self.b1, stride=1, padding=1))
        Q1 = self.maxpool(Q1)

        Q2 = F.relu(F.conv2d(Q1, self.W2, bias=self.b2, stride=2, padding=2))
        Q2 = self.dropout1(self.maxpool(Q2))

        Q3 = F.relu(F.conv2d(Q2, self.W3, bias=self.b3, stride=2, padding=2))

        Q3 = self.dropout2(self.maxpool(Q3))

        Q3flat = Q3.view(
            -1, self.U3flat
        )  # Flatten the output for fully connected layers

        Q4 = F.relu(Q3flat.mm(self.W4) + self.b4)

        Z = Q4.mm(self.W5) + self.b5
        return Z


# Initialize the model

model = FaceCNN()

# Define loss function and optimizer

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.8, patience=3, verbose=True
)
best_val_accuracy = 0
best_model_state = None
num_epochs = 100
PATH = "cnn.pth"
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Training accuracy after each epoch
    train_accuracy = 100 * correct_train / total_train
    # Validation accuracy after each epoch
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        validation_loss = 0
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            validation_loss += F.cross_entropy(outputs, labels)
    scheduler.step(validation_loss)
    val_accuracy = 100 * correct_val / total_val
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Check if current model has higher validation accuracy than the best model
    if val_accuracy > best_val_accuracy:
        print("saving model")
        best_val_accuracy = val_accuracy
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(PATH)  # Save

    # Update the scheduler based on the validation loss


# Load the best model state
if best_model_state is not None:
    model = torch.jit.load(PATH)
    model.eval()
    print(
        "Best model loaded based on validation accuracy, it had accuracy of:"
        + str(best_val_accuracy)
    )
correct_train = 0
total_train = 0
correct_val = 0
total_val = 0
correct_test = 0
total_test = 0
with torch.no_grad():
    for data in train_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    for data in val_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

train_accuracy = 100 * correct_train / total_train
print(f"train Accuracy: {train_accuracy:.2f}%")
val_accuracy = 100 * correct_val / total_val
print(f"Validation Accuracy: {val_accuracy:.2f}%")
test_accuracy = 100 * correct_test / total_test

print(f"Test Accuracy: {test_accuracy:.2f}%")
