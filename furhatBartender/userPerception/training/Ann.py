import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Define transforms for data preprocessing and augmentation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485], std=[0.229]
        ),  # For grayscale, mean and std are single values
    ]
)

# Path to the root folder containing subfolders for each emotion
data_path = "../../../data/DiffusionCropped"

# Create dataset using ImageFolder
emotion_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Use DataLoader to create batches of data
train_size = int(0.7 * len(emotion_dataset))  # 70% of the dataset for training
val_size = int(0.15 * len(emotion_dataset))  # 15% for validation
test_size = len(emotion_dataset) - train_size - val_size  # Remaining for test

# Use random_split to split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(
    emotion_dataset, [train_size, val_size, test_size]
)

# Create DataLoaders for train, validation, and test sets
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flattening = 24 * 28 * 28
        self.fc1 = nn.Linear(self.flattening, 40)
        self.fc2 = nn.Linear(40, 7)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout1(self.pool(torch.relu(self.conv1(x))))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout2(self.pool(torch.relu(self.conv3(x))))
        x = x.view(-1, self.flattening)  # Flatten the output for fully connected layers
        x = self.dropout3(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Initialize the model

model = FaceCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_val_accuracy = 0.0
best_model_state = None
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
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
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Check if current model has higher validation accuracy than the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict()

# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Best model loaded based on validation accuracy.")

# Validation accuracy
correct_val = 0
total_val = 0
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

val_accuracy = 100 * correct_val / total_val
print(f"Validation Accuracy: {val_accuracy:.2f}%")
# Test accuracy and loss after training with the best model
model.eval()
correct_test = 0
total_test = 0
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
average_test_loss = test_loss / len(test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {average_test_loss:.4f}")

# Save the best model's state to a file
if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print("Best model saved.")
