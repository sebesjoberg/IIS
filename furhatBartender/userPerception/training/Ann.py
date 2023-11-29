import torch
import torch.nn as nn
import torch.nn.functional as F
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
            mean=[0.473], std=[0.285]
        ),  # For grayscale, mean and std are single values
    ]
)

# Path to the root folder containing subfolders for each emotion
data_path = "../../../data/DiffusionCropped"

# Create dataset using ImageFolder
emotion_dataset = datasets.ImageFolder(root=data_path, transform=transform)
class_to_idx = emotion_dataset.class_to_idx

# Invert the mapping to get index-to-class mapping
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

# Print the index-to-class mapping
print(idx_to_class)
# Use DataLoader to create batches of data
train_size = int(0.8 * len(emotion_dataset))  # 70% of the dataset for training
val_size = int(0.1 * len(emotion_dataset))  # 15% for validation
test_size = len(emotion_dataset) - train_size - val_size  # Remaining for test

# Use random_split to split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(
    emotion_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

# Create DataLoaders for train, validation, and test sets
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset)
test_loader = DataLoader(test_dataset)


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        U1 = 16
        U2 = 32
        U3 = 64
        self.U3flat = U3 * 26 * 26
        U4 = 80
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

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        Q1 = F.relu(F.conv2d(x, self.W1, bias=self.b1, stride=1, padding=1))
        Q1 = self.maxpool(Q1)

        Q2 = F.relu(F.conv2d(Q1, self.W2, bias=self.b2, stride=1, padding=1))
        Q2 = self.dropout1(self.maxpool(Q2))

        Q3 = F.relu(F.conv2d(Q2, self.W3, bias=self.b3, stride=1, padding=1))
        Q3 = self.maxpool(Q3)

        Q3flat = Q3.view(
            -1, self.U3flat
        )  # Flatten the output for fully connected layers

        Q4 = self.dropout2(F.relu(Q3flat.mm(self.W4) + self.b4))

        Z = Q4.mm(self.W5) + self.b5
        return Z


# Initialize the model

model = FaceCNN()

# Define loss function and optimizer

optimizer = optim.Adam(model.parameters(), lr=0.001)
best_val_accuracy = 0.0
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
        print("saving model")
        best_val_accuracy = val_accuracy
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(PATH)  # Save


# Load the best model state
if best_model_state is not None:
    model = torch.jit.load(PATH)
    model.eval()
    print(
        "Best model loaded based on validation accuracy, it had accuracy of:"
        + str(best_val_accuracy)
    )

# Validation accuracy


correct_train = 0
total_train = 0
with torch.no_grad():
    for data in train_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

train_accuracy = 100 * correct_train / total_train
print(f"train Accuracy: {train_accuracy:.2f}%")

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

correct_test = 0
total_test = 0

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test

print(f"Test Accuracy: {test_accuracy:.2f}%")
