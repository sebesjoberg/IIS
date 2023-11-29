import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transforms for data preprocessing and augmentation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize image tensors
    ]
)

# Path to the root folder containing subfolders for each emotion
data_path = "../../../data/DiffusionCropped"

# Create dataset using ImageFolder
emotion_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Use DataLoader to create batches of data
batch_size = 32
train_loader = DataLoader(emotion_dataset, batch_size=batch_size, shuffle=True)


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 7)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten the output for fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Initialize the model

model = FaceCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Batch [{i + 1}/{len(train_loader)}], "
                f"Loss: {running_loss / 100:.4f}"
            )
            running_loss = 0.0

print("Finished Training")
