import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets

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
train_transform = transforms.Compose(
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
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=0.15)], 0.5
        ),
        transforms.Normalize(
            mean=[0.473], std=[0.285]
        ),  # For grayscale, mean and std are single values
    ]
)
# Path to the root folder containing subfolders for each emotion
data_path = "../../../data/customset"

train_set = datasets.ImageFolder(root=data_path + "/train", transform=train_transform)
val_set = datasets.ImageFolder(root=data_path + "/val", transform=transform)
test_set = datasets.ImageFolder(root=data_path + "/test", transform=transform)


# Create DataLoaders using SubsetRandomSampler
train_loader = DataLoader(train_set, batch_size=32)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        U1 = 16
        U2 = 32
        U3 = 48
        self.U3flat = U3 * 7 * 7
        U4 = 40
        U5 = 4
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

        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        Q1 = F.relu(F.conv2d(x, self.W1, bias=self.b1, stride=1, padding=1))
        Q1 = self.maxpool(Q1)

        Q2 = F.relu(F.conv2d(Q1, self.W2, bias=self.b2, stride=2, padding=2))
        Q2 = self.dropout1(self.maxpool(Q2))

        Q3 = F.relu(F.conv2d(Q2, self.W3, bias=self.b3, stride=2, padding=2))
        Q3 = self.maxpool(Q3)
        Q3flat = Q3.view(
            -1, self.U3flat
        )  # Flatten the output for fully connected layers

        Q4 = F.relu(Q3flat.mm(self.W4) + self.b4)

        Z = Q4.mm(self.W5) + self.b5

        return Z


# Initialize the model
keep_training = True
if keep_training:
    PATH = "../model/cnn4big.pth"
    model = torch.jit.load(PATH)
    total_val = 0
    correct_val = 0
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    best_val_accuracy = 100 * correct_val / total_val
    print(f"Validation Accuracy: {best_val_accuracy:.2f}%, ")

else:
    model = FaceCNN()
    best_val_accuracy = 0
# weights = torch.tensor(
#    [225 / 1188, 243 / 1188, 318 / 1188, 402 / 1188], dtype=torch.float
# )
weights = torch.tensor(
    [2004 / 10468, 1506 / 10468, 2922 / 10468, 4036 / 10468], dtype=torch.float
)
# Define loss function and optimizer

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.6, patience=3, verbose=True
)

best_model_state = None
num_epochs = 100
SAVE_PATH = "cnn.pth"
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs
        labels = labels
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, weight=weights)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
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
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Training Accuracy: {train_accuracy:.2f}%, "
        f"Validation Accuracy: {val_accuracy:.2f}%, "
        f"Total Loss: {running_loss/total_train:.4f}"  # Printing total loss
    )

    # Check if current model has higher validation accuracy than the best model
    if val_accuracy > best_val_accuracy:
        print("saving model")
        best_val_accuracy = val_accuracy
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(SAVE_PATH)  # Save

    # Update the scheduler based on the validation loss


# Load the best model state
if best_model_state is not None:
    model = torch.jit.load(SAVE_PATH)
    model.eval()
    print(
        "Best model loaded based on validation accuracy, it had accuracy of:"
        + str(best_val_accuracy)
    )
train_loader.dataset.transform = train_transform
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
