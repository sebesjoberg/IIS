import torch
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

model = torch.jit.load("cnn.pth")
correct_val = 0
total_val = 0
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data
        outputs = model(inputs)
        print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()
