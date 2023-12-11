import os
import shutil

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets

# Path to the original ImageFolder dataset
original_dataset_path = "../../../data/7EmoDiffusion"


# Load the ImageFolder dataset
full_dataset = datasets.ImageFolder(root=original_dataset_path)

# Get labels and indices
targets = [label for _, label in full_dataset.samples]
train_idx, temp_idx = train_test_split(
    range(len(full_dataset)), test_size=0.2, random_state=42, stratify=targets
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=42,
    stratify=[targets[i] for i in temp_idx],
)

# Create Subset datasets using the indices
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)


def save_dataset(dataset, destination_path):
    os.makedirs(destination_path, exist_ok=True)
    for idx in dataset.indices:
        image_path, label = full_dataset.samples[idx]
        class_folder = os.path.join(destination_path, full_dataset.classes[label])
        os.makedirs(class_folder, exist_ok=True)
        shutil.copy(image_path, class_folder)


# Paths for saving train, validation, and test datasets
output_base_dir = "../../../data/7EmoSet"


# Save datasets to respective directories
save_dataset(train_dataset, output_base_dir + "/train")
save_dataset(val_dataset, output_base_dir + "/val")
save_dataset(test_dataset, output_base_dir + "/test")
