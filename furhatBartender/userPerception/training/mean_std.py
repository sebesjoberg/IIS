import os

import numpy as np
from PIL import Image

# Path to the root folder containing subfolders for each emotion
data_path = "../../../data/customset"


# Function to calculate mean and std for a dataset of grayscale images
def calculate_mean_and_std_grayscale(dataset_path):
    mean = 0.0
    std = 0.0
    total_images = 0

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(
                (".jpg", ".jpeg", ".png")
            ):  # Adjust file extensions as needed
                image_path = os.path.join(root, file)
                image = (
                    np.array(Image.open(image_path).convert("L")) / 255.0
                )  # Convert to grayscale and normalize
                mean += np.mean(image)
                std += np.std(image)
                total_images += 1

    mean /= total_images
    std /= total_images
    print(total_images)
    return mean, std


# Calculate mean and std for your grayscale dataset
mean_value, std_value = calculate_mean_and_std_grayscale(data_path)

print("Mean value:", mean_value)
print("Standard deviation value:", std_value)
