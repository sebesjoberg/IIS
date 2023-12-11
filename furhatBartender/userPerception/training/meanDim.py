from PIL import Image
from torchvision import datasets

# Path to your dataset
data_path = "../../../data/7EmoSet"

# Initialize ImageFolder dataset
image_dataset = datasets.ImageFolder(data_path)

# Calculate mean dimensions
total_width = 0
total_height = 0
total_images = len(image_dataset)


for i in range(total_images):
    image_path = image_dataset.imgs[i][0]  # Get the path of the image
    img = Image.open(image_path)
    width, height = img.size
    total_width += width
    total_height += height

mean_width = total_width / total_images
mean_height = total_height / total_images

print(f"Mean Width: {mean_width}, Mean Height: {mean_height}")
num_classes = len(image_dataset.classes)
print(f"Number of classes: {num_classes}")

# Count the number of images in each class
class_counts = {class_name: 0 for class_name in image_dataset.classes}
for _, class_label in image_dataset:
    class_name = image_dataset.classes[class_label]
    class_counts[class_name] += 1

# Print the number of images in each class
print("Number of images in each class:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Initialize lists to store individual aspect ratios
aspect_ratios = []

# Calculate aspect ratio for each image and store in the list
for i in range(total_images):
    image_path = image_dataset.imgs[i][0]  # Get the path of the image
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height
    aspect_ratios.append(aspect_ratio)

# Calculate the mean aspect ratio
mean_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)

print(f"Mean Aspect Ratio: {mean_aspect_ratio}")
