import timeit

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# this model can be called with an img containing one face to predict the faces emotion
# please try and use as little background as possible, aka only the face
# should put so that gpu is used if available
class ConditionalToPILImage:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return transforms.ToPILImage()(img)
        return img


class EmotionDetector:
    def __init__(self):
        self.model = torch.jit.load("../model/cnn11-30.pth")
        self.model.eval()
        self.transform = transforms.Compose(
            [
                ConditionalToPILImage(),
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.473], std=[0.285]
                ),  # For grayscale, mean and std are single values
            ]
        )
        self.map = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise",
        }

    def predict(self, image):
        _, predicted = torch.max(self.model(self.transform(image)), 1)

        return self.map.get(predicted.item(), "Unknown Emotion")

    def evaluate(self):
        # Path to the root folder containing subfolders for each emotion
        data_path = "../../../data/MyDiffusion"

        # Create dataset using ImageFolder
        emotion_dataset = datasets.ImageFolder(root=data_path, transform=self.transform)

        # Use DataLoader to create batches of data
        train_size = int(0.8 * len(emotion_dataset))  # 80% of the dataset for training
        val_size = int(0.1 * len(emotion_dataset))  # 10% for validation
        test_size = len(emotion_dataset) - train_size - val_size  # Remaining for test

        # Use random_split to split the dataset into train, validation, and test sets
        train_dataset, val_dataset, test_dataset = random_split(
            emotion_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Create DataLoaders for train, validation, and test sets
        train_loader = DataLoader(train_dataset)
        val_loader = DataLoader(val_dataset)
        test_loader = DataLoader(test_dataset)
        correct_train = 0
        total_train = 0
        correct_val = 0
        total_val = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data in train_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            for data in val_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
            for data in test_loader:
                inputs, labels = data
                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f"train Accuracy: {train_accuracy:.2f}%")
        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        test_accuracy = 100 * correct_test / total_test

        print(f"Test Accuracy: {test_accuracy:.2f}%")


def test_time(model):
    def classification(model, image):
        model.predict(image)

    image = Image.open("../../../data/DiffusionCropped/angry/aaaaaaaa_6.png")
    times = 10
    execution_time = timeit.timeit(lambda: classification(model, image), number=times)

    print(f"Execution time: {execution_time/times} seconds")


if __name__ == "__main__":
    model = EmotionDetector()  # previews one prediction from the dataset
    image = Image.open("../../../data/MyDiffusion/disgust/cropped_face_cjdxeady_5.png")
    print(model.predict(image))
    test_time(model)
    model.evaluate()
