import timeit

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} for computations.")

        self.model = torch.jit.load("userPerception/model/cnn4big.pth")
        self.model = self.model.to(self.device)
        self.map = {0: "Aghast", 1: "Furious", 2: "Happy", 3: "Melancholic"}
        # angry+disgusted = furious, happy=happy, sad+neutral=melancholic fear+surprise = aghast
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

    def predict(self, image):
        prob = self.model(self.transform(image).to(self.device))

        _, predicted = torch.max(prob, 1)

        return self.map.get(predicted.item(), "Unknown Emotion")

    def evaluate(self):
        # Path to the root folder containing subfolders for each emotion
        data_path = "../data/4EmoSet"
        train_loader = DataLoader(
            datasets.ImageFolder(root=data_path + "/train", transform=self.transform),
            batch_size=32,
        )
        val_loader = DataLoader(
            datasets.ImageFolder(root=data_path + "/val", transform=self.transform),
            batch_size=32,
        )
        test_loader = DataLoader(
            datasets.ImageFolder(root=data_path + "/test", transform=self.transform),
            batch_size=32,
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
                outputs = self.model(inputs.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels.to(self.device)).sum().item()
            for data in val_loader:
                inputs, labels = data
                outputs = self.model(inputs.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels.to(self.device)).sum().item()
            for data in test_loader:
                inputs, labels = data
                outputs = self.model(inputs.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels.to(self.device)).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f"train Accuracy: {train_accuracy:.2f}%")
        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        test_accuracy = 100 * correct_test / total_test

        print(f"Test Accuracy: {test_accuracy:.2f}%")


def test_time(model, image):
    def classification(model, image):
        model.predict(image)

    times = 10
    execution_time = timeit.timeit(lambda: classification(model, image), number=times)

    print(f"Execution time: {execution_time/times} seconds")


if __name__ == "__main__":
    model = EmotionDetector()
    image = Image.open("../data/4EmoSet/test/happy/happy_ahvsvhfy_1.png")
    print(model.predict(image))  # previews one prediction from the dataset
    test_time(model, image)
    model.evaluate()
