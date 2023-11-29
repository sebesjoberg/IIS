import torch
from PIL import Image
from torchvision import transforms


class EmotionDetector:
    def __init__(self):
        self.model = torch.jit.load("cnn.pth")
        self.transform = transforms.Compose(
            [
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


if __name__ == "__main__":
    model = EmotionDetector()
    image = Image.open("../../../data/DiffusionCropped/angry/aaaaaaaa_6.png")
    print(model.predict(image))
