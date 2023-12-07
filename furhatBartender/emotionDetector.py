import timeit

import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
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
    def __init__(self, e4=True):
        self.e4 = e4
        if e4:
            self.model = torch.jit.load("userPerception/model/cnn4class-strat.pth")
            self.map = {0: "Aghast", 1: "Furious", 2: "Happy", 3: "Melancholic"}
            # angry+disgusted = furious, happy=happy, sad+neutral=melancholic fear+surprise = aghast
        else:
            self.model = torch.jit.load("userPerception/model/cnn11-30-strat.pth")
            self.map = {
                0: "Angry",
                1: "Disgust",
                2: "Fear",
                3: "Happy",
                4: "Neutral",
                5: "Sad",
                6: "Surprise",
            }
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
        prob = self.model(self.transform(image))
        # prob[0, 4] -= 0.5
        _, predicted = torch.max(prob, 1)
        # print(prob)
        return self.map.get(predicted.item(), "Unknown Emotion")

    def evaluate(self):
        # Path to the root folder containing subfolders for each emotion
        if self.e4:
            data_path = "../data/MyAggregatedDiffusion"
        else:
            data_path = "../data/MyDiffusion"

        full_dataset = datasets.ImageFolder(root=data_path, transform=self.transform)

        # Extract labels and indices for stratified split
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

        # Create SubsetRandomSampler for train, validation, and test
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # Create DataLoaders using SubsetRandomSampler
        train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(full_dataset, batch_size=32, sampler=val_sampler)
        test_loader = DataLoader(full_dataset, batch_size=32, sampler=test_sampler)
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


def test_time(model, image):
    def classification(model, image):
        model.predict(image)

    times = 10
    execution_time = timeit.timeit(lambda: classification(model, image), number=times)

    print(f"Execution time: {execution_time/times} seconds")


if __name__ == "__main__":
    e4 = True
    model = EmotionDetector(e4)  # previews one prediction from the dataset
    if e4:
        image = Image.open(
            "../data/MyAggregatedDiffusion/aghast/cropped_face_alzbfyrn_3.png"
        )
    else:
        image = Image.open("../data/MyDiffusion/disgust/cropped_face_ahdmclwl_5.png")
    print(model.predict(image))
    test_time(model, image)
    model.evaluate()
