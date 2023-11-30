import os

import cv2
import pandas as pd
from feat import Detector


def picToCsv(datafolder="../../../data/DiffusionCropped"):
    detector = Detector(device="auto")
    columns = [
        "emotion",
        "AU01",
        "AU02",
        "AU04",
        "AU05",
        "AU06",
        "AU07",
        "AU09",
        "AU10",
        "AU11",
        "AU12",
        "AU14",
        "AU15",
        "AU17",
        "AU20",
        "AU23",
        "AU24",
        "AU25",
        "AU26",
        "AU28",
        "AU43",
    ]
    df = pd.DataFrame(columns=columns)
    print("load detector")
    images = []
    for filename in os.listdir(datafolder):
        emotion = filename
        try:
            for img in os.listdir(datafolder + "/" + emotion):
                image = cv2.imread(datafolder + "/" + emotion + "/" + img)
                print(image.shape)
                images.append((image, emotion))
        except Exception as e:
            print(e)

    print("here")
    i = 0
    for image_emo in images:
        image, emo = image_emo
        detected_faces = detector.detect_faces(image)
        detected_landmarks = detector.detect_landmarks(image, detected_faces)

        detected_aus = detector.detect_aus(image, detected_landmarks)
        try:
            saving = [emo]
            saving.extend(detected_aus[0][0])

            df.loc[len(df)] = saving
            i += 1
            print(i)
        except Exception as e:
            print(e)

    df.to_csv("dataset.csv", index=False)


if __name__ == "__main__":
    picToCsv("../../../data/DiffusionCropped")
