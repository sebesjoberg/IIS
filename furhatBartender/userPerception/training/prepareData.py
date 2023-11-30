import os

import cv2

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("../model/frontal_face_features.xml")

# Path to your dataset folder containing subfolders for different emotions
data_dir = "../../../data/DiffusionWhole"

# Output directory to save cropped face images
output_base_dir = "../../../data/MyDiffusion"
os.makedirs(output_base_dir, exist_ok=True)


# Function to detect faces and crop images
def crop_and_save_largest_face(image_path, output_base_dir):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        # Find the largest face
        largest_face_idx = sorted(
            range(len(faces)), key=lambda i: faces[i][2] * faces[i][3], reverse=True
        )[0]
        x, y, w, h = faces[largest_face_idx]

        # Crop the largest face
        face = img[y : y + h, x : x + w]

        # Get the emotion (assuming the image path contains the emotion attribute)
        emotion = os.path.basename(os.path.dirname(image_path))

        # Create output directory for the emotion if it doesn't exist
        output_emotion_dir = os.path.join(output_base_dir, emotion)
        os.makedirs(output_emotion_dir, exist_ok=True)

        # Save the largest face to the corresponding emotion folder
        output_path = os.path.join(
            output_emotion_dir, f"cropped_face_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, face)


# Process each image in the dataset folders
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(
            ".png"
        ):  # Adjust file extensions as needed
            image_path = os.path.join(root, file)
            crop_and_save_largest_face(image_path, output_base_dir)
