import cv2


class FaceDetector:
    def __init__(self):
        self.model = cv2.CascadeClassifier(
            "userPerception/model/frontal_face_features.xml"
        )

    def find_face(self, image):  # takes grayscale image
        # return biggest face, if no face it returns none
        faces = self.model.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        max_face_size = 0
        largest_face = None
        for x, y, w, h in faces:
            face_size = w * h
            if face_size > max_face_size:
                max_face_size = face_size
                largest_face = (x, y, w, h)

        return largest_face


if __name__ == "__main__":
    from emotionDetector import EmotionDetector

    cam = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    emotion_model = EmotionDetector(True)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF
        face = face_detector.find_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if face is not None:
            x, y, w, h = face
            emotion = emotion_model.predict(frame[y : y + h, x : x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            org = (int(x) + 25, int(y) - 25)

            cv2.putText(
                frame,
                emotion,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("Webcam Feed", frame)
        if key == 27:  # ESC pressed
            break
