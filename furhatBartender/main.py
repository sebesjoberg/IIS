# Main file from where the actual bartender system can be ranned

import queue
import threading
from collections import Counter

import cv2
import speech_recognition as sr
from emotionDetector import EmotionDetector
from faceDetector import FaceDetector
from furhat_remote_api import FurhatRemoteAPI
from furhatInteraction import interaction, set_persona

FD = FaceDetector()
ED = EmotionDetector()
recognizer = sr.Recognizer()
cam = cv2.VideoCapture(0)
FURHAT_IP = "localhost"

furhat = FurhatRemoteAPI(FURHAT_IP)
emotion_queue = queue.Queue(maxsize=1)


def capture_voice_input():
    with sr.Microphone() as source:
        print("Listening...")
        aud = recognizer.listen(source)
    return aud


def convert_voice_to_text(aud):
    try:
        text = recognizer.recognize_google(aud)
        print("You said: " + text)
    except sr.UnknownValueError:
        text = ""

    except sr.RequestError as e:
        text = ""
        print("Error; {0}".format(e))
    return text


def process_voice_command(text):
    if "hello" in text.lower():
        print("Hello! How can I help you?")
    elif "goodbye" in text.lower():
        print("Goodbye! Have a great day!")
        return True
    else:
        print("I didn't understand that command. Please try again.")
    return False


def calculate_emotion():
    global cam, FD, ED, emotion_queue
    last_emotions = []
    nth_frame = 3
    frame_counter = -1  # start from -1 so first frame is frame 0
    while True:
        ret, frame = cam.read()
        frame_counter += 1
        if not ret:
            break
        if frame_counter % nth_frame != 0:
            continue  # Skip frames until the nth_frame is reached

        face = FD.find_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if face is not None:
            x, y, w, h = face
            emotion = ED.predict(frame[y : y + h, x : x + w])
            last_emotions.append(emotion)
        if len(last_emotions) > 10:  # change this to make it more responsive/robust
            last_emotions.pop(0)

        if len(last_emotions) > 0:
            most_common_emotion = Counter(last_emotions).most_common(1)
            if most_common_emotion:
                most_common_emotion = most_common_emotion[0][0]
            else:
                most_common_emotion = "Neutral"
        else:
            most_common_emotion = "Neutral"
        try:
            emotion_queue.put_nowait(most_common_emotion)
        except queue.Full:
            emotion_queue.get()
            emotion_queue.put_nowait(most_common_emotion)


emotion_thread = threading.Thread(target=calculate_emotion)
emotion_thread.daemon = True
emotion_thread.start()

# set persona here
set_persona("Amany", furhat)
interaction_count = 0
context = {}
# could make the whole loop into a function that runs over the specified interaction max count
# run the loop until max count, but only start/restart the loop if a face appears in the frame?
# should probably restart the whole emotionthread in that case too
# do put that emotion on the first face in the queue too(so that it is not empty for first interaction)
# when doing this our abrtender could also be the one initializing the convo
while True:
    aud = capture_voice_input()
    text = convert_voice_to_text(aud)
    try:
        emotion = emotion_queue.get(timeout=1)  # Timeout to avoid blocking indefinitely
    except queue.Empty:
        emotion = "Neutral"  # Default emotion if queue is empty
    print(emotion)
    interaction(text, emotion, furhat, interaction_count, context)
    interaction_count += 1
    if interaction_count == 6:
        interaction_count = 0
        context = {}
        break
