# Main file from where the actual bartender system can be ranned

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


# set persona her
set_persona("Amany", furhat, async_req=True)
interaction_count = 0
context = {}
while True:
    aud = capture_voice_input()
    text = convert_voice_to_text(aud)
    ret, frame = cam.read()
    if not ret:
        break
    face = FD.find_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if face is not None:
        x, y, w, h = face

        emotion = ED.predict(frame[y : y + h, x : x + w])
    else:
        emotion = "Neutral"

    interaction(text, emotion, furhat, interaction_count, context)
    interaction_count += 1
    if interaction_count == 5:
        interaction_count = 0
        context = {}
        break
