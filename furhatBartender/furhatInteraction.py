from time import sleep

from numpy.random import randint

FACES = {"Loo": "Patricia", "Amany": "Nazar"}

VOICES_EN = {"Loo": "BellaNeural", "Amany": "CoraNeural"}

VOICES_NATIVE = {"Loo": "SofieNeural", "Amany": "AmanyNeural"}


def idle_animation(furhat):
    furhat.gesture(name="GazeAway")
    gesture = {
        "frames": [
            {
                "time": [0.33],
                "persist": True,
                "params": {
                    "NECK_PAN": randint(-4, 4),
                    "NECK_TILT": randint(-4, 4),
                    "NECK_ROLL": randint(-4, 4),
                },
            }
        ],
        "class": "furhatos.gestures.Gesture",
    }
    furhat.gesture(body=gesture, blocking=True)


def LOOK_BACK(speed):
    return {
        "frames": [
            {
                "time": [0.33 / speed],
                "persist": True,
                "params": {"LOOK_DOWN": 0, "LOOK_UP": 0, "NECK_TILT": 0},
            },
            {
                "time": [1 / speed],
                "params": {"NECK_PAN": 0, "LOOK_DOWN": 0, "LOOK_UP": 0, "NECK_TILT": 0},
            },
        ],
        "class": "furhatos.gestures.Gesture",
    }


# DO NOT CHANGE
def LOOK_DOWN(speed=1):
    return {
        "frames": [
            {
                "time": [0.33 / speed],
                "persist": True,
                "params": {
                    #                'LOOK_DOWN' : 1.0
                },
            },
            {"time": [1 / speed], "persist": True, "params": {"NECK_TILT": 20}},
        ],
        "class": "furhatos.gestures.Gesture",
    }


def set_persona(persona, furhat):
    furhat.gesture(name="CloseEyes")
    furhat.gesture(body=LOOK_DOWN(speed=1), blocking=True)
    sleep(0.3)
    furhat.set_face(character=FACES[persona], mask="Adult")
    furhat.set_voice(name=VOICES_EN[persona])
    sleep(2)
    furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)


# Say with blocking (blocking say, bsay for short)
def bsay(line, furhat):
    furhat.say(text=line, blocking=True)


def demo_personas(furhat):
    set_persona("Amany")
    # furhat.
    # birthdate = furhat.ask("Which date were you born?")
    # furhat.say(f"You were born on {birthdate}")

    bsay("Lucky is a liitle bitch")
    furhat.set_voice(name=VOICES_NATIVE["Amany"])
    bsay("يسعدني أن ألتقي بكم جميعا!")  # Nice to meet you all
    furhat.set_voice(name=VOICES_EN["Amany"])

    sleep(1)
    idle_animation()
    sleep(1)

    set_persona("Loo")
    furhat.set_voice(name=VOICES_NATIVE["Loo"])
    furhat.gesture(name="Smile")
    bsay("Hej allihopa!")
    furhat.set_voice(name=VOICES_EN["Loo"])
    furhat.gesture(name="Smile")
    bsay("My name is Loo, my pronouns are they them! I speak English and Swedish")


def interaction(text, emotion, furhat, interaction_count):
    bsay(text + " " + emotion, furhat)


if __name__ == "__main__":
    end_program = False

    demo_personas()
    idle_animation()
