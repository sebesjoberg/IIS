import random
from time import sleep

from numpy.random import randint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
FACES = {"Loo": "Patricia", "Amany": "Nazar"}

VOICES_EN = {"Loo": "BellaNeural", "Amany": "CoraNeural"}

VOICES_NATIVE = {"Loo": "SofieNeural", "Amany": "AmanyNeural"}
drinkdict = {
    "Indian Pale Ale": ["Angry", "Disgusted", "Bitter", True],
    "Belgian Double": ["Angry", "Disgusted", "Sweet", True],
    "Russian Imperial Stout": ["Angry", "Disgusted", "Strong", True],
    "Rasberry Fruit Labmic": ["Angry", "Disgusted", "Fruity", True],
    "American Pale Ale": ["Fear", "Surprise", "Bitter", True],
    "Honey Wheat Ale": ["Fear", "Surprise", "Sweet", True],
    "Belgian Triple": ["Fear", "Surprise", "Strong", True],
    "Hefeweizen": ["Fear", "Surprise", "Fruity", True],
    "Porter": ["Sad", "Neutral", "Bitter", True],
    "Milk Stout": ["Sad", "Neutral", "Sweet", True],
    "Barleywine": ["Sad", "Neutral", "Strong", True],
    "Fruit Beer": ["Sad", "Neutral", "Fruity", True],
    "Session IPA": ["Happy", "Bitter", True],
    "Blonde Ale": ["Happy", "Sweet", True],
    "Double IPA": ["Happy", "Strong", True],
    "Fruit-infused Pale Ale": ["Happy", "Fruity", True],
    "Negroni": ["Angry", "Disgusted", "Bitter", False],
    "Bitter lemon drop": ["Angry", "Disgusted", "Sweet", False],
    "Zombie": ["Angry", "Disgusted", "Strong", False],
    "Rasberry Mojito": ["Angry", "Disgusted", "Fruity", False],
    "Espresso Martini": ["Fear", "Surprise", "Bitter", False],
    "Blue Lagoon": ["Fear", "Surprise", "Sweet", False],
    "Long island iced tea": ["Fear", "Surprise", "Strong", False],
    "Mango Tango": ["Fear", "Surprise", "Fruity", False],
    "Americano": ["Sad", "Neutral", "Bitter", False],
    "Amaretto Sour": ["Sad", "Neutral", "Sweet", False],
    "Rusty Nail": ["Sad", "Neutral", "Strong", False],
    "Bellini": ["Sad", "Neutral", "Fruity", False],
    "Aperol Spritz": ["Happy", "Bitter", False],
    "Mai Tai": ["Happy", "Sweet", False],
    "Margarita": ["Happy", "Strong", False],
    "Strawberry Daquiri": ["Happy", "Fruity", False],
}


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


def interaction(text, emotion, furhat, interaction_count, context):
    match interaction_count:
        case 0:
            context = firstInteraction(text, emotion, furhat, context)

        case 1:
            context = secondInteraction(text, emotion, furhat, context)

        case 2:
            context = thirdInteraction(text, emotion, furhat, context)

        case 3:
            context = fourthInteraction(text, emotion, furhat, context)

        case 4:
            context = fifthInteraction(text, emotion, furhat, context)

        case _:
            context = bsay("Out of case")
    return context


def firstInteraction(text, emotion, furhat, context):
    bsay("Hello, what is your name friend?", furhat)
    context["Question"] = "Name"
    return context


map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


def secondInteraction(text, emotion, furhat, context):
    if context["Question"] == "Name":
        context["Name"] = findName(text)
    bsay(f"Welcome, {context['Name']}, you can call me barty the bartender!", furhat)
    if emotion in ["Angry", "Disgust"]:
        bsay("Why so serious?", furhat)

    elif emotion in ["Fear", "Surprise"]:
        bsay("Is something bothering you?", furhat)

    elif emotion in ["Sad", "Neutral"]:
        bsay("It looks like you had a long day?", furhat)

    else:
        bsay("It is a good day is it not?", furhat)
    context["Question"] = None


def thirdInteraction(text, emotion, furhat, context):
    bsay("Well you have come to the right place then", furhat)
    sleep(0.1)
    bsay("Do you like beer?", furhat)
    context["Question"] = "Beer"
    return context


def fourthInteraction(text, emotion, furhat, context):
    if context["Question"] == "Beer":
        vs = analyzer.polarity_scores(text)
        if vs["neg"] > vs["pos"]:
            context["Beer"] = False
            bsay("Cocktail it is!", furhat)
        else:
            context["Beer"] = True

    bsay("Do you feel like something bitter, sweet, strong or fruity?", furhat)
    context["Question"] = "Preference"
    return context


def fifthInteraction(text, emotion, furhat, context):
    if context["Question"] == "Preference":
        vs = analyzer.polarity_scores(text)
        if "bitter" in text.lower():
            context["Preference"] = "Bitter"
        elif "sweet" in text.lower():
            context["Preference"] = "Sweet"
        elif "strong" in text.lower():
            context["Preference"] = "Strong"
        elif "fruity" in text.lower():
            context["Preference"] = "Fruity"
        else:
            context["Preference"] = "None"

        if vs["neg"] > vs["pos"]:
            context[context["Preference"]] = False
        else:
            context[context["Preference"]] = True

    drink = contextToDrink(
        context["Beer"], emotion, context["Preference"], context[context["Preference"]]
    )
    if context["Beer"]:
        drinkchoice = "beer"
    else:
        drinkchoice = "cocktail"
    bsay(
        f"Hmm.. I noticed that you seem to be {emotion}, that you would like a {drinkchoice} and that you would prefer something {context['Preference']}",
        furhat,
    )
    bsay(f"How about a {drink}?", furhat)
    return context


def contextToDrink(beer, emotion, preference, preferencefeeling):
    if preferencefeeling:
        preference = [preference]

    else:
        if preference == "Bitter":
            preference = ["Sweet", "Strong", "Fruity"]
        elif preference == "Sweet":
            preference = ["Bitter", "Strong", "Fruity"]
        elif preference == "Strong":
            preference = ["Bitter", "Sweet", "Fruity"]
        else:
            preference = ["Bitter", "Sweet", "Strong"]

    # Bitter sweet stronf fruity
    matching_drinks = []
    for pref in preference:
        for drink, ingredients in drinkdict.items():
            if all(elem in ingredients for elem in [beer, emotion, pref]):
                matching_drinks.append(drink)
    return random.choice(matching_drinks)


def findName(text):
    try:
        return text.split()[-1]
    except:
        return None


if __name__ == "__main__":
    # end_program = False
    print(contextToDrink(False, "Happy", "Fruity", False))
    # demo_personas()
    # idle_animation()
