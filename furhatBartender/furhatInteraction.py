import random
from time import sleep

from numpy.random import randint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
FACES = {"Loo": "Patricia", "Amany": "Nazar"}

VOICES_EN = {"Loo": "BellaNeural", "Amany": "CoraNeural"}

VOICES_NATIVE = {"Loo": "SofieNeural", "Amany": "AmanyNeural"}
drinkdict = {
    "Indian Pale Ale": ["Furious", "Bitter", True],
    "Belgian Double": ["Furious", "Sweet", True],
    "Russian Imperial Stout": ["Furious", "Strong", True],
    "Rasberry Fruit Labmic": ["Furious", "Fruity", True],
    "American Pale Ale": ["Aghast", "Bitter", True],
    "Honey Wheat Ale": ["Aghast", "Sweet", True],
    "Belgian Triple": ["Aghast", "Strong", True],
    "Hefeweizen": ["Aghast", "Fruity", True],
    "Porter": ["Melancholic", "Bitter", True],
    "Milk Stout": ["Melancholic", "Sweet", True],
    "Barleywine": ["Melancholic", "Strong", True],
    "Fruit Beer": ["Melancholic", "Fruity", True],
    "Session IPA": ["Happy", "Bitter", True],
    "Blonde Ale": ["Happy", "Sweet", True],
    "Double IPA": ["Happy", "Strong", True],
    "Fruit-infused Pale Ale": ["Happy", "Fruity", True],
    "Negroni": ["Furious", "Bitter", False],
    "Bitter lemon drop": ["Furious", "Sweet", False],
    "Zombie": ["Furious", "Strong", False],
    "Rasberry Mojito": ["Furious", "Fruity", False],
    "Espresso Martini": ["Aghast", "Bitter", False],
    "Blue Lagoon": ["Aghast", "Sweet", False],
    "Long island iced tea": ["Aghast", "Strong", False],
    "Mango Tango": ["Aghast", "Fruity", False],
    "Americano": ["Melancholic", "Bitter", False],
    "Amaretto Sour": ["Melancholic", "Sweet", False],
    "Rusty Nail": ["Melancholic", "Strong", False],
    "Bellini": ["Melancholic", "Fruity", False],
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


def interaction(text, emotion, furhat, interaction_count, context):
    match interaction_count:
        case 0:
            # here we could do some check for potential drink order
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


map = {0: "Aghast", 1: "Furious", 2: "Happy", 3: "Melancholic"}


def firstInteraction(text, emotion, furhat, context):
    bsay("Hello, what is your name friend?", furhat)
    context["Question"] = "Name"
    return context


def secondInteraction(text, emotion, furhat, context):
    if context["Question"] == "Name":
        context["Name"] = findName(text)
    bsay(f"Welcome, {context['Name']}, you can call me barty the bartender!", furhat)
    if emotion in ["Furious"]:
        bsay("Why so serious?", furhat)

    elif emotion in ["Aghast"]:
        bsay("Is something bothering you?", furhat)

    elif emotion in ["Melancholic"]:
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
    if context[context["preference"]]:
        line = f"Hmm.. I noticed that you seem to be {emotion}, that you would like a {drinkchoice}, and that you would prefer something {context['Preference']}."
    else:
        line = f"Hmm.. I noticed that you seem to be {emotion}, that you would like a {drinkchoice}, and that you would not prefer something {context['Preference']}."
    bsay(
        line,
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

    # Bitter sweet strong fruity
    matching_drinks = []
    for pref in preference:
        for drink, ingredients in drinkdict.items():
            if all(elem in ingredients for elem in [beer, emotion, pref]):
                matching_drinks.append(drink)
    return random.choice(matching_drinks)


def findName(text):  # this could be more sophisticated
    try:
        return text.split()[-1]
    except:
        return None


if __name__ == "__main__":
    # end_program = False
    print(contextToDrink(True, "Furious", "Fruity", False))
    # demo_personas()
    # idle_animation()
