import random
from time import sleep

import spacy
from numpy.random import randint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_trf")
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
            # here we could do some check for potential drink order using the spacy nl for NOUNS perhaps and matching to our drink_dict
            context = firstInteraction(text, emotion, furhat, context)

        case 1:
            context = secondInteraction(text, emotion, furhat, context)

        case 2:
            context = thirdInteraction(text, emotion, furhat, context)

        case 3:
            context = fourthInteraction(text, emotion, furhat, context)

        case 4:
            context = fifthInteraction(text, emotion, furhat, context)
            
        case 5: 
            context = sixthInteraction(text, emotion, furhat, context)

        case _:
            context = bsay("Out of case")
    return context


map = {0: "Aghast", 1: "Furious", 2: "Happy", 3: "Melancholic"}


def firstInteraction(text, emotion, furhat, context):
    bsay("Hello, what is your name friend?", furhat)
    context["Question"] = "Name"
    print("context 1 : ", context)
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
    print("context 2 : ", context)


def thirdInteraction(text, emotion, furhat, context):
    bsay("Well you have come to the right place then", furhat)
    sleep(0.1)
    bsay("Do you like beer?", furhat)
    context["Question"] = "Beer"
    print("context 3 : ", context)
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
    print("context 4 : ", context)
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
    if context[context["Preference"]]:
        line = f"Hmm.. I noticed that you seem to be {emotion}, that you would like a {drinkchoice}, and that you would prefer something {context['Preference']}."
    else:
        line = f"Hmm.. I noticed that you seem to be {emotion}, that you would like a {drinkchoice}, and that you would not prefer something {context['Preference']}."
    bsay(
        line,
        furhat,
    )
    bsay(f"Sound like you need a {drink} then! Would you like to know something about the {drink}?", furhat)
    context["Drink"] = drink
    print("context 5: ", context)
    return context

def sixthInteraction(text, emotion, furhat, context):

    chosendrink = context["Drink"]

    answer = analyzer.polarity_scores(text)

    if answer["neg"] < answer["pos"]:
        dinfo = drinkinfo(chosendrink)
        bsay(f"{dinfo}", furhat)
        bsay("enjoy your drink!", furhat)
    else: 
        bsay(f"Okay then, enjoy your {chosendrink}", furhat)



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

def drinkinfo(drink):

    drinkinfodict = {
    "Indian Pale Ale": "Indian Pale Ale, or IPA, originated in England and was later adapted by British brewers for export to India. The extra hops helped preserve the beer during the long sea voyage.",
    "Belgian Double": "Belgian Double is a rich and malty beer brewed by Belgian Trappist monks, known for its deep flavors and higher alcohol content.",
    "Russian Imperial Stout": "Russian Imperial Stout, a favorite of Catherine the Great's court in 18th-century Russia, is a robust and dark beer with origins in England.",
    "Rasberry Fruit Labmic": "Raspberry Fruit Lambic is a Belgian beer with a fruity twist. Lambics are fermented through exposure to wild yeast and bacteria, resulting in a unique and refreshing brew.",
    "American Pale Ale": "American Pale Ale, a hop-forward beer style, emerged in the United States during the craft beer revolution, showcasing the vibrant flavors of American hops.",
    "Honey Wheat Ale": "Honey Wheat Ale is a sweet and smooth beer that often incorporates honey during the brewing process, adding a touch of natural sweetness.",
    "Belgian Triple": "Belgian Triple, brewed by Trappist monks in Belgium, is a strong and golden ale known for its complex flavor profile and high alcohol content.",
    "Hefeweizen": "Hefeweizen, a traditional German wheat beer, is prized for its fruity and spicy notes derived from the special yeast strain used in fermentation.",
    "Porter": "Porter, a dark and flavorful beer, has its roots in 18th-century London. It gained popularity among porters and laborers, inspiring its name.",
    "Milk Stout": "Milk Stout, a sweet and creamy beer, includes lactose in its brewing process, providing a smooth texture and a hint of sweetness.",
    "Barleywine": "Barleywine is a strong ale with a high alcohol content, originally brewed in England. Despite its name, it's more of a beer than a wine.",
    "Fruit Beer": "Fruit Beer comes in various styles, incorporating different fruits into the brewing process to create a refreshing and fruity taste.",
    "Session IPA": "Session IPA is a lower-alcohol version of the classic IPA, allowing beer enthusiasts to enjoy the hoppy goodness for an extended 'session' without the heavy alcohol impact.",
    "Blonde Ale": "Blonde Ale is a light and easy-drinking beer with a golden hue, offering a balanced flavor profile that appeals to a wide range of beer drinkers.",
    "Double IPA": "Double IPA, or DIPA, is a bold and hoppy beer with an extra dose of hops, providing a more intense flavor experience than its single IPA counterpart.",
    "Fruit-infused Pale Ale": "Fruit-infused Pale Ale combines the hoppy goodness of a pale ale with the added twist of fruity flavors, creating a delightful and aromatic beverage.",
    "Negroni": "Negroni, a classic cocktail originating in Italy, is a perfect balance of gin, vermouth, and Campari, creating a sophisticated and bitter-sweet flavor profile.",
    "Bitter lemon drop": "Bitter Lemon Drop is a modern twist on the classic cocktail, featuring vodka, triple sec, and a splash of bitter lemon for a refreshing and zesty taste.",
    "Zombie": "Zombie, a Tiki cocktail with a mysterious origin, packs a punch with a mix of various rums and fruit juices, creating a tropical and potent libation.",
    "Rasberry Mojito": "Raspberry Mojito is a fruity and refreshing take on the classic Cuban cocktail, blending rum, mint, lime, and raspberries for a delightful summertime drink.",
    "Espresso Martini": "Espresso Martini is a caffeinated cocktail that combines vodka, coffee liqueur, and freshly brewed espresso, creating a perfect blend of bold flavors and a caffeine kick.",
    "Blue Lagoon": "Blue Lagoon, a vibrant and tropical cocktail, features vodka, blue curaçao, and lemonade, transporting you to a turquoise paradise with each sip.",
    "Long island iced tea": "Long Island Iced Tea, despite its name, contains no actual tea. Instead, it's a potent mix of vodka, rum, gin, tequila, triple sec, sour mix, and a splash of cola.",
    "Mango Tango": "Mango Tango is a fruity and exotic cocktail that combines mango vodka, triple sec, and mango nectar, creating a dance of tropical flavors on your palate.",
    "Americano": "Americano is a classic Italian cocktail with a simple yet elegant combination of Campari, sweet vermouth, and soda water, resulting in a refreshing and bittersweet drink.",
    "Amaretto Sour": "Amaretto Sour is a sweet and tangy cocktail featuring amaretto liqueur, lemon juice, and simple syrup, creating a perfect balance of flavors.",
    "Rusty Nail": "Rusty Nail is a Scotch-based cocktail with a robust and smoky flavor, combining Scotch whisky and Drambuie for a warming and sophisticated drink.",
    "Bellini": "Bellini, created in Venice, is a sparkling cocktail made with Prosecco and peach purée, offering a delightful and elegant drink perfect for celebrations.",
    "Aperol Spritz": "Aperol Spritz, a popular Italian aperitif, combines Aperol, Prosecco, and soda water, creating a light and refreshing drink with a vibrant orange hue.",
    "Mai Tai": "Mai Tai, a tropical cocktail with roots in Polynesia, features a blend of rum, lime juice, orgeat syrup, and orange liqueur, creating a balanced and flavorful drink.",
    "Margarita": "Margarita, a classic Mexican cocktail, combines tequila, triple sec, and lime juice, served in a salt-rimmed glass for a perfect balance of sweet, sour, and salty.",
    "Strawberry Daiquiri": "Strawberry Daiquiri is a fruity and icy cocktail made with rum, lime juice, simple syrup, and fresh strawberries, offering a sweet and refreshing treat.",
}
    return drinkinfodict[drink]



def findName(text):
    doc = nlp(text)
    return [ent for ent in doc.ents if ent.label_ == "PERSON"][0]

    # legacy code down below
    # try:
    # return text.split()[-1]
    # except:
    # return None


if __name__ == "__main__":
    # end_program = False
    # print(contextToDrink(True, "Furious", "Fruity", False))
    # demo_personas()
    # idle_animation()
    print(findName("My name is Sebastian"))
