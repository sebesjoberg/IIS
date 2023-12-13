# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# analyzer = SentimentIntensityAnalyzer()

# vs = analyzer.polarity_scores("I like cocktails")
# can be used to classify answers
# print(vs)
import spacy

# Load English tokenizer, tagger, parser, NER, etc.
nlp = spacy.load("en_core_web_trf")
# Example text
text = "Can I have a margarita and a beer, please,strong?"
# Process the text
doc = nlp(text)

# Find nouns in the text that might represent drinks


drink_orders = [token.text for token in doc if token.pos_ == "NOUN"]
print(drink_orders)

doc = nlp("My name is sebastian, barkeep, bartender, cat,john")
ents = list(doc.ents)
for ent in ents:
    print(ent.label_)
    print(ent)
