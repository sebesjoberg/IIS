from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

vs = analyzer.polarity_scores("I am not sure, but on second tought that would be nice")
# can be used to classify answers
print(vs)
