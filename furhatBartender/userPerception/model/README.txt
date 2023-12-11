here is where the acutal ml model will be stored
cnn7small is a cnn model trained only on the images from DiffusionFER, and it can predict the 7 emotions, 
angry, disgusted, happy, sad, neutral, fear, surprise
accuracy on train-val-test is 89.05-69.75-79.83
cnn4big is a cnn model trained on images both from DiffusionFER but also from FER2013 and it can predict 4 emotions 
angry+disgusted = furious, happy=happy, sad+neutral=melancholic fear+surprise = aghast
accuracy on train-val-test is 87.04-88.63-87.8

execution time for both models lies around 0.012 seconds on my laptop (no GPU)

The prefered model to use is the cnn4big, it has a bigger dataset, performs better on it's dataset than the other model 
and after real life testing seems to be responsive to the four emotions it sets out to predict.
Thre is of course moments where the cnn7small might be better, perhaps when seven emotions are needed.