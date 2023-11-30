here is where the acutal ml model will be stored
cnn11-30.pth is a CNN model, it had a training accuracy of:92.42 ,validation accuracy of:70.34 and test accuracy of:77.5
bear in mind that images were slightly distorted and also flipped during training, something which might affect the accuracy and such.
Training data was also not stratified, something which might be a problem, seeing as there are "only" 1285 images and 7 classes,
resulting in an average image per class to be 184, with training data beeing 80% there is some chance that some lasses might not be 
as present in part of train/val/test as others are.