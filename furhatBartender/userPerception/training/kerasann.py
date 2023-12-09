import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.compat.v1.keras.models import Sequential, load_model
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator

# Define paths to your image folders for train, validation, and test

train_data_dir = "../../../data/customset/train"
val_data_dir = "../../../data/customset/val"
test_data_dir = "../../../data/customset/test"

# Define data generators with augmentation for train and validation sets
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode="categorical",
    color_mode="grayscale",
)

validation_generator = test_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    color_mode="grayscale",
)


# Define the CNN model
def conv_model(device):
    with tf.device(device):
        model = Sequential(
            [
                Conv2D(16, (5, 5), activation="relu", input_shape=(224, 224, 1)),
                MaxPooling2D((2, 2)),
                Conv2D(32, (5, 5), activation="relu"),
                MaxPooling2D((2, 2)),
                Dropout(0.2),
                Conv2D(48, (4, 4), activation="relu"),
                MaxPooling2D((2, 2)),
                Dropout(
                    0.4,
                ),
                Flatten(),
                Dense(40, activation="relu"),
                Dense(4),
            ]
        )

        model.compile(
            optimizer=Adam(lr=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


model = conv_model("DML")
# Callback to save the best model based on validation accuracy
checkpoint = ModelCheckpoint(
    "best_model.bh5", monitor="val_acc", verbose=2, save_best_only=True, mode="max"
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    callbacks=[checkpoint],
)


# Load the best model for evaluation
best_model = load_model("best_model.bh5")
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    color_mode="grayscale",
)

test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
