from time import time

import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

tf.disable_v2_behavior()


def conv_model(device, filters, kz):
    with tf.device(device):
        model = Sequential()
        model.add(
            Conv2D(
                filters,
                kernel_size=3,
                padding="same",
                activation="relu",
                input_shape=(28, 28, 1),
            )
        )
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["acc"]
        )
        model.summary()

        return model


def run(model, x_train, y_train, epochs=128, batch_size=32):
    start = time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    end = time()

    return end - start


(x_train, y_train), (_, _) = mnist.load_data()

x_train = x_train / 255.0  # Normalize x train
x_train = x_train.reshape(-1, 28, 28, 1)  # Add the cnannel axis
y_train = to_categorical(y_train, num_classes=10)  # convert to one-hot

cpu_model = conv_model("CPU", 64, 3)  # 64 filters, kernel size of 3
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices() # GET GPU DEVICE NAME IF 'GPU' DO NOT WORK
gpu_model = conv_model("DML", 64, 3)  # 64 filters, kernel size of 3 , change for

epochs = 1
bz = 16

conv_cpu_time = run(cpu_model, x_train, y_train, epochs=epochs, batch_size=bz)
conv_gpu_time = run(gpu_model, x_train, y_train, epochs=epochs, batch_size=bz)

print("Time to train on CPU for 8 epochs: {} seconds".format(conv_cpu_time))
print("Time to train on GPU for 8 epochs: {} seconds".format(conv_gpu_time))
