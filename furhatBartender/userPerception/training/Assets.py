import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():  # loads data into 70-20-10 split between train,validation and test data
    data = pd.read_csv("dataset.csv")
    labels = data["emotion"]
    inputs = data.drop(labels="emotion", axis=1)
    data_in, test_in, data_out, test_out = train_test_split(
        inputs, labels, test_size=0.1, stratify=labels, random_state=42
    )
    train_in, val_in, train_out, val_out = train_test_split(
        data_in, data_out, test_size=0.2 / 0.9, stratify=data_out, random_state=42
    )

    return (train_in, train_out), (val_in, val_out), (test_in, test_out)
