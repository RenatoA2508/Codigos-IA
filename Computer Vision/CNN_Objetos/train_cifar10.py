import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


DATA_DIR = Path("data/cifar-10-batches-py")
MODEL_PATH = "cifar10_model.h5"


def load_batch(path):
    with path.open("rb") as file:
        batch = pickle.load(file, encoding="latin1")

    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch["labels"]).reshape(-1, 1)
    return images, labels


def load_cifar10():
    if DATA_DIR.exists():
        x_train_parts = []
        y_train_parts = []

        for number in range(1, 6):
            images, labels = load_batch(DATA_DIR / f"data_batch_{number}")
            x_train_parts.append(images)
            y_train_parts.append(labels)

        x_train = np.concatenate(x_train_parts)
        y_train = np.concatenate(y_train_parts)
        x_test, y_test = load_batch(DATA_DIR / "test_batch")
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    model = build_model()
    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_test, y_test),
    )

    model.save(MODEL_PATH)
    print(f"Modelo guardado como {MODEL_PATH}")


if __name__ == "__main__":
    main()
