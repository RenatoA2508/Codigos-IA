import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


CLASS_NAMES = [
    "avion",
    "automovil",
    "ave",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camion",
]


def load_cifar10_data():
    """Carga CIFAR-10, normaliza imagenes y convierte etiquetas a one-hot."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Construye una CNN sencilla y clara para imagenes CIFAR-10 de 32x32."""
    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(epochs, batch_size, model_path):
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    model = build_model()

    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Exactitud final en test: {accuracy:.4f}")
    print(f"Perdida final en test: {loss:.4f}")

    model.save(model_path)
    print(f"Modelo guardado en: {Path(model_path).resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena una CNN con CIFAR-10 y guarda el modelo en H5."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epocas de entrenamiento.")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamano de lote.")
    parser.add_argument(
        "--model-path",
        default="cifar10_model.h5",
        help="Archivo donde se guardara el modelo entrenado.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path,
    )
