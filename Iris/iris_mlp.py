

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input


DATASET_PATH = Path("Iris.csv")
RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_iris_data(csv_path: Path = DATASET_PATH) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Carga Iris.csv local. Si no existe, usa el dataset Iris incluido en scikit-learn."""
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            feature_names = [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ]
            rows = list(reader)

        x = np.array(
            [[float(row[feature]) for feature in feature_names] for row in rows],
            dtype=np.float32,
        )
        species = np.array([row["Species"] for row in rows])
        encoder = LabelEncoder()
        y = encoder.fit_transform(species).astype(np.int64)
        target_names = encoder.classes_.tolist()
        return x, y, feature_names, target_names

    iris = load_iris()
    x = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)
    return x, y, feature_names, target_names


def preprocess_data(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Divide el dataset y escala usando solo estadisticas del conjunto de entrenamiento."""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


def build_model(
    input_dim: int,
    num_classes: int,
    hidden_layers: Iterable[int] = (16, 8),
    activation: str = "relu",
    dropout_rate: float = 0.0,
) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 150,
    batch_size: int = 16,
) -> tf.keras.callbacks.History:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )
    return model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )


def evaluate_model(
    model: Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
) -> tuple[float, float]:
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    matrix = confusion_matrix(y_test, predictions)
    class_metrics = calculate_sensitivity_specificity(matrix, target_names)

    print("\nEvaluacion del modelo principal")
    print(f"Perdida en prueba: {loss:.4f}")
    print(f"Precision en prueba: {accuracy:.4f}")
    print("\nMatriz de confusion:")
    print(matrix)
    print("\nSensibilidad y especificidad por clase:")
    print(f"{'Clase':<18} {'Sensibilidad':<15} {'Especificidad':<15}")
    for metric in class_metrics:
        print(
            f"{metric['class_name']:<18} "
            f"{metric['sensitivity']:<15.4f} "
            f"{metric['specificity']:<15.4f}"
        )

    macro_sensitivity = np.mean([metric["sensitivity"] for metric in class_metrics])
    macro_specificity = np.mean([metric["specificity"] for metric in class_metrics])
    print(f"{'Promedio macro':<18} {macro_sensitivity:<15.4f} {macro_specificity:<15.4f}")
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, predictions, target_names=target_names))
    return float(loss), float(accuracy)


def calculate_sensitivity_specificity(
    matrix: np.ndarray,
    target_names: list[str],
) -> list[dict[str, float | str]]:
    metrics: list[dict[str, float | str]] = []
    total = np.sum(matrix)

    for class_index, class_name in enumerate(target_names):
        true_positive = matrix[class_index, class_index]
        false_negative = np.sum(matrix[class_index, :]) - true_positive
        false_positive = np.sum(matrix[:, class_index]) - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        sensitivity_denominator = true_positive + false_negative
        specificity_denominator = true_negative + false_positive
        sensitivity = (
            true_positive / sensitivity_denominator
            if sensitivity_denominator > 0
            else 0.0
        )
        specificity = (
            true_negative / specificity_denominator
            if specificity_denominator > 0
            else 0.0
        )

        metrics.append(
            {
                "class_name": class_name,
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
            }
        )

    return metrics


def permutation_feature_importance(
    model: Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    repeats: int = 30,
) -> list[tuple[str, float]]:
    """Calcula importancia por permutacion: caida promedio de precision al mezclar cada variable."""
    rng = np.random.default_rng(RANDOM_STATE)
    baseline_predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    importances: list[tuple[str, float]] = []

    for column_index, feature_name in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            x_permuted = x_test.copy()
            x_permuted[:, column_index] = rng.permutation(x_permuted[:, column_index])
            permuted_predictions = np.argmax(model.predict(x_permuted, verbose=0), axis=1)
            permuted_accuracy = accuracy_score(y_test, permuted_predictions)
            drops.append(baseline_accuracy - permuted_accuracy)

        importances.append((feature_name, float(np.mean(drops))))

    return sorted(importances, key=lambda item: item[1], reverse=True)


def run_architecture_experiments(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    num_classes: int,
) -> list[dict[str, object]]:
    experiments = [
        {"name": "MLP 8 relu", "layers": (8,), "activation": "relu"},
        {"name": "MLP 16-8 relu", "layers": (16, 8), "activation": "relu"},
        {"name": "MLP 32-16 relu", "layers": (32, 16), "activation": "relu"},
        {"name": "MLP 16 tanh", "layers": (16,), "activation": "tanh"},
        {"name": "MLP 16 sigmoid", "layers": (16,), "activation": "sigmoid"},
    ]
    results: list[dict[str, object]] = []

    print("\nExperimentos con arquitecturas y activaciones")
    print(f"{'Modelo':<18} {'Capas':<12} {'Activacion':<10} {'Loss':<10} {'Accuracy':<10}")

    for config in experiments:
        set_seed()
        model = build_model(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=config["layers"],
            activation=str(config["activation"]),
        )
        train_model(model, x_train, y_train)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        result = {
            "name": config["name"],
            "layers": config["layers"],
            "activation": config["activation"],
            "loss": float(loss),
            "accuracy": float(accuracy),
        }
        results.append(result)
        print(
            f"{result['name']:<18} {str(result['layers']):<12} "
            f"{result['activation']:<10} {result['loss']:<10.4f} {result['accuracy']:<10.4f}"
        )

    return results


def main() -> None:
    set_seed()
    x, y, feature_names, target_names = load_iris_data()
    x_train, x_test, y_train, y_test, _ = preprocess_data(x, y)

    print(f"Dataset cargado: {x.shape[0]} muestras, {x.shape[1]} caracteristicas")
    print(f"Clases: {', '.join(target_names)}")

    model = build_model(
        input_dim=x_train.shape[1],
        num_classes=len(target_names),
        hidden_layers=(16, 8),
        activation="relu",
    )
    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test, target_names)

    importances = permutation_feature_importance(model, x_test, y_test, feature_names)
    print("\nImportancia de caracteristicas por permutacion")
    for feature_name, importance in importances:
        print(f"{feature_name:<15} caida media de precision: {importance:.4f}")

    run_architecture_experiments(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_dim=x_train.shape[1],
        num_classes=len(target_names),
    )


if __name__ == "__main__":
    main()
