import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from shopping import evaluate
from shopping import load_data

TEST_SIZE = 0.2


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping_80_20.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])

    # Calculate accuracy with 5-fold cross-validation
    cv_model = train_model(evidence, labels)
    cv_scores = cross_val_score(cv_model, evidence, labels, cv=5, scoring="accuracy")

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Confusion matrix and related metrics
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print()
    print("Confusion Matrix:")
    print(matrix)
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print()
    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall / Sensitivity: {100 * recall:.2f}%")
    print(f"Specificity: {100 * specificity:.2f}%")
    print(f"F1 Score: {100 * f1:.2f}%")
    print()
    print("5-Fold Cross-Validation Accuracy:")
    for fold, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold}: {100 * score:.2f}%")
    print(f"Average: {100 * np.mean(cv_scores):.2f}%")
    print(f"Standard Deviation: {100 * np.std(cv_scores):.2f}%")


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


if __name__ == "__main__":
    main()
