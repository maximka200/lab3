"""Воспроизводимая подготовка данных, оценка моделей и запись результатов."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import compare_classical_models, train_linear_regression, train_logistic_classifier


ROOT = Path(__file__).resolve().parent
SEED = 42


def main():
    regression = pd.read_csv(ROOT / "Student_Performance.csv")
    regression["Extracurricular Activities"] = regression["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    x = regression.drop(columns="Performance Index").to_numpy(dtype=float)
    y = regression["Performance Index"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    regression_data = RegressionInput(x_train, y_train, x_test, y_test, learning_rate=0.05, epochs=800)
    regression_result = train_linear_regression(regression_data)
    regression_mse = float(np.mean((y_test - regression_result.predictions) ** 2))

    classification = pd.read_csv(ROOT / "german.csv", sep=";")
    x = classification.drop(columns="Creditability").to_numpy(dtype=float)
    y = classification["Creditability"].to_numpy(dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    classification_data = ClassificationInput(x_train, y_train, x_test, y_test, learning_rate=0.1, epochs=1000)
    own = train_logistic_classifier(classification_data)
    baselines = compare_classical_models(classification_data)

    def metrics(item):
        return {name: float(getattr(item, name)) for name in ("roc_auc", "accuracy", "precision", "recall", "f1")}

    result = {
        "linear_regression": {"test_mse": regression_mse},
        "classification": {
            "own_logistic": metrics_from_predictions(y_test, own.probabilities),
            "logistic_regression": metrics(baselines.logistic_regression),
            "decision_tree": metrics(baselines.decision_tree),
            "knn": metrics(baselines.knn),
        },
    }
    (ROOT / "experiment_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def metrics_from_predictions(y_true, probabilities):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


if __name__ == "__main__":
    main()
