"""Реализация линейной регрессии и логистической классификации градиентным спуском."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _matrix(values):
    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Признаки должны быть конечной двумерной матрицей.")
    return result


def _targets(values, n):
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) != n or not np.isfinite(result).all():
        raise ValueError("Цели должны содержать по одному конечному значению на строку.")
    return result


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучает собственную линейную регрессию пакетным градиентным спуском."""
    x = _matrix(data.train_features)
    y = _targets(data.train_target, len(x))
    test = _matrix(data.test_features)
    if test.shape[1] != x.shape[1]:
        raise ValueError("Train и test должны содержать одинаковое число признаков.")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    z_test = (test - mean) / scale
    y_mean = y.mean()
    y_scale = y.std() or 1.0
    target = (y - y_mean) / y_scale
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    history = []
    lr = float(data.learning_rate)
    for _ in range(int(data.epochs)):
        error = z @ weights + bias - target
        history.append(float(np.mean(error ** 2) * y_scale ** 2))
        weights -= lr * (2.0 / len(x)) * (z.T @ error)
        bias -= lr * 2.0 * float(error.mean())
    # Return parameters in original feature and target units.
    original_weights = weights * y_scale / scale
    original_bias = y_mean + y_scale * bias - float(mean @ original_weights)
    predictions = (z_test @ weights + bias) * y_scale + y_mean
    return RegressionResult(original_weights, float(original_bias), history, predictions)


def _sigmoid(logits):
    # Stable sigmoid for both large positive and negative values.
    output = np.empty_like(logits, dtype=float)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучает собственный бинарный классификатор пакетным градиентным спуском."""
    x = _matrix(data.train_features)
    y = _targets(data.train_target, len(x))
    test = _matrix(data.test_features)
    if test.shape[1] != x.shape[1] or not np.isin(y, [0, 1]).all():
        raise ValueError("Ожидаются совпадающие признаки и бинарные метки 0/1.")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z, z_test = (x - mean) / scale, (test - mean) / scale
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    history = []
    lr = float(data.learning_rate)
    for _ in range(int(data.epochs)):
        probabilities = _sigmoid(z @ weights + bias)
        clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
        history.append(float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))))
        error = probabilities - y
        weights -= lr * (z.T @ error) / len(x)
        bias -= lr * float(error.mean())
    probs = _sigmoid(z_test @ weights + bias)
    original_weights = weights / scale
    original_bias = bias - float(mean @ original_weights)
    return ClassificationResult(original_weights, float(original_bias), history, probs, (probs >= 0.5).astype(int))


def _metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    auc = roc_auc_score(y_true, probabilities) if np.unique(y_true).size == 2 else float("nan")
    return ClassificationMetrics(
        float(auc), float(accuracy_score(y_true, predictions)),
        float(precision_score(y_true, predictions, zero_division=0)),
        float(recall_score(y_true, predictions, zero_division=0)),
        float(f1_score(y_true, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Обучает собственную логистическую модель и sklearn baseline на одном split."""
    x, test = _matrix(data.train_features), _matrix(data.test_features)
    y = _targets(data.train_target, len(x)).astype(int)
    y_test = _targets(data.test_target, len(test)).astype(int)
    if x.shape[1] != test.shape[1] or not np.isin(np.r_[y, y_test], [0, 1]).all():
        raise ValueError("Ожидаются совпадающие признаки и бинарные метки 0/1.")
    own = train_logistic_classifier(data)
    # Standardize for distance based and linear models, using train statistics only.
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z, z_test = (x - mean) / scale, (test - mean) / scale
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression(max_iter=2000, random_state=42).fit(z, y)
    tree = DecisionTreeClassifier(random_state=42).fit(x, y)
    neighbors = KNeighborsClassifier(n_neighbors=min(5, len(x))).fit(z, y)
    own_metrics = _metrics(y_test, own.probabilities)
    logistic_metrics = _metrics(y_test, logistic.predict_proba(z_test)[:, 1])
    tree_metrics = _metrics(y_test, tree.predict_proba(test)[:, 1])
    knn_metrics = _metrics(y_test, neighbors.predict_proba(z_test)[:, 1])
    return BaselineComparisonResult(logistic_metrics, tree_metrics, knn_metrics)
