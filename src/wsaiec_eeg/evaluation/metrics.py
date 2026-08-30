"""The six metrics and arithmetic composite score used in Tables 2, 3, and 8."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from wsaiec_eeg.constants import METRIC_COLUMNS


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    settings: dict[str, Any],
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the article's metrics with explicit binary/multiclass rules."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = (
        np.arange(probabilities.shape[1], dtype=np.int64)
        if labels is None
        else np.asarray(labels)
    )
    if classes.ndim != 1 or len(classes) == 0 or len(np.unique(classes)) != len(classes):
        raise ValueError("labels must be a non-empty one-dimensional unique array")
    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError(
            f"Expected probability shape {(len(y_true), len(classes))}, got {probabilities.shape}"
        )
    if not np.isin(y_true, classes).all() or not np.isin(y_pred, classes).all():
        raise ValueError("Predictions and targets must be contained in labels")
    average = settings["average_binary"] if len(classes) == 2 else settings["average_multiclass"]
    common = {
        "average": average,
        "labels": classes,
        "zero_division": int(settings["zero_division"]),
    }
    auc_probabilities = probabilities
    if settings.get("auc_input") == "hard_predictions":
        auc_probabilities = np.column_stack([(y_pred == label).astype(float) for label in classes])
    auc_values: list[float] = []
    auc_weights: list[float] = []
    for index, label in enumerate(classes):
        binary_true = (y_true == label).astype(np.int64)
        if len(np.unique(binary_true)) != 2:
            continue
        auc_values.append(float(roc_auc_score(binary_true, auc_probabilities[:, index])))
        auc_weights.append(float(binary_true.sum()))
    if not auc_values:
        auc = 0.5
    elif str(settings["average_multiclass"]) == "weighted":
        auc = float(np.average(auc_values, weights=auc_weights))
    else:
        auc = float(np.mean(auc_values))
    kappa = (
        0.0
        if len(np.unique(np.concatenate([y_true, y_pred]))) < 2
        else float(cohen_kappa_score(y_true, y_pred, labels=classes))
    )
    if not np.isfinite(kappa):
        kappa = 0.0
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **common)),
        "recall": float(recall_score(y_true, y_pred, **common)),
        "f1": float(f1_score(y_true, y_pred, **common)),
        "auc_roc": float(auc),
        "kappa": kappa,
    }
    result["score"] = float(np.mean([result[name] for name in METRIC_COLUMNS]))
    return result
