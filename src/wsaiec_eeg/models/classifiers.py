"""Factories for all 16 classifiers listed in Table 2."""

from __future__ import annotations

from typing import Any

import numpy as np

from wsaiec_eeg.constants import CLASSIFIER_ORDER
from wsaiec_eeg.models.mlp import TorchMLPClassifier


def _parameters(config: dict[str, Any], name: str) -> dict[str, Any]:
    parameters = dict(config[name])
    if name == "MLP":
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
    return parameters


def make_classifier(
    name: str,
    classifier_config: dict[str, Any],
    seed: int,
    n_jobs: int = 1,
) -> Any:
    """Create a fresh classifier with every stochastic source seeded."""

    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if name not in CLASSIFIER_ORDER:
        raise KeyError(f"Unknown classifier {name!r}; choose from {list(CLASSIFIER_ORDER)}")
    p = _parameters(classifier_config, name)
    factories = {
        "LDA": lambda: LinearDiscriminantAnalysis(**p),
        "LR": lambda: LogisticRegression(**p, random_state=seed),
        "PC": lambda: Perceptron(**p, random_state=seed),
        "SGD": lambda: SGDClassifier(**p, random_state=seed),
        "RC": lambda: RidgeClassifier(**p),
        "SVM": lambda: SVC(**p, random_state=seed),
        "SVM_rbf": lambda: SVC(**p, random_state=seed),
        "KN": lambda: KNeighborsClassifier(**p, n_jobs=n_jobs),
        "NB": lambda: GaussianNB(**p),
        "DT": lambda: DecisionTreeClassifier(**p, random_state=seed),
        "RF": lambda: RandomForestClassifier(**p, random_state=seed, n_jobs=n_jobs),
        "ET": lambda: ExtraTreesClassifier(**p, random_state=seed, n_jobs=n_jobs),
        "GB": lambda: GradientBoostingClassifier(**p, random_state=seed),
        "AB": lambda: AdaBoostClassifier(**p, random_state=seed),
        "QDA": lambda: QuadraticDiscriminantAnalysis(**p),
        "MLP": lambda: TorchMLPClassifier(**p, random_state=seed),
    }
    return factories[name]()


def aligned_probabilities(estimator: Any, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return class-aligned probabilities for every classifier family."""

    classes = np.asarray(classes)
    estimator_classes = np.asarray(getattr(estimator, "classes_", classes))
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    elif hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X), dtype=np.float64)
        if decision.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-np.clip(decision, -40, 40)))
            raw = np.column_stack([1.0 - positive, positive])
        else:
            shifted = decision - decision.max(axis=1, keepdims=True)
            exponent = np.exp(np.clip(shifted, -40, 40))
            raw = exponent / exponent.sum(axis=1, keepdims=True)
    else:
        predicted = np.asarray(estimator.predict(X))
        raw = np.column_stack([(predicted == label).astype(float) for label in estimator_classes])

    output = np.zeros((len(X), len(classes)), dtype=np.float64)
    for source_column, label in enumerate(estimator_classes):
        destinations = np.flatnonzero(classes == label)
        if len(destinations) != 1:
            raise RuntimeError(f"Classifier emitted unknown class {label!r}")
        output[:, int(destinations[0])] = raw[:, source_column]
    row_sums = output.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0
    if np.any(zero_rows):
        output[zero_rows] = 1.0 / len(classes)
        row_sums = output.sum(axis=1, keepdims=True)
    return output / row_sums
