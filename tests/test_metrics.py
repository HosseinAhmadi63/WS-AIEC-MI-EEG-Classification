from __future__ import annotations

import numpy as np
import pytest

from wsaiec_eeg.evaluation.metrics import classification_metrics


def test_weighted_recall_equals_accuracy_and_auc_uses_hard_predictions(paper_config) -> None:
    y_true = np.asarray([0, 0, 0, 1, 1, 1])
    y_pred = np.asarray([0, 0, 1, 1, 0, 1])
    probabilities = np.asarray(
        [[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
    )
    metrics = classification_metrics(
        y_true, y_pred, probabilities, paper_config.section("evaluation")
    )
    assert metrics["recall"] == metrics["accuracy"]
    assert metrics["auc_roc"] == pytest.approx(2 / 3)
    assert metrics["score"] == np.mean(
        [metrics[name] for name in ["accuracy", "precision", "recall", "f1", "auc_roc", "kappa"]]
    )


def test_metrics_remain_finite_when_validation_fold_omits_a_class(paper_config) -> None:
    y_true = np.asarray([0, 0, 1, 1])
    y_pred = np.asarray([0, 1, 1, 1])
    probabilities = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    metrics = classification_metrics(
        y_true,
        y_pred,
        probabilities,
        paper_config.section("evaluation"),
        labels=np.asarray([0, 1, 2]),
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["auc_roc"] == pytest.approx(0.75)


def test_single_class_validation_fold_uses_neutral_auc_and_finite_kappa(paper_config) -> None:
    y_true = np.asarray([2, 2, 2])
    y_pred = np.asarray([2, 2, 2])
    probabilities = np.asarray([[0.0, 0.0, 1.0]] * 3)
    metrics = classification_metrics(
        y_true,
        y_pred,
        probabilities,
        paper_config.section("evaluation"),
        labels=np.asarray([0, 1, 2]),
    )
    assert metrics["auc_roc"] == 0.5
    assert metrics["kappa"] == 0.0
    assert all(np.isfinite(value) for value in metrics.values())
